"""Haan model -- subclasses the Moshi model from `transformers`.

Haan reuses Moshi's whole stack (Temporal + Depth Transformer, generation loop, delay
pattern) and overrides the three places where the architecture actually differs:

  - the Temporal input side. Moshi holds `2 * K` separate audio embedding tables, one set
    per stream; Haan holds `K` shared tables and marks the stream with a role signal
    instead. `_embed_audio_codes` is the single hook -- both `forward` and the generation
    loop route audio embedding through it.
  - the Depth Transformer. Moshi's per-index parameters run `2 * K` wide (one index per
    predicted codebook, both streams laid end to end); Haan's run `K` wide and are shared
    across roles, with the role added onto the per-index projection of `z_s`.
  - attention. Qwen3 replaces Helium as the backbone and adds QK-Norm, so `HaanAttention`
    / `HaanDecoderLayer` / `HaanModel` override Moshi's -- but on the *backbone* only. The
    Depth Transformer keeps Moshi's plain attention.

Everything else -- the delay pattern, `generate` -- is inherited untouched.
"""

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.integrations.hub_kernels import use_kernelized_func
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.moshi.modeling_moshi import (
    MoshiAttention,
    MoshiDecoderLayer,
    MoshiDepthDecoderForCausalLM,
    MoshiDepthDecoderModel,
    MoshiFlexibleLinear,
    MoshiForConditionalGeneration,
    MoshiModel,
    MoshiRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    get_codebook_idx,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_haan import HaanConfig, HaanDepthConfig
from .generation_haan import HaanGenerationMixin

__all__ = [
    "RoleEmbedding",
    "HaanAttention",
    "HaanDecoderLayer",
    "HaanModel",
    "HaanDepthDecoderModel",
    "HaanDepthDecoderForCausalLM",
    "HaanForConditionalGeneration",
]


def _sequence_width(input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> int:
    """Sequence length of whichever input was actually given, or 0 if neither was.

    Both are checked because the depth decoder's layout guard has to hold for an `inputs_embeds`
    caller too: gating on `input_ids` alone let a `2 * K`-wide embedding sequence walk the banned
    sequential path with no error, leaking the assistant's frame-internal choice into the user
    logits.
    """
    if input_ids is not None:
        return input_ids.shape[1]
    if inputs_embeds is not None:
        return inputs_embeds.shape[1]
    return 0


class RoleEmbedding(nn.Module):
    """The role signal: one learned parameter per role.

    New in Haan -- Moshi has no role signal, because it keeps a separate embedding table
    per stream and so never needs one.

    **On `role_mode="additive"`, the literal "one learned vector per role, added on" form.**
    The Temporal input at frame `t` is a *sum* over every codebook of both streams (an
    elementwise sum, not a concat). With shared tables that is

        h_t = TextEmb(w_t) + sum_k E_k(A_self[t,k]) + sum_k E_k(A_user[t,k]) + r_self + r_user

    and swapping the two streams leaves `h_t` bit-identical: the code sums commute and
    `r_self + r_user` is a constant that does not depend on which stream is which. So an
    additive role token cannot distinguish "I said X, you said Y" from "I said Y, you
    said X" -- it does not merely encode the role more weakly than Moshi's separate tables,
    it carries exactly zero role information at the Temporal input. Adding a constant does
    not avert the collapse the shared-table design would otherwise cause.

    `role_mode="scale"` is the minimal repair that keeps everything else about the design
    -- shared tables, two learned vectors, no change to the backbone's RoPE. A per-role
    elementwise gain makes the per-stream contributions `s_self * sum_k E_k(A_self)` and
    `s_user * sum_k E_k(A_user)`, which differ under a swap whenever `s_self != s_user`.
    Both modes are kept so the role-signal ablation can measure the gap between them.

    Both modes initialise to the identity, so a Moshi warm-start reproduces the copied
    tables undisturbed at step 0. The two roles therefore start identical; under "scale",
    symmetry breaks on the first step because each role's gain multiplies a different
    stream's codes and so receives a different gradient.
    """

    def __init__(self, hidden_size: int, num_roles: int = 2, role_mode: str = "scale") -> None:
        super().__init__()
        self.role_mode = role_mode
        self.num_roles = num_roles
        if role_mode == "scale":
            self.role_scale = nn.Parameter(torch.ones(num_roles, hidden_size))
        elif role_mode == "additive":
            self.role_emb = nn.Parameter(torch.zeros(num_roles, hidden_size))
        else:
            raise ValueError(f"`role_mode={role_mode!r}` must be one of ('scale', 'additive').")

    def forward(self, hidden_states: torch.Tensor, role_ids: torch.Tensor | int) -> torch.Tensor:
        """Apply the role signal.

        Args:
            hidden_states (`torch.Tensor` of shape `(..., hidden_size)`):
                Audio embeddings for one role, or a sequence whose positions each carry a role.
            role_ids (`torch.Tensor` or `int`):
                A single role for the whole tensor, or one role per position. A per-position
                tensor of shape `(sequence_length,)` indexes to `(sequence_length, hidden_size)`,
                which broadcasts against a `(batch, sequence_length, hidden_size)` input.
        """
        parameter = self.role_scale if self.role_mode == "scale" else self.role_emb
        selected = parameter[role_ids]
        return hidden_states * selected if self.role_mode == "scale" else hidden_states + selected


@use_kernelized_func(apply_rotary_pos_emb)
class HaanAttention(MoshiAttention):
    """[`MoshiAttention`] plus Qwen3's QK-Norm.

    Haan replaces Helium with Qwen3 as the Temporal backbone, and Qwen3 differs from Helium inside
    attention in exactly one way that carries weights: it RMS-normalizes the query and the key over
    the head dimension, per head, immediately before RoPE.

        query = q_norm(q_proj(x).view(..., heads, head_dim))
        key   = k_norm(k_proj(x).view(..., heads, head_dim))

    `MoshiAttention` has no such module, so loading Qwen3 into it would drop `q_norm.weight` and
    `k_norm.weight` for every layer (72 tensors for Qwen3-8B) and run the remaining weights at a
    q/k scale they were never trained under. That is not a small numerical difference: QK-Norm
    controls the magnitude entering the softmax.

    Gated on `config.use_qk_norm` because the same class also serves the Moshi warm-start, where
    the norms must NOT be present -- an RMSNorm whose weight is all ones still rescales its input,
    so switching it on under Moshi weights would corrupt an otherwise exact transfer. The
    parameters are named `q_norm` / `k_norm` to match Qwen3's checkpoints exactly.
    """

    def __init__(self, config, layer_idx: int | None = None, use_flexible_linear: bool = False):
        super().__init__(config, layer_idx=layer_idx, use_flexible_linear=use_flexible_linear)

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            # Over the HEAD dimension only -- not hidden_size. Qwen3 uses its own RMSNorm here;
            # MoshiRMSNorm is the same function (both are T5-style) and keeps this file on one
            # norm implementation.
            self.q_norm = MoshiRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = MoshiRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        codebook_idx: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: this is `MoshiAttention.forward` verbatim except for the two `q_norm`/`k_norm`
        # lines marked below. There is no hook between the projections and RoPE, and the norm has
        # to sit exactly there, so the method is copied rather than wrapped -- keep it in sync
        # when the upstream Moshi attention changes.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states, codebook_idx).view(hidden_shape)
        key_states = self.k_proj(hidden_states, codebook_idx).view(hidden_shape)
        if self.use_qk_norm:
            # <-- the QK-Norm delta. Applied on the (..., heads, head_dim) view, before the
            #     transpose and before RoPE, exactly where Qwen3 applies it. RMSNorm reduces over
            #     the last axis, so norm-then-transpose and transpose-then-norm are identical.
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states, codebook_idx).view(hidden_shape).transpose(1, 2)

        # rotary embeddings are not used in the depth decoder, where `position_embeddings` is None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, codebook_idx)
        return attn_output, attn_weights


class HaanDecoderLayer(MoshiDecoderLayer):
    """[`MoshiDecoderLayer`] with [`HaanAttention`] in place of [`MoshiAttention`].

    Everything else -- the MLP, both RMSNorms, the residual structure -- is Moshi's. The attention
    is rebuilt on top of the parent's construction rather than in place of it, so the rest of the
    parent `__init__` stays the single source of truth (the same pattern
    [`HaanForConditionalGeneration`] uses for its embeddings).
    """

    def __init__(self, config, layer_idx: int, use_flexible_linear: bool):
        super().__init__(config, layer_idx, use_flexible_linear)
        self.self_attn = HaanAttention(config, layer_idx=layer_idx, use_flexible_linear=use_flexible_linear)


class HaanModel(MoshiModel):
    """The Haan Temporal Transformer -- [`MoshiModel`] with QK-Norm-capable attention.

    The only structural difference from Moshi is [`HaanDecoderLayer`], and that only matters when
    `config.use_qk_norm` is on (the Qwen3 backbone). With it off, this is `MoshiModel`
    parameter for parameter, which is what keeps the Moshi warm-start exact.
    """

    config: HaanConfig

    def __init__(self, config: HaanConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HaanDecoderLayer(config, layer_idx, use_flexible_linear=False) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class HaanDepthDecoderModel(MoshiDepthDecoderModel):
    """The Haan depth decoder.

    Same sequence layout as [`MoshiDepthDecoderModel`] -- position `i` is codebook `i` of one
    main-decoder timestep, both streams laid end to end -- but the per-index parameters behind
    those positions are only `codebooks_per_role` wide and shared across roles. A position's
    role is applied to the per-index projection of the main decoder's hidden state:

        z_depth[k, role] = RoleEmb[role](depformer_in[k](z_s))
    """

    config: HaanDepthConfig

    def __init__(self, config: HaanDepthConfig):
        codebooks_per_role = config.codebooks_per_role

        # The projections are one per codebook index k, each shared between the two roles. Every
        # per-index parameter here -- the input tables, `input_projections`, and each decoder layer's
        # flexible linears -- is sized by `config.num_codebooks`, which counts BOTH streams. Haan wants
        # one stream's worth, shared across roles and indexed by `codebook_idx % codebooks_per_role`.
        #
        # So `num_codebooks` is narrowed across the whole of `super().__init__` and restored after,
        # rather than letting the parent build full-width modules here that are then thrown away.
        #
        # This trims the innermost of three build-and-discard levels, not all of them.
        # `MoshiForConditionalGeneration.__init__` and `MoshiDepthDecoderForCausalLM.__init__` still
        # each build a full-width module and drop it, because both hardcode the class they construct
        # and offer no hook; removing those would mean bypassing the parent `__init__` and restating
        # its body, which drifts on every upstream change. The discarded modules are collected --
        # this is transient peak, not a leak.
        #
        # The saving is in peak construction memory: narrowing the axis sizes the input tables,
        # `input_projections`, and each layer's flexible linears to one stream's width, so the
        # full-width versions are never allocated in the first place.
        #
        # Narrowed in place, not on a `deepcopy`: a copy would leave every layer holding a *different*
        # config object than the one `forward` passes to `create_causal_mask`, and
        # `MoshiAttention.forward` reads `self.config._attn_implementation` at run time -- so the mask
        # and the attention kernel would drift apart. `set_attn_implementation` after construction
        # would then be silently ignored by the layers, `output_attentions=True` would record nothing,
        # and a flex-attention mask would reach an SDPA kernel.
        #
        # Safe because `num_codebooks` is only read while modules are being sized, never in `forward`.
        # `max_position_embeddings` is deliberately not touched: the depth *sequence* still spans both
        # streams, it is only the *parameter* axis that shrinks.
        full_num_codebooks = config.num_codebooks
        config.num_codebooks = codebooks_per_role
        try:
            super().__init__(config)
            self.role_embedding = RoleEmbedding(config.hidden_size, config.num_roles, config.role_mode)
        finally:
            config.num_codebooks = full_num_codebooks

        self.codebooks_per_role = codebooks_per_role

        # The parent's `num_codebooks - 1` input tables (here `K - 1`) are exactly right. Every row is
        # `[text, cb_0 .. cb_{K-2}]`: the last codebook of a stream is only ever a target, never fed
        # back in, so under the role-parallel layout (`_split_roles`, one stream per row) there is no
        # input for a `K`-th slot.

        self.post_init()

    def _slot_and_role(self, codebook_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a position index into (which shared per-index parameter, which role)."""
        return codebook_idx % self.codebooks_per_role, codebook_idx // self.codebooks_per_role

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        position_ids: torch.LongTensor | None = None,
        role_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens. The first element of the sequence must be the text token associated to
            the audio codebooks. The rest of the elements must be flattened audio codebooks.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the main decoder. Used to contextualize
            `input_ids`.
        role_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Which role each *row* carries, for the batch-parallel rollout. When given, the
            sequence covers one stream (`codebooks_per_role` positions) and the role comes from the batch axis.
            When omitted, both streams are laid end to end along the sequence and the role is read off the
            position instead -- the teacher-forced training layout that [`MoshiDepthDecoderModel`] uses.
        """
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        codebook_idx = get_codebook_idx(input_ids, inputs_embeds, past_seen_tokens)
        slot_idx, role_idx = self._slot_and_role(codebook_idx)
        if role_ids is not None:
            # Validated rather than trusted, because the failure is silent in one direction: indexing
            # a parameter with `-1` is legal Python and aliases to the LAST role, so an upstream
            # convention where -1 means "unknown/padding" would train and generate as role 1 with no
            # error anywhere. A row count that disagrees with the batch broadcasts just as quietly.
            batch = inputs_embeds.shape[0] if input_ids is None else input_ids.shape[0]
            if role_ids.dim() != 1 or role_ids.shape[0] != batch:
                raise ValueError(
                    f"`role_ids` must be one role per row -- shape ({batch},), got {tuple(role_ids.shape)}."
                )
            # Gate the range check to the prefill step. `role_ids` is regenerated identically for every
            # step of a cached rollout (see `HaanDepthDecoderForCausalLM.generate`), so re-checking it per
            # step buys nothing but a GPU->CPU sync. The previous `0 <= role_ids.min() and role_ids.max()
            # < N` forced TWO 0-dim-tensor `__bool__()` syncs (and a torch.compile graph break) on every
            # call -- the same per-step sync the input-embedding loop below was rewritten to remove. Fold
            # both bounds into one reduction; `and` short-circuits so the sync fires only at prefill (which
            # is every training forward, since training has no cache, and once per generated frame).
            #
            # What the gate gives up: a *different* `role_ids` handed in on a warm cache is not
            # range-checked, and `-1` is a legal index that would alias to the last role silently.
            # Nothing does that here -- the only producers are the two `torch.arange` sites below, and
            # `generate` builds one tensor per frame and reuses it for every step of that frame, so
            # checking at the frame's first step checks the tensor every later step uses. The shape
            # check above stays ungated (it is free).
            if past_seen_tokens == 0 and bool(((role_ids < 0) | (role_ids >= self.config.num_roles)).any()):
                raise ValueError(
                    f"`role_ids` must lie in [0, {self.config.num_roles}), got "
                    f"[{int(role_ids.min())}, {int(role_ids.max())}]. Negative ids silently alias to the "
                    "last role rather than raising."
                )
            # Role per row, not per position: index to `(batch, 1, hidden)` so it broadcasts over the sequence.
            role_idx = role_ids[:, None]

        if position_ids is None:
            position_ids = codebook_idx.unsqueeze(0)

        # If inputs_embeds is provided, it has the priority over input_ids, which won't be used
        if inputs_embeds is None:
            # `codebook_idx = arange(seq_len) + past_seen_tokens`, so position `local_idx` in this call
            # is absolute position `past_seen_tokens + local_idx`. Both are already Python ints, so the
            # loop control touches no tensor -- where the previous `for position_idx in codebook_idx:
            # position_idx.item()` forced a GPU->CPU sync every step (8 per training forward, one per
            # generated codebook) and a torch.compile graph break each time. Iterating `range` instead
            # is bit-identical and drops those syncs to zero. `codebook_idx` stays a tensor for
            # `_slot_and_role` / `position_ids`,
            # which index it whole and never sync per element.
            inputs_embeds = []
            for local_idx in range(input_ids.shape[1]):
                position_idx = past_seen_tokens + local_idx
                if position_idx == 0:
                    inputs_embeds.append(self.text_embed_tokens(input_ids[:, [local_idx]]))
                else:
                    # Position `p > 0` takes codebook `p - 1` as input, so the shared table is picked by the
                    # *input* codebook's slot, one behind the position's own.
                    slot = (position_idx - 1) % self.codebooks_per_role
                    inputs_embeds.append(self.embed_tokens[slot](input_ids[:, [local_idx]]))

            inputs_embeds = torch.cat(inputs_embeds, dim=1)

        # The role rides on the projection of z_s, not on the codebook embedding.
        # Out-of-place (Moshi accumulates in place) so a caller's `inputs_embeds` is never mutated.
        projected = self.role_embedding(self.input_projections(last_hidden_state, slot_idx), role_idx)
        inputs_embeds = inputs_embeds + projected

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                codebook_idx=slot_idx,
            )

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HaanDepthDecoderForCausalLM(MoshiDepthDecoderForCausalLM):
    """The Haan depth decoder with a per-codebook audio language modelling head on top.

    The heads are shared across roles the same way the body is.
    """

    config: HaanDepthConfig

    def __init__(self, config: HaanDepthConfig):
        super().__init__(config)
        self.model = HaanDepthDecoderModel(config)
        # One head per codebook of a single stream, shared across roles -- indexed with the same
        # `codebook_idx % codebooks_per_role` the body uses.
        self.codebooks_per_role = config.codebooks_per_role
        self.lm_heads = MoshiFlexibleLinear(config.hidden_size, config.audio_vocab_size, self.codebooks_per_role)
        # How many roles `generate` rolls out. Flipped by `HaanGenerationMixin.set_depth_mode`
        # (q16 for simulation -- both roles; q8 for live conversation -- self only).
        self.num_generation_roles = config.num_roles
        self.post_init()

    @torch.no_grad()
    def generate(self, last_hidden_state=None, input_ids=None, **kwargs):
        """Roll out one frame with the roles in parallel.

        Moshi walks all `2 * K` codebooks as one autoregressive sequence, so the user codebooks are
        conditioned on the assistant's within the frame. Haan instead stacks the roles on the
        **batch** axis and walks `K` steps, which is both ~2x shorter and the factorization
        `p(self_t | z_s) * p(user_t | z_s)`: each role reads the same context and neither sees the
        other's frame-internal choice.

        The return layout is deliberately Moshi's -- `(batch, 1 + roles * K)`, the text token
        followed by the assistant's codebooks then the user's. `MoshiGenerationMixin` consumes it
        from two call sites and both keep working untouched.
        """
        roles = self.num_generation_roles
        codebooks = self.codebooks_per_role
        batch = input_ids.shape[0]

        # One row per (role, sequence) pair. `repeat_interleave` on the role index pairs with
        # `repeat` (tiling) on the rows below: rows `[0, batch)` are role 0, `[batch, 2 * batch)`
        # role 1 -- the order the reshape at the end unstacks.
        role_ids = torch.arange(roles, device=input_ids.device).repeat_interleave(batch)

        # How long the rollout is, is architecture, not a caller preference: exactly one step per
        # codebook of one stream plus the leading text token. Every knob that would change the step
        # count is dropped, not just the two `MoshiGenerationMixin.generate` happens to set -- it
        # sizes them for Moshi's `2 * K`-long sequence, and a survivor silently wins over the values
        # passed below and lands as an unreadable reshape error.
        for knob in ("min_length", "max_length", "min_new_tokens", "max_new_tokens", "max_time", "stop_strings"):
            kwargs.pop(knob, None)
        # These change the number of *rows* instead, which the role unstacking cannot absorb.
        for knob in ("num_return_sequences", "num_beams"):
            if kwargs.get(knob, 1) != 1:
                raise ValueError(
                    f"`{knob}` is not supported on the depth decoder: its rows are the role axis, "
                    f"so multiplying them would make role and sample indistinguishable. "
                    f"Pass it to the main `generate` instead."
                )

        outputs = super().generate(
            last_hidden_state=last_hidden_state.repeat(roles, 1, 1),
            input_ids=input_ids.repeat(roles, 1),
            role_ids=role_ids,
            min_length=codebooks + 1,
            max_length=codebooks + 1,
            **kwargs,
        )

        # Fail here rather than in the reshape, which reports only a shape and names nothing.
        if outputs.shape != (roles * batch, codebooks + 1):
            raise RuntimeError(
                f"the depth rollout produced {tuple(outputs.shape)}, expected "
                f"{(roles * batch, codebooks + 1)} ({roles} roles x {batch} rows, {codebooks} codebooks + "
                "the text token). Something overrode the rollout length or the batch size."
            )

        # (roles * batch, 1 + K) -> text token once, then each role's codebooks end to end.
        codes = outputs[:, 1:].reshape(roles, batch, codebooks)
        return torch.cat([outputs[:batch, :1], *codes], dim=1)

    def _split_roles(self, input_ids, labels, last_hidden_state):
        """Re-lay Moshi's sequential depth batch as the role-parallel one.

        `MoshiForConditionalGeneration.forward` builds one row per frame holding both streams end to
        end -- `[text, cb_0 .. cb_{roles*K-2}]`, `roles * K` wide. Walked as a single causal sequence
        that trains `p(user | z_s, self)`: the user positions attend to the assistant codebooks
        sitting in front of them. Generation samples `p(user | z_s)`, so left alone the user head
        would be trained on conditioning it never gets at inference.

        This turns that one row into `roles` rows of `K`, each `[text, cb_{r,0} .. cb_{r,K-2}]`, and
        tags them by role. Row order is role-major, matching `generate`. Returns `None` when the input
        is not that layout (already role-parallel, single-role, or a caller passing something else),
        in which case the sequential path runs unchanged.
        """
        roles, codebooks = self.config.num_roles, self.codebooks_per_role
        if roles < 2 or input_ids is None or input_ids.shape[1] != self.config.num_codebooks:
            return None
        if last_hidden_state is None or (labels is not None and labels.shape[1] != self.config.num_codebooks):
            return None

        text, audio_in = input_ids[:, :1], input_ids[:, 1:]
        frames = input_ids.shape[0]
        # Role `r` owns codebooks `[r*K, (r+1)*K)`. Its inputs are that stream's text token followed by
        # its own first `K-1` codebooks -- the last one is only ever a target, never an input.
        rows = [torch.cat([text, audio_in[:, r * codebooks : (r + 1) * codebooks - 1]], dim=1) for r in range(roles)]
        stacked_ids = torch.cat(rows, dim=0)

        stacked_labels = None
        if labels is not None:
            stacked_labels = labels.view(frames, roles, codebooks).transpose(0, 1).reshape(-1, codebooks)

        role_ids = torch.arange(roles, device=input_ids.device).repeat_interleave(frames)
        return stacked_ids, stacked_labels, last_hidden_state.repeat(roles, 1, 1), role_ids

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        role_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens. The first element of the sequence must be the text token associated to
            the audio codebooks. The rest of the elements must be flattened audio codebooks.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the main decoder. Used to contextualize
            `input_ids`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the audio language modeling loss. Indices should either be in
            `[0, ..., config.audio_vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to
            `-100` are ignored (masked).
        """
        # Teacher-forced training arrives in Moshi's sequential layout -- one `roles * K`-long row per
        # frame, both streams end to end. Re-lay it as one `K`-long row per (frame, role) so training
        # uses the same factorization generation does; see `_split_roles`. Skipped
        # when the caller already supplies `role_ids` (the `generate` path, already role-parallel).
        stacked = None
        if role_ids is None and inputs_embeds is None and past_key_values is None:
            stacked = self._split_roles(input_ids, labels, last_hidden_state)
        if stacked is not None:
            # The split changes both axes -- `frames` rows of `2K` become `roles * frames` rows of
            # `K` -- so anything indexed by the OLD layout no longer describes this batch. Neither is
            # restacked: measured, a caller mask is dropped bit-for-bit (blanking half the sequence
            # left the logits identical to no mask at all), and a stale `position_ids` likewise. Both
            # reach here from the public API as `depth_decoder_attention_mask` / `_position_ids`.
            #
            # Rejected rather than restacked because no caller needs them: the depth sequence is
            # dense and causal over codebooks, Moshi's own path never passes either, and inventing a
            # remap would add surface that nothing exercises.
            for name, value in (("attention_mask", attention_mask), ("position_ids", position_ids)):
                if value is not None:
                    raise ValueError(
                        f"`{name}` cannot be combined with the role-parallel depth layout: the rows and the "
                        f"sequence length both change under the role split, so a {tuple(value.shape)} tensor "
                        f"built for the caller's layout would silently not apply. Drop it, or pass per-role "
                        "rows with `role_ids` and size it to those."
                    )
            input_ids, labels, last_hidden_state, role_ids = stacked
        elif (width := _sequence_width(input_ids, inputs_embeds)) > self.codebooks_per_role:
            # Anything wider than ONE stream that could not be re-laid. Rejected rather than run,
            # because the sequential walk it would fall back to is no longer a supported layout: it
            # trains `p(user | z_s, self)` where generation samples `p(user | z_s)`, and it indexes
            # an input table for slot `K - 1` that no longer exists (see `__init__`). Both failures
            # are quiet -- the first trains the wrong conditional, the second raises an `IndexError`
            # naming nothing.
            #
            # The bound is `> codebooks_per_role`, not `== num_codebooks`: guarding only the exact
            # `2K` width left every width between `K + 1` and `2K - 1` (a miscounted collator, say)
            # falling through to that same `IndexError`. Widths at or below `K` are legitimate --
            # a role-parallel row, or a single step during incremental decoding.
            reason = (
                "`role_ids` was supplied" if role_ids is not None
                else "`inputs_embeds` was supplied" if inputs_embeds is not None
                else "`past_key_values` was supplied" if past_key_values is not None
                else "`last_hidden_state` is missing, or `labels` does not match the input width"
            )
            raise ValueError(
                f"a {width}-wide depth sequence cannot be used here: {reason}, so it could not be re-laid as "
                f"one {self.codebooks_per_role}-wide row per role. Haan's depth decoder runs the roles in "
                "parallel over one stream each; pass either the full sequential batch on "
                "its own (`input_ids` + `last_hidden_state` + matching `labels`, no `role_ids`), or per-role "
                "rows together with `role_ids`."
            )

        # Same indices the body uses, computed before the backbone call advances `past_key_values`.
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        slot_idx, _ = self.model._slot_and_role(get_codebook_idx(input_ids, inputs_embeds, past_seen_tokens))

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            role_ids=role_ids,
            **kwargs,
        )

        logits = self.lm_heads(outputs.last_hidden_state, slot_idx)

        loss = None
        if labels is not None:
            logits = logits.float()
            labels = labels.masked_fill(labels == self.config.audio_vocab_size, -100).reshape(-1)
            labels = labels.to(logits.device)
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.config.audio_vocab_size), labels)

        if stacked is not None:
            # Hand the caller back the layout it passed in -- `(frames, roles * K, vocab)`, both streams
            # end to end -- so `MoshiForConditionalGeneration.forward` and everything reading
            # `audio_logits` is unaffected by the restructuring above.
            roles, frames = self.config.num_roles, logits.shape[0] // self.config.num_roles
            logits = logits.view(roles, frames, -1, logits.shape[-1]).transpose(0, 1).reshape(
                frames, -1, logits.shape[-1]
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HaanForConditionalGeneration(HaanGenerationMixin, MoshiForConditionalGeneration):
    """The Haan model, for full-duplex speech-to-speech.

    Shared audio embeddings + role signal on the Temporal side, and a role-shared depth
    decoder. The delay pattern and the Mimi wiring are [`MoshiForConditionalGeneration`]'s,
    untouched; the generation-side deltas live in [`HaanGenerationMixin`], which is listed
    first so its hooks win over Moshi's.
    """

    config: HaanConfig
    # The BACKBONE's layers are `HaanDecoderLayer` (carrying QK-Norm); the depth decoder's
    # are still `MoshiDecoderLayer`, and Moshi's own declaration covers those. transformers unions
    # the two -- `PreTrainedModel.__init__` walks the submodules and `update()`s their sets -- so
    # the effective value here is `{HaanDecoderLayer, MoshiDecoderLayer}`. This ADDS to the parent
    # list; replacing it would make the depth layers splittable again.
    #
    # Only accelerate reads this (`from_pretrained(device_map=...)`); the FSDP2 path in
    # `utils/train.py` matches by attribute path instead.
    _no_split_modules = ["HaanDecoderLayer"]

    def __init__(self, config: HaanConfig):
        super().__init__(config)

        # `K` shared audio tables, where Moshi builds `2 * K` separate ones. Rebuilt on top of
        # the parent's construction rather than in place of it, so the rest of the parent `__init__` (the
        # depth decoder wiring, `predicts_user_stream`, the head) stays the single source of truth.
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(config.num_codebooks)]
        )
        self.role_embedding = RoleEmbedding(config.hidden_size, config.num_roles, config.role_mode)
        self.depth_decoder = HaanDepthDecoderForCausalLM._from_config(config.depth_decoder_config)

        # `MoshiForConditionalGeneration.__init__` hardcodes `self.model = MoshiModel(config)`, so the
        # backbone has to be replaced here too -- subclassing `MoshiModel` is not enough to get it used.
        # With `HaanDecoderLayer` carrying QK-Norm, leaving the parent's `MoshiModel` in place would
        # leave the norm silently absent from every layer while `config.use_qk_norm` reported True.
        self.model = HaanModel(config)

        self.post_init()

    def forward(self, *, input_ids=None, assistant_audio_codes=None, user_audio_codes=None, inputs_embeds=None, **kwargs):
        """[`MoshiForConditionalGeneration.forward`], plus a text-only path for the anchor batches.

        A small pure-text loss runs for the whole schedule -- it is the only guard against the Qwen3
        backbone forgetting its multilingual ability, and `configs/data/loader.yaml` pins it at 5%
        with a hard floor because if it silently drops out the later cross-lingual transfer is void.
        Those batches carry NO audio at all: `datasets/text_collator.py` refuses to fabricate silence
        to match shapes, since a 2048-token anchor would mean ~33k invented codes per sample.

        Moshi cannot take them. Its `forward` runs `torch.cat([assistant_audio_codes, user_audio_codes])`
        unconditionally, so two `None`s raise `TypeError` before reaching the `if audio_codes is not None`
        branch that would have handled them. Rather than copy the method to move one line, this embeds
        the text itself and hands the result over as `inputs_embeds`, which the parent already treats as
        taking priority over both other inputs -- the audio branch is then skipped rather than repaired.

        Going through the parent's `forward` (rather than reaching for `self.model` from the trainer) is
        what keeps this correct under FSDP2: `fully_shard` hangs its all-gather on the ROOT module's
        forward hook, so a text-only batch arriving first would otherwise read never-gathered
        `lm_head` / `embed_tokens` shards.
        """
        # Both streams together, or neither. Exactly one present falls through to Moshi's unconditional
        # `torch.cat([assistant_audio_codes, user_audio_codes])`, which raises deep in the parent with
        # `expected Tensor ... got NoneType` -- naming neither which stream is missing nor that the two
        # are a pair. Fail here with that context instead.
        if (assistant_audio_codes is None) != (user_audio_codes is None):
            missing, present = (
                ("user_audio_codes", "assistant_audio_codes")
                if user_audio_codes is None
                else ("assistant_audio_codes", "user_audio_codes")
            )
            raise ValueError(
                f"`{present}` was given but `{missing}` is None. The assistant and user streams must be "
                "passed together (both present) or both omitted (a text-only anchor batch); a single "
                "stream is not a supported layout."
            )

        if inputs_embeds is None and assistant_audio_codes is None and user_audio_codes is None:
            if input_ids is None:
                raise ValueError("provide `input_ids`, `inputs_embeds`, or the audio codes.")
            inputs_embeds = self.model.embed_tokens(input_ids)
            input_ids = None  # the parent ignores it once `inputs_embeds` is set; do not imply otherwise

        return super().forward(
            input_ids=input_ids,
            assistant_audio_codes=assistant_audio_codes,
            user_audio_codes=user_audio_codes,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
