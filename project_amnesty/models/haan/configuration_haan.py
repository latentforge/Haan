from huggingface_hub.dataclasses import strict
from transformers.models.moshi.configuration_moshi import MoshiConfig, MoshiDepthConfig

__all__ = ["HaanConfig", "HaanDepthConfig"]


# Why a role signal is needed at all: with shared tables, self and user are summed into one input
# vector, so without a signal code 42 from either stream reaches the transformer as the same vector
# and the two speakers are indistinguishable. `role_mode` selects how that signal is applied.
#
#   "scale"    -- per-role elementwise gain on each audio embedding (default)
#   "additive" -- one learned vector per role, added to the embedding sum
#
# "additive" is degenerate **on the Temporal side only**, and is kept there so the role-signal
# ablation can run it. See `RoleEmbedding` in modeling_haan.py for the derivation; the short version
# is that adding a per-role constant to a sum over both streams leaves that sum invariant under
# swapping the two streams, so it carries zero role information there.
#
# The Depth Transformer is a different situation and takes a different default. There the role is
# applied to the per-index projection `Proj_k(z_s)` of a row that carries exactly ONE stream (the
# batch-2 layout, one role per row), not to a sum over both -- so distinct role vectors give
# distinct rows and nothing collapses. The depth step is written additively,
#
#     z_depth[k, role] = depformer_in_k(z_s) + RoleEmb[role]
#
# and additive is preferred there. So `role_mode` (Temporal) defaults to "scale" and
# `depth_role_mode` defaults to "additive": the degeneracy repair is applied where the degeneracy
# is, and the additive form is used where it is well-posed.
ROLE_MODES = ("scale", "additive")


@strict
class HaanDepthConfig(MoshiDepthConfig):
    """Depth Transformer config.

    `num_codebooks` keeps its Moshi meaning -- how many codebooks the depth decoder
    *predicts*, both streams included (`2 * K`). What Haan changes is that the per-index
    parameters behind those positions are only `K` wide and shared across the two roles,
    so `num_codebooks` must divide evenly by `num_roles`.
    """

    model_type = "haan_depth"

    num_roles: int = 2
    # The depth step `depformer_in_k(z_s) + RoleEmb[role]`. Additive is well-posed here -- each row
    # carries one stream, so the Temporal-side degeneracy does not apply. Set from
    # `HaanConfig.depth_role_mode`, which is a separate knob from the Temporal `role_mode`.
    role_mode: str = "additive"

    @property
    def codebooks_per_role(self) -> int:
        """Width of one audio stream -- the size of the shared per-index parameter axis."""
        return self.num_codebooks // self.num_roles

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if self.role_mode not in ROLE_MODES:
            raise ValueError(f"`role_mode={self.role_mode!r}` must be one of {ROLE_MODES}.")
        if self.num_roles < 1:
            raise ValueError(f"`num_roles={self.num_roles}` must be at least 1.")
        if self.num_codebooks % self.num_roles:
            raise ValueError(
                f"`num_codebooks={self.num_codebooks}` must be divisible by `num_roles={self.num_roles}`: the "
                "depth decoder's per-index parameters are shared across roles, so every role must cover the same "
                "number of codebooks."
            )


@strict
class HaanConfig(MoshiConfig):
    """Top-level Haan config.

    `num_codebooks` is Moshi's: the width of a *single* audio stream (`K`, 8 for Mimi).
    Haan holds `K` shared audio embedding tables rather than Moshi's `2 * K` separate
    ones, and tells the two streams apart with the role signal instead.
    """

    model_type = "haan"
    # Resolved through `sub_configs` so the depth config is built as a `HaanDepthConfig`
    # rather than silently falling back to Moshi's.
    sub_configs = {**MoshiConfig.sub_configs, "depth_decoder_config": HaanDepthConfig}

    num_roles: int = 2
    # Temporal-side role signal. "scale" because the Temporal input sums both streams and an additive
    # tag cancels out of that sum -- see `ROLE_MODES` above.
    role_mode: str = "scale"
    # Depth-side role signal. Deliberately a separate knob, and deliberately a different default: the
    # depth rows carry one stream each, so nothing collapses and the additive form holds. Mirrored
    # onto `depth_decoder_config['role_mode']`.
    depth_role_mode: str = "additive"

    # Full causal attention, where Moshi defaults to a 3000-frame sliding window.
    #
    # This overrides `MoshiConfig.sliding_window = 3000`:
    #
    #   - Moshi's ~5 minute context limit is a teacher weakness Haan must not inherit. Keeping the
    #     window would transplant that limit *structurally* -- by construction, not through KD -- so
    #     no amount of checking the trained student afterwards could find it absent.
    #   - A mismatch between the Qwen3 backbone's sequence handling and Moshi's frame-rate / window
    #     assumptions is the kind of bug that happens silently at code level. This is exactly that
    #     mismatch: Qwen3 is full-attention over 40960 positions.
    #
    # `MoshiModel.forward` selects
    # `create_sliding_window_causal_mask` whenever this is non-None, and `DynamicCache` builds
    # window layers that physically evict KV past the window. But a sliding window of W is
    # bit-identical to full causal for any sequence of length <= W, and training runs at 750 frames
    # (60 s), so every test passes and the divergence would appear only in conversations past 240 s.
    #
    # Also note this must be correct BEFORE the first cache is built -- mutating the config later
    # leaves already-constructed cache layers sliding.
    sliding_window: int | None = None

    # QK-Norm, the one weight-carrying difference the Helium -> Qwen3 backbone substitution adds
    # inside attention.
    #
    # Qwen3 RMS-normalizes the query and key over the head dimension, per head, immediately before
    # RoPE, and ships a `q_norm`/`k_norm` weight per layer. Moshi (Helium) does not, and
    # `MoshiAttention` has nowhere to put those tensors -- so a Qwen3 backbone loaded into a stock
    # Moshi attention would silently drop them and run at a q/k scale the weights were never
    # trained under. `HaanAttention` adds them; this flag says whether they exist.
    #
    # Defaults to True: Qwen3 is Haan's backbone, so QK-Norm is Haan's normal shape and the default
    # should describe Haan rather than its Moshi parent.
    #
    # Set it False ONLY for a Moshi-backbone build (a Moshi-backbone baseline, or a Moshi-only
    # warm-start). An RMSNorm whose weight is all ones still rescales its input, so leaving it on
    # under Moshi backbone weights corrupts an otherwise exact transfer, and turning it off under
    # Qwen3 weights discards two tensors per layer.
    use_qk_norm: bool = True

    def __post_init__(self, **kwargs):
        # `depth_role_mode` is mirrored onto the depth sub-config's `role_mode` -- NOT the Temporal `role_mode`,
        # which describes a different mechanism in a different place (see `ROLE_MODES`). `num_roles` is *derived*
        # rather than mirrored, because the word means two different things on the two configs:
        #
        #   here          how many streams share the audio tables and the Depth parameters -- 2
        #   depth config  how many streams the depth decoder actually *predicts* -- 1 (live: self only)
        #                 or 2 (training/simulation: self and user)
        #
        # Those coincide when the depth decoder predicts both streams, and they do not when it predicts only
        # its own (Moshi's released layout, `predict_user_stream=False` in the warm-start). Mirroring `2` onto a
        # self-only depth decoder made `codebooks_per_role` come out as K/2, which silently split one stream in
        # half and labelled its upper codebooks role=1 -- no error, just self cb4..7 treated as the user.
        # Deriving it from the codebook counts makes that unrepresentable.
        #
        # `None` is normalised to `{}` first, exactly as MoshiConfig does for the same reason: left as `None` the
        # depth decoder silently keeps `HaanDepthConfig`'s own defaults instead of what was asked for. An already-built sub-config is normalised back to a dict so
        # it takes the same path -- otherwise it skips this block entirely and `save_pretrained` /
        # `from_pretrained` (which always goes through the dict path) would silently *change* the architecture.
        if self.depth_decoder_config is None:
            self.depth_decoder_config = {}
        elif not isinstance(self.depth_decoder_config, dict):
            self.depth_decoder_config = self.depth_decoder_config.to_dict()
            self.depth_decoder_config.pop("model_type", None)

        # Moshi defaults this to `2 * num_codebooks`; the number of streams is Haan's to decide.
        self.depth_decoder_config.setdefault("num_codebooks", self.num_roles * self.num_codebooks)
        predicted = self.depth_decoder_config["num_codebooks"]
        self.depth_decoder_config.update(
            {
                "role_mode": self.depth_role_mode,
                "num_roles": max(predicted // self.num_codebooks, 1) if self.num_codebooks else self.num_roles,
            }
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        for field in ("role_mode", "depth_role_mode"):
            value = getattr(self, field)
            if value not in ROLE_MODES:
                raise ValueError(f"`{field}={value!r}` must be one of {ROLE_MODES}.")

        # `num_roles` is not a free knob on the Temporal side: `MoshiForConditionalGeneration.forward`
        # unconditionally builds the audio input as `cat([assistant_audio_codes, user_audio_codes])`, and
        # `generation_haan._embed_audio_codes` reads the role back as `codebook // num_codebooks`. Both are
        # hard-wired to exactly two streams. `num_roles=1` used to validate, and then died on the first
        # forward with `IndexError: index 1 is out of bounds` from `RoleEmbedding` -- a config-shaped bug
        # surfacing as a runtime crash. Rejected here instead.
        if self.num_roles != 2:
            raise ValueError(
                f"`num_roles={self.num_roles}` must be 2: the Temporal input is built by concatenating the "
                "assistant and user streams and the role is recovered from the codebook index, so the "
                "self/user pair is structural rather than configurable."
            )

        # `num_codebooks` is a stream width, so it has to be positive. Without this, `num_codebooks=0`
        # validated cleanly (the axis check below compares 0 != 0 and passes) and `-4` produced
        # `codebooks_per_role=-4` and `max_position_embeddings=-7`, both of which survived a
        # save_pretrained/from_pretrained round trip.
        if self.num_codebooks < 1:
            raise ValueError(f"`num_codebooks={self.num_codebooks}` must be at least 1 (it is a stream width).")

        # The real invariant behind the depth decoder's shared parameter axis: one role covers exactly one
        # stream. `HaanDepthConfig` can only check that its own `num_codebooks` divides by its own
        # `num_roles`, which is a weaker proxy. In practice `__post_init__` derives the depth's `num_roles`
        # from the codebook counts, so the only configuration that actually trips this is a depth
        # `num_codebooks` that is not a positive multiple of the parent's.
        depth = self.depth_decoder_config
        if depth is not None and not isinstance(depth, dict) and depth.codebooks_per_role != self.num_codebooks:
            raise ValueError(
                f"the depth decoder's shared parameter axis is {depth.codebooks_per_role} wide but one audio "
                f"stream is {self.num_codebooks} codebooks (`num_codebooks`). Those must match: the axis is "
                "shared across roles, so each role has to cover exactly one stream. Set "
                f"`depth_decoder_config['num_codebooks']` to a multiple of {self.num_codebooks} "
                f"(got {depth.num_codebooks})."
            )
