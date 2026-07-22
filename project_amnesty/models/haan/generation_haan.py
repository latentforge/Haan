"""Haan generation -- subclasses [`MoshiGenerationMixin`].

Mirrors the layout of `transformers/models/moshi/generation_moshi.py`: the audio
embedding hook and everything about how a frame is rolled out live here, not in
`modeling_haan.py`.

Two things differ from Moshi, and nothing else is touched -- the outer loop, the delay
pattern, the user-stream bookkeeping and `generate` itself are all inherited:

  - `_embed_audio_codes`: `K` shared tables + a role signal, where Moshi indexes `2 * K`
    separate ones. This is the only place audio codes become embeddings, for both
    training and generation.
  - the depth rollout runs the two roles as a **batch of 2 over `K` steps** instead of one
    sequential `2 * K`-step rollout, and `set_depth_mode` drops the user role entirely for
    live conversation.

The batch-2 rollout itself is implemented in `HaanDepthDecoderForCausalLM.generate`
(`modeling_haan.py`). That is deliberate: `MoshiGenerationMixin` calls
`self.depth_decoder.generate(...)` from two separate places -- once per step in
`prepare_inputs_for_generation`, once more for the trailing frame in `generate` -- and
both consume the result as `(batch, 1 + roles * K)`, text token first. Reassembling into
that exact layout inside the depth decoder keeps both call sites working untouched,
which is why neither of those two long methods is overridden here.
"""

import torch
from transformers.models.moshi.generation_moshi import MoshiGenerationMixin

__all__ = ["HaanGenerationMixin"]


class HaanGenerationMixin(MoshiGenerationMixin):
    """Generation loop for [`HaanForConditionalGeneration`]."""

    def _embed_audio_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Sum the per-codebook audio embeddings, marking each stream with its role.

        `audio_codes` spans both streams, the assistant's codebooks followed by the user's (the
        layout `_split_predicted_streams` documents). Moshi reads that straight off `2 * K` tables;
        Haan folds it onto `K` shared tables -- codebook `c` uses table `c % K` -- and recovers the
        stream (the role) from `c // K`.

        Like Moshi's, each embedding's output is moved to a single device before summing, so a
        `ModuleList` split across a device map still works.
        """
        target_device = self.embed_tokens[0].weight.device
        codebooks_per_role = self.num_codebooks

        # Both streams are laid end to end, each `codebooks_per_role` (K) wide, so the only valid width is
        # `num_roles * K`. The mis-widths are silent otherwise: `role = c // K` folds a width in `(K, 2*K)`
        # onto role 1 with no error (a miscounted collator quietly training the wrong stream), and a width
        # of `2*K` or more indexes a role the `RoleEmbedding` has no row for, raising an `IndexError` that
        # names neither this tensor nor `audio_codes`. The depth decoder guards the identical hazard
        # (`HaanDepthDecoderModel.forward`); the Temporal side must not be the quiet one.
        num_roles = self.config.num_roles
        width = audio_codes.shape[1]
        if width != num_roles * codebooks_per_role:
            raise ValueError(
                f"`audio_codes` is {width} codebooks wide, but must be `num_roles * codebooks_per_role` "
                f"= {num_roles} * {codebooks_per_role} = {num_roles * codebooks_per_role}: the assistant's "
                f"{codebooks_per_role} codebooks followed by the user's. A width in "
                f"({codebooks_per_role}, {num_roles * codebooks_per_role}) would silently fold onto the user "
                "role rather than raise."
            )

        embeds = None
        for codebook in range(audio_codes.shape[1]):
            slot, role = codebook % codebooks_per_role, codebook // codebooks_per_role
            codebook_embeds = self.embed_tokens[slot](audio_codes[:, codebook]).to(target_device)
            codebook_embeds = self.role_embedding(codebook_embeds, role)
            embeds = codebook_embeds if embeds is None else embeds + codebook_embeds

        return embeds

    def set_depth_mode(self, mode: str) -> None:
        """Switch the depth decoder between its two rollout modes.

        | mode           | predicts          | depth batch | use                                                |
        |----------------|-------------------|-------------|----------------------------------------------------|
        | `"live"`       | self only (q8)    | 1           | real conversation -- the user stream is real input |
        | `"simulation"` | self + user (q16) | 2           | offline eval, synthetic dialogue                   |

        The user prediction is a training target, but in a live conversation the actual user audio is
        used instead and the prediction is thrown away. Rolling it out anyway costs a second batch
        element for an output nobody reads, so `"live"` skips it outright.

        Training always runs both roles -- it goes through `forward`, not `generate`, and does not
        consult this switch.
        """
        if mode not in ("live", "simulation"):
            raise ValueError(f"`mode={mode!r}` must be 'live' or 'simulation'.")

        roles = 1 if mode == "live" else self.config.num_roles
        self.depth_decoder.num_generation_roles = roles
        # `_split_predicted_streams` slices the user half off the depth decoder's output, so it has to
        # agree with what was actually rolled out. Left alone, "live" would hand it a self-only tensor
        # and it would slice an empty user stream out of thin air.
        self.predicts_user_stream = roles > 1
