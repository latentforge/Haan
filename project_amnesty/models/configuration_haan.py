"""Haan model configuration -- subclasses the vendored Moshi config.

Haan IS a Moshi variant (shared audio embeddings + Role Token + shared batch-2
Depth, backbone swapped to Qwen3-8B), so its config inherits Moshi's config from
`project_amnesty.models.moshi` (which now imports transformers via absolute paths).

Grounded in docs/contexts/ARCHITECTURE.md:
  - 1    backbone: Helium -> Qwen3-8B (Temporal Transformer)
  - 3.3  Role Token: RoleEmb[role], one learned additive vector per role (2 roles)
  - 4.1  Mimi split-RVQ: 1 semantic + 7 acoustic, audio cardinality 2048 (frozen)
  - 5.0 / 5.4  Depth Transformer (shared across self/user, batch-2 parallel role split)

The Haan-specific fields (share_scope / init_source / num_roles / Qwen3 backbone dims)
are a TODO on top of the inherited Moshi fields.
"""

from __future__ import annotations

from project_amnesty.models.moshi import MoshiConfig, MoshiDepthConfig

__all__ = ["HaanConfig", "HaanDepthConfig"]


class HaanDepthConfig(MoshiDepthConfig):
    """Depth Transformer config (ARCHITECTURE 5.0 eq.2, 5.4).

    Inherits MoshiDepthConfig. TODO(ARCH 5.4): shared Depth + batch-2 parallel role
    split (Moshi is 16-step sequential; Haan is shared + 8-step parallel), depth_mode
    q16(train)/q8(live, 5.0.3).
    """

    model_type = "haan_depth"


class HaanConfig(MoshiConfig):
    """Top-level Haan config (ARCHITECTURE 1 / 3 / 4 / 5).

    Inherits MoshiConfig (audio_vocab_size, audio_encoder_config=Mimi,
    depth_decoder_config, backbone dims). TODO(ARCH 1/3.3/5.4.2): retarget the
    backbone dims to Qwen3-8B, set depth_decoder_config to HaanDepthConfig, and add
    Haan fields (share_scope, init_source, num_roles=2, moshi_ckpt for warm-start).
    """

    model_type = "haan"
    sub_configs = {**MoshiConfig.sub_configs, "depth_decoder_config": HaanDepthConfig}
