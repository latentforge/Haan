"""Haan model -- subclasses the vendored Moshi model.

Haan reuses Moshi's whole stack (Temporal + Depth Transformer, generation, Mimi
audio machinery) and overrides only the Haan-specific pieces:
  - shared audio embeddings (K) + Role Token, replacing Moshi's 2*K separate tables (ARCH 3.3)
  - shared Depth with batch-2 parallel role split, replacing Moshi's 16-step sequential (ARCH 5.4)
  - backbone swapped to Qwen3-8B (ARCH 1)
  - Moshi warm-start in from_pretrained (emb.8~15 -> shared, depformer/linears, adapter; ARCH 5.4.1)

Moshi classes come from `project_amnesty.models.moshi` (now standalone-importable:
its transformers-internal relative imports were rewritten to absolute `transformers.`).
The Haan overrides are still TODO; the inheritance/wiring is what's established here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from project_amnesty.models.moshi import (
    MoshiDepthDecoderModel,
    MoshiForConditionalGeneration,
    MoshiModel,
)

from project_amnesty.models.configuration_haan import HaanConfig, HaanDepthConfig

__all__ = [
    "RoleEmbedding",
    "HaanModel",
    "HaanDepthDecoder",
    "HaanForConditionalGeneration",
]


class RoleEmbedding(nn.Module):
    """Role Token (ARCHITECTURE 3.3): RoleEmb[role_id], role_id in {0=self, 1=user}.

    New in Haan -- Moshi has no role token (it uses 2*K separate embedding tables).
    Two learned additive vectors, added to the shared audio-embedding sum.
    """

    def __init__(self, dim: int, num_roles: int = 2) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, dim)

    def forward(self, hidden: torch.Tensor, role_ids: torch.Tensor) -> "Any":
        # TODO(ARCH 3.3): add self.role_emb(role_ids) to the shared audio-embedding sum.
        raise NotImplementedError("RoleEmbedding.forward is a stub (ARCHITECTURE 3.3)")


class HaanModel(MoshiModel):
    """Temporal Transformer (ARCHITECTURE 5.0 eq.1): backbone -> z_s.

    Inherits MoshiModel. TODO(ARCH 1): swap the decoder backbone to Qwen3-8B.
    """

    config_class = HaanConfig


class HaanDepthDecoder(MoshiDepthDecoderModel):
    """Depth Transformer (ARCHITECTURE 5.0 eq.2, 5.4).

    Inherits MoshiDepthDecoderModel. TODO(ARCH 5.4): shared Depth + Role Token
    injection + batch-2 parallel role split (vs Moshi's 16-step sequential).
    """

    config_class = HaanDepthConfig


class HaanForConditionalGeneration(MoshiForConditionalGeneration):
    """Top Haan module (ARCHITECTURE 5): text + self/user audio heads.

    Inherits MoshiForConditionalGeneration (and, transitively, MoshiGenerationMixin
    -> generate()). TODO(ARCH 3.3/5.4/1): in __init__ replace `embed_tokens` (2*K
    separate) with shared K + RoleEmbedding, set `depth_decoder` to HaanDepthDecoder,
    and swap `self.model` to a Qwen3-8B backbone.
    """

    config_class = HaanConfig

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "HaanForConditionalGeneration":
        # TODO(ARCH 5.4.1/5.4.2): assemble Qwen3 backbone + shared emb(K) + RoleEmb(2) + shared
        #   Depth, then warm-start from Moshi -- emb.8~15(user) -> shared audio emb, depformer body /
        #   linears.0~7 copied, depformer_in(backbone_dim->depth_dim) init-then-retrain adapter.
        #   (Overrides Moshi's from_pretrained because the warm-start is bespoke;
        #   project_amnesty/models/moshi/convert_moshi_to_hf.py has the checkpoint conversion.)
        raise NotImplementedError(
            "HaanForConditionalGeneration.from_pretrained is a stub -- "
            "Qwen3 assembly + Moshi warm-start (ARCHITECTURE 5.4.1)"
        )
