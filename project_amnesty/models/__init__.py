"""Model definitions.

Two subpackages:
  haan/   the Haan model -- Moshi subclasses carrying the Haan deltas
          (shared audio embeddings + role signal, role-shared Depth Transformer)
  moshi/  a thin re-export of the stock Moshi classes from `transformers`
          (the pinned fork, pyproject.toml: latentforge/transformers-moshi)

Re-exported here so callers can use the short path,
`from project_amnesty.models import HaanForConditionalGeneration`.

`utils/` deliberately imports the long path (`from project_amnesty.models.haan import ...`)
instead: with two subpackages under this one, the long path says which model is meant at the
import site, and it keeps the training/eval entry points independent of this re-export layer.
"""

from project_amnesty.models.haan import (
    HaanConfig,
    HaanDepthConfig,
    HaanDepthDecoderForCausalLM,
    HaanDepthDecoderModel,
    HaanForConditionalGeneration,
    HaanGenerationMixin,
    HaanModel,
    HaanProcessor,
    RoleEmbedding,
)

__all__ = [
    "HaanConfig",
    "HaanDepthConfig",
    "HaanModel",
    "HaanDepthDecoderModel",
    "HaanDepthDecoderForCausalLM",
    "RoleEmbedding",
    "HaanGenerationMixin",
    "HaanForConditionalGeneration",
    "HaanProcessor",
]
