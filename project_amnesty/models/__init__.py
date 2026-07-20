"""Haan model package. Grounded in ARCHITECTURE.md 1/3/5.

The classes subclass the real transformers base classes (PretrainedConfig /
PreTrainedModel / GenerationMixin / ProcessorMixin) -- transformers is a declared
dependency (pyproject.toml: latentforge/transformers-moshi fork). The configs are
functional; the modeling forward / warm-start / processor bodies are still TODO and
raise NotImplementedError. Exported so utils/ can import
`from project_amnesty.models import HaanForConditionalGeneration`.
"""

from project_amnesty.models.configuration_haan import HaanConfig, HaanDepthConfig
from project_amnesty.models.modeling_haan import (
    HaanDepthDecoder,
    HaanForConditionalGeneration,
    HaanModel,
    RoleEmbedding,
)
from project_amnesty.models.processing_haan import HaanProcessor

__all__ = [
    "HaanConfig",
    "HaanDepthConfig",
    "HaanModel",
    "HaanDepthDecoder",
    "RoleEmbedding",
    "HaanForConditionalGeneration",
    "HaanProcessor",
]
