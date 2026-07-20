# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vendored transformers Moshi model, made standalone.

Originally lived at transformers/src/transformers/models/moshi, so its modules
used transformers-internal relative imports (`from ...configuration_utils`, ...).
Now that it lives under project_amnesty/models/moshi, those are rewritten to
absolute `transformers.` imports, and the transformers `_LazyModule` machinery is
replaced with plain eager imports so `project_amnesty.models.moshi` imports on its
own. Haan (project_amnesty/models/*_haan.py) subclasses the classes re-exported here.
"""

from .configuration_moshi import MoshiConfig, MoshiDepthConfig
from .generation_moshi import MoshiGenerationMixin
from .modeling_moshi import (
    MoshiDepthDecoderForCausalLM,
    MoshiDepthDecoderModel,
    MoshiForCausalLM,
    MoshiForConditionalGeneration,
    MoshiModel,
    MoshiPreTrainedModel,
)
from .processing_moshi import MoshiProcessor

__all__ = [
    "MoshiConfig",
    "MoshiDepthConfig",
    "MoshiGenerationMixin",
    "MoshiPreTrainedModel",
    "MoshiModel",
    "MoshiForCausalLM",
    "MoshiDepthDecoderModel",
    "MoshiDepthDecoderForCausalLM",
    "MoshiForConditionalGeneration",
    "MoshiProcessor",
]
