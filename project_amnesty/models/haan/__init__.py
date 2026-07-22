from transformers import (
    AutoModel, AutoModelForCausalLM,
    AutoTokenizer, AutoProcessor, AutoConfig
)

from .configuration_haan import HaanConfig
from .tokenization_haan import HaanTokenizer
from .processing_haan import HaanProcessor
from .modeling_haan import HaanModel, HaanForConditionalGeneration


AutoConfig.register("haan", HaanModel)
AutoTokenizer.register(HaanConfig, HaanTokenizer)
AutoProcessor.register(HaanConfig, AutoProcessor)
AutoModel.register(HaanConfig, HaanModel)
AutoModelForCausalLM.register(HaanConfig, HaanForConditionalGeneration)
