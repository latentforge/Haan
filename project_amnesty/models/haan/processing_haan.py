"""Haan processor and tokenizer.

`HaanTokenizer` subclasses the Qwen2 tokenizer (Qwen3 reuses `Qwen2Tokenizer`) and adds a `<pad>`
token used to fill the text stream between enunciations (Moshi's tokenizer carries `<pad>`;
Qwen3's does not). `<pad>` gets its own id and does not reuse Qwen3's `<|endoftext|>` (its
pad/bos/eos token) or the ChatML markers.

`HaanProcessor` subclasses [`MoshiProcessor`] (Mimi codec, dual audio streams, text padded to one
token per audio frame) and uses `<pad>` as the id that fills the text stream between enunciations.
"""

from transformers.models.moshi.processing_moshi import MoshiProcessor
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

__all__ = ["HaanProcessor", "HaanTokenizer"]

STREAM_PAD_TOKEN = "<pad>"


class HaanTokenizer(Qwen2Tokenizer):
    """Qwen3 tokenizer (`Qwen2Tokenizer`) plus a `<pad>` text-stream fill token."""

    stream_pad_token = STREAM_PAD_TOKEN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add the text-stream fill token if the loaded vocab does not already carry it.
        if self.stream_pad_token not in self.get_vocab():
            self.add_special_tokens({"additional_special_tokens": [self.stream_pad_token]})

    @property
    def stream_pad_id(self) -> int:
        """Id of the text-stream PAD token (`<pad>`)."""
        return self.convert_tokens_to_ids(self.stream_pad_token)


class HaanProcessor(MoshiProcessor):
    r"""Mimi codec + Haan tokenizer.

    The text stream is padded to one token per audio frame; frames with no word being spoken are
    filled with the `<pad>` token (added to the Qwen3 tokenizer by `HaanTokenizer`).

    Args:
        stream_epad_id (`int`, *optional*):
            Id of the EPAD token, inserted one frame before a word starts. Carried here so the
            collator reads one definition of it; the processor does not insert it, since EPAD
            placement needs word-level timestamps.
    """

    stream_pad_token = STREAM_PAD_TOKEN

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        num_codebooks=8,
        stream_epad_id: int | None = None,
    ):
        self.stream_epad_id = stream_epad_id
        super().__init__(feature_extractor, tokenizer, audio_tokenizer, num_codebooks=num_codebooks)

    def _pad_token_id(self) -> int:
        """Id of the stream PAD token (`<pad>`). Overrides Moshi's `pad_token_id` lookup."""
        return self.tokenizer.convert_tokens_to_ids(self.stream_pad_token)
