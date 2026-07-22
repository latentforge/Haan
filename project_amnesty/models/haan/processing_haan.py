"""Haan processor -- subclasses the Moshi processor from `transformers`.

Inherits everything from [`MoshiProcessor`] (Mimi codec, dual audio streams, text
padded to one token per audio frame) and changes exactly one thing: which id fills the
text stream between enunciations.

Scope note. The Zone A / B / C instruction template (voice-prompt segment, role text, and
the dialogue body) is *not* built here -- the collator assembles `[Zone A | Zone B | Zone C]`
per batch so the voice prompt can vary per epoch. What the processor owns is the token
contract those layers share.
"""

from transformers.models.moshi.processing_moshi import MoshiProcessor

__all__ = ["HaanProcessor"]


class HaanProcessor(MoshiProcessor):
    r"""Mimi codec + Qwen3 tokenizer, with Moshi's stream PAD kept distinct from batch padding.

    Everything here turns on one distinction that Moshi's tokenizers make for free and
    Qwen3's does not:

      - **stream PAD** -- "not speaking, session continues". It fills roughly 65% of the
        text channel, is a prediction target, and is down-weighted (x0.3) rather than
        masked.
      - **batch pad** -- length alignment across a batch. Masked out of the loss entirely.

    Qwen3 has no spare `<pad>`, and HF convention points `pad_token_id` at `<|im_end|>`.
    Inheriting `MoshiProcessor._pad_token_id` unchanged would therefore fill the stream
    with the *end-of-message* token -- the exact failure this class exists to prevent: a
    token that means "generation over" would come to occupy 65% of the text channel and
    overwrite the instruction-following behaviour the Qwen3 backbone was chosen for.

    So the stream PAD is passed in explicitly. It must be a newly assigned reserved slot,
    never `<|im_end|>` / `<|im_start|>`.

    Args:
        stream_pad_id (`int`, *optional*):
            Id of the stream PAD token (`configs/tokens.yaml: text_pad_id`).
        stream_epad_id (`int`, *optional*):
            Id of the EPAD token (`configs/tokens.yaml: text_epad_id`), inserted one frame
            before a word starts. Carried here so the collator and the processor read one
            source of truth; the processor itself does not insert it, since EPAD placement
            needs word-level timestamps.
    """

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        num_codebooks=8,
        stream_pad_id: int | None = None,
        stream_epad_id: int | None = None,
    ):
        self.stream_pad_id = stream_pad_id
        self.stream_epad_id = stream_epad_id
        super().__init__(feature_extractor, tokenizer, audio_tokenizer, num_codebooks=num_codebooks)

    def _pad_token_id(self) -> int:
        """The id used to fill the text stream between enunciations.

        Overrides Moshi's `pad_token_id` -> `<pad>` lookup, which on a Qwen3 tokenizer would
        resolve to the end-of-message token. Fails loudly instead of falling back: a silently
        wrong id here does not crash, it just quietly trains the wrong turn-boundary behaviour.
        """
        if self.stream_pad_id is None:
            raise ValueError(
                "`stream_pad_id` is unset, so the text stream cannot be padded. The stream PAD is a "
                "distinct token from the batch pad: assign it to a reserved Qwen3 slot and pass it here, "
                "rather than reusing `<|im_end|>`/`<|im_start|>`. Source of truth: "
                "`configs/tokens.yaml: text_pad_id`."
            )
        banned = self._banned_stream_pad_ids()
        if self.stream_pad_id in banned:
            raise ValueError(
                f"`stream_pad_id={self.stream_pad_id}` is {banned[self.stream_pad_id]}, which cannot also be the "
                "stream PAD. The stream PAD marks 'not speaking, session continues' and fills ~65% of the text "
                "channel as a down-weighted prediction target; these tokens mean 'this is over' and are either "
                "masked out of the loss (batch pad) or terminate generation. Reusing one overwrites that "
                "association across most of the channel and damages the very instruction-following the Qwen3 "
                "backbone was chosen for. Assign a reserved slot instead."
            )

        if self.stream_pad_id < 0:
            raise ValueError(f"`stream_pad_id={self.stream_pad_id}` must be non-negative.")

        # No UPPER bound is checked here, deliberately. The stream PAD is meant to live in a
        # reserved slot -- an embedding row that exists but carries no token (Qwen3-8B has 151936
        # rows for 151669 tokens, and `configs/tokens.yaml` uses three of that gap). Only the MODEL
        # knows how many rows there are, and a processor must not have to be told: it is a data-side
        # object, and reaching for `config.vocab_size` here would invert the dependency. Bounding by
        # `len(tokenizer)` instead is worse than not checking -- it rejects every reserved slot that
        # exists, i.e. exactly what the docstring above tells the caller to assign.
        #
        # What the tokenizer alone CAN settle is the direction that actually corrupts training:
        # below `len(tokenizer)` the id is a token the backbone was already trained on, and filling
        # ~65% of the text channel with it overwrites whatever it meant. Same damage as reusing
        # `<|im_end|>`, only quieter, since a rare token gives no clue at review time.
        real_tokens = len(self.tokenizer)
        if self.stream_pad_id < real_tokens:
            token = self.tokenizer.convert_ids_to_tokens(self.stream_pad_id)
            raise ValueError(
                f"`stream_pad_id={self.stream_pad_id}` is the existing token {token!r}, not a reserved slot. "
                "The stream PAD occupies most of the text channel, so that token's learned meaning would be "
                f"overwritten. Use an id at or above {real_tokens} -- an embedding row that "
                "exists but carries no token. Source of truth: `configs/tokens.yaml`."
            )
        return self.stream_pad_id

    def _banned_stream_pad_ids(self) -> dict[int, str]:
        """Ids that must never serve as the stream PAD, mapped to why.

        Resolved from the tokenizer rather than hardcoded: `<|im_end|>` is 151645 on Qwen3 and
        something else everywhere else, so a literal would silently stop guarding the moment the
        tokenizer changed. `convert_tokens_to_ids` returns None for a token the vocabulary does not
        have, so a missing marker simply drops out of the set.
        """
        banned: dict[int, str] = {}
        for attribute, reason in (
            ("pad_token_id", "the tokenizer's batch `pad_token_id`"),
            ("eos_token_id", "the tokenizer's `eos_token_id`"),
        ):
            token_id = getattr(self.tokenizer, attribute, None)
            if token_id is not None:
                banned.setdefault(token_id, reason)

        # The two ChatML markers, named explicitly. Kept separate from the eos lookup above because
        # they coincide on Qwen3 (`eos == <|im_end|>`) and need not elsewhere -- and because naming
        # the marker makes the error say what the caller actually did.
        for marker in ("<|im_start|>", "<|im_end|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(marker)
            if token_id is not None:
                banned.setdefault(token_id, f"the ChatML marker `{marker}`")
        return banned
