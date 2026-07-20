"""Text family: plain-text jsonl → token sequences (text_anchor).

Anchor for retaining text ability. There is no audio, so it holds a plain token
sequence without frame alignment (codes_a is an empty array). No separate build
stage — tokenization happens on the fly in iter_samples, since the source is
already in its final form (jsonl) and there is nothing to bake.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Sample
from ..base import BaseDataset
from ..mixins import TextTokCfg


class TextAnchorDataset(BaseDataset):
    """text_anchor: tokenize a {"text": ...} jsonl and emit Samples."""

    name = "text_anchor"
    source = "text_anchor"
    lang = "ko"
    sample_type = "text_anchor"

    def __init__(
        self,
        jsonl: str | Path = "data/raw/text_anchor.jsonl",
        tokenizer_name: str = "Qwen/Qwen3-8B",
        max_len: int = 2048,
        out_dir: str | Path = "data/raw",   # no artifacts — kept for the convention
    ):
        super().__init__(out_dir)
        self.jsonl = Path(jsonl)
        self.tokenizer_name = tokenizer_name
        self.max_len = max_len

    @classmethod
    def from_cli(cls, args) -> "TextAnchorDataset":
        # take the tokenizer from --text-config so the CLI path can't silently
        # diverge from the rest of the pipeline once the team tokenizer lands
        return cls(
            jsonl=args.jsonl,
            tokenizer_name=TextTokCfg.from_yaml(args.text_config).tokenizer_name,
            out_dir=args.out_dir or "data/raw",
        )

    def build(self, limit: int | None = None) -> dict:
        """No build stage (on-the-fly tokenization). No-op to satisfy the contract."""
        return {}

    def iter_samples(self) -> Iterator[Sample]:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        for i, line in enumerate(open(self.jsonl)):
            text = json.loads(line)["text"]
            ids = tok.encode(text)[: self.max_len]
            yield Sample(
                sample_type=self.sample_type, lang=self.lang,
                codes_a=np.zeros((8, 0), dtype=np.int16),
                text_tokens_a=np.asarray(ids, dtype=np.int32),
                sample_uid=hashlib.sha1(f"anchor:{i}".encode()).hexdigest()[:16],
                speaker="",   # plain text: there is no speaker to record
            )
