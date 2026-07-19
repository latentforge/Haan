"""Collator for the `text_anchor` path -- deliberately separate from KDCollator.

text_anchor rows carry `text_flat`, an unaligned token sequence, and T=0 frames.
They are never mixed into an audio batch (plan section 5.4). Four independent reasons,
any one of which is sufficient:

* T means different things. Audio T is 12.5 Hz frames; anchor T is tokens (up to
  2048). Sharing the axis lets one 2048-token anchor pad every audio row in the
  batch out to 2048 frames -- 164 s of nothing, ~7x compute for zero signal.
* The delay pattern is undefined: there is no codebook axis to offset.
* `codes_a` is `zeros((8, 0))`. Materializing 2048 frames x 2 roles x 8 codebooks
  of fake silence to match shapes fabricates ~33k tokens per sample, which is the
  plan section 2.5 failure mode two orders of magnitude larger.
* Loss routing is fully disjoint: no PAD/EPAD frame semantics, no Zone A, no
  audio branch, no KD.

Mixing happens at the **step** level (alternating micro-batches inside a
grad_accum window), not at the batch level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import TokenConfig
from .item import LANG_IDS, SAMPLE_TYPE_IDS, KDSample

_AUDIO_ROW_MSG = (
    "TextAnchorCollator received a non-text_anchor row ({uid!r}, sample_type="
    "{stype!r}). Frame-aligned audio rows belong to "
    "training.datasets.collator.KDCollator."
)


@dataclass
class TextAnchorCollatorConfig:
    tokens: TokenConfig = field(default_factory=TokenConfig)
    pad_to_multiple_of: int = 8
    max_text_len: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.tokens, dict):
            self.tokens = TokenConfig(**self.tokens)
        assert self.pad_to_multiple_of >= 1


class TextAnchorCollator:
    """list[KDSample] (text_anchor only) -> a plain padded LM batch."""

    def __init__(self, cfg: TextAnchorCollatorConfig | None = None, **kwargs) -> None:
        self.cfg = cfg if cfg is not None else TextAnchorCollatorConfig(**kwargs)
        # Only the batch pad id matters here: there is no stream PAD/EPAD in an
        # anchor sequence. require() still cross-checks pad != batch_pad.
        self.cfg.tokens.require("batch_pad_id")

    def __call__(self, rows: list[KDSample]) -> dict:
        cfg, tok = self.cfg, self.cfg.tokens
        assert len(rows) > 0, "empty batch"
        for r in rows:
            if r["sample_type"] != "text_anchor" or not bool(r["is_text_only"]):
                raise ValueError(
                    _AUDIO_ROW_MSG.format(uid=r["sample_uid"], stype=r["sample_type"])
                )

        lens = [int(r["text_flat"].shape[0]) for r in rows]
        if cfg.max_text_len is not None:
            lens = [min(L, cfg.max_text_len) for L in lens]
        assert all(L > 0 for L in lens), "text_anchor row with empty text_flat"

        B = len(rows)
        m = cfg.pad_to_multiple_of
        L = ((max(lens) + m - 1) // m) * m
        dev = rows[0]["text_flat"].device

        text_tokens = torch.full((B, L), int(tok.batch_pad_id), dtype=torch.int64, device=dev)
        attention_mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        sample_type_id = torch.zeros((B,), dtype=torch.int64, device=dev)
        lang_id = torch.zeros((B,), dtype=torch.int64, device=dev)

        for b, r in enumerate(rows):
            n = lens[b]
            text_tokens[b, :n] = r["text_flat"][:n].to(torch.int64)
            attention_mask[b, :n] = True
            sample_type_id[b] = SAMPLE_TYPE_IDS[r["sample_type"]]
            lang_id[b] = LANG_IDS[r["lang"]]

        return {
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            # Uniform weight: anchor loss is a directly loggable scalar, not a
            # masked subset of a mixed loss. Its coefficient is tuned separately.
            "text_loss_weight": attention_mask.float(),
            "text_lengths": torch.tensor(lens, dtype=torch.int64, device=dev),
            # Carried even though the value is constant here, so the loss can
            # assert its routing instead of assuming it.
            "sample_type_id": sample_type_id,
            "lang_id": lang_id,
            "sample_uid": [r["sample_uid"] for r in rows],
            "target_aligned": True,
        }
