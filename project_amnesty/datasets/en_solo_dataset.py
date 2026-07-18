"""Derived family: EN multi-turn (generated) → pseudo-singleton crops.

Purpose (slide 23, risk 1): break the "Korean=singleton, English=multi-turn"
shortcut correlation by mixing some English data cropped into singleton form
into the batches.

Method: determine speaker-activity spans from non-PAD activity in the text stream,
crop windows where one speaker talks continuously for at least min_sec while the
other stays silent, and store them in the same shape as ko_tts
(codes_a + text_tokens_a, lang="en").
Teacher logits are dropped — these samples train via the CE path (same treatment as KO TTS).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import BaseDataset
from .mixins import NpzPairIOMixin, TextTokCfg


def find_solo_windows(
    text_self: np.ndarray, text_other: np.ndarray,
    pad_id: int, min_frames: int,
) -> list[tuple[int, int]]:
    """Find contiguous spans where self is active and other is silent.

    Silence is judged with a moving-window smoothing (2 s = 25 frames) rather than
    per frame — prevents a couple of backchannel tokens from shattering the window.
    """
    w = 25
    self_active = np.convolve((text_self != pad_id).astype(float), np.ones(w) / w, "same") > 0.05
    other_silent = np.convolve((text_other != pad_id).astype(float), np.ones(w) / w, "same") < 0.02
    solo = self_active & other_silent

    windows, start = [], None
    for t, ok in enumerate(solo):
        if ok and start is None:
            start = t
        elif not ok and start is not None:
            if t - start >= min_frames:
                windows.append((start, t))
            start = None
    if start is not None and len(solo) - start >= min_frames:
        windows.append((start, len(solo)))
    return windows


class EnSoloDataset(NpzPairIOMixin, BaseDataset):
    """en_solo: crop solo-speech spans from en_kd artifacts and store in singleton form."""

    name = "en_solo"
    source = "en_solo"
    lang = "en"                  # distinguished by lang → separable in mixing/analysis
    sample_type = "ko_tts"       # treated as the singleton (CE) path by shape

    def __init__(
        self,
        out_dir: str | Path = "data/generated",
        kd_dir: str | Path | None = None,       # en_kd artifact directory (needed only for build)
        text_cfg: TextTokCfg | None = None,     # injects text_pad_id (needed only for build)
        min_sec: float = 6.0,
    ):
        super().__init__(out_dir)
        self.kd_dir = Path(kd_dir) if kd_dir is not None else None
        self.text_cfg = text_cfg
        self.min_sec = min_sec

    @classmethod
    def from_cli(cls, args) -> "EnSoloDataset":
        return cls(
            out_dir=args.out_dir,
            kd_dir=args.root,
            text_cfg=TextTokCfg.from_yaml(args.text_config),
            min_sec=args.min_sec,
        )

    def build(self, limit: int | None = None) -> dict:
        assert self.kd_dir is not None and self.text_cfg is not None, \
            "build() requires kd_dir and text_cfg"
        pad_id = self.text_cfg.text_pad_id
        # None would make every activity mask False → zero crops, silently
        assert pad_id is not None, \
            "text_pad_id is unset (configs/data/text_tok.yaml) - finalize the PAD/EPAD mapping first"
        min_frames = int(self.min_sec * 12.5)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        accepted = set(json.loads((self.kd_dir / "accepted.json").read_text()))

        n_crops = 0
        for uid in sorted(accepted):
            if limit and n_crops >= limit:
                break
            data = dict(np.load(self.kd_dir / f"{uid}.npz"))
            for me, other in (("a", "b"), ("b", "a")):
                wins = find_solo_windows(
                    data[f"text_tokens_{me}"], data[f"text_tokens_{other}"],
                    pad_id, min_frames,
                )
                for wi, (s, e) in enumerate(wins):
                    self.save_pair(
                        f"{uid}-solo-{me}{wi}",
                        data[f"codes_{me}"][:, s:e],
                        data[f"text_tokens_{me}"][s:e],
                        {"lang": self.lang, "src_dialogue": uid, "frames": int(e - s)},
                    )
                    n_crops += 1
        stats = {"crops": n_crops, "dialogues": len(accepted)}
        print(f"[{self.name}] {n_crops} crops from {len(accepted)} dialogues → {self.out_dir}")
        return stats

    # iter_samples is the NpzPairIOMixin default implementation (npz pairs → Sample)
