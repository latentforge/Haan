"""Base classes organized by operating principle: BaseDataset + family bases + registry.

Folder layout — one file per concrete dataset, named after it:

  base.py                      BaseDataset (contract + auto-registry)
                               + AudioSourceDataset (audio-source family base) + RawEntry
  mixins.py                    overlapping functionality (MimiEncoder / TextAlign / NpzPairIO)
  kss_dataset.py               KSSDataset
  common_voice_ko_dataset.py   CommonVoiceKoDataset
  zeroth_ko_dataset.py         ZerothKoDataset
  en_kd_dataset.py             EnKDDialogueDataset (self-talk generation + quality filter)
  en_solo_dataset.py           EnSoloDataset (solo-speech crops from en_kd artifacts)
  seed_prompt_dataset.py       SeedPromptDataset (short EN clips → Mimi codes, en_kd priming)
  text_anchor_dataset.py       TextAnchorDataset (plain-text jsonl → token sequences)

Contract shared by every dataset:
  * build():        raw data → tokenized artifacts under out_dir/<name>/
  * iter_samples(): artifacts → unified-schema Sample (prepare_dataset converts to Arrow)
  * name   = registry key (CLI name). Family bases with an empty name are not registered.
  * source = Arrow group (en_kd | en_solo | ko_tts | text_anchor).
             mixing_sampler draws at this granularity, so it must match the
             training configs' sources.

The training-time Dataset (MixedDataset) only reads the unified Arrow data and
never sees the source — the inheritance hierarchy exists only in this offline layer.

Caution: naming this a top-level `datasets/` package would shadow the HF `datasets`
import, so it must stay the `project_amnesty.datasets` subpackage.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .schema import FRAME_RATE_HZ, SAMPLE_RATE, Sample
from .mixins import MimiEncoderMixin, NpzPairIOMixin, TextAlignMixin, TextTokCfg

REGISTRY: dict[str, type["BaseDataset"]] = {}


class BaseDataset(ABC):
    """Abstract root of all source datasets.

    Subclass contract:
      - set the `name` (registry key) and `source` (Arrow group) class attributes
      - implement build() / iter_samples() (or just inherit when a family base with
        the same operating principle already exists)
    """

    name: str = ""
    source: str = ""
    lang: str = "ko"
    sample_type: str = "ko_tts"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:  # family bases (no name) are not registered
            REGISTRY[cls.name] = cls

    def __init__(self, out_dir: str | Path = "data/tokenized"):
        # artifact location convention: <out_dir>/<name>/
        self.out_dir = Path(out_dir) / self.name

    # ---------------- contract ----------------

    @abstractmethod
    def build(self, limit: int | None = None) -> dict:
        """Raw data → tokenized artifacts in out_dir. Returns a stats dict."""

    @abstractmethod
    def iter_samples(self) -> Iterator[Sample]:
        """Artifacts → unified-schema Sample. Input for prepare_dataset's Arrow conversion."""

    # ---------------- CLI adapter ----------------

    @classmethod
    def from_cli(cls, args) -> "BaseDataset":
        """__main__'s shared argparse namespace → instance. Overridden per family."""
        raise NotImplementedError(f"{cls.__name__} has no from_cli")


def build_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# ---------------- audio-source family ----------------

@dataclass
class RawEntry:
    """Standard unit yielded by a subclass's iter_entries().

    Either audio_path or waveform is required (supports sources that provide
    arrays directly, e.g. the HF hub).
    """
    text: str
    audio_path: str | None = None
    waveform: np.ndarray | None = None      # (S,) or (C, S), at sample_rate
    sample_rate: int | None = None          # required when using waveform
    speaker: str = ""
    word_timestamps: list[dict] | None = None   # for "aligned" mode (if available)
    extra: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        key = self.audio_path or f"{self.speaker}:{hashlib.sha1(self.text.encode()).hexdigest()}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]


class AudioSourceDataset(MimiEncoderMixin, TextAlignMixin, NpzPairIOMixin, BaseDataset):
    """Family base for audio sources: raw audio+transcript → Mimi encoding + text
    alignment (KO TTS). Not registered (no name).

    Per-source differences = only the "parsing" of raw data (directory layout,
    transcript format, sample rate) → all common logic (cache skip, load/resample,
    batched encoding, alignment, save) lives here; subclasses implement only
    iter_entries().

    To add a new source:
      1. Create <name>_dataset.py with an AudioSourceDataset subclass — implement
         iter_entries() only
      2. The `name` class attribute automatically becomes the registry key
         (import it in __init__.py)
      3. Run: python -m project_amnesty.datasets <name> --root ...
    Nothing changes on the training side.
    """

    source = "ko_tts"
    lang = "ko"
    sample_type = "ko_tts"

    def __init__(
        self,
        out_dir: str | Path = "data/tokenized",
        root: str | Path | None = None,        # needed only for build (iter_samples uses out_dir only)
        text_cfg: TextTokCfg | None = None,    # needed only for build
        device: str = "cuda",
        batch_size: int = 16,
    ):
        super().__init__(out_dir)
        self.root = Path(root) if root is not None else None
        self.text_cfg = text_cfg
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_cli(cls, args) -> "AudioSourceDataset":
        return cls(
            out_dir=args.out_dir or "data/tokenized",
            root=args.root,
            text_cfg=TextTokCfg.from_yaml(args.text_config),
            device=args.device,
            batch_size=args.batch_size,
        )

    # ---------------- implemented/overridden by subclasses ----------------

    def iter_entries(self) -> Iterator[RawEntry]:
        """Raw data → RawEntry. Per-source differences should, in principle, live only here."""
        raise NotImplementedError

    def load_audio(self, entry: RawEntry) -> np.ndarray:
        """Return (1, S) float32 @ 24kHz. Override only for sources that need resampling."""
        if entry.waveform is not None:
            wav, sr = np.atleast_2d(entry.waveform)[:1], entry.sample_rate
        else:
            import sphn
            wav, sr = sphn.read(str(self.root / entry.audio_path))
            wav = np.atleast_2d(wav)[:1]
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr)
        return wav.astype(np.float32)

    # ---------------- common logic (overriding discouraged) ----------------

    def build(self, limit: int | None = None) -> dict:
        """Run the full pipeline: cache skip → batched Mimi encoding → text alignment → save."""
        assert self.root is not None and self.text_cfg is not None, \
            "build() requires root and text_cfg"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": 0, "cached": 0, "encoded": 0, "failed": 0}

        batch: list[RawEntry] = []
        for entry in self.iter_entries():
            if limit and stats["total"] >= limit:
                break
            stats["total"] += 1
            if self.is_cached(entry.uid):        # cache: skip on re-run
                stats["cached"] += 1
                continue
            batch.append(entry)
            if len(batch) >= self.batch_size:
                stats["encoded"] += self._process_batch(batch, stats)
                batch = []
        if batch:
            stats["encoded"] += self._process_batch(batch, stats)

        (self.out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[{self.name}] {stats}")
        return stats

    def _process_batch(self, batch: list[RawEntry], stats: dict) -> int:
        wavs = []
        ok_entries = []
        for e in batch:
            try:
                wavs.append(self.load_audio(e))
                ok_entries.append(e)
            except Exception as ex:
                stats["failed"] += 1
                print(f"[{self.name}] load failed {e.audio_path or e.uid}: {ex}")

        if not ok_entries:
            return 0
        codes_list = self.encode_audio(wavs)      # list of (K, T)
        for e, codes in zip(ok_entries, codes_list):
            T = codes.shape[-1]
            stream = self.align_text(e.text, e.word_timestamps, T)
            self.save_pair(e.uid, codes, stream, {
                "path": e.audio_path or "", "text": e.text,
                "source": self.name, "speaker": e.speaker,
                "duration": T / FRAME_RATE_HZ, **e.extra,
            })
        return len(ok_entries)
