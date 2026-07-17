"""SeedPromptDataset: English speech clips → Mimi codes → seed prompts for en_kd.

EnKDDialogueDataset primes the first frames of self-talk generation with external
audio (prevents mode collapse). This dataset builds those seeds: sample short
English clips, Mimi-encode them, and save one {uid}.safetensors per clip
(tensor "codes": (K, T) int16) into GenConfig.seed_prompt_dir.

Not a training source — iter_samples() is empty and `source` stays "" so the
prepare/Arrow flow never sees it. Only build() matters here.

Clip sources (--seed-source):
  * "libriheavy": mythicinfinity/libriheavy via HF streaming (Apache 2.0, not
    gated, LibriVox read speech). No local download needed.
  * "local": any local folder of audio files (wav/flac/mp3), e.g. the speech
    subset of HKUSTAudio/Audio-FLAN-Dataset (Apache 2.0, gated — accept the terms
    on the hub, `huggingface-cli login`, download audio_files/ for the speech
    domain, then pass the folder as --root).

Usage:
  python -m data_pipeline.datasets seed_prompts --seed-source libriheavy --limit 500
  python -m data_pipeline.datasets seed_prompts --seed-source local \
      --root data/raw/audio_flan_speech --limit 500
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import SAMPLE_RATE
from .base import BaseDataset
from .mixins import MimiEncoderMixin

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".opus", ".ogg")


class SeedPromptDataset(MimiEncoderMixin, BaseDataset):
    """seed_prompts: short EN clips → Mimi codes as safetensors (en_kd priming)."""

    name = "seed_prompts"
    source = ""          # not a training source: excluded from prepare/Arrow
    lang = "en"

    def __init__(
        self,
        out_dir: str | Path = "data",       # artifacts land in <out_dir>/seed_prompts/
        seed_source: str = "libriheavy",    # "libriheavy" | "local"
        root: str | Path | None = None,     # required for seed_source="local"
        seed_sec: float = 10.0,             # clip length fed to engine.prime()
        min_sec: float = 6.0,               # skip clips shorter than this
        device: str = "cuda",
        batch_size: int = 8,
    ):
        super().__init__(out_dir)
        self.seed_source = seed_source
        self.root = Path(root) if root is not None else None
        self.seed_sec = seed_sec
        self.min_sec = min_sec
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_cli(cls, args) -> "SeedPromptDataset":
        return cls(
            out_dir=args.out_dir,
            seed_source=args.seed_source,
            root=args.root,
            seed_sec=args.seed_sec,
            device=args.device,
            batch_size=args.batch_size,
        )

    # ---------------- clip sources ----------------

    def _iter_clips(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yields (uid, wav) with wav (1, S) float32 @ 24 kHz, trimmed to seed_sec."""
        if self.seed_source == "libriheavy":
            yield from self._iter_libriheavy()
        elif self.seed_source == "local":
            yield from self._iter_local()
        else:
            raise ValueError(f"unknown seed_source '{self.seed_source}'")

    def _trim(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        """Mono, resampled to 24 kHz, first seed_sec seconds. None if too short."""
        wav = np.atleast_2d(np.asarray(wav, dtype=np.float32))[:1]
        if wav.shape[-1] < self.min_sec * sr:
            return None
        wav = wav[:, : int(self.seed_sec * sr)]
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr)
        return wav.astype(np.float32)

    def _iter_libriheavy(self) -> Iterator[tuple[str, np.ndarray]]:
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset  # HF datasets (unrelated to our subpackage)

        ds = load_dataset("mythicinfinity/libriheavy", "small", split="train",
                          streaming=True)
        # decode=False + soundfile: avoids the torchcodec dependency that newer
        # `datasets` versions require for built-in audio decoding
        ds = ds.cast_column("audio", Audio(decode=False))
        for ex in ds:
            if ex.get("audio_duration", 0.0) < self.min_sec:
                continue
            wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            wav = self._trim(wav.T, sr)   # (S,) or (S, C) → channel-first
            if wav is None:
                continue
            yield hashlib.sha1(f"libriheavy:{ex['id']}".encode()).hexdigest()[:16], wav

    def _iter_local(self) -> Iterator[tuple[str, np.ndarray]]:
        assert self.root is not None, "seed_source='local' requires --root"
        import torchaudio
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            try:
                wav, sr = torchaudio.load(str(p))
            except Exception as ex:
                print(f"[{self.name}] load failed {p.name}: {ex}")
                continue
            wav = self._trim(wav.numpy(), sr)
            if wav is None:
                continue
            yield hashlib.sha1(f"local:{p.relative_to(self.root)}".encode()).hexdigest()[:16], wav

    # ---------------- contract ----------------

    def build(self, limit: int | None = None) -> dict:
        """Encode up to `limit` (default 500) clips into {uid}.safetensors seeds."""
        from safetensors.numpy import save_file

        n_target = limit or 500
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stats = {"target": n_target, "cached": 0, "encoded": 0}

        batch: list[tuple[str, np.ndarray]] = []

        def flush(batch):
            codes_list = self.encode_audio([w for _, w in batch])
            for (uid, _), codes in zip(batch, codes_list):
                save_file({"codes": codes}, str(self.out_dir / f"{uid}.safetensors"))
            stats["encoded"] += len(batch)

        for uid, wav in self._iter_clips():
            # count the pending batch too, or the last flush overshoots the target
            if stats["cached"] + stats["encoded"] + len(batch) >= n_target:
                break
            if (self.out_dir / f"{uid}.safetensors").exists():   # cache: skip on re-run
                stats["cached"] += 1
                continue
            batch.append((uid, wav))
            if len(batch) >= self.batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

        (self.out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[{self.name}] {stats} → {self.out_dir}")
        return stats

    def iter_samples(self) -> Iterator:
        """Seeds are not training samples — nothing to export."""
        return iter(())
