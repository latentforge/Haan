"""CommonVoiceKoDataset: Mozilla Common Voice Korean corpus."""

from __future__ import annotations

import csv
from typing import Iterator

import numpy as np

from ..schema import SAMPLE_RATE
from .base import AudioSourceDataset, RawEntry


class CommonVoiceKoDataset(AudioSourceDataset):
    """Common Voice Korean: validated.tsv (client_id, path, sentence, ...), mp3 48 kHz."""

    name = "common_voice_ko"

    def iter_entries(self) -> Iterator[RawEntry]:
        with open(self.root / "validated.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                yield RawEntry(
                    audio_path=f"clips/{row['path']}",
                    text=row["sentence"],
                    speaker=row.get("client_id", "")[:12],
                )

    def load_audio(self, entry: RawEntry) -> np.ndarray:
        """mp3 is loaded with torchaudio, not sphn."""
        import torchaudio
        wav, sr = torchaudio.load(str(self.root / entry.audio_path))
        wav = wav[:1].numpy()
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr)
        return wav.astype(np.float32)
