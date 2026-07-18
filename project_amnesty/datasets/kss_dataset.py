"""KSSDataset: KSS v1.4 Korean single-speaker TTS corpus."""

from __future__ import annotations

from typing import Iterator

from .base import AudioSourceDataset, RawEntry


class KSSDataset(AudioSourceDataset):
    """KSS v1.4: transcript.v.1.4.txt, pipe-separated (path|raw|normalized|...), 44.1 kHz."""

    name = "kss"

    def iter_entries(self) -> Iterator[RawEntry]:
        with open(self.root / "transcript.v.1.4.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                # parts[2] = transcription with numbers normalized to Hangul — use this for TTS
                yield RawEntry(audio_path=parts[0], text=parts[2], speaker="kss")
