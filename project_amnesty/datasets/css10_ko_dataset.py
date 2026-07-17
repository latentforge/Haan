"""CSS10KoDataset: CSS10 Korean single-speaker corpus."""

from __future__ import annotations

from typing import Iterator

from .base import AudioSourceDataset, RawEntry


class CSS10KoDataset(AudioSourceDataset):
    """CSS10 Korean: transcript.txt, pipe-separated, 22.05 kHz single speaker."""

    name = "css10_ko"

    def iter_entries(self) -> Iterator[RawEntry]:
        with open(self.root / "transcript.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                yield RawEntry(audio_path=parts[0], text=parts[2], speaker="css10")
