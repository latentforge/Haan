"""ZerothKoDataset: Zeroth-Korean ASR corpus (HF hub)."""

from __future__ import annotations

from typing import Iterator

from .base import AudioSourceDataset, RawEntry


class ZerothKoDataset(AudioSourceDataset):
    """Zeroth-Korean: loaded from the HF hub (audio comes as arrays → uses the waveform path)."""

    name = "zeroth_ko"

    def iter_entries(self) -> Iterator[RawEntry]:
        from datasets import load_dataset  # HF datasets (unrelated to our subpackage)
        ds = load_dataset("kresnik/zeroth_korean", split="train", streaming=True)
        for ex in ds:
            yield RawEntry(
                text=ex["text"],
                waveform=ex["audio"]["array"],
                sample_rate=ex["audio"]["sampling_rate"],
                speaker=str(ex.get("speaker_id", "")),
            )
