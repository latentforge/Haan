"""ZerothKoDataset: Zeroth-Korean ASR corpus (HF hub)."""

from __future__ import annotations

from typing import Iterator

from ..base import AudioSourceDataset, RawEntry


class ZerothKoDataset(AudioSourceDataset):
    """Zeroth-Korean: loaded from the HF hub (audio comes as arrays → uses the waveform path)."""

    name = "zeroth_ko"

    def iter_entries(self) -> Iterator[RawEntry]:
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset  # HF datasets (unrelated to our subpackage)

        ds = load_dataset("kresnik/zeroth_korean", split="train", streaming=True)
        # decode=False + soundfile, matching seed_prompt_dataset: datasets>=4 routes
        # automatic audio decoding through torchcodec, which needs FFmpeg shared
        # libraries that are not present here. soundfile reads the flac bytes directly.
        ds = ds.cast_column("audio", Audio(decode=False))
        for ex in ds:
            wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            yield RawEntry(
                text=ex["text"],
                waveform=wav,
                sample_rate=sr,
                speaker=str(ex.get("speaker_id", "")),
            )
