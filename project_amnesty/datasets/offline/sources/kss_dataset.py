"""KSSDataset: KSS v1.4 Korean single-speaker TTS corpus."""

from __future__ import annotations

from typing import Iterator

from ..base import AudioSourceDataset, RawEntry

# Unofficial but faithful HF mirror of the Kaggle original (same columns, same
# CC BY-NC-SA 4.0 license). Used automatically when --root has no local copy.
HF_REPO = "Bingsu/KSS_Dataset"


class KSSDataset(AudioSourceDataset):
    """KSS v1.4: transcript.v.1.4.txt, pipe-separated (path|raw|normalized|...), 44.1 kHz.

    Raw data resolution order:
      1. local: <root>/transcript.v.1.4.txt + wav folders (Kaggle download)
      2. HF hub: Bingsu/KSS_Dataset — fetched into the HF cache on first build,
         no manual download needed. `expanded_script` == transcript column 2.
    """

    name = "kss"

    def iter_entries(self) -> Iterator[RawEntry]:
        if self.root is not None and (self.root / "transcript.v.1.4.txt").exists():
            yield from self._iter_local()
        else:
            # ASCII only: Windows consoles may still run cp949
            print(f"[{self.name}] no transcript.v.1.4.txt under {self.root} -- "
                  f"loading from HF hub ({HF_REPO})")
            yield from self._iter_hub()

    def _iter_local(self) -> Iterator[RawEntry]:
        with open(self.root / "transcript.v.1.4.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                # parts[2] = transcription with numbers normalized to Hangul — use this for TTS
                yield RawEntry(audio_path=parts[0], text=parts[2], speaker="kss")

    def _iter_hub(self) -> Iterator[RawEntry]:
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset  # HF datasets (unrelated to our subpackage)

        # Non-streaming: ~4.3 GB lands in the HF cache once, re-runs are free.
        ds = load_dataset(HF_REPO, split="train")
        # decode=False + soundfile, matching zeroth/seed_prompt: datasets>=4 routes
        # automatic audio decoding through torchcodec, which needs FFmpeg shared
        # libraries that are not present here. soundfile reads the wav bytes directly.
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, ex in enumerate(ds):
            wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            if wav.ndim == 2:
                # soundfile is frames-first (S, C); RawEntry.waveform wants (C, S)
                wav = wav.T
            yield RawEntry(
                # synthetic path (parquet carries none): keeps uid unique even for
                # duplicate scripts (single speaker, so the speaker:text fallback
                # could collide) and stable across runs. Audio comes from waveform.
                audio_path=f"hf/{i:05d}",
                text=ex["expanded_script"],
                waveform=wav,
                sample_rate=sr,
                speaker="kss",
            )
