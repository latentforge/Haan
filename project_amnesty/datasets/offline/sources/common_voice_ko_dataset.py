"""CommonVoiceKoDataset: Mozilla Common Voice Korean corpus."""

from __future__ import annotations

import csv
import tarfile
from itertools import count
from typing import Iterator

import numpy as np

from project_amnesty.datasets.shared.schema import SAMPLE_RATE
from ..base import AudioSourceDataset, RawEntry

# Mozilla pulled the data from its own HF repo in October 2025 (now behind the
# Mozilla Data Collective login). Common Voice is CC0, so this pre-existing full
# mirror stays a legal, ungated source. Layout: audio/ko/<split>/ko_<split>_<i>.tar
# + transcript/ko/validated.tsv.
HF_REPO = "fsicoli/common_voice_17_0"
# validated.tsv spans these splits; "other"/"invalidated" are unvalidated clips.
TAR_SPLITS = ("train", "dev", "test")


class CommonVoiceKoDataset(AudioSourceDataset):
    """Common Voice Korean: validated.tsv (client_id, path, sentence, ...), mp3 48 kHz.

    build() auto-downloads the ko subset (~35 MB) from the HF mirror into --root
    when <root>/validated.tsv is missing.
    """

    name = "common_voice_ko"

    def ensure_downloaded(self) -> None:
        if (self.root / "validated.tsv").exists():
            return
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        # ASCII only: Windows consoles may still run cp949
        print(f"[{self.name}] no validated.tsv under {self.root} -- "
              f"downloading ko subset from HF ({HF_REPO})")
        clips = self.root / "clips"
        clips.mkdir(parents=True, exist_ok=True)
        for split in TAR_SPLITS:
            for i in count():
                try:
                    tar_path = hf_hub_download(
                        HF_REPO, f"audio/ko/{split}/ko_{split}_{i}.tar",
                        repo_type="dataset")
                except EntryNotFoundError:
                    break
                with tarfile.open(tar_path) as tf:
                    n = 0
                    for m in tf:
                        if not (m.isfile() and m.name.endswith(".mp3")):
                            continue
                        # flatten: tar members carry a wrapper dir, iter_entries
                        # expects clips/<basename> per validated.tsv `path`
                        dest = clips / m.name.rsplit("/", 1)[-1]
                        with tf.extractfile(m) as src:
                            dest.write_bytes(src.read())
                        n += 1
                    print(f"[{self.name}] {split} shard {i}: {n} clips")
        tsv = hf_hub_download(HF_REPO, "transcript/ko/validated.tsv",
                              repo_type="dataset")
        # written last: acts as the completion marker for the skip check above
        (self.root / "validated.tsv").write_bytes(open(tsv, "rb").read())

    def iter_entries(self) -> Iterator[RawEntry]:
        skipped = 0
        with open(self.root / "validated.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                # validated.tsv is a superset of the train/dev/test tars: a few
                # validated clips belong to no split and ship no audio
                if not (self.root / "clips" / row["path"]).exists():
                    skipped += 1
                    continue
                yield RawEntry(
                    audio_path=f"clips/{row['path']}",
                    text=row["sentence"],
                    speaker=row.get("client_id", "")[:12],
                )
        if skipped:
            print(f"[{self.name}] {skipped} validated rows without audio on disk -- skipped")

    def load_audio(self, entry: RawEntry) -> np.ndarray:
        """mp3 is loaded with torchaudio, not sphn."""
        import torchaudio
        wav, sr = torchaudio.load(str(self.root / entry.audio_path))
        wav = wav[:1].numpy()
        if sr != SAMPLE_RATE:
            wav = self._resample(wav, sr)
        return wav.astype(np.float32)
