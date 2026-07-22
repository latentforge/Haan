"""Tests for the two project_amnesty defects fixed alongside them.

1. `speaker` used to be collected by every dataset and then silently dropped at
   the Arrow boundary, because schema.Sample / to_row / arrow_features had no
   such field. Voice-prompt conditioning picks the reference audio for a ko_tts
   sample from *the same speaker's other utterance* -- another row of the same
   dataset -- so speaker identity has to survive into Arrow or that lookup is
   impossible. These tests pin both ends of the path: the in-memory round trip
   and the npz-artifact -> iter_samples read.

2. MimiEncoderMixin reached for `moshi.models.loaders`, a package that is not
   installed; the code had never run. It now builds a standalone transformers
   MimiModel out of the HF Moshi checkpoint's `audio_encoder.*` tensors. The
   smoke test at the bottom is what proves that path is actually alive -- it
   loads the real codec and encodes real signals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_amnesty.datasets.offline.mixins import (  # noqa: E402
    DEFAULT_MIMI_CKPT,
    MimiEncoderMixin,
    NpzPairIOMixin,
)
from project_amnesty.datasets.shared.schema import (  # noqa: E402
    FRAME_RATE_HZ,
    NUM_CODEBOOKS,
    SAMPLE_RATE,
    SCHEMA_VERSION,
    Sample,
    arrow_features,
    row_to_arrays,
)

# ------------------------------------------------------------------ speaker


def _sample(speaker: str, uid: str = "uid0") -> Sample:
    T = 5
    return Sample(
        sample_type="ko_tts",
        lang="ko",
        codes_a=np.arange(NUM_CODEBOOKS * T, dtype=np.int16).reshape(NUM_CODEBOOKS, T),
        text_tokens_a=np.arange(T, dtype=np.int32),
        sample_uid=uid,
        speaker=speaker,
    )


def test_arrow_features_declares_speaker():
    assert "speaker" in arrow_features(), (
        "speaker missing from arrow_features(); it would be dropped when "
        "artifacts become Arrow rows"
    )


@pytest.mark.parametrize("speaker", ["spk_042", "", "화자/1 with spaces"])
def test_speaker_round_trips_through_arrow(tmp_path, speaker):
    """Sample -> to_row -> real Arrow dataset -> row_to_arrays keeps speaker.

    Goes through an actual datasets.Dataset rather than just the dict, so a
    Features/row mismatch (the failure mode that dropped the field originally)
    surfaces here instead of in prepare_dataset.
    """
    from datasets import Dataset as HFDataset

    s = _sample(speaker)
    row = s.to_row()
    assert row["speaker"] == speaker

    ds = HFDataset.from_list([row], features=arrow_features())
    assert "speaker" in ds.column_names
    assert ds[0]["speaker"] == speaker

    out = row_to_arrays(ds[0])
    assert out["speaker"] == speaker


def test_row_to_arrays_tolerates_a_row_without_speaker():
    """Defensive: a hand-built row lacking the column reads back as "", not KeyError."""
    row = _sample("ignored").to_row()
    row.pop("speaker")
    assert row_to_arrays(row)["speaker"] == ""


def test_speaker_survives_the_npz_artifact_path(tmp_path):
    """save_pair writes speaker into the sidecar json; iter_samples must read it back."""

    class _Store(NpzPairIOMixin):
        lang = "ko"
        sample_type = "ko_tts"

        def __init__(self, out_dir: Path):
            self.out_dir = out_dir

    store = _Store(tmp_path)
    T = 4
    codes = np.full((NUM_CODEBOOKS, T), 7, dtype=np.int16)
    text = np.arange(T, dtype=np.int32)

    store.save_pair("uidA", codes, text, {"speaker": "spk_A", "text": "안녕"})
    store.save_pair("uidB", codes, text, {"speaker": "spk_B", "text": "hello"})
    # An artifact from before the speaker column existed.
    store.save_pair("uidC", codes, text, {"text": "legacy"})
    # A stray non-sample file that iter_samples must keep skipping.
    (tmp_path / "build_stats.json").write_text(json.dumps({"total": 3}))

    by_uid = {s.sample_uid: s for s in store.iter_samples()}
    assert set(by_uid) == {"uidA", "uidB", "uidC"}
    assert by_uid["uidA"].speaker == "spk_A"
    assert by_uid["uidB"].speaker == "spk_B"
    assert by_uid["uidC"].speaker == ""   # absent, not crashing

    # ...and it still makes it out the Arrow end.
    assert by_uid["uidA"].to_row()["speaker"] == "spk_A"


def test_schema_version_bumped_past_the_pre_speaker_value():
    """Adding a column to arrow_features() must invalidate stale prepared data.

    prepare.py stamps SCHEMA_VERSION into each group's _SUCCESS.json and rebuilds
    when it no longer matches. Version 1 is the pre-speaker schema, so reading a
    v1 corpus back through these features would yield rows with no speaker at all.
    """
    assert SCHEMA_VERSION > 1, (
        "SCHEMA_VERSION is still 1 (the pre-speaker value); corpora prepared "
        "without a speaker column would be silently reused"
    )


# --------------------------------------------------------------------- mimi


def test_mimi_default_checkpoint_is_the_cached_hf_conversion():
    """The old default (kyutai/moshiko-pytorch-bf16) needs the uninstalled `moshi`
    package. It must be the HF conversion, and it must be overridable."""
    assert DEFAULT_MIMI_CKPT == "kmhf/hf-moshiko"
    assert MimiEncoderMixin.mimi_ckpt_id == DEFAULT_MIMI_CKPT

    class _Custom(MimiEncoderMixin):
        mimi_ckpt_id = "some/other-moshi"

    assert _Custom.mimi_ckpt_id == "some/other-moshi"


def test_importing_mixins_loads_no_model():
    """The load must stay lazy: constructing the mixin must not touch the hub."""

    class _Enc(MimiEncoderMixin):
        device = "cpu"

    enc = _Enc()
    assert getattr(enc, "_mimi", None) is None


def _mimi_is_cached() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(DEFAULT_MIMI_CKPT, "config.json", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _mimi_is_cached(), reason=f"{DEFAULT_MIMI_CKPT} not in HF cache")
def test_mimi_encoder_actually_encodes():
    """The test that proves the previously-dead Mimi path runs.

    Encodes 2 s of silence and 1 s of a 440 Hz tone in one batch and checks the
    shape contract encode_audio promises (K, T) with T = round(L/24000*12.5), the
    int16 dtype the Arrow schema stores, and -- the part that actually shows the
    codec is doing something -- that the two signals do not produce the same codes.
    """

    class _Enc(MimiEncoderMixin):
        device = "cpu"

    enc = _Enc()

    silence = np.zeros((1, 2 * SAMPLE_RATE), dtype=np.float32)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)[None, :]

    codes_silence, codes_tone = enc.encode_audio([silence, tone])

    assert codes_silence.shape == (NUM_CODEBOOKS, round(2 * FRAME_RATE_HZ))   # (8, 25)
    assert codes_tone.shape == (NUM_CODEBOOKS, round(1 * FRAME_RATE_HZ))      # (8, 12)
    assert codes_silence.dtype == np.int16 and codes_tone.dtype == np.int16
    assert codes_silence.min() >= 0 and codes_tone.min() >= 0

    # Batching must not leak: the tone is shorter, so its tail was zero-padded and
    # truncated away. If padding bled through, the two would agree on the overlap.
    n = codes_tone.shape[1]
    assert not np.array_equal(codes_silence[:, :n], codes_tone), (
        "silence and a 440 Hz tone encoded to identical codes -- the encoder is "
        "not actually running"
    )


# ------------------------------------------------- sharded / chunked writing
#
# `write()` used to buffer every row of every split in a Python list and call
# Dataset.from_list once, while Sample.to_row emitted Python lists -- an 18x
# blowup over the packed dtype (a 2-byte int16 becomes a 28-byte Python int,
# which pyarrow then re-packs into 2 bytes). Fine for the 150 rows on disk,
# ~430 GB for the ~10k en_kd dialogues section 4.7's exposure budget needs.
# These pin the streaming path, which nothing else exercises: the fixtures are
# all far below ROWS_PER_CHUNK, so the flush never fires in the other tests.


def test_to_row_emits_arrays_not_python_lists():
    """The 18x memory blowup, pinned at the source."""
    row = _sample("spk").to_row()
    for key in ("codes_a", "text_tokens_a", "teacher_topk_val_a"):
        assert isinstance(row[key], np.ndarray), (
            f"{key} came back as {type(row[key]).__name__}; a Python list here "
            f"costs ~18x and pyarrow immediately undoes it"
        )
    assert row["codes_a"].dtype == np.int16
    assert row["teacher_topk_val_a"].size == 0, "absent tensors must stay empty"


def test_write_chunks_without_changing_the_result(tmp_path, monkeypatch):
    """Many small chunks must produce exactly what one big write would."""
    from datasets import load_from_disk

    from project_amnesty.datasets.offline import prepare_dataset as pd_mod

    n = 25
    rows = [_sample(f"spk{i}", uid=f"uid{i:03d}").to_row() for i in range(n)]

    def run(chunk_size: int, out: Path) -> dict:
        monkeypatch.setattr(pd_mod, "ROWS_PER_CHUNK", chunk_size)
        return pd_mod.write(iter(list(rows)), out, "ko_tts", holdout_ratio=0.0)

    one = tmp_path / "one"
    many = tmp_path / "many"
    counts_one = run(10_000, one)      # single chunk: the old behaviour
    counts_many = run(3, many)         # forces 9 flushes

    assert counts_one == counts_many == {"train": n}

    a = load_from_disk(str(one / "ko_tts" / "train"))
    b = load_from_disk(str(many / "ko_tts" / "train"))
    assert a.column_names == b.column_names
    assert sorted(a["sample_uid"]) == sorted(b["sample_uid"]) == sorted(
        r["sample_uid"] for r in rows
    )
    by_uid = {r["sample_uid"]: r for r in b}
    for row in a:
        other = by_uid[row["sample_uid"]]
        assert row["codes_a"] == other["codes_a"]
        assert row["speaker"] == other["speaker"]
        assert row["num_frames"] == other["num_frames"]


def test_write_removes_its_chunk_scratch(tmp_path, monkeypatch):
    """The chunks live under the destination; leaving them doubles corpus size
    and makes `_chunks` look like a prepared group to anything globbing the root."""
    from project_amnesty.datasets.offline import prepare_dataset as pd_mod

    monkeypatch.setattr(pd_mod, "ROWS_PER_CHUNK", 2)
    rows = [_sample("s", uid=f"u{i}").to_row() for i in range(7)]
    pd_mod.write(iter(rows), tmp_path, "ko_tts", holdout_ratio=0.0)

    assert not (tmp_path / "ko_tts" / "_chunks").exists()
    assert sorted(p.name for p in (tmp_path / "ko_tts").iterdir()) == ["train"]


def test_write_still_splits_train_and_probe_in_one_pass(tmp_path, monkeypatch):
    """`rows` is a generator over iter_samples(); a second pass would re-read
    every npz, so the split has to happen while streaming."""
    from project_amnesty.datasets.offline import prepare_dataset as pd_mod

    monkeypatch.setattr(pd_mod, "ROWS_PER_CHUNK", 2)
    rows = [_sample("s", uid=f"u{i:03d}").to_row() for i in range(40)]
    consumed = 0

    def counting():
        nonlocal consumed
        for r in rows:
            consumed += 1
            yield r

    counts = pd_mod.write(counting(), tmp_path, "ko_tts", holdout_ratio=0.5)

    assert consumed == len(rows), "rows were iterated more than once"
    assert set(counts) <= {"train", "probe"}
    assert sum(counts.values()) == len(rows)
    assert counts.get("probe", 0) > 0 and counts.get("train", 0) > 0


def test_rows_per_chunk_stays_small_enough_to_bound_memory():
    """ROWS_PER_CHUNK *is* the memory bound -- raising it far enough restores the
    unbounded behaviour with no test failing, because the correctness tests set
    it explicitly. en_kd sizes it: ~3 MB of teacher top-k per dialogue at
    K=8/T=1500/topk=32, so 512 rows is ~1.5 GB of buffer before a flush."""
    from project_amnesty.datasets.offline.prepare_dataset import ROWS_PER_CHUNK

    assert 1 <= ROWS_PER_CHUNK <= 4096, (
        f"ROWS_PER_CHUNK={ROWS_PER_CHUNK} no longer bounds the write buffer"
    )
