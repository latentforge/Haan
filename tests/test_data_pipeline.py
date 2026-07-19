"""Tests for the two data_pipeline defects fixed alongside them.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.datasets.mixins import (  # noqa: E402
    DEFAULT_MIMI_CKPT,
    MimiEncoderMixin,
    NpzPairIOMixin,
)
from data_pipeline.schema import (  # noqa: E402
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
