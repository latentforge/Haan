"""Shared fixtures for the project_amnesty/datasets test suite.

Fixtures build real Arrow datasets through Sample.to_row() + arrow_features() --
the same path prepare_dataset.write() uses. Arrow is never mocked: the flat-store
/ reshape-by-num_frames round-trip is precisely what these tests exist to check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tests live in per-package subdirectories (tests/datasets, tests/losses, tests/tools),
# so pytest puts *their* directory on sys.path, not this one. The `from conftest
# import make_*_sample` in those files needs tests/ itself.
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from project_amnesty.datasets.shared.schema import CODEBOOK_SIZE, NUM_CODEBOOKS, Sample, arrow_features  # noqa: E402
from project_amnesty.datasets.runtime.config import DataConfig, TokenConfig  # noqa: E402

# Dummy ids standing in for the still-unassigned reserved Qwen3 slots. Tests must
# never depend on the real values -- that is the whole point of config injection.
DUMMY_TEXT_PAD = 151_000
DUMMY_TEXT_EPAD = 151_001
DUMMY_BATCH_PAD = 151_002
DUMMY_SILENCE = (11, 22, 33, 44, 55, 66, 77, 88)

# The silence fill is a (K, P) bank of real frames, tiled with a per-sample phase
# -- not one constant per codebook. P > 1 here on purpose: with P == 1 the tiling
# is the identity and a dropped phase (or a transposed index) would still pass.
# bank[k][j] == DUMMY_SILENCE[k] + j, so both axes are identifiable in a failure.
DUMMY_SILENCE_PERIOD = 5
DUMMY_SILENCE_BANK = tuple(
    tuple(c + j for j in range(DUMMY_SILENCE_PERIOD)) for c in DUMMY_SILENCE
)


@pytest.fixture
def tokens() -> TokenConfig:
    return TokenConfig(
        text_pad_id=DUMMY_TEXT_PAD,
        text_epad_id=DUMMY_TEXT_EPAD,
        batch_pad_id=DUMMY_BATCH_PAD,
        audio_init_id=CODEBOOK_SIZE,
        silence_bank=DUMMY_SILENCE_BANK,
        mimi_ckpt_id="test/dummy-mimi",
    )


@pytest.fixture
def data_cfg(tokens: TokenConfig, tmp_path: Path) -> DataConfig:
    return DataConfig(
        root=str(tmp_path / "prepared"),
        tokens=tokens,
        max_frames=64,
        crop_mode="random",
        double_ab=True,
        seed=0,
        debug_validate=True,
    )


def ramp_codes(K: int, T: int, offset: int = 0) -> np.ndarray:
    """codes[k, t] = (t + offset) % CODEBOOK_SIZE.

    A per-frame serial number: any crop/delay/slice misalignment shows up as a
    wrong value rather than as a plausible-looking wrong tensor.
    """
    t = (np.arange(T, dtype=np.int64) + offset) % CODEBOOK_SIZE
    return np.broadcast_to(t, (K, T)).astype(np.int16).copy()


def ramp_text(T: int, pad_id: int = DUMMY_TEXT_PAD, every: int = 3) -> np.ndarray:
    """Frame-aligned text: a serial number every `every` frames, PAD elsewhere."""
    out = np.full((T,), pad_id, dtype=np.int32)
    idx = np.arange(0, T, every)
    out[idx] = idx.astype(np.int32)
    return out


def ramp_teacher(K: int, T: int, topk: int, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Teacher top-k whose slot 0 is exactly the sampled token at that frame.

    Mirrors reality (the token was drawn from these very logits) and makes the
    teacher/student frame-alignment check assertable: after collation, slot 0 must
    still line up with codebook 0 at the same output position.
    """
    idx = np.zeros((K, T, topk), dtype=np.int16)
    idx[:, :, 0] = codes
    for j in range(1, topk):
        idx[:, :, j] = (codes.astype(np.int64) + j * 7) % CODEBOOK_SIZE
    val = np.linspace(4.0, -4.0, topk, dtype=np.float32)
    val = np.broadcast_to(val, (K, T, topk)).copy()
    return val, idx


def make_en_kd_sample(uid: str, T: int, topk: int = 4, K: int = NUM_CODEBOOKS) -> Sample:
    codes_a = ramp_codes(K, T, offset=0)
    codes_b = ramp_codes(K, T, offset=1000)
    val_a, idx_a = ramp_teacher(K, T, topk, codes_a)
    val_b, idx_b = ramp_teacher(K, T, topk, codes_b)
    return Sample(
        sample_type="en_kd", lang="en",
        codes_a=codes_a, codes_b=codes_b,
        text_tokens_a=ramp_text(T), text_tokens_b=ramp_text(T, every=4),
        teacher_topk_val_a=val_a, teacher_topk_idx_a=idx_a,
        teacher_topk_val_b=val_b, teacher_topk_idx_b=idx_b,
        gen_meta={"seed": 1, "gen_temperature": 0.8, "gen_top_k": 250, "seed_prompt_id": "p0"},
        sample_uid=uid,
    )


def make_solo_sample(
    uid: str,
    T: int,
    lang: str = "ko",
    K: int = NUM_CODEBOOKS,
    speaker: str = "",
    offset: int = 0,
) -> Sample:
    """ko_tts / en_solo shape: mono, no B stream, no teacher.

    `offset` shifts the code ramp so two rows by the same speaker are
    distinguishable -- a voice-prompt test that cannot tell the reference from
    the sample proves nothing.
    """
    return Sample(
        sample_type="ko_tts", lang=lang,
        codes_a=ramp_codes(K, T, offset=offset), text_tokens_a=ramp_text(T),
        sample_uid=uid, speaker=speaker,
    )


def make_text_anchor_sample(uid: str, L: int, K: int = NUM_CODEBOOKS) -> Sample:
    return Sample(
        sample_type="text_anchor", lang="en",
        codes_a=np.zeros((K, 0), dtype=np.int16),
        text_tokens_a=np.arange(L, dtype=np.int32),
        sample_uid=uid,
    )


def write_prepared(root: Path, source: str, split: str, samples: list[Sample]) -> Path:
    """Write samples as prepared Arrow, exactly as prepare_dataset.write() does."""
    from datasets import Dataset as HFDataset

    path = Path(root) / source / split
    ds = HFDataset.from_list([s.to_row() for s in samples], features=arrow_features())
    ds.save_to_disk(str(path))
    return path


@pytest.fixture
def prepared(tmp_path: Path):
    """Factory: prepared(source, split, samples) -> path under a temp prepared root."""

    def _make(source: str, split: str, samples: list[Sample]) -> Path:
        return write_prepared(tmp_path / "prepared", source, split, samples)

    return _make
