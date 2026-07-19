"""The silence *bank*: config validation, JSON loading, and phase-tiled filling.

Background, because the shape of these tests only makes sense with it. The user
channel of a solo sample used to be filled with one constant Mimi code per
codebook. Measured against the real codec (`kmhf/hf-moshiko`) that turns out to
be wrong: only cb0, the WavLM-distilled semantic VQ, is near-constant on silence
(share 0.943); cb1..7 are acoustic residuals over the noise floor with modal
shares of 0.24-0.55, and they do not cycle either -- cb2 sits on plateaus of
20-120 frames while cb4/cb6 alternate among a handful of values with runs of 1-3.

So the fill is a `(K, P)` bank of real consecutive silence frames, tiled with a
per-sample random phase. Two properties have to hold and neither is visible in a
loss curve if it breaks:

* the fill is a *rotation* of the bank -- one phase shared by all K codebooks,
  advancing one frame per timestep. Filling every frame with `bank[:, 0]`, or
  drawing an independent phase per codebook, both produce a tensor that looks
  fine and is not silence.
* the phase is a pure function of (seed, epoch, index), like the crop. A phase
  drawn from global RNG or from worker state makes the dataset depend on
  `num_workers`, which no test of the *values* would catch.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from conftest import DUMMY_SILENCE, DUMMY_SILENCE_BANK, make_solo_sample
from project_amnesty.datasets.schema import CODEBOOK_SIZE, NUM_CODEBOOKS
from project_amnesty.datasets.config import DataConfig, TokenConfig
from project_amnesty.datasets.dataset import MoshiKDDataset

K = NUM_CODEBOOKS
P = len(DUMMY_SILENCE_BANK[0])

# The degenerate bank: P == 1 is exactly the old one-code-per-codebook contract,
# so it must remain expressible and must produce a constant fill.
CONSTANT_BANK = tuple((c,) for c in DUMMY_SILENCE)


def _tokens(**kw) -> TokenConfig:
    base = dict(
        text_pad_id=151_000,
        text_epad_id=151_001,
        batch_pad_id=151_002,
        audio_init_id=CODEBOOK_SIZE,
        silence_bank=DUMMY_SILENCE_BANK,
        mimi_ckpt_id="test/dummy-mimi",
    )
    base.update(kw)
    return TokenConfig(**base)


# --------------------------------------------------------------- config shape


def test_bank_array_is_k_by_p_int16():
    arr = _tokens().silence_bank_array()
    assert arr.shape == (K, P)
    assert arr.dtype == np.int16
    np.testing.assert_array_equal(arr[:, 0], np.asarray(DUMMY_SILENCE, np.int16))


def test_bank_of_period_one_is_accepted():
    arr = _tokens(silence_bank=CONSTANT_BANK).silence_bank_array()
    assert arr.shape == (K, 1)


def test_require_rejects_a_bank_with_the_wrong_codebook_count():
    t = _tokens(silence_bank=DUMMY_SILENCE_BANK[:-1])
    with pytest.raises(AssertionError, match="codebooks, expected"):
        t.require("silence_bank")


def test_require_rejects_out_of_range_codes():
    bad = list(list(r) for r in DUMMY_SILENCE_BANK)
    bad[3][2] = CODEBOOK_SIZE
    with pytest.raises(AssertionError, match="out of range"):
        _tokens(silence_bank=bad).require("silence_bank")


def test_a_ragged_bank_is_rejected_at_construction():
    """numpy would turn a ragged bank into a 1-D object array and `bank[:, j]`
    would then return nonsense rather than raise."""
    ragged = [list(r) for r in DUMMY_SILENCE_BANK]
    ragged[2] = ragged[2][:-1]
    with pytest.raises(AssertionError, match="ragged"):
        _tokens(silence_bank=ragged)


def test_an_empty_bank_is_rejected():
    with pytest.raises(AssertionError, match=r"P=0|empty"):
        _tokens(silence_bank=tuple(() for _ in range(K)))


def test_require_still_names_the_config_file_when_the_bank_is_unset():
    with pytest.raises(AssertionError, match="configs/tokens.yaml"):
        TokenConfig().require("silence_bank")


# ----------------------------------------------------------- loading the json


def _payload(**kw) -> dict:
    base = {
        "silence_bank": [list(r) for r in DUMMY_SILENCE_BANK],
        "mimi_ckpt_id": "test/dummy-mimi",
        "bank_period": P,
    }
    base.update(kw)
    return base


def test_bank_is_loaded_from_the_derived_json(tmp_path):
    p = tmp_path / "mimi_silence.json"
    p.write_text(json.dumps(_payload()))
    t = TokenConfig(silence_bank_path=str(p), mimi_ckpt_id="test/dummy-mimi")
    np.testing.assert_array_equal(
        t.silence_bank_array(), np.asarray(DUMMY_SILENCE_BANK, np.int16)
    )


def test_a_bank_from_a_different_codec_is_refused(tmp_path):
    """The exact failure mimi_ckpt_id exists for: silence that is plausible for
    the wrong model produces a user channel nothing downstream will flag."""
    p = tmp_path / "mimi_silence.json"
    p.write_text(json.dumps(_payload(mimi_ckpt_id="someone/other-mimi")))
    with pytest.raises(AssertionError, match="codec mismatch"):
        TokenConfig(silence_bank_path=str(p), mimi_ckpt_id="test/dummy-mimi")


def test_the_old_pre_bank_format_is_refused(tmp_path):
    """A stale mimi_silence.json carries `silence_codes`, not `silence_bank`.
    Silently treating a missing key as "unset" would defer the failure to a
    require() far from the cause."""
    p = tmp_path / "mimi_silence.json"
    p.write_text(json.dumps({"silence_codes": list(DUMMY_SILENCE), "mimi_ckpt_id": "x"}))
    with pytest.raises(AssertionError, match="pre-bank"):
        TokenConfig(silence_bank_path=str(p))


def test_a_missing_bank_file_names_the_derivation_command(tmp_path):
    with pytest.raises(AssertionError, match="derive_silence_codes"):
        TokenConfig(silence_bank_path=str(tmp_path / "nope.json"))


def test_an_inline_bank_wins_over_the_path(tmp_path):
    """Tests inject the bank directly; that must not require a file on disk."""
    t = TokenConfig(
        silence_bank=CONSTANT_BANK, silence_bank_path=str(tmp_path / "nope.json")
    )
    assert t.silence_bank_array().shape == (K, 1)


# ------------------------------------------------------------ the actual fill


@pytest.fixture
def solo_ds_factory(prepared, data_cfg):
    def _make(cfg: DataConfig | None = None, n: int = 12, T: int = 30):
        cfg = cfg or data_cfg
        prepared("ko_tts", "train", [make_solo_sample(f"ko{i}", T=T) for i in range(n)])
        return MoshiKDDataset(cfg.root, "ko_tts", "train", cfg=cfg)

    return _make


def _phase_of(codes: np.ndarray, bank: np.ndarray) -> int:
    """Recover the phase from a filled channel, asserting it really is a rotation."""
    Pb = bank.shape[1]
    matches = [
        j for j in range(Pb)
        if np.array_equal(codes, bank[:, (j + np.arange(codes.shape[1])) % Pb])
    ]
    assert matches, (
        "the filled user channel is not any rotation of the bank -- it is not the "
        "bank tiled frame-by-frame"
    )
    return matches[0]


def test_fill_is_a_rotation_of_the_bank(solo_ds_factory):
    ds = solo_ds_factory()
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)
    for i in range(len(ds)):
        codes = ds[i]["codes_other"].numpy()
        assert codes.shape[1] == int(ds[i]["num_frames"])
        _phase_of(codes, bank)  # raises if it is not a clean rotation


def test_fill_advances_one_frame_per_timestep(solo_ds_factory):
    """The failure this catches: `bank[:, phase]` broadcast over T instead of
    `bank[:, (phase + t) % P]`. Every frame identical, phase still 'random'."""
    ds = solo_ds_factory()
    codes = ds[0]["codes_other"].numpy()
    assert codes.shape[1] > P
    assert not np.array_equal(codes[:, 0], codes[:, 1]), (
        "consecutive frames are identical -- the bank is not being tiled"
    )
    # And the same phase applies to every codebook, not one per row.
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)
    j = _phase_of(codes, bank)
    np.testing.assert_array_equal(codes[:, 0], bank[:, j])


def test_phase_varies_across_samples(solo_ds_factory):
    """A single fixed phase is as separable as a constant fill was."""
    ds = solo_ds_factory(n=16)
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)
    phases = {_phase_of(ds[i]["codes_other"].numpy(), bank) for i in range(len(ds))}
    assert len(phases) > 1, f"every sample got the same phase {phases}"


def test_phase_is_reproducible_across_worker_counts(solo_ds_factory):
    """Same guarantee as the crop: a pure function of (seed, epoch, index).

    A phase drawn from global numpy RNG passes every value test above and still
    makes the dataset depend on num_workers.
    """
    from torch.utils.data import DataLoader

    ds = solo_ds_factory(n=16)
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)

    def run(nw):
        dl = DataLoader(ds, batch_size=None, num_workers=nw, collate_fn=None, shuffle=False)
        return [_phase_of(it["codes_other"].numpy(), bank) for it in dl]

    ref = run(0)
    assert len(set(ref)) > 1, "fixture too weak: all phases identical"
    for nw in (2, 4):
        assert run(nw) == ref, f"phases differ at num_workers={nw}"


def test_phase_follows_the_epoch(solo_ds_factory):
    ds = solo_ds_factory(n=16)
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)

    def phases():
        return [_phase_of(ds[i]["codes_other"].numpy(), bank) for i in range(len(ds))]

    ds.set_epoch(0)
    e0 = phases()
    ds.set_epoch(1)
    assert phases() != e0
    ds.set_epoch(0)
    assert phases() == e0, "epoch 0 not reproducible"


def test_period_one_bank_reduces_to_the_old_constant_fill(prepared, data_cfg):
    """P == 1 must be exactly the pre-bank behaviour, so nothing that used to be
    expressible was lost in the change."""
    cfg = dataclasses.replace(data_cfg, tokens=_tokens(silence_bank=CONSTANT_BANK))
    prepared("ko_tts", "train", [make_solo_sample(f"ko{i}", T=30) for i in range(4)])
    ds = MoshiKDDataset(cfg.root, "ko_tts", "train", cfg=cfg)

    for i in range(len(ds)):
        codes = ds[i]["codes_other"].numpy()
        T = codes.shape[1]
        expected = np.broadcast_to(np.asarray(DUMMY_SILENCE, np.int16)[:, None], (K, T))
        np.testing.assert_array_equal(codes, expected)


def test_fill_is_never_all_zeros(solo_ds_factory):
    """0 is a valid Mimi code; an all-zero user channel is the original bug."""
    ds = solo_ds_factory()
    assert not np.all(ds[0]["codes_other"].numpy() == 0)
