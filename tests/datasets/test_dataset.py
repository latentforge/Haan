"""Dataset-layer tests -- plan section 6, items 6 through 24.

The through-line: every assertion here is about something that fails *silently*.
A wrong crop offset, a teacher that follows the wrong speaker across a swap, a
solo row filled with code 0 instead of the real silence code -- none of these
crash. They train.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections import Counter

import numpy as np
import pytest
import torch

from conftest import (
    DUMMY_SILENCE_BANK,
    DUMMY_TEXT_PAD,
    make_en_kd_sample,
    make_solo_sample,
    make_text_anchor_sample,
)
from project_amnesty.datasets.shared.schema import CODEBOOK_SIZE, NUM_CODEBOOKS, row_to_arrays
from project_amnesty.datasets.runtime.crop import Window, choose_window
from project_amnesty.datasets.runtime.dataset import MoshiKDDataset, build_source_datasets
from project_amnesty.datasets.runtime.item import ITEM_KEYS

K = NUM_CODEBOOKS


def _cfg(base, **over):
    return dataclasses.replace(base, **over)


@pytest.fixture
def en_kd_ds(prepared, data_cfg):
    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=40 + 7 * i) for i in range(4)])
    return MoshiKDDataset(data_cfg.root, "en_kd", "train", cfg=data_cfg)


@pytest.fixture
def solo_ds(prepared, data_cfg):
    prepared("ko_tts", "train", [make_solo_sample(f"ko{i}", T=30 + 5 * i) for i in range(3)])
    return MoshiKDDataset(data_cfg.root, "ko_tts", "train", cfg=data_cfg)


@pytest.fixture
def anchor_ds(prepared, data_cfg):
    prepared("text_anchor", "train", [make_text_anchor_sample(f"tx{i}", L=17 + i) for i in range(3)])
    return MoshiKDDataset(data_cfg.root, "text_anchor", "train", cfg=data_cfg)


# --- 6, 7: the item contract -------------------------------------------------


def test_06_item_keys_dtypes_and_shapes(en_kd_ds, solo_ds, anchor_ds):
    for ds in (en_kd_ds, solo_ds, anchor_ds):
        item = ds[0]
        assert set(item) == set(ITEM_KEYS), f"{ds.source}: key set drift"

        for k in ("sample_uid", "source", "sample_type", "lang"):
            assert isinstance(item[k], str)
        for k in ("swapped", "is_text_only", "has_teacher", "use_kd", "use_ce_audio", "use_ce_text"):
            assert item[k].dtype is torch.bool and item[k].ndim == 0
        for k in ("num_frames", "topk"):
            assert item[k].dtype is torch.int32 and item[k].ndim == 0

        assert item["codes_self"].dtype is torch.int16
        assert item["codes_other"].dtype is torch.int16
        assert item["text_self"].dtype is torch.int32
        assert item["text_other"].dtype is torch.int32
        assert item["text_flat"].dtype is torch.int32
        assert item["teacher_val"].dtype is torch.float16
        assert item["teacher_idx"].dtype is torch.int16


def test_07_frame_axes_agree_with_num_frames(en_kd_ds, solo_ds):
    for ds in (en_kd_ds, solo_ds):
        for i in range(len(ds)):
            it = ds[i]
            T = int(it["num_frames"])
            assert it["codes_self"].shape == (K, T)
            assert it["codes_other"].shape == (K, T)
            assert it["text_self"].shape == (T,)
            assert it["text_other"].shape == (T,)
            if bool(it["has_teacher"]):
                assert it["teacher_val"].shape == (K, T, int(it["topk"]))
                assert it["teacher_idx"].shape == (K, T, int(it["topk"]))


# --- 8, 9: the fast path is the only thing that reads Arrow ------------------


def test_08_fast_path_matches_row_to_arrays_oracle(prepared, data_cfg):
    """The zero-copy buffer read has exactly one oracle: the reference reshape."""
    samples = [make_en_kd_sample(f"kd{i}", T=20 + 3 * i, topk=4) for i in range(3)]
    prepared("en_kd", "train", samples)
    cfg = _cfg(data_cfg, max_frames=10_000, crop_mode="head")  # no cropping
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)

    hf = ds._hf()
    for row in range(len(samples)):
        oracle = row_to_arrays(hf[row])
        for swapped in (False, True):
            it = ds[row * 2 + int(swapped)]
            lo, hi = ("b", "a") if swapped else ("a", "b")
            np.testing.assert_array_equal(it["codes_self"].numpy(), oracle[f"codes_{lo}"])
            np.testing.assert_array_equal(it["codes_other"].numpy(), oracle[f"codes_{hi}"])
            np.testing.assert_array_equal(it["text_self"].numpy(), oracle[f"text_tokens_{lo}"])
            np.testing.assert_array_equal(it["text_other"].numpy(), oracle[f"text_tokens_{hi}"])
            np.testing.assert_array_equal(
                it["teacher_val"].numpy(), oracle[f"teacher_topk_val_{lo}"]
            )
            np.testing.assert_array_equal(
                it["teacher_idx"].numpy(), oracle[f"teacher_topk_idx_{lo}"]
            )


def test_08b_fast_path_matches_oracle_under_crop(prepared, data_cfg):
    """Same equality, but through the crop -- catches a window applied to one
    array and not another."""
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=100, topk=3)])
    cfg = _cfg(data_cfg, max_frames=37, crop_mode="center")
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)
    oracle = row_to_arrays(ds._hf()[0])
    w = choose_window(100, 37, None, "center")

    it = ds[0]
    np.testing.assert_array_equal(it["codes_self"].numpy(), oracle["codes_a"][:, w.start : w.end])
    np.testing.assert_array_equal(
        it["teacher_val"].numpy(), oracle["teacher_topk_val_a"][:, w.start : w.end, :]
    )
    np.testing.assert_array_equal(
        it["teacher_idx"].numpy(), oracle["teacher_topk_idx_a"][:, w.start : w.end, :]
    )


def test_09_does_not_use_datasets_getitem(en_kd_ds, monkeypatch):
    """Regression guard against reverting to full-row materialization.

    hf_ds[i] pulls every column -- both teachers -- into Python objects. If this
    test ever passes again after someone "simplifies" the reader, the crop-early
    optimization is dead and the payload has doubled.
    """
    import datasets

    en_kd_ds._columns()  # warm the pid-keyed cache through the legal path

    def boom(self, key):
        raise AssertionError("datasets.Dataset.__getitem__ used on the hot path")

    monkeypatch.setattr(datasets.Dataset, "__getitem__", boom)
    for i in range(len(en_kd_ds)):
        en_kd_ds[i]


# --- 10, 11, 12: per-source normalization ------------------------------------


def test_10_solo_other_channel_is_silence_not_zero(solo_ds):
    it = solo_ds[0]
    T = int(it["num_frames"])
    got = it["codes_other"].numpy()
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)
    P = bank.shape[1]
    # The fill is the bank tiled from *some* phase; which phase is the RNG's
    # business, but it must be the same phase for every codebook (one rotation of
    # the bank, not eight independent ones) and it must tile, not repeat frame 0.
    phase = int(np.flatnonzero(bank[0] == got[0, 0])[0])
    expected = bank[:, (phase + np.arange(T)) % P]
    np.testing.assert_array_equal(got, expected)
    # 0 is a perfectly valid Mimi code, so "all zeros" is the failure this guards.
    assert not np.all(got == 0)
    np.testing.assert_array_equal(
        it["text_other"].numpy(), np.full((T,), DUMMY_TEXT_PAD, np.int32)
    )
    assert not bool(it["has_teacher"]) and int(it["topk"]) == 0
    assert not bool(it["use_kd"]) and bool(it["use_ce_audio"])


def test_11_en_solo_source_is_not_sample_type(prepared, data_cfg):
    """The trap: en_solo rows carry sample_type='ko_tts' and lang='en'."""
    prepared("en_solo", "train", [make_solo_sample("es0", T=30, lang="en")])
    ds = MoshiKDDataset(data_cfg.root, "en_solo", "train", cfg=data_cfg)
    it = ds[0]
    assert it["source"] == "en_solo"
    assert it["sample_type"] == "ko_tts"
    assert it["lang"] == "en"
    assert ds.double_ab is False  # no B stream despite cfg.double_ab=True
    # Behavior branched on sample_type: it got the solo silence fill. Column 0 is
    # some column of the bank (the phase is per-sample), which is enough to
    # distinguish "silence-filled" from "left as zeros" or "copied from codes_a".
    bank = np.asarray(DUMMY_SILENCE_BANK, np.int16)
    col0 = it["codes_other"].numpy()[:, 0]
    assert any(np.array_equal(col0, bank[:, j]) for j in range(bank.shape[1]))


def test_12_text_anchor_is_genuinely_empty(anchor_ds):
    it = anchor_ds[0]
    assert bool(it["is_text_only"])
    assert int(it["num_frames"]) == 0
    assert it["codes_self"].shape == (K, 0)
    assert it["codes_other"].shape == (K, 0)
    assert it["text_self"].shape == (0,)
    assert it["text_other"].shape == (0,)
    assert it["teacher_val"].numel() == 0 and it["teacher_idx"].numel() == 0
    assert it["text_flat"].numel() == 17
    np.testing.assert_array_equal(it["text_flat"].numpy(), np.arange(17, dtype=np.int32))
    assert not bool(it["use_kd"])
    assert not bool(it["use_ce_audio"])
    assert not bool(it["use_ce_text"])  # anchors drive loss off text_flat, separate path
    assert anchor_ds.double_ab is False


# --- 13 through 16: the A/B axis ---------------------------------------------


def test_13_len_doubles_for_en_kd_only(en_kd_ds, solo_ds, anchor_ds):
    assert len(en_kd_ds) == 2 * en_kd_ds._n_rows
    assert len(solo_ds) == solo_ds._n_rows
    assert len(anchor_ds) == anchor_ds._n_rows


def test_13b_double_ab_can_be_disabled(prepared, data_cfg):
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=30)])
    ds = MoshiKDDataset(data_cfg.root, "en_kd", "train", cfg=data_cfg, double_ab=False)
    assert len(ds) == 1 and ds[0]["swapped"].item() is False


def test_14_swap_is_an_involution(en_kd_ds):
    for row in range(en_kd_ds._n_rows):
        a, b = en_kd_ds[row * 2], en_kd_ds[row * 2 + 1]
        assert a["sample_uid"] == b["sample_uid"]
        assert not bool(a["swapped"]) and bool(b["swapped"])
        # Same crop (same epoch, but different index -> different window), so
        # compare the un-cropped identity instead: self/other are exchanged.
        assert a["codes_self"].shape == a["codes_other"].shape


def test_14b_swap_exchanges_self_and_other(prepared, data_cfg):
    """With cropping disabled the exchange is exact, elementwise."""
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=30)])
    cfg = _cfg(data_cfg, max_frames=10_000)
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)
    a, b = ds[0], ds[1]
    np.testing.assert_array_equal(a["codes_self"].numpy(), b["codes_other"].numpy())
    np.testing.assert_array_equal(a["codes_other"].numpy(), b["codes_self"].numpy())
    np.testing.assert_array_equal(a["text_self"].numpy(), b["text_other"].numpy())
    np.testing.assert_array_equal(a["text_other"].numpy(), b["text_self"].numpy())


def test_15_teacher_follows_the_self_side(prepared, data_cfg):
    """The highest-value test in the file.

    Keeping the A-side teacher after a swap does not crash and does not change
    any shape -- it just points the KD loss at the speaker the model is not
    predicting. The fixture makes teacher slot 0 equal the sampled token, so
    "teacher follows self" is directly assertable.
    """
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=45, topk=4)])
    cfg = _cfg(data_cfg, max_frames=10_000)
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)
    oracle = row_to_arrays(ds._hf()[0])

    unswapped, swapped = ds[0], ds[1]

    np.testing.assert_array_equal(
        unswapped["teacher_idx"].numpy(), oracle["teacher_topk_idx_a"]
    )
    np.testing.assert_array_equal(swapped["teacher_idx"].numpy(), oracle["teacher_topk_idx_b"])

    # And the structural statement, independent of the oracle: slot 0 of the
    # teacher is the token in codes_self, in both directions.
    for it in (unswapped, swapped):
        np.testing.assert_array_equal(
            it["teacher_idx"][:, :, 0].numpy(), it["codes_self"].numpy()
        )
    # The A/B streams are distinguishable (offset 1000), so the swapped case is
    # not passing by coincidence.
    assert not np.array_equal(
        unswapped["teacher_idx"].numpy(), swapped["teacher_idx"].numpy()
    )


def test_15b_swapped_teacher_is_not_carried_from_both_sides(en_kd_ds):
    """Only one side's teacher is materialized -- shape proves the other was
    dropped rather than concatenated."""
    it = en_kd_ds[1]
    assert it["teacher_val"].shape[0] == K
    assert it["teacher_val"].ndim == 3


def test_16_every_uid_seen_in_both_directions_once_per_epoch(en_kd_ds):
    seen = Counter()
    for i in range(len(en_kd_ds)):
        it = en_kd_ds[i]
        seen[(it["sample_uid"], bool(it["swapped"]))] += 1
    assert len(seen) == 2 * en_kd_ds._n_rows
    assert set(seen.values()) == {1}


# --- 17 through 20: cropping --------------------------------------------------


def test_17_crop_length_and_alignment(prepared, data_cfg):
    """Ramp data turns any off-by-one into a wrong value, not a plausible tensor.

    Both slice paths are checked: (K,T) codes and (K,T,topk) teacher.
    """
    T, topk = 200, 3
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=T, topk=topk)])
    cfg = _cfg(data_cfg, max_frames=64, crop_mode="random")
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)

    it = ds[0]
    assert int(it["num_frames"]) == 64
    assert it["codes_self"].shape == (K, 64)
    assert it["teacher_val"].shape == (K, 64, topk)

    # codes_a is ramp offset 0: value at cropped position i is (start + i).
    start = int(it["codes_self"][0, 0])
    assert 0 <= start <= T - 64
    expected = (np.arange(start, start + 64) % CODEBOOK_SIZE).astype(np.int16)
    np.testing.assert_array_equal(it["codes_self"].numpy(), np.broadcast_to(expected, (K, 64)))
    # The teacher window must be the *same* window, not an independently drawn one.
    np.testing.assert_array_equal(it["teacher_idx"][:, :, 0].numpy(), it["codes_self"].numpy())
    # text is frame-aligned: ramp_text puts the absolute frame index every 3 frames.
    text = it["text_self"].numpy()
    for i in range(64):
        if (start + i) % 3 == 0:
            assert text[i] == start + i
        else:
            assert text[i] == DUMMY_TEXT_PAD


def test_17b_short_rows_are_not_cropped(prepared, data_cfg):
    prepared("ko_tts", "train", [make_solo_sample("ko0", T=10)])
    ds = MoshiKDDataset(data_cfg.root, "ko_tts", "train", cfg=data_cfg)  # max_frames=64
    assert int(ds[0]["num_frames"]) == 10


def test_18_probe_crop_is_deterministic_and_centered(prepared, data_cfg):
    prepared("en_kd", "probe", [make_en_kd_sample("kd0", T=200)])
    cfg = _cfg(data_cfg, crop_mode="random")  # config says random...
    ds = MoshiKDDataset(cfg.root, "en_kd", "probe", cfg=cfg)
    assert ds.crop_mode == "center"  # ...__init__ hard-forces center for probe

    starts = set()
    for epoch in range(3):
        ds.set_epoch(epoch)
        starts.add(int(ds[0]["codes_self"][0, 0]))
    assert starts == {(200 - 64) // 2}


def test_19_crop_output_is_contiguous_and_small(prepared, data_cfg):
    """A slice is a view that keeps the whole mmapped parent row alive; shipping
    one across the worker boundary undoes the crop entirely."""
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=400, topk=8)])
    ds = MoshiKDDataset(data_cfg.root, "en_kd", "train", cfg=data_cfg)
    it = ds[0]
    for key in ("codes_self", "codes_other", "text_self", "text_other", "teacher_val", "teacher_idx"):
        assert it[key].is_contiguous(), f"{key} is not contiguous"
    # Payload is proportional to the crop, not the row.
    assert it["teacher_val"].numel() == K * 64 * 8
    assert len(pickle.dumps(it)) < K * 400 * 8 * 2


def test_19b_uncropped_rows_still_own_their_memory(prepared, data_cfg):
    """The case ascontiguousarray alone gets wrong.

    When the window covers the whole row the slice is already contiguous, so
    ascontiguousarray is a no-op and the tensor stays a view onto the read-only
    mmapped Arrow column -- parent buffer pinned, tensor non-writable. Every
    ko_tts sample takes this path.
    """
    prepared("ko_tts", "train", [make_solo_sample("ko0", T=20)])
    ds = MoshiKDDataset(data_cfg.root, "ko_tts", "train", cfg=data_cfg)  # max_frames=64 > T
    it = ds[0]
    assert int(it["num_frames"]) == 20  # confirms no crop happened
    for key in ("codes_self", "codes_other", "text_self", "text_other"):
        assert it[key].numpy().flags.writeable, f"{key} is a read-only mmap view"

    # The decisive check: writing must not reach back into the Arrow buffer, i.e.
    # a second read of the same row is unaffected.
    original = int(it["codes_self"][0, 3])
    it["codes_self"][0, 3] = 1234
    assert int(ds[0]["codes_self"][0, 3]) == original, "item aliases the mmapped column"


def test_20_crop_identical_across_worker_counts(prepared, data_cfg):
    from torch.utils.data import DataLoader

    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=300) for i in range(4)])
    ds = MoshiKDDataset(data_cfg.root, "en_kd", "train", cfg=data_cfg)

    def run(nw):
        dl = DataLoader(ds, batch_size=None, num_workers=nw, collate_fn=None, shuffle=False)
        return [(it["sample_uid"], bool(it["swapped"]), int(it["codes_self"][0, 0])) for it in dl]

    ref = run(0)
    assert len({s for _, _, s in ref}) > 1, "fixture too weak: all crops identical"
    for nw in (2, 4):
        assert run(nw) == ref, f"crops differ at num_workers={nw}"


def test_20b_set_epoch_changes_crops_deterministically(prepared, data_cfg):
    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=300) for i in range(4)])
    ds = MoshiKDDataset(data_cfg.root, "en_kd", "train", cfg=data_cfg)

    def starts():
        return [int(ds[i]["codes_self"][0, 0]) for i in range(len(ds))]

    ds.set_epoch(0)
    e0 = starts()
    ds.set_epoch(1)
    e1 = starts()
    assert e0 != e1, "set_epoch did not change the crop -- the persistent_workers failure"
    ds.set_epoch(0)
    assert starts() == e0, "epoch 0 not reproducible"


# --- 21 through 24: determinism, transport, teacher semantics ----------------


def test_21_getitem_does_not_touch_global_rng(en_kd_ds):
    np.random.seed(1234)
    torch.manual_seed(1234)
    np_before = np.random.get_state()
    torch_before = torch.random.get_rng_state()

    for i in range(len(en_kd_ds)):
        en_kd_ds[i]

    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    np.testing.assert_array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_22_dataset_is_picklable_and_payload_is_small(en_kd_ds):
    en_kd_ds[0]  # force the Arrow handle + column cache open
    assert en_kd_ds._hf_ds is not None and en_kd_ds._cols is not None

    blob = pickle.dumps(en_kd_ds)
    assert len(blob) < 8192, f"pickle carries {len(blob)} bytes -- an Arrow table leaked in"

    clone = pickle.loads(blob)
    assert clone._hf_ds is None and clone._cols is None
    assert len(clone) == len(en_kd_ds)
    np.testing.assert_array_equal(
        clone[3]["codes_self"].numpy(), en_kd_ds[3]["codes_self"].numpy()
    )
    assert en_kd_ds._hf_ds is not None  # original handle untouched by the clone's reopen


def test_23_teacher_values_are_raw_logits(prepared, data_cfg):
    """No temperature, no softmax. A helpful softmax here would be invisible
    downstream and would silently bake in a hyperparameter that is meant to be swept.
    """
    topk = 4
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=30, topk=topk)])
    cfg = _cfg(data_cfg, max_frames=10_000)
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)
    val = ds[0]["teacher_val"].float().numpy()

    expected = np.linspace(4.0, -4.0, topk, dtype=np.float32).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(val[0, 0], expected, rtol=0, atol=0)
    assert val.min() < 0.0, "negative logits gone -- something normalized them"
    assert not np.allclose(val.sum(axis=-1), 1.0), "top-k slots sum to 1 -- softmax was applied"


def test_24_no_delay_is_applied(prepared, data_cfg):
    """Storage is delay-free; delay is the collator's job. The ramp makes a
    one-frame shift in any stream immediately visible.
    """
    T = 50
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=T, topk=2)])
    cfg = _cfg(data_cfg, max_frames=10_000, crop_mode="head")
    ds = MoshiKDDataset(cfg.root, "en_kd", "train", cfg=cfg)
    it = ds[0]

    ramp = (np.arange(T) % CODEBOOK_SIZE).astype(np.int16)
    # Every codebook starts at frame 0 -- no acoustic delay applied.
    np.testing.assert_array_equal(it["codes_self"].numpy(), np.broadcast_to(ramp, (K, T)))
    # codes_b is offset by 1000, unshifted in time.
    np.testing.assert_array_equal(
        it["codes_other"].numpy(), np.broadcast_to((ramp + 1000) % CODEBOOK_SIZE, (K, T))
    )
    # Text is not shifted relative to codes.
    text = it["text_self"].numpy()
    np.testing.assert_array_equal(text[::3], np.arange(0, T, 3, dtype=np.int32))
    # Teacher slot 0 sits on the same frame as the code it was sampled from.
    np.testing.assert_array_equal(it["teacher_idx"][:, :, 0].numpy(), it["codes_self"].numpy())
    # No head padding: nothing equals audio_init_id / batch pad at frame 0.
    assert int(it["codes_self"][0, 0]) == 0 and int(it["text_self"][0]) == 0


# --- 25: Zone B voice-prompt reference lookup (ARCHITECTURE 7.2 / 7.4) --------
#
# The failure mode this whole block guards is that a *wrong* reference trains
# perfectly happily. Prompting a sample with its own audio makes voice cloning
# look solved while the model has only learned to copy; prompting with a
# stranger teaches the opposite of the intended conditioning. Neither crashes.


def _voice_ds(prepared, data_cfg, spec, split="train", **over):
    """spec: list of (uid, speaker, T). Rows get distinct code ramps.

    `split` is a plain directory name here, used to give each dataset in a test
    its own Arrow file: rewriting a path that a live MoshiKDDataset has already
    memory-mapped is a bus error, not a test failure.
    """
    samples = [
        make_solo_sample(uid, T=T, speaker=spk, offset=100 * (i + 1))
        for i, (uid, spk, T) in enumerate(spec)
    ]
    prepared("ko_tts", split, samples)
    cfg = _cfg(data_cfg, **over)
    return MoshiKDDataset(cfg.root, "ko_tts", split, cfg=cfg)


def _uid_of(ds, ref_codes):
    """Recover which row a reference came from via its unique code ramp offset."""
    hf = ds._hf()
    first = int(ref_codes[0, 0])
    for row in range(len(hf)):
        codes = np.asarray(hf[row]["codes_a"], dtype=np.int64).reshape(K, -1)
        if first in set(codes[0].tolist()):
            return str(hf[row]["sample_uid"])
    raise AssertionError("reference does not match any row")


def test_25_reference_is_a_different_row_by_the_same_speaker(prepared, data_cfg):
    spec = [(f"s{i}", "spkA" if i < 3 else "spkB", 40 + i) for i in range(6)]
    ds = _voice_ds(prepared, data_cfg, spec)
    by_uid = {uid: spk for uid, spk, _ in spec}

    for i in range(len(ds)):
        it = ds[i]
        assert bool(it["has_ref"]), f"{it['sample_uid']}: no reference despite 3 rows/speaker"
        ref_uid = _uid_of(ds, it["ref_codes"])
        assert ref_uid != it["sample_uid"], "sample was prompted with its own audio"
        assert by_uid[ref_uid] == it["speaker"], "reference came from another speaker"


def test_25b_no_reference_without_a_usable_speaker(prepared, data_cfg):
    # "" speaker, and a speaker with exactly one row: both must yield empty_ref.
    ds = _voice_ds(prepared, data_cfg, [("anon", "", 40), ("lone", "solo", 40)])
    for i in range(2):
        it = ds[i]
        assert not bool(it["has_ref"])
        assert it["ref_codes"].shape == (K, 0)
        assert it["ref_text"].shape == (0,)
        assert it["ref_codes"].dtype is torch.int16 and it["ref_text"].dtype is torch.int32


def test_25c_voice_prompt_false_disables_the_lookup(prepared, data_cfg):
    spec = [(f"s{i}", "spkA", 40) for i in range(3)]
    ds = _voice_ds(prepared, data_cfg, spec, voice_prompt=False)
    for i in range(len(ds)):
        assert not bool(ds[i]["has_ref"])
        assert ds[i]["ref_codes"].shape == (K, 0)


def test_25d_reference_is_frame_aligned_and_respects_the_cap(prepared, data_cfg):
    spec = [(f"s{i}", "spkA", 200) for i in range(3)]
    ds = _voice_ds(prepared, data_cfg, spec, voice_prompt_frames=32)
    for i in range(len(ds)):
        it = ds[i]
        R = int(it["ref_codes"].shape[1])
        assert R == 32, f"reference is {R} frames, cap is 32"
        assert it["ref_text"].shape == (R,), "ref_text is not frame-aligned to ref_codes"

    # A row shorter than the cap is used whole, not padded.
    short = _voice_ds(prepared, data_cfg, [(f"t{i}", "spkA", 20) for i in range(3)],
                      split="short", voice_prompt_frames=64)
    assert int(short[0]["ref_codes"].shape[1]) == 20

    # Alignment is real, not just equal lengths. The fixture ties the two streams
    # to a common frame index f: codes[k, f] == f + offset and text[f] == f every
    # third frame. So recovering `offset` from any one labelled frame pins the
    # window, and every other position must then agree -- an off-by-one between
    # the two crops shows up immediately.
    it = ds[0]
    codes = it["ref_codes"][0].numpy().astype(np.int64)
    text = it["ref_text"].numpy().astype(np.int64)
    labelled = np.flatnonzero(text != DUMMY_TEXT_PAD)
    assert labelled.size >= 2, "fixture too weak: fewer than 2 labelled frames"
    offset = codes[labelled[0]] - text[labelled[0]]
    frames = codes - offset
    np.testing.assert_array_equal(frames, frames[0] + np.arange(len(frames)))
    expected = np.where(frames % 3 == 0, frames, DUMMY_TEXT_PAD)
    np.testing.assert_array_equal(text, expected)


def test_25e_reference_choice_is_reproducible_across_workers(prepared, data_cfg):
    from torch.utils.data import DataLoader

    spec = [(f"s{i}", "spkA", 200) for i in range(8)]
    ds = _voice_ds(prepared, data_cfg, spec, voice_prompt_frames=32)

    def run(nw):
        dl = DataLoader(ds, batch_size=None, num_workers=nw, collate_fn=None, shuffle=False)
        return [(it["sample_uid"], it["ref_codes"].numpy().tobytes()) for it in dl]

    ref = run(0)
    assert len({r for _, r in ref}) > 1, "fixture too weak: every reference identical"
    for nw in (2, 4):
        assert run(nw) == ref, f"reference choice differs at num_workers={nw}"


def test_25f_reference_varies_with_epoch(prepared, data_cfg):
    """Phase 2's "varied reference voice per sample" is exactly this."""
    spec = [(f"s{i}", "spkA", 200) for i in range(8)]
    ds = _voice_ds(prepared, data_cfg, spec, voice_prompt_frames=32)

    def refs():
        return [ds[i]["ref_codes"].numpy().tobytes() for i in range(len(ds))]

    ds.set_epoch(0)
    e0 = refs()
    ds.set_epoch(1)
    assert refs() != e0, "the reference is frozen across epochs -- no augmentation"
    ds.set_epoch(0)
    assert refs() == e0, "epoch 0 not reproducible"


def test_25g_reference_lookup_does_not_move_the_crop(prepared, data_cfg):
    """Draw order regression. The reference must be drawn AFTER the crop window,
    or enabling Zone B silently re-crops the entire corpus and the prompted /
    unprompted ablation stops being a controlled comparison."""
    spec = [(f"s{i}", "spkA", 400) for i in range(6)]
    on = _voice_ds(prepared, data_cfg, spec, split="on", voice_prompt=True, max_frames=64)
    off = _voice_ds(prepared, data_cfg, spec, split="off", voice_prompt=False, max_frames=64)

    starts = {int(on[i]["codes_self"][0, 0]) for i in range(len(on))}
    assert len(starts) > 1, "fixture too weak: crop is not actually random"

    for i in range(len(on)):
        a, b = on[i], off[i]
        assert bool(a["has_ref"]) and not bool(b["has_ref"])
        np.testing.assert_array_equal(a["codes_self"].numpy(), b["codes_self"].numpy())
        np.testing.assert_array_equal(a["codes_other"].numpy(), b["codes_other"].numpy())
        np.testing.assert_array_equal(a["text_self"].numpy(), b["text_self"].numpy())


def test_25h_speaker_index_reads_only_the_speaker_column(prepared, data_cfg, monkeypatch):
    """Same spirit as test 9: the index must not drag the code/teacher columns in.

    `speaker` is a handful of bytes per row; codes_* and teacher_* are megabytes.
    Building a speaker map through the list-column path would materialize the
    whole corpus to answer a string question.
    """
    import project_amnesty.datasets.runtime.dataset as dsmod

    ds = _voice_ds(prepared, data_cfg, [(f"s{i}", "spkA", 40) for i in range(3)])
    ds._spk = None  # nothing cached yet

    def boom(*a, **kw):
        raise AssertionError("_speaker_index touched a list column")

    monkeypatch.setattr(dsmod, "_ListColumn", boom)
    monkeypatch.setattr(dsmod.MoshiKDDataset, "_columns", boom)

    index = ds._speaker_index()
    assert index == {"spkA": [0, 1, 2]}


def test_25i_en_kd_has_no_reference(en_kd_ds):
    """A dialogue row carries two voices, so its speaker is "" by construction and
    a single-voice prompt has no meaning for it."""
    for i in range(len(en_kd_ds)):
        it = en_kd_ds[i]
        assert it["speaker"] == "", "en_kd row acquired a speaker id"
        assert not bool(it["has_ref"])


def test_25j_text_anchor_has_no_reference(anchor_ds):
    for i in range(len(anchor_ds)):
        assert not bool(anchor_ds[i]["has_ref"])


def test_25k_validate_item_passes_on_every_source(prepared, data_cfg):
    from project_amnesty.datasets.runtime.item import validate_item

    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=80)])
    prepared("text_anchor", "train", [make_text_anchor_sample("tx0", L=20)])
    prepared("ko_tts", "train",
             [make_solo_sample(f"ko{i}", T=80, speaker="spkA", offset=100 * i) for i in range(3)])
    cfg = _cfg(data_cfg, debug_validate=False)  # validate explicitly, not implicitly
    for source in ("en_kd", "ko_tts", "text_anchor"):
        ds = MoshiKDDataset(cfg.root, source, "train", cfg=cfg)
        for i in range(len(ds)):
            validate_item(ds[i])


def test_25l_probe_crops_the_reference_from_the_head(prepared, data_cfg):
    """Probe already forces a non-random crop; the prompt window follows suit, so
    the eval prompt is a fixed opening segment of whichever row is chosen rather
    than a random slice of it. (Which row is chosen still follows the epoch --
    only the window within it is pinned.)"""
    samples = [make_solo_sample(f"p{i}", T=200, speaker="spkA", offset=100 * (i + 1))
               for i in range(4)]
    prepared("ko_tts", "probe", samples)
    ds = MoshiKDDataset(data_cfg.root, "ko_tts", "probe", cfg=data_cfg)
    assert ds.crop_mode == "center"
    for i in range(len(ds)):
        it = ds[i]
        assert int(it["ref_codes"].shape[1]) == data_cfg.voice_prompt_frames
        # Head crop => the window starts at frame 0, so the first code is exactly
        # the row's ramp offset and the first text token is frame 0.
        assert int(it["ref_codes"][0, 0]) % 100 == 0, "reference was not head-cropped"
        assert int(it["ref_text"][0]) == 0


# --- crop.py units + factory --------------------------------------------------


def test_choose_window_modes():
    assert choose_window(10, 64, None, "center") == Window(0, 10)
    assert choose_window(100, 64, None, "head") == Window(0, 64)
    assert choose_window(100, 64, None, "center") == Window(18, 82)
    rng = np.random.default_rng(0)
    ws = {choose_window(100, 64, rng, "random").start for _ in range(50)}
    assert len(ws) > 1 and all(0 <= s <= 36 for s in ws)
    assert len(choose_window(100, 64, None, "head")) == 64
    with pytest.raises(AssertionError):
        choose_window(100, 64, None, "diagonal")


def test_build_source_datasets(prepared, data_cfg):
    prepared("en_kd", "train", [make_en_kd_sample("kd0", T=30)])
    prepared("ko_tts", "train", [make_solo_sample("ko0", T=30)])
    prepared("text_anchor", "train", [make_text_anchor_sample("tx0", L=12)])
    out = build_source_datasets(data_cfg, "train")
    assert set(out) == {"en_kd", "ko_tts", "text_anchor"}
    assert out["en_kd"].double_ab and not out["ko_tts"].double_ab
    assert len(out["en_kd"]) == 2
