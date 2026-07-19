"""Collator tests -- plan section 6 items 25-32.

The load-bearing one is the delay round-trip inversion: everything else in this
file exists because plan section 7.8 misalignment converges, never raises, and only
shows up as ~80 ms of turn-taking skew in the finished model.

Inputs are built by a local helper that mimics what MoshiKDDataset produces
(torch tensors, delay-free, roles already resolved). training.data.dataset is
deliberately not imported: it is being written concurrently, and the collator's
contract is `list[KDSample]`, not "whatever the dataset happens to return".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_pipeline.schema import CODEBOOK_SIZE, NUM_CODEBOOKS
from tests.conftest import (
    DUMMY_BATCH_PAD,
    DUMMY_SILENCE,
    DUMMY_SILENCE_BANK,
    DUMMY_TEXT_EPAD,
    DUMMY_TEXT_PAD,
    ramp_codes,
    ramp_teacher,
    ramp_text,
)
from training.data.collator import (
    ZONE_A,
    ZONE_B,
    ZONE_C,
    ZONE_PAD,
    DelayConfig,
    KDCollator,
    KDCollatorConfig,
)
from training.data.item import validate_item
from training.data.text_collator import TextAnchorCollator, TextAnchorCollatorConfig

K = NUM_CODEBOOKS

# Stand-in for the ChatML system prefix of ARCHITECTURE section 7.2. Values are
# arbitrary but must not collide with PAD/EPAD/batch-pad, so that "Zone A text is
# dense" is distinguishable from "Zone A text is padding".
SYS = (9001, 9002, 9003, 9004, 9005, 9006)


# --------------------------------------------------------------------- helpers
def _t(a, dtype):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


def kd_row(
    uid: str,
    T: int,
    *,
    topk: int = 4,
    with_teacher: bool = True,
    sample_type: str = "en_kd",
    lang: str = "en",
    text_self: np.ndarray | None = None,
    text_other: np.ndarray | None = None,
    offset: int = 0,
    ref: int = 0,
    ref_offset: int = 500,
    **extra,
) -> dict:
    """A KDSample as the Dataset emits it: torch, delay-free, roles resolved."""
    codes_self = ramp_codes(K, T, offset=offset)
    codes_other = ramp_codes(K, T, offset=offset + 1000)
    ts = ramp_text(T) if text_self is None else text_self
    to = ramp_text(T, every=4) if text_other is None else text_other

    if with_teacher:
        val, idx = ramp_teacher(K, T, topk, codes_self)
        tv, ti = _t(val, torch.float16), _t(idx, torch.int16)
    else:
        topk = 0
        tv = torch.empty((0, 0, 0), dtype=torch.float16)
        ti = torch.empty((0, 0, 0), dtype=torch.int16)

    row = {
        "sample_uid": uid, "source": sample_type, "sample_type": sample_type, "lang": lang,
        "speaker": "spk0",
        "swapped": torch.tensor(False), "is_text_only": torch.tensor(False),
        "has_teacher": torch.tensor(with_teacher),
        "num_frames": torch.tensor(T, dtype=torch.int32),
        "topk": torch.tensor(topk, dtype=torch.int32),
        "codes_self": _t(codes_self, torch.int16),
        "codes_other": _t(codes_other, torch.int16),
        "text_self": _t(ts, torch.int32), "text_other": _t(to, torch.int32),
        "text_flat": torch.empty((0,), dtype=torch.int32),
        # Zone B voice prompt, as the Dataset supplies it: another row by the same
        # speaker, plus that row's frame-aligned transcript.
        "ref_codes": _t(ramp_codes(K, ref, offset=ref_offset), torch.int16),
        "ref_text": _t(ramp_text(ref, every=2), torch.int32),
        "has_ref": torch.tensor(ref > 0),
        "teacher_val": tv, "teacher_idx": ti,
        "use_kd": torch.tensor(with_teacher),
        "use_ce_audio": torch.tensor(True), "use_ce_text": torch.tensor(True),
    }
    validate_item(row)          # the helper must itself satisfy the item contract
    row.update(extra)           # per-test overrides, applied after the contract check
    return row


def solo_row(uid: str, T: int, lang: str = "ko", **kw) -> dict:
    """ko_tts / en_solo: the Dataset has already silence-filled the user channel."""
    row = kd_row(uid, T, with_teacher=False, sample_type="ko_tts", lang=lang, **kw)
    sil = np.asarray(DUMMY_SILENCE, dtype=np.int16)[:, None].repeat(T, axis=1)
    row["codes_other"] = _t(sil, torch.int16)
    row["text_other"] = torch.full((T,), DUMMY_TEXT_PAD, dtype=torch.int32)
    return row


def anchor_row(uid: str, L: int) -> dict:
    return {
        "sample_uid": uid, "source": "text_anchor", "sample_type": "text_anchor", "lang": "en",
        "speaker": "",
        "swapped": torch.tensor(False), "is_text_only": torch.tensor(True),
        "has_teacher": torch.tensor(False),
        "num_frames": torch.tensor(0, dtype=torch.int32),
        "topk": torch.tensor(0, dtype=torch.int32),
        "codes_self": torch.empty((K, 0), dtype=torch.int16),
        "codes_other": torch.empty((K, 0), dtype=torch.int16),
        "text_self": torch.empty((0,), dtype=torch.int32),
        "text_other": torch.empty((0,), dtype=torch.int32),
        "text_flat": torch.arange(L, dtype=torch.int32),
        "ref_codes": torch.empty((K, 0), dtype=torch.int16),
        "ref_text": torch.empty((0,), dtype=torch.int32),
        "has_ref": torch.tensor(False),
        "teacher_val": torch.empty((0, 0, 0), dtype=torch.float16),
        "teacher_idx": torch.empty((0, 0, 0), dtype=torch.int16),
        "use_kd": torch.tensor(False), "use_ce_audio": torch.tensor(False),
        "use_ce_text": torch.tensor(True),
    }


@pytest.fixture
def coll_cfg(tokens):
    def _make(**kw):
        delay = kw.pop("delay", DelayConfig(acoustic_delay=2))
        return KDCollatorConfig(tokens=tokens, delay=delay, **kw)
    return _make


@pytest.fixture
def collator(coll_cfg):
    return KDCollator(coll_cfg())


# ============================================================ 25: delay round-trip
@pytest.mark.parametrize("tau", [0, 1, 2])
@pytest.mark.parametrize("text_delay", [-8, 0, 8])
@pytest.mark.parametrize("edge_mode", ["extend", "truncate"])
def test_delay_roundtrip_inversion(coll_cfg, tau, text_delay, edge_mode):
    """THE test: invert the delay and recover the delay-free row exactly."""
    delay = DelayConfig(acoustic_delay=tau, semantic_delay=0, text_delay_frames=text_delay)
    c = KDCollator(coll_cfg(delay=delay, edge_mode=edge_mode))
    rows = [kd_row("a", 40), kd_row("b", 23, offset=7)]
    out = c(rows)

    off = [int(x) for x in out["delay_offsets"]]
    assert off == delay.offsets()
    max_off = max(off)

    for b, r in enumerate(rows):
        L = int(r["num_frames"])
        extent = L + max_off if edge_mode == "extend" else L
        for k in range(K):
            o = off[k + 1]
            n = min(L, extent - o)
            for role, key in ((0, "codes_self"), (1, "codes_other")):
                got = out["codes"][b, role, k, o:o + n]
                assert torch.equal(got, r[key][k, :n].to(torch.int64)), (
                    f"tau={tau} td={text_delay} {edge_mode} b={b} role={role} k={k}"
                )
                # everything outside [o, o+n) must be flagged invalid
                assert not out["stream_valid"][b, role, k, :o].any()
                assert not out["stream_valid"][b, role, k, o + n:].any()
                assert out["stream_valid"][b, role, k, o:o + n].all()
        ot = off[0]
        nt = min(L, extent - ot)
        assert torch.equal(out["text_tokens"][b, ot:ot + nt],
                           r["text_self"][:nt].to(torch.int64))

    # extend never drops supervision; truncate drops exactly max_off frames of it
    if edge_mode == "extend":
        for b, r in enumerate(rows):
            assert int(out["stream_valid"][b, 0, K - 1].sum()) == int(r["num_frames"])


def test_padding_and_dtypes(collator):
    out = collator([kd_row("a", 40), kd_row("b", 23)])
    B, T = 2, out["codes"].shape[-1]
    assert T % 8 == 0 and T >= 40 + collator.cfg.delay.max_offset
    assert out["codes"].shape == (B, 2, K, T) and out["codes"].dtype is torch.int64
    assert out["role_ids"].shape == (B, 2)
    assert torch.equal(out["role_ids"][0], torch.tensor([0, 1]))
    assert out["text_tokens"].shape == (B, T) and out["text_tokens"].dtype is torch.int64
    assert out["stream_valid"].shape == (B, 2, K, T) and out["stream_valid"].dtype is torch.bool
    assert out["attention_mask"].shape == (B, T) and out["attention_mask"].dtype is torch.bool
    assert out["zone_ids"].shape == (B, T) and out["zone_ids"].dtype is torch.uint8
    assert out["audio_loss_weight"].shape == (B, 2, K, T)
    assert out["audio_loss_weight"].dtype is torch.float32
    assert out["text_loss_weight"].shape == (B, T)
    assert out["teacher_topk_val"].shape == (B, 2, T, 4)
    assert out["teacher_topk_val"].dtype is torch.float16
    assert out["teacher_topk_idx"].shape == (B, 2, T, 4)
    assert out["teacher_topk_idx"].dtype is torch.int64
    assert out["kd_valid"].shape == (B, 2, T)
    assert out["kd_frame_weight"].shape == (B, 2, T)
    for k in ("sample_type_id", "lang_id", "has_teacher", "num_frames"):
        assert out[k].shape == (B,)
    assert out["sample_uid"] == ["a", "b"]
    assert out["delay_offsets"].shape == (K + 1,)
    assert out["target_aligned"] is True


def test_teacher_absent_is_minus_one_sentinel(coll_cfg):
    """-1 is not a valid Mimi code, so gathering on it fails loudly downstream."""
    c = KDCollator(coll_cfg())
    out = c([kd_row("a", 20), solo_row("s", 20)])
    assert (out["teacher_topk_idx"][:, 1] == -1).all()      # user role never has a teacher
    assert (out["teacher_topk_idx"][1] == -1).all()         # ko_tts row has none at all
    assert not out["kd_valid"][1].any()
    assert not out["kd_valid"][:, 1].any()


# ================================================= 26: offsets normalized non-negative
def test_offsets_match_plan_3_6_tables():
    """The two hardcoded tables from plan section 3.6, K=3 (text, cb0, cb1)."""
    d0 = DelayConfig(acoustic_delay=1, semantic_delay=0, text_delay_frames=0, num_codebooks=2)
    assert d0.raw() == [0, 0, 1]
    assert d0.offsets() == [0, 0, 1]
    assert d0.max_offset == 1                      # T_out = 5 + 1 = 6

    d1 = DelayConfig(acoustic_delay=1, semantic_delay=0, text_delay_frames=-1, num_codebooks=2)
    assert d1.raw() == [-1, 0, 1]
    assert d1.offsets() == [0, 1, 2]               # cb0 moved even though semantic_delay==0
    assert d1.max_offset == 2                      # T_out = 5 + 2 = 7


@pytest.mark.parametrize("tau", [0, 1, 2])
@pytest.mark.parametrize("td", [-8, -1, 0, 1, 8])
@pytest.mark.parametrize("sem", [0, 1])
def test_offsets_always_non_negative_and_min_zero(tau, td, sem):
    off = DelayConfig(acoustic_delay=tau, semantic_delay=sem, text_delay_frames=td).offsets()
    assert len(off) == K + 1
    assert min(off) == 0 and all(o >= 0 for o in off)
    # normalization is a pure translation: gaps are preserved
    raw = DelayConfig(acoustic_delay=tau, semantic_delay=sem, text_delay_frames=td).raw()
    assert [a - b for a, b in zip(off, raw)] == [-min(raw)] * (K + 1)


def test_plan_3_6_worked_example_layout(coll_cfg):
    """Reproduce both plan section 3.6 tables position-by-position on a real batch.

    K=8 rather than the doc's K=3, so we check text/cb0/cb1 which are the three
    rows the tables actually name.
    """
    for text_delay, exp_off, T_out in ((0, [0, 0, 1], 6), (-1, [0, 1, 2], 7)):
        delay = DelayConfig(acoustic_delay=1, semantic_delay=0, text_delay_frames=text_delay)
        c = KDCollator(coll_cfg(delay=delay))
        r = kd_row("x", 5)
        out = c([r])
        off = [int(v) for v in out["delay_offsets"]]
        assert off[:3] == exp_off, f"text_delay={text_delay}"
        assert 5 + max(off) == T_out

        ot, o0, o1 = off[0], off[1], off[2]
        # text row: w0..w4 at [ot, ot+5)
        assert torch.equal(out["text_tokens"][0, ot:ot + 5], r["text_self"].to(torch.int64))
        # cb0 row: a0..a4 at [o0, o0+5)
        assert torch.equal(out["codes"][0, 0, 0, o0:o0 + 5], r["codes_self"][0].to(torch.int64))
        # cb1 row: init then b0..b4 at [o1, o1+5)
        assert torch.equal(out["codes"][0, 0, 1, o1:o1 + 5], r["codes_self"][1].to(torch.int64))
        assert (out["codes"][0, 0, 1, :o1] == c.cfg.tokens.audio_init_id).all()
        # teacher(cb0): P0..P4 at the SAME offset as cb0
        assert torch.equal(out["teacher_topk_idx"][0, 0, o0:o0 + 5, 0],
                           r["teacher_idx"][0, :, 0].to(torch.int64))


# ================================== 27: teacher shifted identically to codebook 0
@pytest.mark.parametrize("tau", [0, 1, 2])
@pytest.mark.parametrize("text_delay", [-8, -1, 0, 8])
@pytest.mark.parametrize("sem", [0, 1])
def test_teacher_shifted_with_cb0(coll_cfg, tau, text_delay, sem):
    """Fails loudly for any implementation that hardcodes o0 = 0.

    ramp_teacher puts the sampled token in slot 0, so "teacher slot 0 == cb0" at
    every output position is an exact, position-wise identity.
    """
    delay = DelayConfig(acoustic_delay=tau, semantic_delay=sem, text_delay_frames=text_delay)
    c = KDCollator(coll_cfg(delay=delay))
    rows = [kd_row("a", 30)]
    out = c(rows)
    off = [int(v) for v in out["delay_offsets"]]
    o0 = off[1]
    if text_delay < 0:
        assert o0 > 0, "negative text delay must push codebook 0 forward"

    valid = out["kd_valid"][0, 0]
    cb0 = out["codes"][0, 0, 0]
    slot0 = out["teacher_topk_idx"][0, 0, :, 0]
    assert torch.equal(cb0[valid], slot0[valid])
    assert not valid[:o0].any() and valid[o0]
    # kd_valid is derived, not recomputed
    assert torch.equal(out["kd_valid"][0, 0], out["stream_valid"][0, 0, 0])


def test_kd_valid_is_derived_from_stream_valid(collator):
    out = collator([kd_row("a", 30), kd_row("b", 17)])
    sv0 = out["stream_valid"][:, :, 0]
    assert torch.equal(out["kd_valid"], sv0 & out["kd_valid"])
    assert torch.equal(out["kd_valid"][:, 0], sv0[:, 0])    # role 0 has the teacher


# ============================================= 28: shift scan + mutation test
def test_shift_scan_peaks_at_zero(coll_cfg):
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-1)))
    rows = [kd_row("a", 60), kd_row("b", 45, offset=13)]
    rates = c.debug_alignment_check(c(rows), rows)
    assert max(rates, key=rates.get) == 0
    assert rates[0] > 0.9
    assert rates[1] < rates[0] and rates[-1] < rates[0]


@pytest.mark.parametrize("roll", [-1, 1])
def test_shift_scan_mutation_raises(coll_cfg, roll):
    """Roll the teacher by one frame -- the detector must refuse the batch."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-1)))
    rows = [kd_row("a", 60), kd_row("b", 45, offset=13)]
    out = c(rows)
    out["teacher_topk_idx"] = torch.roll(out["teacher_topk_idx"], shifts=roll, dims=2)
    with pytest.raises(AssertionError, match="misalignment"):
        c.debug_alignment_check(out, rows)


def test_hardcoded_o0_bug_is_caught(coll_cfg):
    """Simulate the plan section 7.8 bug directly: place the teacher at 0 under a
    negative text delay, i.e. exactly what `o0 = 0` would produce."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-1)))
    rows = [kd_row("a", 60)]
    out = c(rows)
    o0 = int(out["delay_offsets"][1])
    assert o0 == 1
    out["teacher_topk_idx"] = torch.roll(out["teacher_topk_idx"], shifts=-o0, dims=2)
    with pytest.raises(AssertionError, match="misalignment"):
        c.debug_alignment_check(out, rows)


def test_debug_alignment_check_passes_all_delays(coll_cfg):
    for tau in (0, 1, 2):
        for td in (-8, 0, 8):
            c = KDCollator(coll_cfg(
                delay=DelayConfig(acoustic_delay=tau, text_delay_frames=td)))
            rows = [kd_row("a", 50), kd_row("b", 36, offset=5)]
            c.debug_alignment_check(c(rows), rows)


# ============================== 29 / 30: weight values at hand-checked positions
def test_weight_values_exact(coll_cfg):
    T = 24
    za = 4
    # text: EPAD at 6, PAD everywhere else except a real token at 10
    ts = np.full((T,), DUMMY_TEXT_PAD, dtype=np.int32)
    ts[6] = DUMMY_TEXT_EPAD
    ts[10] = 12345
    delay = DelayConfig(acoustic_delay=2, semantic_delay=0, text_delay_frames=0)
    # Zone A is built by the collator, so it is turned on here by config, not by
    # smuggling a zone_a_frames key into the row that production never provides.
    c = KDCollator(coll_cfg(delay=delay, system_prompt_ids=SYS[:za],
                            zone_a_sources=("en_kd",)))
    r = kd_row("a", T, text_self=ts)
    out = c([r])
    off = [int(v) for v in out["delay_offsets"]]
    assert off[0] == 0 and off[1] == 0 and off[2] == 2
    total = za + T

    aw, tw, zid = out["audio_loss_weight"], out["text_loss_weight"], out["zone_ids"]
    assert (zid[0, :za] == ZONE_A).all()
    assert (zid[0, total + max(off):] == ZONE_PAD).all()

    # --- Zone A: semantic AND acoustic are 0.0, not 0.02 (item 30) ---
    assert aw[0, 0, 0, 0].item() == 0.0
    for k in range(1, K):
        assert aw[0, 0, k, 3].item() == 0.0, "Zone A acoustic frame must be 0.0, not 0.02"
    assert tw[0, 0].item() == 0.0

    # --- Zone C, semantic = 1.0; non-semantic codebooks = 0.02 ---
    assert aw[0, 0, 0, za + 10].item() == pytest.approx(1.0)
    for k in range(1, K):
        assert aw[0, 0, k, za + 10].item() == pytest.approx(0.02)

    # --- text: stream PAD x0.3, EPAD x1.0, real token x1.0 ---
    assert tw[0, za + 5].item() == pytest.approx(0.30)
    assert tw[0, za + 6].item() == pytest.approx(1.00), "EPAD must NOT be down-weighted"
    assert tw[0, za + 10].item() == pytest.approx(1.00)

    # --- batch pad is exactly 0 via stream_valid, on every axis ---
    pad = slice(total + max(off), aw.shape[-1])
    assert aw[..., pad].abs().max().item() == 0.0
    assert tw[..., pad].abs().max().item() == 0.0
    assert (out["text_tokens"][0, pad] == DUMMY_BATCH_PAD).all()
    # ...and the delay head, which is inside the row but not real data
    assert aw[0, 0, 1, :off[2]].abs().max().item() == 0.0


def test_batch_pad_weight_exactly_zero_short_row(collator):
    out = collator([kd_row("long", 48), kd_row("short", 9)])
    T = out["codes"].shape[-1]
    extent = 9 + collator.cfg.delay.max_offset
    assert out["audio_loss_weight"][1, :, :, extent:].abs().max().item() == 0.0
    assert out["text_loss_weight"][1, extent:].abs().max().item() == 0.0
    assert not out["attention_mask"][1, extent:].any()
    assert out["attention_mask"][1, :extent].all()
    assert (out["zone_ids"][1, extent:T] == ZONE_PAD).all()


def test_epad_weight_is_configurable_for_ablation(coll_cfg):
    ts = np.full((16,), DUMMY_TEXT_EPAD, dtype=np.int32)
    c = KDCollator(coll_cfg(w_stream_epad_text=0.5))
    out = c([kd_row("a", 16, text_self=ts)])
    assert out["text_loss_weight"][0, 3].item() == pytest.approx(0.5)


def test_synthetic_user_channel_masked(coll_cfg):
    c = KDCollator(coll_cfg())
    out = c([kd_row("kd", 20), solo_row("solo", 20)])
    assert out["audio_loss_weight"][0, 1, 0, 5].item() > 0.0     # real user stream
    assert out["audio_loss_weight"][1, 1].abs().max().item() == 0.0
    assert out["audio_loss_weight"][1, 0, 0, 5].item() > 0.0     # self stream still trains
    # codes are still present -- the Dataset silence-filled them, we only unweight
    assert (out["codes"][1, 1, :, 5] == torch.tensor(DUMMY_SILENCE)).all()


def test_attention_mask_is_the_union(coll_cfg):
    """Under a negative text delay, text starts before cb0 -- the mask is their union."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-3)))
    out = c([kd_row("a", 20)])
    off = [int(v) for v in out["delay_offsets"]]
    assert off[0] == 0 and off[1] == 3
    assert out["attention_mask"][0, 0]                     # text only
    assert not out["stream_valid"][0, :, :, 0].any()
    assert out["attention_mask"][0, :20].all()


# ----------------------------------------------------- KD transition weighting
def test_kd_transition_weight_is_pre_delay_then_shifted(coll_cfg):
    """Silence, then speech from frame 20: the onset bump must land at 20 + off[1]."""
    T = 48
    ts = np.full((T,), DUMMY_TEXT_PAD, dtype=np.int32)
    ts[20:30] = 777
    hw = 3
    for td in (0, -4):
        c = KDCollator(coll_cfg(
            delay=DelayConfig(acoustic_delay=2, text_delay_frames=td),
            kd_transition_halfwidth=hw, kd_transition_weight=2.0))
        out = c([kd_row("a", T, text_self=ts)])
        o0 = int(out["delay_offsets"][1])
        w = out["kd_frame_weight"][0, 0]
        assert w[20 + o0].item() == pytest.approx(2.0)          # onset
        assert w[20 - hw + o0].item() == pytest.approx(2.0)     # dilation edge
        assert w[20 - hw - 1 + o0].item() == pytest.approx(1.0)
        assert w[30 + o0].item() == pytest.approx(2.0)          # offset
        assert w[30 + hw + 1 + o0].item() == pytest.approx(1.0)
        assert w[10 + o0].item() == pytest.approx(1.0)


def test_kd_transition_weight_disabled(coll_cfg):
    c = KDCollator(coll_cfg(kd_transition_weight=1.0))
    out = c([kd_row("a", 30)])
    assert (out["kd_frame_weight"] == 1.0).all()


# ============================================ Zone A / Zone B assembly (ARCH 7.1-7.4)
def _sil(n: int) -> torch.Tensor:
    """(K, n) int64 silence, tiled from the bank at phase 0 -- what the collator does."""
    bank = np.asarray(DUMMY_SILENCE_BANK, dtype=np.int64)
    return torch.as_tensor(bank[:, np.arange(n) % bank.shape[1]])


def _zero_delay() -> DelayConfig:
    """Zones are indexed from output position 0; zero delay keeps the arithmetic
    in these tests about the prefix rather than about the delay (which has its own
    tests, plus the prefix-aware round-trip in debug_alignment_check)."""
    return DelayConfig(acoustic_delay=0, semantic_delay=0, text_delay_frames=0)


def test_zone_a_is_dense_text_over_silence_and_fully_masked(coll_cfg):
    """ARCH 7.2: system prompt text, one token per frame, silence on BOTH channels,
    loss masked. 7.1 calls this region 'dense' -- every frame carries a real token."""
    T, N = 20, len(SYS)
    c = KDCollator(coll_cfg(system_prompt_ids=SYS, delay=_zero_delay()))
    out = c([solo_row("s", T)])

    assert int(out["zone_a_frames"][0]) == N
    assert (out["zone_ids"][0, :N] == ZONE_A).all()
    assert (out["zone_ids"][0, N:N + T] == ZONE_C).all()

    assert torch.equal(out["text_tokens"][0, :N], torch.tensor(SYS, dtype=torch.int64))
    for role in (0, 1):
        assert torch.equal(out["codes"][0, role, :, :N], _sil(N)), f"role {role} must be silence"

    # the whole point of w_zone_a: the prompt is a condition, not a target
    assert out["audio_loss_weight"][0, :, :, :N].abs().max().item() == 0.0
    assert out["text_loss_weight"][0, :N].abs().max().item() == 0.0
    # ...and it is not masked because it looks like padding
    assert out["attention_mask"][0, :N].all()
    assert out["stream_valid"][0, :, :, :N].all()


def test_zone_b_carries_ref_audio_and_transcript_and_is_predicted(coll_cfg):
    """ARCH 7.4: unlike PersonaPlex, the voice prompt keeps its inner monologue and
    is NOT loss-masked -- masking it would undo the 'Zone C is a continuation' claim.
    TRAINING_CURRICULUM line 50 masks Zone A only."""
    T, R, N = 20, 7, len(SYS)
    c = KDCollator(coll_cfg(system_prompt_ids=SYS, delay=_zero_delay()))
    r = solo_row("s", T, ref=R)
    out = c([r])

    assert int(out["zone_b_frames"][0]) == R
    assert (out["zone_ids"][0, N:N + R] == ZONE_B).all()
    assert torch.equal(out["codes"][0, 0, :, N:N + R], r["ref_codes"].to(torch.int64))
    assert torch.equal(out["codes"][0, 1, :, N:N + R], _sil(N + R)[:, N:])
    assert torch.equal(out["text_tokens"][0, N:N + R], r["ref_text"].to(torch.int64))

    assert out["audio_loss_weight"][0, 0, 0, N:N + R].min().item() > 0.0
    assert out["text_loss_weight"][0, N:N + R].min().item() > 0.0


def test_zone_b_absent_when_has_ref_false(coll_cfg):
    """Zero-length Zone B must reproduce today's behaviour byte for byte."""
    T = 20
    c_on = KDCollator(coll_cfg())
    c_off = KDCollator(coll_cfg(voice_prompt_sources=()))
    rows = [solo_row("s", T, ref=0)]
    on, off = c_on(rows), c_off(rows)
    assert int(on["zone_a_frames"][0]) == 0 and int(on["zone_b_frames"][0]) == 0
    assert torch.equal(on["codes"], off["codes"])
    assert torch.equal(on["text_tokens"], off["text_tokens"])
    assert torch.equal(on["zone_ids"], off["zone_ids"])
    assert (on["zone_ids"][0, :T] == ZONE_C).all()


def test_zone_a_covers_en_kd_too_but_zone_b_does_not(coll_cfg):
    """Zone A on every audio source; Zone B on ko_tts only.

    Excluding en_kd from Zone A would make "no system prompt" perfectly correlated
    with "this is the turn-taking task" -- a cleaner shortcut cue than language,
    and the exact correlation RISKS_AND_DIAGNOSTICS section 1 names as the project's
    most severe risk. Zone B is a different matter: an en_kd row carries two
    voices, so it has no single speaker to prompt with.
    """
    c = KDCollator(coll_cfg(system_prompt_ids=SYS, delay=_zero_delay()))
    out = c([kd_row("kd", 20, ref=5), solo_row("s", 20, ref=5)])
    assert out["zone_a_frames"].tolist() == [len(SYS), len(SYS)]
    assert out["zone_b_frames"].tolist() == [0, 5]


def test_prefix_source_gating_is_configurable(coll_cfg):
    """The gate itself still works -- an empty tuple disables the prefix."""
    c = KDCollator(coll_cfg(system_prompt_ids=SYS, zone_a_sources=(),
                            voice_prompt_sources=(), delay=_zero_delay()))
    out = c([kd_row("kd", 20, ref=5), solo_row("s", 20, ref=5)])
    assert out["zone_a_frames"].tolist() == [0, 0]
    assert out["zone_b_frames"].tolist() == [0, 0]
    assert (out["zone_ids"][0, :20] == ZONE_C).all()


def test_total_length_is_zone_a_plus_zone_b_plus_zone_c(coll_cfg):
    T, R, N = 30, 7, len(SYS)
    c = KDCollator(coll_cfg(system_prompt_ids=SYS))
    out = c([solo_row("s", T, ref=R)])
    exp = N + R + T + c.cfg.delay.max_offset       # edge_mode="extend"
    assert out["codes"].shape[-1] == ((exp + 7) // 8) * 8
    assert out["attention_mask"][0, :exp].all()
    assert not out["attention_mask"][0, exp:].any()
    assert int(out["num_frames"][0]) == T, "num_frames stays Zone C, the supervised span"


def test_kd_valid_is_false_across_zone_a_and_zone_b(coll_cfg):
    T, R, N = 30, 5, len(SYS)
    c = KDCollator(coll_cfg(system_prompt_ids=SYS, delay=_zero_delay(),
                            zone_a_sources=("en_kd",), voice_prompt_sources=("en_kd",)))
    out = c([kd_row("a", T, ref=R)])
    zid = out["zone_ids"][0]
    kv = out["kd_valid"][0, 0]
    assert not kv[zid == ZONE_A].any(), "no teacher exists over the system prompt"
    assert not kv[zid == ZONE_B].any(), "no teacher exists over the reference utterance"
    assert kv[N + R:N + R + T].all()
    assert not kv[zid == ZONE_PAD].any()


def test_teacher_alignment_survives_the_prefix(coll_cfg):
    """RISKS 7.8's exact shape: the teacher is aligned to Zone C, so in output
    coordinates it starts at off[1] + len(zone_a) + R. Dropping the prefix term
    converges and never raises, which is why the mutation half is mandatory."""
    T, R, N = 40, 6, len(SYS)
    c = KDCollator(coll_cfg(system_prompt_ids=SYS,
                            zone_a_sources=("en_kd",), voice_prompt_sources=("en_kd",),
                            delay=DelayConfig(acoustic_delay=2, text_delay_frames=-1)))
    rows = [kd_row("a", T, ref=R)]
    out = c(rows)
    o0 = int(out["delay_offsets"][1])
    assert o0 == 1, "negative text delay must push codebook 0 forward"

    valid = out["kd_valid"][0, 0]
    cb0 = out["codes"][0, 0, 0]
    slot0 = out["teacher_topk_idx"][0, 0, :, 0]
    assert valid.any()
    assert not valid[:o0 + N + R].any() and valid[o0 + N + R]
    # ramp_teacher puts the sampled token in slot 0 -> a position-wise identity
    assert torch.equal(cb0[valid], slot0[valid])
    c.debug_alignment_check(out, rows)

    # --- mutation: place the teacher WITHOUT the prefix offset, nothing else ---
    mutated = dict(out)
    mutated["teacher_topk_idx"] = torch.roll(out["teacher_topk_idx"], shifts=-(N + R), dims=2)
    bad_slot0 = mutated["teacher_topk_idx"][0, 0, :, 0]
    assert not torch.equal(cb0[valid], bad_slot0[valid]), (
        "the alignment assertion passes under the very bug it exists to catch"
    )
    with pytest.raises(AssertionError, match="misalignment"):
        c.debug_alignment_check(mutated, rows)


def test_kd_transition_detector_does_not_fire_on_zone_a(coll_cfg):
    """Zone A is dense, so a detector run on the assembled text would read every
    prompt frame as speech. It must be computed on Zone C only, then shifted."""
    T, N, hw = 40, len(SYS), 3
    ts = np.full((T,), DUMMY_TEXT_PAD, dtype=np.int32)
    ts[20:30] = 777
    c = KDCollator(coll_cfg(system_prompt_ids=SYS,
                            delay=DelayConfig(acoustic_delay=2, text_delay_frames=0),
                            kd_transition_halfwidth=hw, kd_transition_weight=2.0))
    out = c([solo_row("s", T, ref=4, text_self=ts)])
    o0 = int(out["delay_offsets"][1])
    P = N + 4
    w = out["kd_frame_weight"][0, 0]
    assert (w[:P] == 1.0).all(), "the dense prefix must not register as a speech onset"
    assert w[P + o0 + 20].item() == pytest.approx(2.0)          # real onset, shifted
    assert w[P + o0 + 20 - hw - 1].item() == pytest.approx(1.0)
    assert w[P + o0 + 30].item() == pytest.approx(2.0)          # offset


# ==================================================== 31: text_delay_sec = 0.6 raises
def test_text_delay_sec_0_6_raises():
    with pytest.raises(AssertionError, match="not close enough to an integer"):
        DelayConfig(text_delay_sec=0.6)


@pytest.mark.parametrize("sec,frames", [(0.0, 0), (0.08, 1), (-0.08, -1), (0.64, 8), (-0.64, -8)])
def test_text_delay_sec_exact_values(sec, frames):
    assert DelayConfig(text_delay_sec=sec).text_delay_frames == frames


def test_text_delay_sec_and_frames_conflict():
    with pytest.raises(AssertionError, match="not both"):
        DelayConfig(text_delay_sec=0.08, text_delay_frames=1)


# ============================ 32: text_anchor rejected, pointing at TextAnchorCollator
def test_text_anchor_rejected_by_kd_collator(collator):
    with pytest.raises(ValueError, match="TextAnchorCollator"):
        collator([kd_row("a", 20), anchor_row("t", 64)])


def test_audio_row_rejected_by_text_anchor_collator(tokens):
    c = TextAnchorCollator(TextAnchorCollatorConfig(tokens=tokens))
    with pytest.raises(ValueError, match="KDCollator"):
        c([anchor_row("t", 64), kd_row("a", 20)])


def test_text_anchor_collator_batch(tokens):
    c = TextAnchorCollator(TextAnchorCollatorConfig(tokens=tokens))
    out = c([anchor_row("t0", 20), anchor_row("t1", 7)])
    assert out["text_tokens"].shape == (2, 24)          # pad_to_multiple_of=8
    assert out["text_tokens"].dtype is torch.int64
    assert torch.equal(out["text_tokens"][0, :20], torch.arange(20, dtype=torch.int64))
    assert (out["text_tokens"][1, 7:] == DUMMY_BATCH_PAD).all()
    assert out["attention_mask"][1, :7].all() and not out["attention_mask"][1, 7:].any()
    assert out["text_loss_weight"][1, 7:].abs().max().item() == 0.0
    assert out["target_aligned"] is True
    assert out["sample_uid"] == ["t0", "t1"]


# --------------------------------------------------------------- config asserts
def test_missing_token_ids_name_the_file():
    from training.data.config import TokenConfig
    with pytest.raises(AssertionError, match="configs/tokens.yaml"):
        KDCollator(KDCollatorConfig(tokens=TokenConfig()))


def test_stream_pad_equal_batch_pad_raises():
    from training.data.config import TokenConfig
    bad = TokenConfig(
        text_pad_id=5, text_epad_id=6, batch_pad_id=5, audio_init_id=CODEBOOK_SIZE,
        silence_bank=tuple((c,) for c in DUMMY_SILENCE), mimi_ckpt_id="x",
    )
    with pytest.raises(AssertionError, match="must differ"):
        KDCollator(KDCollatorConfig(tokens=bad))


def test_bad_edge_mode_raises(tokens):
    with pytest.raises(AssertionError, match="edge_mode"):
        KDCollatorConfig(tokens=tokens, edge_mode="wrap")


def test_set_delay_without_rebuilding(coll_cfg):
    """Phase 1 -> Phase 2 flips tau and text_delay in place."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-1)))
    rows = [kd_row("a", 20)]
    # raw = [-1, 0, 2...] -> min -1 -> normalized [0, 1, 3...]
    assert [int(v) for v in c(rows)["delay_offsets"]] == [0, 1] + [3] * (K - 1)
    c.set_delay(DelayConfig(acoustic_delay=1, text_delay_frames=0))
    assert [int(v) for v in c(rows)["delay_offsets"]] == [0, 0] + [1] * (K - 1)
    c.debug_alignment_check(c(rows), rows)


def test_shape_mismatch_in_user_channel_is_loud(collator):
    r = kd_row("a", 20)
    r["codes_other"] = r["codes_other"][:, :19]
    with pytest.raises(AssertionError, match="silence-filling"):
        collator([r])


def test_mixed_lengths_and_types_in_one_batch(collator):
    out = collator([kd_row("a", 40), solo_row("s", 12, lang="en"), kd_row("c", 31)])
    assert out["sample_type_id"].tolist() == [0, 1, 0]      # en_kd, ko_tts, en_kd
    assert out["lang_id"].tolist() == [0, 0, 0]
    assert out["has_teacher"].tolist() == [True, False, True]
    assert out["num_frames"].tolist() == [40, 12, 31]


# ==================================================== ASR direction (DATA_STRATEGY 4.2)
# Bidirectional reuse: the same (text, audio) pair trained twice, once with text
# leading (TTS) and once with text lagging (ASR). ARCHITECTURE 5.0.2 claims this
# costs "no changes in the loss, architecture, or training data" -- only the
# delay -- so these tests are mostly about pinning that *nothing else* moved.


def asr_row(uid: str, T: int, **kw) -> dict:
    """A ko_asr row: a ko_tts row whose mixing group says ASR direction."""
    return solo_row(uid, T, source="ko_asr", **kw)


def test_asr_direction_places_text_after_audio(coll_cfg):
    """Positive text delay -> text lags the audio. That is the whole mechanism."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=0)))
    off = [int(v) for v in c([asr_row("a", 20)])["delay_offsets"]]
    # default asr_text_delay_frames=8, raw = [8, 0, 2...] -> min 0 -> unchanged
    assert off == [8, 0] + [2] * (K - 1)
    assert off[0] > off[1], "ASR: text must be placed after codebook 0"


def test_tts_direction_places_text_before_audio(coll_cfg):
    """The mirror image: a ko_tts row under a negative (text-leading) delay."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-8)))
    off = [int(v) for v in c([solo_row("a", 20)])["delay_offsets"]]
    assert off == [0, 8] + [10] * (K - 1)
    assert off[0] < off[1], "TTS: text must be placed before codebook 0"


def test_asr_delay_is_the_sign_flip_of_the_tts_delay(coll_cfg):
    """+-|text| about the same audio delays, i.e. Moshi Table 1's +-0.6 s pair."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-8)))
    tts = [int(v) for v in c([solo_row("t", 20)])["delay_offsets"]]
    asr = [int(v) for v in c([asr_row("a", 20)])["delay_offsets"]]
    assert asr == [8, 0] + [2] * (K - 1)
    # audio streams keep their *relative* spacing; only text moves side
    assert [o - min(tts[1:]) for o in tts[1:]] == [o - min(asr[1:]) for o in asr[1:]]


def test_asr_delay_tracks_set_delay(coll_cfg):
    """Phase 1 -> Phase 2 must move both directions, not just the TTS one."""
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=0)))
    assert [int(v) for v in c([asr_row("a", 20)])["delay_offsets"]] == [8, 0] + [2] * (K - 1)
    c.set_delay(DelayConfig(acoustic_delay=1, text_delay_frames=0))
    assert [int(v) for v in c([asr_row("a", 20)])["delay_offsets"]] == [8, 0] + [1] * (K - 1)


def test_explicit_asr_delay_overrides_the_derivation(coll_cfg):
    c = KDCollator(coll_cfg(
        delay=DelayConfig(acoustic_delay=2, text_delay_frames=-8),
        asr_delay=DelayConfig(acoustic_delay=1, text_delay_frames=3),
    ))
    assert [int(v) for v in c([asr_row("a", 20)])["delay_offsets"]] == [3, 0] + [1] * (K - 1)
    # the TTS side is untouched by the override
    assert [int(v) for v in c([solo_row("t", 20)])["delay_offsets"]] == [0, 8] + [10] * (K - 1)


def test_same_row_both_directions_differs_only_by_the_shift(coll_cfg):
    """THE load-bearing one. ARCHITECTURE 5.0.2: "no changes in the loss,
    architecture, or training data" -- so the identical row pushed through both
    directions must yield the identical codes and the identical text, placed at
    different offsets and nothing more. Un-shift each and demand equality.
    """
    c = KDCollator(coll_cfg(delay=DelayConfig(acoustic_delay=2, text_delay_frames=-8)))
    T = 37
    tts_row = solo_row("dual", T)
    asr = {**tts_row, "source": "ko_asr"}          # literally the same tensors

    out_t, out_a = c([tts_row]), c([asr])
    off_t = [int(v) for v in out_t["delay_offsets"]]
    off_a = [int(v) for v in out_a["delay_offsets"]]
    assert off_t != off_a, "the two directions must not collapse to one delay"

    # --- codes: un-shift by each direction's own per-codebook offset ---
    for k in range(K):
        for role in (0, 1):
            a = out_t["codes"][0, role, k, off_t[k + 1]:off_t[k + 1] + T]
            b = out_a["codes"][0, role, k, off_a[k + 1]:off_a[k + 1] + T]
            assert torch.equal(a, b), f"codes differ between directions at k={k} role={role}"
            assert torch.equal(a, tts_row["codes_self" if role == 0 else "codes_other"][k].long())

    # --- text: same tokens, only the offset moved ---
    a = out_t["text_tokens"][0, off_t[0]:off_t[0] + T]
    b = out_a["text_tokens"][0, off_a[0]:off_a[0] + T]
    assert torch.equal(a, b)
    assert torch.equal(a, tts_row["text_self"].long())

    # --- and the direction really is opposite, on the same underlying row ---
    assert off_t[0] < off_t[1] and off_a[0] > off_a[1]

    # Nothing else about the batch is direction-dependent: same zones, same
    # per-row bookkeeping, same loss-weight *values* over the valid region.
    assert torch.equal(out_t["num_frames"], out_a["num_frames"])
    assert torch.equal(out_t["sample_type_id"], out_a["sample_type_id"])
    assert torch.equal(out_t["kd_valid"], out_a["kd_valid"])       # both empty: no teacher
    for k in range(K):
        wa = out_t["audio_loss_weight"][0, 0, k, off_t[k + 1]:off_t[k + 1] + T]
        wb = out_a["audio_loss_weight"][0, 0, k, off_a[k + 1]:off_a[k + 1] + T]
        assert torch.equal(wa, wb)
    wt = out_t["text_loss_weight"][0, off_t[0]:off_t[0] + T]
    wa_ = out_a["text_loss_weight"][0, off_a[0]:off_a[0] + T]
    assert torch.equal(wt, wa_)


def test_teacher_lands_on_its_own_frames_under_the_asr_delay(coll_cfg):
    """The prefix/offset bug class of section 7.8, re-checked in the ASR direction.

    en_kd is the only source with a teacher, so this builds an en_kd-shaped row and
    tags it into asr_sources: the point is that a positive text delay does not
    disturb the teacher, which hangs off codebook 0 plus the prefix, not off text.
    """
    c = KDCollator(coll_cfg(
        delay=DelayConfig(acoustic_delay=2, text_delay_frames=0),
        asr_sources=("ko_asr", "en_kd_asr"),
        system_prompt_ids=SYS,
        zone_a_sources=("en_kd",),
    ))
    rows = [kd_row("kd", 30, source="en_kd_asr")]
    out = c(rows)
    off = [int(v) for v in out["delay_offsets"]]
    assert off[0] == 8 and off[1] == 0          # ASR delay actually applied

    P = len(SYS)
    o = off[1] + P
    got = out["teacher_topk_idx"][0, 0, o:o + 30]
    assert torch.equal(got, rows[0]["teacher_idx"][0].long())
    # and nothing landed outside the teacher's own Zone C frames
    assert not out["kd_valid"][0, 0, :o].any()
    assert not out["kd_valid"][0, 0, o + 30:].any()
    assert out["kd_valid"][0, 0, o:o + 30].all()
    c.debug_alignment_check(out, rows)


def test_batch_mixing_directions_raises(collator):
    with pytest.raises(AssertionError, match="mixes ASR-direction"):
        collator([solo_row("t", 20), asr_row("a", 20)])


def test_batch_mixing_sources_within_one_direction_is_fine(collator):
    """The homogeneity check is on *direction*, not on source: a batch of en_kd +
    ko_tts rows is legal and must stay legal."""
    out = collator([kd_row("a", 20), solo_row("s", 20)])
    assert out["codes"].shape[0] == 2


def test_negative_asr_text_delay_frames_raises(tokens):
    with pytest.raises(AssertionError, match="asr_text_delay_frames"):
        KDCollatorConfig(tokens=tokens, asr_text_delay_frames=-1)
