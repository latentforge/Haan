"""Sampler / schedule tests -- plan §6 items 33-41 plus the statistical ratio tests.

Most tests build a `GroupIndex` straight from synthetic cost arrays: the cost
array *is* the sampler's entire view of the corpus, so an Arrow round-trip would
only slow the suite down. `test_group_index_from_prepared` is the one test that
needs real Arrow, because `cost = num_frames or len(text_tokens_a)` is a claim
about the stored schema.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from project_amnesty.datasets.runtime.sampler import GroupIndex, MixingBatchSampler, assert_group_sync
from project_amnesty.datasets.runtime.schedule import MixSchedule

GROUPS = ("en_kd", "en_solo", "ko_tts", "text_anchor")

RAMP_CFG = {
    "interp": "linear",
    "unit": "global_step",
    "anchors": [
        {"at": 0, "weights": {"en_kd": 0.85, "en_solo": 0.05, "ko_tts": 0.05, "text_anchor": 0.05}},
        {"at": 4000, "weights": {"en_kd": 0.75, "en_solo": 0.10, "ko_tts": 0.10, "text_anchor": 0.05}},
        {"at": 20000, "weights": {"en_kd": 0.55, "en_solo": 0.10, "ko_tts": 0.30, "text_anchor": 0.05}},
        {"at": 45000, "weights": {"en_kd": 0.35, "en_solo": 0.10, "ko_tts": 0.50, "text_anchor": 0.05}},
    ],
    "constraints": {
        "text_anchor": {"min": 0.03, "max": 0.08},
        "require_groups": ["en_kd", "ko_tts", "text_anchor"],
    },
}


def ramp_schedule() -> MixSchedule:
    return MixSchedule.from_cfg(RAMP_CFG)


def flat_schedule(**weights: float) -> MixSchedule:
    return MixSchedule.from_cfg({"anchors": [{"at": 0, "weights": weights}]})


def costs(n: int, seed: int, lo: int = 250, hi: int = 1500) -> np.ndarray:
    """The real T distribution: 250..1500 frames (20 s .. 120 s)."""
    return np.random.default_rng(seed).integers(lo, hi + 1, size=n, dtype=np.int64)


def synth_index(n: int = 4000, **overrides: int) -> GroupIndex:
    sizes = {g: n for g in GROUPS}
    sizes.update(overrides)
    return GroupIndex.from_costs({g: costs(sizes[g], seed=i) for i, g in enumerate(GROUPS)})


def make_sampler(index=None, schedule=None, **kw) -> MixingBatchSampler:
    params = dict(
        steps_per_epoch=200, token_budget=6000, max_batch=16,
        bucket_width=100, grad_accum=4, seed=1234,
    )
    params.update(kw)
    return MixingBatchSampler(index or synth_index(), schedule or ramp_schedule(), **params)


# =========================================================== schedule (§5.1) ==
def test_weights_are_normalized_and_clamped():
    s = ramp_schedule()
    for step in (0, 1, 3999, 4000, 12345, 45000, 90000, -5):
        w = s.weights_at(step)
        assert set(w) == set(GROUPS)
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    # Clamped outside the anchor range -- an overrun job must not extrapolate.
    assert s.weights_at(10**9) == s.weights_at(45000)
    assert s.weights_at(-1) == s.weights_at(0)
    # Piecewise linear between anchors, on the unnormalized weights.
    mid = s.weights_at(2000)
    assert math.isclose(mid["en_kd"], 0.80, abs_tol=1e-9)


def test_unnormalized_weights_are_normalized_at_read_time():
    s = flat_schedule(a=3.0, b=1.0)
    assert s.weights_at(0) == pytest.approx({"a": 0.75, "b": 0.25})


def test_mean_weights_over_tracks_the_ramp():
    s = ramp_schedule()
    m = s.mean_weights_over(0, 4000)
    # The mean over a ramp sits strictly between the endpoints, unlike weights_at(0).
    assert s.weights_at(0)["ko_tts"] < m["ko_tts"] < s.weights_at(4000)["ko_tts"]
    assert math.isclose(sum(m.values()), 1.0, abs_tol=1e-9)
    with pytest.raises(AssertionError):
        s.mean_weights_over(10, 10)


# --- §6 item 5: schedule validation ------------------------------------------
def test_reject_non_monotonic_anchors():
    with pytest.raises(AssertionError, match="strictly increasing"):
        MixSchedule.from_cfg({"anchors": [
            {"at": 0, "weights": {"a": 1.0}},
            {"at": 500, "weights": {"a": 1.0}},
            {"at": 500, "weights": {"a": 1.0}},
        ]})


def test_reject_missing_anchor_at_zero():
    with pytest.raises(AssertionError, match="must be at step 0"):
        MixSchedule.from_cfg({"anchors": [{"at": 100, "weights": {"a": 1.0}}]})


def test_reject_group_set_mismatch_between_anchors():
    # A missing key is an error, not an implicit 0: that is exactly how a phase
    # silently loses text_anchor and makes Phase 4 meaningless.
    with pytest.raises(AssertionError, match="names groups"):
        MixSchedule.from_cfg({"anchors": [
            {"at": 0, "weights": {"en_kd": 0.9, "text_anchor": 0.1}},
            {"at": 100, "weights": {"en_kd": 0.9}},
        ]})


def test_reject_constraint_violated_between_anchors():
    # text_anchor's unnormalized weight is constant while en_kd ramps 1 -> 9, so
    # its normalized share decays 0.333 -> 0.0526 and crosses the 0.06 floor at
    # step ~854. The failure must be reported at a step strictly *between* the
    # anchors -- an anchor-only check would name step 1000 and hide the fact that
    # the run is already out of band for the last 15% of the segment.
    cfg = {
        "anchors": [
            {"at": 0, "weights": {"en_kd": 1.0, "text_anchor": 0.5}},
            {"at": 1000, "weights": {"en_kd": 9.0, "text_anchor": 0.5}},
        ],
        "constraints": {"text_anchor": {"min": 0.06, "max": 1.0}},
    }
    with pytest.raises(AssertionError, match=r"violated at step 900\b"):
        MixSchedule.from_cfg(cfg)


def test_constraint_sweep_granularity_is_the_stride():
    # Companion to the above, pinning that the sweep -- not the anchor set -- is
    # what found it. NOTE (deviation from plan §5.1): for a min/max band on a
    # single group under *linear* interpolation of unnormalized weights, the
    # normalized share is a ratio of two affine functions and is therefore
    # monotone inside each segment, so its extrema always land on anchors. A
    # violation strictly between anchors with both anchors in-band is impossible
    # today. The dense sweep is kept anyway (it is ~450 evaluations at build
    # time) because it stops being redundant the moment interp gains a non-linear
    # mode or a constraint becomes non-monotone, and because it reports the first
    # offending step rather than the anchor after it.
    cfg = {
        "anchors": [
            {"at": 0, "weights": {"en_kd": 1.0, "text_anchor": 0.5}},
            {"at": 1000, "weights": {"en_kd": 9.0, "text_anchor": 0.5}},
        ],
        "constraints": {"text_anchor": {"min": 0.06, "max": 1.0}},
    }
    coarse = dict(cfg, sweep_stride=2000)  # only steps 0 and 1000 get checked
    with pytest.raises(AssertionError, match=r"violated at step 1000\b"):
        MixSchedule.from_cfg(coarse)


def test_reject_require_groups_not_declared():
    with pytest.raises(AssertionError, match="require_groups"):
        MixSchedule.from_cfg({
            "anchors": [{"at": 0, "weights": {"a": 1.0}}],
            "constraints": {"require_groups": ["text_anchor"]},
        })


# ============================================================ sampler basics ==
def test_len_is_steps_per_epoch():                                  # §6 item 33
    s = make_sampler(steps_per_epoch=137)
    assert len(s) == 137
    assert sum(1 for _ in s) == 137


def test_token_budget_and_padding_waste():                          # §6 item 34
    s = make_sampler(steps_per_epoch=400)
    waste = []
    for batch in s:
        g = s.index.group_of(batch[0])
        cost = s.index.cost(g)[np.asarray(batch) - s.index.offset(g)]
        padded = len(batch) * int(cost.max())
        assert padded <= s.token_budget, f"budget blown: {padded} > {s.token_budget}"
        assert len(batch) <= s.max_batch
        waste.append(1.0 - cost.sum() / padded)
    assert np.mean(waste) < 0.15, f"padding waste {np.mean(waste):.3f} -- bucketing broken"


def test_no_batch_mixes_groups():                                   # §6 item 35
    s = make_sampler(steps_per_epoch=500)
    for batch in s:
        gs = {s.index.group_of(i) for i in batch}
        assert len(gs) == 1, f"batch spans groups {gs}"


def test_group_constant_across_grad_accum_window():                 # §6 item 36
    s = make_sampler(grad_accum=4)
    seq = [s.group_at(t) for t in range(4000)]
    for w in range(0, 4000, 4):
        assert len(set(seq[w : w + 4])) == 1, f"group changed inside window {w}"
    # ...and it is not a constant sequence pretending to pass.
    assert len(set(seq)) > 1


def test_all_ranks_draw_the_same_group_for_10k_steps():             # §6 item 37
    # The §4.5 regression: a rank-dependent group draw makes rank 1 issue an
    # all_gather (Depth Transformer / linears.0-7) that rank 0 never issues on a
    # text_anchor batch -> silent 100%-GPU hang. Intermittent, so a smoke test
    # passes; hence 10k steps across all 4 ranks.
    idx, sch = synth_index(), ramp_schedule()
    samplers = [make_sampler(idx, sch, rank=r, world_size=4) for r in range(4)]
    seqs = [[s.group_at(t) for t in range(10_000)] for s in samplers]
    assert seqs[0] == seqs[1] == seqs[2] == seqs[3]
    # Same holds for the batches actually emitted, not just the pure function.
    emitted = [[s.index.group_of(b[0]) for b in iter(s)] for s in samplers]
    assert emitted[0] == emitted[1] == emitted[2] == emitted[3]


def test_rank_partitions_are_disjoint_and_complete():               # §6 item 38
    idx, sch = synth_index(n=1500), ramp_schedule()
    ws = 4
    samplers = [make_sampler(idx, sch, rank=r, world_size=ws, steps_per_epoch=300)
                for r in range(ws)]
    for batches in zip(*[iter(s) for s in samplers]):
        rows = [set(b) for b in batches]
        assert sum(len(r) for r in rows) == len(set().union(*rows)), "ranks overlap"

    # Completeness is a property of the cut, before the trailing block is dropped:
    # every row of the group-epoch appears exactly once across all ranks.
    s0 = samplers[0]
    for g in GROUPS:
        flat = np.concatenate(s0._build(g, 0))
        n = idx.size(g)
        dropped = n - flat.size
        assert dropped < ws * s0.max_batch, "dropped more than one trailing block"
        assert len(set(flat.tolist())) == flat.size, "a row was emitted twice"
        assert set(flat.tolist()) <= set(range(n))


def test_set_epoch_reshuffles_rows_but_not_the_group_sequence():    # §6 item 39
    idx, sch = synth_index(), ramp_schedule()
    a = make_sampler(idx, sch, steps_per_epoch=300)
    b = make_sampler(idx, sch, steps_per_epoch=300)
    b.set_epoch(1)
    ba, bb = list(a), list(b)
    assert [idx.group_of(x[0]) for x in ba] == [idx.group_of(x[0]) for x in bb]
    assert ba != bb, "set_epoch did not reshuffle rows"


def test_state_dict_resume_reproduces_batches_index_for_index():    # §6 item 40
    idx, sch = synth_index(), ramp_schedule()
    ref = make_sampler(idx, sch, steps_per_epoch=1000)
    it = iter(ref)
    first = [next(it) for _ in range(500)]
    state = ref.state_dict()
    assert state["group_cursor"] and state["group_epoch"]
    rest = list(it)
    assert len(rest) == 500

    resumed = make_sampler(idx, sch, steps_per_epoch=500)
    resumed.load_state_dict(state)
    assert list(resumed) == rest, "resume diverged from the original stream"
    assert first != rest  # guard against a trivially constant stream


def test_resume_without_group_cursor_would_diverge():
    # Pins *why* group_cursor is in state_dict: dropping it silently restarts
    # every group stream at 0 and re-shows the small group's head.
    idx, sch = synth_index(), ramp_schedule()
    ref = make_sampler(idx, sch, steps_per_epoch=1000)
    it = iter(ref)
    for _ in range(500):
        next(it)
    rest = list(it)
    naive = make_sampler(idx, sch, steps_per_epoch=500)
    naive.set_step(500)  # step only, no cursors
    assert list(naive) != rest


def test_recycle_exposure_is_uniform_within_one():                  # §6 item 41
    idx = GroupIndex.from_costs({"en_kd": costs(600, seed=7)})
    sch = flat_schedule(en_kd=1.0)
    s = make_sampler(idx, sch, steps_per_epoch=400, world_size=1)
    seen = Counter(i for batch in s for i in batch)
    assert s.group_epoch["en_kd"] >= 2, "test did not recycle the group"
    assert max(seen.values()) - min(seen.values()) <= 1, (
        "without-replacement recycling must give every row floor(n) or ceil(n) "
        f"exposures, got {min(seen.values())}..{max(seen.values())}"
    )
    assert len(seen) == idx.size("en_kd")


def test_with_replacement_fails_the_same_uniformity_assertion():    # §6 item 41b
    # If the two modes ever silently coincide, this is the test that notices.
    idx = GroupIndex.from_costs({"en_kd": costs(600, seed=7)})
    sch = flat_schedule(en_kd=1.0)
    s = make_sampler(idx, sch, steps_per_epoch=400, world_size=1,
                     recycle="with_replacement")
    seen = Counter(i for batch in s for i in batch)
    assert max(seen.values()) - min(seen.values()) > 1, (
        "with_replacement produced a uniform exposure histogram -- the two "
        "recycle modes have silently become the same thing"
    )


def test_repeat_factor_warns_once_and_never_raises():
    idx = GroupIndex.from_costs({"en_kd": costs(200, seed=3)})
    s = make_sampler(idx, flat_schedule(en_kd=1.0), steps_per_epoch=400,
                     world_size=1, max_repeat_factor=1.0)
    with pytest.warns(RuntimeWarning, match="max_repeat_factor"):
        list(s)
    assert s.repeat_factor("en_kd") > 1.0
    assert s.group_epochs()["en_kd"] >= 1
    # Second pass: already warned, so no further warnings for this group.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        list(s)


def test_realized_ratios_reports_both_spaces():
    s = make_sampler(steps_per_epoch=200)
    list(s)
    r = s.realized_ratios()
    assert set(r) == {"steps", "frames"}
    for space in r.values():
        assert math.isclose(sum(space.values()), 1.0, abs_tol=1e-9)


def test_assert_group_sync_is_a_noop_without_distributed():
    assert_group_sync("en_kd", 0, GROUPS) is None


# ================================================ statistical ratio tests ====
def test_realized_step_ratios_match_mean_weights():
    # Written exactly as plan §6 specifies. The target is mean_weights_over, not
    # weights_at(step): inside a ramp the weights move across the window, so a
    # left-edge comparison manufactures a systematic bias that looks like a bug.
    idx, sch = synth_index(), ramp_schedule()
    W, step = 4000, 3000
    s = make_sampler(idx, sch, steps_per_epoch=W)
    s.set_step(step)

    counts = Counter(s.group_at(t) for t in range(step, step + W))
    n_updates = W // s.grad_accum   # the number of *independent* draws, not W
    for g, p in sch.mean_weights_over(step, step + W).items():
        se = math.sqrt(max(p * (1 - p), 1e-6) / n_updates)
        assert abs(counts[g] / W - p) < 4 * se + 0.01, (
            f"{g}: realized {counts[g] / W:.4f} vs target {p:.4f}"
        )


def test_realized_frame_ratios_match_mean_weights():
    # The companion in *frame* space, 3x looser. This is the one that catches the
    # §4.3 trap: "50% of steps are Korean" while Korean batches are 8x shorter, so
    # Korean is 12% of the gradient mass. What the curriculum means by "Korean
    # ratio" is frame occupancy. Every group shares one length distribution here,
    # so any deviation is the sampler's, not the corpus's.
    idx = GroupIndex.from_costs({g: costs(4000, seed=100 + i) for i, g in enumerate(GROUPS)})
    sch = ramp_schedule()
    W, step = 4000, 3000
    s = make_sampler(idx, sch, steps_per_epoch=W)
    s.set_step(step)
    list(s)

    frames = s.realized_ratios()["frames"]
    n_updates = W // s.grad_accum
    for g, p in sch.mean_weights_over(step, step + W).items():
        se = math.sqrt(max(p * (1 - p), 1e-6) / n_updates)
        assert abs(frames[g] - p) < 3 * (4 * se + 0.01), (
            f"{g}: frame occupancy {frames[g]:.4f} vs step target {p:.4f} -- "
            f"token-budget batching should keep these close"
        )


# ================================================== GroupIndex over Arrow ====
def test_group_index_from_prepared_uses_cost_not_num_frames(prepared, tmp_path):
    from tests.conftest import make_en_kd_sample, make_solo_sample, make_text_anchor_sample

    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=60 + i) for i in range(4)])
    prepared("ko_tts", "train", [make_solo_sample(f"ko{i}", T=30 + i) for i in range(3)])
    prepared("text_anchor", "train", [make_text_anchor_sample(f"tx{i}", L=500 + i) for i in range(3)])
    root = tmp_path / "prepared"

    idx = GroupIndex.from_prepared(root, ["en_kd", "ko_tts", "text_anchor"])
    assert idx.cost("en_kd").tolist() == [60, 61, 62, 63]
    assert idx.cost("ko_tts").tolist() == [30, 31, 32]
    # text_anchor has num_frames == 0; cost must fall back to the token length,
    # or every anchor row would have cost 0 and the whole group would collapse
    # into a single max_batch batch.
    assert idx.cost("text_anchor").tolist() == [500, 501, 502]

    # Global ids are contiguous per group and invertible.
    assert idx.offset("en_kd") == 0 and idx.offset("ko_tts") == 4
    assert idx.global_ids("ko_tts", [0, 2]).tolist() == [4, 6]
    assert idx.group_of(6) == "ko_tts" and idx.group_of(7) == "text_anchor"
    assert len(idx) == 10


def test_group_index_double_ab_matches_dataset_index_doubling(prepared, tmp_path):
    from tests.conftest import make_en_kd_sample

    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=60 + i) for i in range(3)])
    idx = GroupIndex.from_prepared(tmp_path / "prepared", ["en_kd"], double_ab=["en_kd"])
    # Dataset-side doubling maps global id i -> row i // 2, both directions.
    assert idx.cost("en_kd").tolist() == [60, 60, 61, 61, 62, 62]
