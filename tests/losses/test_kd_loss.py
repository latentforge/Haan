"""Semantic KD loss + the shared alignment rules.

The properties asserted here are the ones whose violation is silent: a loss that
still descends while supervising the wrong frames, the wrong subset, or nothing
at all.
"""

import pytest
import torch

from project_amnesty.datasets.runtime.kd_align import (
    assert_aligned,
    derive_kd_valid,
    shift_scan,
    teacher_offset,
    transition_weight,
)
from project_amnesty.losses import semantic_kd_loss

V = 2048


def _teacher_from_probs(probs: torch.Tensor, k: int):
    """(B,T,V) target distribution -> (val, idx) top-k raw-logit dump."""
    top = probs.topk(k, dim=-1)
    return top.values.log().to(torch.float16), top.indices.to(torch.int64)


def _uniform_batch(B=2, T=6, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(B, T, V, generator=g), dim=-1)
    val, idx = _teacher_from_probs(probs, k)
    return probs, val, idx


# --------------------------------------------------------------- loss basics

def test_zero_when_student_matches_teacher_on_support():
    """A student whose distribution IS the teacher's top-k gives ~0 loss.

    Guards the direction of the KL and the teacher-side renormalization together:
    get either wrong and the floor is not zero.
    """
    _, val, idx = _uniform_batch()
    B, T, k = idx.shape
    # Build student logits that reproduce the teacher's renormalized top-k exactly.
    student = torch.full((B, T, V), -1e4)
    student.scatter_(-1, idx, val.to(torch.float32))
    kd_valid = torch.ones(B, T, dtype=torch.bool)

    loss, m = semantic_kd_loss(student, val, idx, kd_valid, tau=1.0)
    assert loss.item() < 1e-3, f"expected ~0, got {loss.item()}"
    assert m["kd/supervised_frames"] == B * T


def test_loss_is_positive_and_finite_for_mismatched_student():
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    student = torch.randn(B, T, V)
    loss, _ = semantic_kd_loss(student, val, idx, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(loss) and loss.item() > 0


def test_masked_frames_do_not_contribute():
    """kd_valid=False frames must not move the loss, whatever they contain.

    This is the property that lets Zone A/B and batch padding sit in the same
    tensor as real supervision.
    """
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    student = torch.randn(B, T, V)
    kd_valid = torch.ones(B, T, dtype=torch.bool)
    kd_valid[:, T // 2:] = False

    ref, _ = semantic_kd_loss(student, val, idx, kd_valid)

    poisoned = student.clone()
    poisoned[:, T // 2:] = torch.randn(B, T - T // 2, V) * 50
    got, _ = semantic_kd_loss(poisoned, val, idx, kd_valid)
    assert torch.allclose(ref, got), "masked frames leaked into the loss"


def test_full_vocab_normalization_penalizes_offsupport_mass():
    """Mass parked outside the teacher's top-k must cost something.

    If the student were log_softmax'd over the top-k subset instead of the full
    vocab, these two would score identically and the loss would stop constraining
    anything outside the support.
    """
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape

    on_support = torch.full((B, T, V), -1e4)
    on_support.scatter_(-1, idx, val.to(torch.float32))

    leaky = on_support.clone()
    # A vocab entry that is guaranteed not to be in any frame's top-k.
    spare = torch.full((B, T, 1), V - 1, dtype=torch.int64)
    assert not (idx == V - 1).any()
    leaky.scatter_(-1, spare, torch.full((B, T, 1), 10.0))

    l_on, _ = semantic_kd_loss(on_support, val, idx, torch.ones(B, T, dtype=torch.bool))
    l_leak, _ = semantic_kd_loss(leaky, val, idx, torch.ones(B, T, dtype=torch.bool))
    assert l_leak > l_on + 1e-3, "off-support mass is free -- student log_softmax is not full-vocab"


def test_sentinel_index_is_inert():
    """A -1 slot must be equivalent to not having that slot at all.

    Two failure modes it rules out: gathering at -1 (which wraps to the last vocab
    entry and silently supervises a random token), and letting the dead slot take
    softmax mass away from the real ones.
    """
    _, val, idx = _uniform_batch()
    B, T, k = idx.shape
    student = torch.randn(B, T, V)
    kd_valid = torch.ones(B, T, dtype=torch.bool)

    marked = idx.clone()
    marked[:, :, -1] = -1                   # last slot absent everywhere
    with_sentinel, _ = semantic_kd_loss(student, val, marked, kd_valid)
    # The same teacher with that slot genuinely dropped.
    dropped, _ = semantic_kd_loss(student, val[:, :, :-1], idx[:, :, :-1], kd_valid)

    assert torch.isfinite(with_sentinel)
    assert torch.allclose(with_sentinel, dropped, atol=1e-4), (
        f"sentinel slot is not inert: {with_sentinel.item()} vs {dropped.item()}"
    )


def test_empty_teacher_batch_is_flagged_not_crashed():
    """An all-ko_tts batch is legal and must report that KD contributed nothing."""
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    loss, m = semantic_kd_loss(
        torch.randn(B, T, V), val, idx, torch.zeros(B, T, dtype=torch.bool)
    )
    assert torch.isfinite(loss) and loss.item() == 0.0
    assert m.get("kd/empty_batch") == 1.0
    assert m["kd/supervised_frames"] == 0


def test_frame_weight_normalizes_by_weight_sum():
    """Scaling every weight by c must leave the loss unchanged.

    Normalizing by frame count instead would make the loss proportional to c, so
    the share of onset frames in a crop would swing the effective learning rate.
    """
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    student = torch.randn(B, T, V)
    kd_valid = torch.ones(B, T, dtype=torch.bool)
    w = torch.rand(B, T) + 0.5

    a, _ = semantic_kd_loss(student, val, idx, kd_valid, w)
    b, _ = semantic_kd_loss(student, val, idx, kd_valid, w * 3.0)
    assert torch.allclose(a, b, atol=1e-5), "loss is not scale-invariant in the frame weights"


def test_frame_weight_actually_reweights():
    """Upweighting frames the student is bad at must raise the loss."""
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    student = torch.full((B, T, V), -1e4)
    student.scatter_(-1, idx, val.to(torch.float32))
    student[:, 0] = torch.randn(B, V) * 5          # frame 0 is now wrong
    kd_valid = torch.ones(B, T, dtype=torch.bool)

    flat = torch.ones(B, T)
    peaked = torch.ones(B, T)
    peaked[:, 0] = 5.0
    lo, _ = semantic_kd_loss(student, val, idx, kd_valid, flat)
    hi, _ = semantic_kd_loss(student, val, idx, kd_valid, peaked)
    assert hi > lo


def test_fp16_teacher_survives_small_tau():
    """tau < 1 on fp16 logits must not overflow (plan section 2.8, note 3)."""
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    val = (val.to(torch.float32) * 200).to(torch.float16)   # large-magnitude logits
    loss, _ = semantic_kd_loss(
        torch.randn(B, T, V), val, idx, torch.ones(B, T, dtype=torch.bool), tau=0.1
    )
    assert torch.isfinite(loss), "fp16 teacher overflowed before the log-sum-exp"


def test_tau_squared_keeps_gradient_scale_stable():
    """Gradient magnitude must not collapse as tau grows."""
    _, val, idx = _uniform_batch()
    B, T, _ = idx.shape
    grads = []
    for tau in (1.0, 4.0):
        s = torch.randn(B, T, V, requires_grad=True)
        loss, _ = semantic_kd_loss(s, val, idx, torch.ones(B, T, dtype=torch.bool), tau=tau)
        loss.backward()
        grads.append(s.grad.abs().mean().item())
    # Without the tau^2 term this ratio would be ~1/16.
    assert grads[1] / grads[0] > 0.1, f"gradient scale collapsed with tau: {grads}"


# --------------------------------------------------------- alignment sharing

def test_teacher_offset_includes_the_prefix():
    assert teacher_offset(2, 0, 0) == 2
    assert teacher_offset(2, 10, 5) == 17
    # The online path's case: live logits are already delayed.
    assert teacher_offset(0, 10, 5) == 15


def test_derive_kd_valid_is_a_subset_of_cb0():
    B, R, T = 2, 2, 5
    cb0 = torch.rand(B, R, T) > 0.3
    teacher_row = torch.tensor([[True, False], [True, False]])
    zone_c = torch.rand(B, T) > 0.2
    kd = derive_kd_valid(cb0, teacher_row, zone_c)
    assert torch.equal(kd, cb0 & kd), "kd_valid escaped codebook 0's validity"
    assert not kd[:, 1].any(), "role 1 must never carry a teacher"


def test_transition_weight_marks_onsets_only():
    pad, epad = 0, 1
    text = torch.tensor([pad, pad, 5, 6, pad, pad, pad, pad])
    w = transition_weight(text, pad_id=pad, epad_id=epad, weight=3.0, halfwidth=0)
    assert w[2].item() == pytest.approx(3.0)      # onset
    assert w[4].item() == pytest.approx(3.0)      # offset
    assert w[0].item() == pytest.approx(1.0)      # steady silence
    assert w[7].item() == pytest.approx(1.0)

    # weight == 1.0 short-circuits to all ones
    flat = transition_weight(text, pad_id=pad, epad_id=epad, weight=1.0, halfwidth=6)
    assert torch.allclose(flat, torch.ones_like(flat))


def test_shift_scan_peaks_at_zero_when_aligned():
    B, T, k = 2, 40, 8
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, V, (B, T), generator=g)
    idx = torch.randint(0, V, (B, T, k), generator=g)
    idx[:, :, 0] = codes                          # the sampled token is in its own top-k
    kd_valid = torch.ones(B, T, dtype=torch.bool)

    rates = shift_scan(codes, idx, kd_valid)
    assert max(rates, key=rates.get) == 0
    assert_aligned(rates, min_hit_rate=0.5)


def test_shift_scan_catches_a_double_shift():
    """The online path's characteristic bug: delay applied twice."""
    B, T, k = 2, 40, 8
    g = torch.Generator().manual_seed(4)
    codes = torch.randint(0, V, (B, T), generator=g)
    idx = torch.randint(0, V, (B, T, k), generator=g)
    idx[:, :, 0] = codes
    # Shift the teacher one frame late, as a re-applied delay would.
    idx = torch.roll(idx, shifts=1, dims=1)
    kd_valid = torch.ones(B, T, dtype=torch.bool)

    rates = shift_scan(codes, idx, kd_valid)
    assert max(rates, key=rates.get) != 0
    with pytest.raises(AssertionError, match="misalignment"):
        assert_aligned(rates, min_hit_rate=0.5)
