"""KD teacher alignment: the rules both the offline and the online path obey.

The offline path stores teacher top-k delay-free and lets `KDCollator` re-apply
the delay at batch time. The online path (on-policy KD, `TRAINING_CURRICULUM.md`
Phase 5) has no artifact and never touches the collator: a live teacher scores
the student's own rollout and hands back logits directly. That is the whole
reason this module exists separately from collator.py.

If the online path re-derives these rules inline, the two derivations drift, and
they drift *silently* -- a teacher shifted by one frame is still a valid
distribution, the loss still falls, and the student simply learns to predict the
wrong frame. The symptom surfaces much later as "quality degrades as we raise the
on-policy fraction", which reads as an exposure-bias result rather than a bug and
would invalidate the Phase 5 ablation it was meant to measure
(`RISKS_AND_DIAGNOSTICS.md` section 7.8).

Three rules, one implementation each:

* `teacher_offset` -- where codebook 0's teacher lands in output coordinates.
  Offline: `delay_offsets[1] + prefix`. Online: 0, because a live teacher already
  emits in the model's delayed coordinates; applying the shift again is a double
  application, exactly the class of bug `ingest_ab_selfplay` fights on the Colab
  capture side.
* `derive_kd_valid` -- KD supervision is a *restriction* of codebook 0's
  validity, never an independent recomputation.
* `transition_weight` -- the section 7.4 class-imbalance correction, built in
  pre-delay coordinates and shifted like the teacher it weights.

Plus `shift_scan`, the detector. It reads only a finished batch dict, so it runs
against an online batch just as well as an offline one -- and on the online path
it is the *only* thing standing between a double shift and a silently wrong
ablation. Run it there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def teacher_offset(cb0_offset: int, zone_a_frames: int, zone_b_frames: int) -> int:
    """Output-coordinate position where a row's Zone C teacher starts.

    The teacher generated its dialogue without any prefix, so its frames align to
    Zone C -- not to output position 0. Dropping the prefix term is the same class
    of bug as hardcoding ``cb0_offset = 0``: it converges and never raises.

    The online path passes ``cb0_offset=0`` (live logits are already delayed) but
    still needs the prefix term whenever Zone A/B are prepended to the rollout.
    """
    assert cb0_offset >= 0 and zone_a_frames >= 0 and zone_b_frames >= 0
    return cb0_offset + zone_a_frames + zone_b_frames


def derive_kd_valid(
    cb0_valid: torch.Tensor,
    teacher_row: torch.Tensor,
    zone_c: torch.Tensor,
) -> torch.Tensor:
    """(B, R, T) bool. KD-supervised positions.

    Args:
        cb0_valid:   (B, R, T) codebook 0's validity. The base set -- KD is a
                     subset of it by construction, which `assert_kd_valid_subset`
                     re-checks on the finished batch.
        teacher_row: (B, R) which (row, role) pairs carry a teacher at all. Only
                     role 0 does: the other stream is input, not a target.
        zone_c:      (B, T) the teacher's own footprint. Zone A is a system prompt
                     and Zone B a reference utterance; the teacher saw neither, so
                     it has nothing to say about those positions.
    """
    assert cb0_valid.dim() == 3 and teacher_row.dim() == 2 and zone_c.dim() == 2
    assert cb0_valid.shape[:2] == teacher_row.shape
    assert cb0_valid.shape[0] == zone_c.shape[0] and cb0_valid.shape[2] == zone_c.shape[1]
    return cb0_valid & teacher_row[:, :, None] & zone_c[:, None, :]


def transition_weight(
    text: torch.Tensor,
    *,
    pad_id: int,
    epad_id: int,
    weight: float,
    halfwidth: int,
) -> torch.Tensor:
    """(L,) fp32 KD frame weight in **pre-delay** coordinates.

    Section 7.4: most frames are silence, so an unweighted per-frame KL converges
    happily to "always predict silence". Speech onsets and offsets get `weight`,
    everything else 1.0.

    Pre-delay is not an implementation detail. The weight is defined on the text
    stream's own time base, and the caller shifts it by the same offset as the
    teacher it weights -- computing it post-delay means reconstructing the delay,
    which is the canonical section 7.8 bug.
    """
    assert text.dim() == 1
    assert halfwidth >= 0
    L = text.shape[0]
    w = torch.ones((L,), dtype=torch.float32, device=text.device)
    if L == 0 or weight == 1.0:
        return w
    active = (text != pad_id) & (text != epad_id)
    prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=text.device), active[:-1]])
    trans = (active & ~prev) | (~active & prev)          # onset | offset
    if halfwidth > 0:
        trans = F.max_pool1d(
            trans.float()[None, None], kernel_size=2 * halfwidth + 1, stride=1, padding=halfwidth
        )[0, 0] > 0
    return 1.0 + (weight - 1.0) * trans.float()


def assert_kd_valid_subset(kd_valid: torch.Tensor, stream_valid: torch.Tensor) -> None:
    """kd_valid must be a subset of codebook 0's validity, not a parallel derivation."""
    cb0 = stream_valid[:, :, 0]
    assert torch.equal(kd_valid, cb0 & kd_valid), (
        "kd_valid is not a subset of stream_valid[:, :, 0] -- it must be derived from it"
    )


def shift_scan(
    codes_cb0: torch.Tensor,
    teacher_idx: torch.Tensor,
    kd_valid: torch.Tensor,
    shifts=(-2, -1, 0, 1, 2),
) -> dict[int, float]:
    """Per-shift teacher hit rate. Peak must be at 0.

    A stored sample token was drawn from that frame's own logits, so it must
    appear in that frame's teacher top-k. `topk_dump=32` against `gen_top_k=250`
    means the rate at zero shift is well below 1.0, which is why the signal is the
    **argmax over shift**, not an absolute threshold.

    Catches a hardcoded cb0 offset, a dropped prefix term, a skipped normalization
    under negative text delay, a crop off-by-one, and a double shift -- the last
    being the online path's characteristic failure.

    Args:
        codes_cb0:   (B, T) int, self-stream codebook 0 at student position p.
        teacher_idx: (B, T, k) int, teacher top-k indices.
        kd_valid:    (B, T) bool, self-role KD mask.
    """
    assert codes_cb0.dim() == 2 and teacher_idx.dim() == 3 and kd_valid.dim() == 2
    T = codes_cb0.shape[-1]
    rates: dict[int, float] = {}
    for d in shifts:
        lo, hi = max(0, -d), min(T, T - d)
        if hi <= lo:
            rates[d] = 0.0
            continue
        cb0 = codes_cb0[:, lo:hi]                         # student position p
        tk = teacher_idx[:, lo + d:hi + d]                # teacher position p+d
        m = kd_valid[:, lo:hi] & kd_valid[:, lo + d:hi + d]
        hit = (tk == cb0[..., None]).any(-1) & m
        rates[d] = float(hit.sum()) / max(1, int(m.sum()))
    return rates


def assert_aligned(rates: dict[int, float], min_hit_rate: float, context: str = "") -> None:
    """Fail on a shift_scan whose peak is not at zero shift."""
    peak = max(rates, key=rates.get)
    assert peak == 0, (
        f"section 7.8 misalignment: shift-scan hit rate peaks at {peak} (rates={rates}). {context}"
    )
    assert rates[0] > min_hit_rate, (
        f"KD hit rate at zero shift is {rates[0]:.3f} <= {min_hit_rate}; "
        f"the teacher dump is likely corrupt. {context}"
    )
