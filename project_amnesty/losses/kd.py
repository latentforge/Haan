"""Semantic KD loss: top-k KL between the teacher and the student's codebook 0.

Scope (`ARCHITECTURE.md` section 5.1): codebook 0 only. Acoustic codebooks 1..7 are
excluded from the default KD path because they carry the teacher's fixed speaker
timbre; they are dumped anyway so Phase 3.5's localized acoustic graft and the
Phase 5 KD-scope ablation do not require regenerating the corpus. Text is SeqKD,
not logit KD -- different tokenizer, so it never reaches this file.

The function takes **tensors, not a batch dict**. That is the load-bearing design
choice: the online path (on-policy KD) produces teacher logits from a live
teacher over the student's own rollout and never builds a collator batch. Both
paths call `semantic_kd_loss` with the same four teacher-side arguments, so the
KD objective cannot drift between the two arms of the Phase 5 ablation that
compares them. Alignment is the caller's job and is shared through
`project_amnesty/datasets/kd_align.py`.
"""

from __future__ import annotations

import torch


def semantic_kd_loss(
    student_logits: torch.Tensor,
    teacher_topk_val: torch.Tensor,
    teacher_topk_idx: torch.Tensor,
    kd_valid: torch.Tensor,
    kd_frame_weight: torch.Tensor | None = None,
    *,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted top-k KL(teacher || student) over codebook 0.

    Args:
        student_logits:   (B, T, V) fp32/bf16 raw logits for codebook 0, already in
                          **target alignment** -- the model owns the 1-step
                          autoregressive shift (collator ships `target_aligned=True`
                          as the tripwire).
        teacher_topk_val: (B, T, k) raw pre-softmax teacher logits. No temperature
                          baked in; see plan section 2.8.
        teacher_topk_idx: (B, T, k) int, vocab indices. `-1` where absent.
        kd_valid:         (B, T) bool, self-role KD mask (kd_align.derive_kd_valid).
        kd_frame_weight:  (B, T) fp32 section 7.4 transition weights, or None for 1.0.
        tau:              KD temperature. Applied here, never at dump time.

    Returns:
        (loss, metrics). `loss` is a scalar; `metrics` carries the diagnostics that
        distinguish "KD is working" from "KD is on but supervising nothing".
    """
    assert student_logits.dim() == 3, f"expected (B, T, V), got {tuple(student_logits.shape)}"
    assert teacher_topk_val.shape == teacher_topk_idx.shape
    assert student_logits.shape[:2] == teacher_topk_val.shape[:2] == kd_valid.shape
    assert tau > 0

    B, T, V = student_logits.shape

    # --- teacher distribution over the top-k support -------------------------
    # fp32 BEFORE dividing by tau: teacher_val is fp16 on the wire, and a small
    # tau overflows fp16 ahead of the log-sum-exp (plan section 2.8, note 3).
    t_logits = teacher_topk_val.to(torch.float32) / tau

    # `-1` is a sentinel, not a vocab id. Two things guard it: the gather index is
    # clamped so it can never wrap to the last row of the vocab, and the slot is
    # masked out of the softmax so it carries no probability mass either way.
    slot_valid = teacher_topk_idx >= 0
    idx = teacher_topk_idx.clamp_min(0).to(torch.int64)

    t_logits = t_logits.masked_fill(~slot_valid, float("-inf"))
    # Renormalized over the top-k support only. The teacher's truncated tail mass
    # is discarded rather than modeled -- a top-k dump cannot represent it, and
    # softmax over a subset is not the tempered distribution anyway (plan 2.8).
    # An all-invalid row would produce NaN here, so those rows are zeroed after.
    row_ok = slot_valid.any(dim=-1)
    t_logits = torch.where(row_ok[..., None], t_logits, torch.zeros_like(t_logits))
    t_prob = torch.softmax(t_logits, dim=-1)
    t_prob = t_prob * slot_valid
    t_logp = torch.log(t_prob + eps)

    # --- student log-probs at the teacher's support --------------------------
    # log_softmax over the FULL vocab, then gather. Normalizing over the top-k
    # subset instead would let the student put arbitrary mass outside the support
    # at zero cost, which removes the only pressure this loss applies.
    s_logp_full = torch.log_softmax(student_logits.to(torch.float32) / tau, dim=-1)
    s_logp = s_logp_full.gather(-1, idx)

    # --- KL and weighting ----------------------------------------------------
    kl = (t_prob * (t_logp - s_logp)).sum(dim=-1)              # (B, T)

    w = kd_valid.to(torch.float32)
    if kd_frame_weight is not None:
        assert kd_frame_weight.shape == kd_valid.shape
        w = w * kd_frame_weight.to(torch.float32)
    w = w * row_ok.to(torch.float32)

    # Normalize by the WEIGHT sum, not the frame count: dividing by frames would
    # let the section 7.4 transition weights swing the effective learning rate
    # batch to batch, since the share of onset frames varies with the crop.
    denom = w.sum()
    # tau^2 keeps the gradient scale independent of tau, so sweeping tau does not
    # implicitly re-tune the learning rate alongside it.
    loss = (w * kl).sum() / denom.clamp_min(eps) * (tau ** 2)

    n_sup = int(kd_valid.sum())
    metrics = {
        "kd/loss": float(loss.detach()),
        "kd/tau": float(tau),
        # Coverage, not cosmetics: a correct-looking loss over 2% of frames means
        # the mask collapsed somewhere upstream, and nothing else reports it.
        "kd/supervised_frames": n_sup,
        "kd/supervised_frac": n_sup / max(1, B * T),
        "kd/weight_sum": float(denom.detach()),
        # Teacher confidence retained inside the dumped support. If this drifts
        # toward 0, topk_dump is too small for the sampling temperature used.
        "kd/teacher_support_mass": float(
            (t_prob.sum(-1) * (w > 0)).sum().detach() / max(1.0, float((w > 0).sum()))
        ),
    }
    if denom <= 0:
        # Not an assert: a legitimate all-ko_tts batch has no teacher at all. But
        # it must be visible, because "KD silently contributed nothing" and "KD ran"
        # produce the same loss curve.
        metrics["kd/empty_batch"] = 1.0
    return loss, metrics


def semantic_kd_loss_from_batch(
    student_logits: torch.Tensor,
    batch: dict,
    *,
    tau: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Offline-path convenience wrapper: pulls role 0 out of a collator batch.

    Role 0 is the modeled speaker; role 1 is input conditioning and carries no
    teacher (collator sets `teacher_row[:, 1] = False`). The online path does not
    use this wrapper -- it calls `semantic_kd_loss` directly with live tensors.
    """
    assert batch.get("target_aligned") is True, (
        "batch is not target-aligned; the model owns the 1-step shift and applying "
        "it twice is an 80 ms timing distortion that trains happily"
    )
    return semantic_kd_loss(
        student_logits,
        batch["teacher_topk_val"][:, 0],
        batch["teacher_topk_idx"][:, 0],
        batch["kd_valid"][:, 0],
        batch["kd_frame_weight"][:, 0],
        tau=tau,
    )
