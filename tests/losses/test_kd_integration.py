"""Collator batch -> KD loss, end to end.

The unit tests in test_kd_loss.py feed the loss hand-built tensors. This file
feeds it what the collator actually emits, which is the only way to catch a
contract mismatch between the two (role axis, dtype, sentinel convention,
zone masking) rather than a mismatch inside either one.
"""

import pytest
import torch

from tests.datasets.test_collator import kd_row
from project_amnesty.datasets.collator import ZONE_A, KDCollator, KDCollatorConfig, DelayConfig
from project_amnesty.losses import semantic_kd_loss_from_batch

V = 2048


@pytest.fixture
def coll_cfg(tokens):
    """Local copy: test_collator's fixture is not importable across modules."""
    def _make(**kw):
        delay = kw.pop("delay", DelayConfig(acoustic_delay=2))
        return KDCollatorConfig(tokens=tokens, delay=delay, **kw)
    return _make


def _batch(coll_cfg, **kw):
    c = KDCollator(coll_cfg(**kw))
    return c, c([kd_row("a", 40), kd_row("b", 23, offset=7)])


def test_loss_runs_on_a_real_batch(coll_cfg):
    c, batch = _batch(coll_cfg)
    B, T = batch["text_tokens"].shape
    student = torch.randn(B, T, V)

    loss, m = semantic_kd_loss_from_batch(student, batch, tau=2.0)
    assert torch.isfinite(loss) and loss.item() > 0
    assert m["kd/supervised_frames"] > 0
    assert m["kd/supervised_frac"] < 1.0, "padding positions were supervised"


def test_teacher_aligned_student_scores_near_zero(coll_cfg):
    """Build the student from the batch's own teacher: the loss must bottom out.

    This is the integration-level alignment check. If the collator's teacher
    landed on the wrong frames, this still passes -- but combined with
    debug_alignment_check (which asserts the frames) it pins both halves.
    """
    c, batch = _batch(coll_cfg)
    B, T = batch["text_tokens"].shape
    t_val, t_idx = batch["teacher_topk_val"][:, 0], batch["teacher_topk_idx"][:, 0]

    student = torch.full((B, T, V), -1e4)
    student.scatter_(-1, t_idx.clamp_min(0), t_val.to(torch.float32))

    loss, _ = semantic_kd_loss_from_batch(student, batch, tau=1.0)
    assert loss.item() < 1e-2, f"teacher-matched student scored {loss.item()}"


def test_zone_a_positions_are_never_supervised(coll_cfg):
    """The teacher never saw the system prompt, so KD must not reach Zone A."""
    c, batch = _batch(coll_cfg, system_prompt_ids=(11, 12, 13, 14))
    zone_ids, kd_valid = batch["zone_ids"], batch["kd_valid"]
    assert (zone_ids == ZONE_A).any(), "test built no Zone A to check"
    assert not kd_valid[:, 0][zone_ids == ZONE_A].any(), "KD supervised Zone A"


def test_batch_without_teacher_contributes_nothing(coll_cfg):
    """A ko_tts-only batch: legal, zero loss, and flagged."""
    c = KDCollator(coll_cfg())
    batch = c([kd_row("a", 30, with_teacher=False, sample_type="ko_tts", lang="ko")])
    B, T = batch["text_tokens"].shape
    loss, m = semantic_kd_loss_from_batch(torch.randn(B, T, V), batch)
    assert loss.item() == 0.0
    assert m.get("kd/empty_batch") == 1.0


def test_wrapper_rejects_a_non_target_aligned_batch(coll_cfg):
    c, batch = _batch(coll_cfg)
    batch["target_aligned"] = False
    B, T = batch["text_tokens"].shape
    try:
        semantic_kd_loss_from_batch(torch.randn(B, T, V), batch)
    except AssertionError as e:
        assert "target-aligned" in str(e)
    else:
        raise AssertionError("wrapper accepted a batch that had already been shifted")


def test_collator_alignment_check_still_passes_after_extraction(coll_cfg):
    """kd_align's shift_scan, driven through the collator's own detector."""
    c = KDCollator(coll_cfg())
    rows = [kd_row("a", 40), kd_row("b", 23, offset=7)]
    rates = c.debug_alignment_check(c(rows), rows)
    assert max(rates, key=rates.get) == 0
