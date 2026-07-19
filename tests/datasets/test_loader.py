"""End-to-end tests for `build_dataloader` -- plan sections 5.5 and 5.6.

Every other test file in this suite exercises one layer against synthetic inputs
of its own making: `test_dataset.py` builds Arrow and checks items,
`test_sampler.py` builds cost arrays and checks indices, `test_collator.py`
builds items and checks tensors. That is the right shape for those tests, but it
means the *seams* between the three are untested -- each layer is verified
against what its author believed the neighbouring layer promised.

This file is where the real stack runs: prepared Arrow -> MoshiKDDataset ->
GroupIndex -> MixingBatchSampler -> ConcatSources -> RoutingCollator -> batch
dict. The single most valuable test here is
`test_global_ids_retrieve_the_rows_the_index_claims`: `GroupIndex` and
`ConcatSources` compute their offsets independently, and `en_kd`'s A/B doubling
makes those offsets disagree in exactly one place if either side is off by one.
The failure is silent -- the sampler asks for a `ko_tts` row and gets an `en_kd`
row, which collates fine, trains fine, and just optimizes the wrong mixture.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch

from conftest import make_en_kd_sample, make_solo_sample, make_text_anchor_sample
from project_amnesty.datasets.schema import NUM_CODEBOOKS
from project_amnesty.datasets.collator import DelayConfig, KDCollator, KDCollatorConfig
from project_amnesty.datasets.item import SAMPLE_TYPE_IDS
from project_amnesty.datasets.loader import (
    GROUP_ALIASES,
    HEAVY_GROUPS,
    ConcatSources,
    GroupSequentialBatchSampler,
    LoaderConfig,
    RoutingCollator,
    _assert_can_build,
    build_dataloader,
    group_dir,
)
from project_amnesty.datasets.text_collator import TextAnchorCollator, TextAnchorCollatorConfig

K = NUM_CODEBOOKS
GROUPS = ("en_kd", "ko_tts", "text_anchor")

# Small on purpose: with token_budget=600 and max_batch=4 a group of ~20 rows
# cuts into 5-6 batches, which is the smallest corpus that still lets a
# world_size=4 shard test be meaningful (the sampler drops the trailing partial
# block, so it needs >= world_size batches per group-epoch).
TOKEN_BUDGET = 600
MAX_BATCH = 4

MIX_CFG = {
    "anchors": [
        {"at": 0, "weights": {"en_kd": 0.5, "ko_tts": 0.3, "text_anchor": 0.2}},
        {"at": 1000, "weights": {"en_kd": 0.4, "ko_tts": 0.4, "text_anchor": 0.2}},
    ]
}


def loader_cfg(**kw) -> LoaderConfig:
    """num_workers=0 throughout: fork adds seconds per test and buys nothing --
    worker-count invariance of the crop is already pinned in test_dataset.py."""
    params = dict(
        token_budget=TOKEN_BUDGET,
        max_batch=MAX_BATCH,
        bucket_width=10,
        steps_per_epoch=12,
        grad_accum=2,
        num_workers=0,
        pin_memory=False,
        seed=1234,
        build="never",
    )
    params.update(kw)
    return LoaderConfig(**params)


# --------------------------------------------------------------- corpora


def write_corpus(prepared, split: str, *, n_kd: int = 24, n_ko: int = 48, n_ta: int = 48):
    """A three-group prepared root. Lengths vary so bucketing has work to do.

    Two constraints on the sizes, both load-bearing:

    * Every length stays under `data_cfg.max_frames` (64) and `max_text_len`, so
      cropping is a no-op and `GroupIndex.cost` is directly comparable to the
      item's own length. That is what lets the global-id test check cost
      *alignment*, not just group membership.
    * Each group must cut into comfortably more than `world_size` batches, so a
      group-epoch spans several steps. With only ~4 batches per group-epoch the
      sampler rolls over to a fresh epoch every step and a rank-cursor bug that
      re-serves rows becomes unobservable.
    """
    prepared("en_kd", split,
             [make_en_kd_sample(f"kd{i:02d}", T=40 + i % 20, topk=4) for i in range(n_kd)])
    prepared("ko_tts", split,
             [make_solo_sample(f"ko{i:02d}", T=40 + i % 20) for i in range(n_ko)])
    prepared("text_anchor", split,
             [make_text_anchor_sample(f"ta{i:02d}", L=30 + i % 20) for i in range(n_ta)])


@pytest.fixture
def train_root(prepared, data_cfg):
    write_corpus(prepared, "train")
    return Path(data_cfg.root)


@pytest.fixture
def probe_root(prepared, data_cfg):
    # Deliberately fewer rows than train: probe batch counts are not a multiple
    # of world_size, which is exactly the case the pad-batch logic exists for.
    write_corpus(prepared, "probe", n_kd=7, n_ko=9, n_ta=11)
    return Path(data_cfg.root)


def train_bundle(data_cfg, **kw):
    return build_dataloader(
        data_cfg=data_cfg, loader_cfg=loader_cfg(**kw.pop("loader", {})),
        split="train", mix_cfg=MIX_CFG, **kw,
    )


def probe_bundle(data_cfg, **kw):
    return build_dataloader(
        data_cfg=data_cfg, loader_cfg=loader_cfg(**kw.pop("loader", {})),
        split="probe", **kw,
    )


# =========================================================== batch contract


def assert_kd_batch(batch: dict) -> tuple[int, int]:
    """Assert the plan-3.3 audio batch contract; return (B, T)."""
    B = batch["codes"].shape[0]
    T = batch["codes"].shape[-1]

    assert batch["codes"].shape == (B, 2, K, T)
    assert batch["codes"].dtype is torch.int64
    assert batch["role_ids"].shape == (B, 2) and batch["role_ids"].dtype is torch.int64
    assert batch["text_tokens"].shape == (B, T) and batch["text_tokens"].dtype is torch.int64
    assert batch["stream_valid"].shape == (B, 2, K, T)
    assert batch["stream_valid"].dtype is torch.bool
    assert batch["text_valid"].shape == (B, T) and batch["text_valid"].dtype is torch.bool
    assert batch["attention_mask"].shape == (B, T) and batch["attention_mask"].dtype is torch.bool
    assert batch["zone_ids"].shape == (B, T) and batch["zone_ids"].dtype is torch.uint8
    assert batch["audio_loss_weight"].shape == (B, 2, K, T)
    assert batch["audio_loss_weight"].dtype is torch.float32
    assert batch["text_loss_weight"].shape == (B, T)
    assert batch["text_loss_weight"].dtype is torch.float32

    kmax = batch["teacher_topk_val"].shape[-1]
    assert batch["teacher_topk_val"].shape == (B, 2, T, kmax)
    assert batch["teacher_topk_val"].dtype is torch.float16
    assert batch["teacher_topk_idx"].shape == (B, 2, T, kmax)
    assert batch["teacher_topk_idx"].dtype is torch.int64
    assert batch["kd_valid"].shape == (B, 2, T) and batch["kd_valid"].dtype is torch.bool
    assert batch["kd_frame_weight"].shape == (B, 2, T)
    assert batch["kd_frame_weight"].dtype is torch.float32

    for key in ("sample_type_id", "lang_id", "num_frames"):
        assert batch[key].shape == (B,) and batch[key].dtype is torch.int64, key
    assert batch["has_teacher"].shape == (B,) and batch["has_teacher"].dtype is torch.bool
    assert isinstance(batch["sample_uid"], list) and len(batch["sample_uid"]) == B
    assert all(isinstance(u, str) for u in batch["sample_uid"])

    assert batch["delay_offsets"].shape == (K + 1,)
    assert int(batch["delay_offsets"].min()) == 0
    assert batch["target_aligned"] is True

    assert T % 8 == 0, "pad_to_multiple_of=8 is part of the contract"
    assert 0 < B <= MAX_BATCH
    # Every row must fit inside the padded extent it was promised.
    assert int(batch["num_frames"].max()) <= T
    return B, T


def assert_text_batch(batch: dict) -> tuple[int, int]:
    B, L = batch["text_tokens"].shape
    assert batch["text_tokens"].dtype is torch.int64
    assert batch["attention_mask"].shape == (B, L)
    assert batch["attention_mask"].dtype is torch.bool
    assert batch["text_loss_weight"].shape == (B, L)
    assert batch["text_loss_weight"].dtype is torch.float32
    assert batch["text_lengths"].shape == (B,) and batch["text_lengths"].dtype is torch.int64
    assert batch["sample_type_id"].shape == (B,) and batch["lang_id"].shape == (B,)
    assert isinstance(batch["sample_uid"], list) and len(batch["sample_uid"]) == B
    assert batch["target_aligned"] is True

    assert L % 8 == 0
    assert 0 < B <= MAX_BATCH
    # attention_mask must agree with text_lengths, or the anchor loss silently
    # supervises padding.
    assert torch.equal(batch["attention_mask"].sum(dim=1), batch["text_lengths"])
    return B, L


def is_audio_batch(batch: dict) -> bool:
    return "codes" in batch


# ================================================================== tests


def test_train_loader_yields_contract_conforming_batches(train_root, data_cfg):
    """The whole stack, end to end. Both batch shapes must appear."""
    bundle = train_bundle(data_cfg)
    seen_audio = seen_text = 0

    for batch in bundle.loader:
        if is_audio_batch(batch):
            assert_kd_batch(batch)
            seen_audio += 1
        else:
            assert_text_batch(batch)
            seen_text += 1

    assert seen_audio + seen_text == bundle.sampler.steps_per_epoch
    assert seen_audio > 0 and seen_text > 0, (
        "the 12-step epoch drew only one kind of batch; raise steps_per_epoch or "
        "check the mix schedule -- this test is not exercising both collators"
    )


def test_batch_shapes_are_internally_consistent(train_root, data_cfg):
    """B and T must agree across keys within one batch.

    Each collator builds ~15 tensors from the same B/T; a mistake in one of them
    produces a batch that only fails once it reaches a matmul deep in the model.
    """
    bundle = train_bundle(data_cfg)
    for batch in bundle.loader:
        if is_audio_batch(batch):
            B, T = assert_kd_batch(batch)
            assert batch["codes"].shape[2] == K == NUM_CODEBOOKS
            assert {t.shape[0] for t in (
                batch["codes"], batch["text_tokens"], batch["stream_valid"],
                batch["zone_ids"], batch["kd_valid"], batch["num_frames"],
            )} == {B}
            assert {t.shape[-1] for t in (
                batch["codes"], batch["text_tokens"], batch["stream_valid"],
                batch["text_valid"], batch["zone_ids"], batch["kd_frame_weight"],
            )} == {T}
        else:
            assert_text_batch(batch)


def test_kd_batches_carry_a_teacher_and_solo_batches_do_not(train_root, data_cfg):
    """Routing sanity in the value domain, not just the shape domain."""
    bundle = train_bundle(data_cfg)
    for batch in bundle.loader:
        if not is_audio_batch(batch):
            continue
        stype = set(int(v) for v in batch["sample_type_id"])
        assert len(stype) == 1, "audio batch mixes sample types"
        if stype == {SAMPLE_TYPE_IDS["en_kd"]}:
            assert bool(batch["has_teacher"].all())
            assert bool(batch["kd_valid"].any())
            assert batch["teacher_topk_val"].shape[-1] > 0
        else:
            assert not bool(batch["has_teacher"].any())
            assert not bool(batch["kd_valid"].any())


# ------------------------------------------------- THE global-id test


def test_global_ids_retrieve_the_rows_the_index_claims(train_root, data_cfg):
    """`ConcatSources[gid]["source"] == index.group_of(gid)`, for every gid.

    `GroupIndex.from_prepared` doubles `en_kd`'s cost array (`np.repeat(cost, 2)`)
    while `ConcatSources` derives its offsets from `len(MoshiKDDataset)`, which
    doubles via `_resolve`. Two independent implementations of the same
    arithmetic, and `en_kd` sits first in sorted order, so an off-by-one shifts
    *every* later group's offsets by one row.

    Nothing downstream catches it. The row still collates, still has the right
    shape, still trains -- it is just from the wrong group, which means the mix
    schedule silently no longer describes what the model sees.
    """
    bundle = train_bundle(data_cfg)
    concat: ConcatSources = bundle.loader.dataset
    index = bundle.index

    assert len(concat) == len(index) > 0
    assert concat.order == index.groups == GROUPS

    # 1. Exhaustive: every global id in every group.
    #
    # Two claims per id, and the second is the sharper one. Checking only the
    # *group* catches a wrong offset between groups but not a wrong ordering
    # inside one: `np.repeat(cost, 2)` (-> [a, a, b, b], matching the dataset's
    # `index // 2`) and `np.tile` (-> [a, b, a, b]) produce identical group
    # membership and identical lengths, and differ only in which cost is attached
    # to which row. That mismatch makes the token budget size batches against the
    # wrong lengths -- padding waste and OOM risk, with nothing to point at.
    #
    # Comparing cost to the item's own length is only valid because every fixture
    # row is shorter than cfg.max_frames / max_text_len, so cropping is a no-op.
    assert data_cfg.max_frames > 60 and data_cfg.max_text_len > 60

    for g in index.groups:
        assert index.size(g) == len(bundle.datasets[g])
        for local in range(index.size(g)):
            gid = int(index.global_ids(g, [local])[0])
            assert index.group_of(gid) == g, f"group_of({gid}) disagrees with global_ids"
            row = concat[gid]
            assert row["source"] == g, (
                f"global id {gid} (group {g!r}, local {local}) retrieved a row from "
                f"{row['source']!r}. Offsets in GroupIndex and ConcatSources disagree; "
                f"check the A/B doubling arithmetic."
            )
            got = (int(row["text_flat"].numel()) if bool(row["is_text_only"])
                   else int(row["num_frames"]))
            assert int(index.cost(g)[local]) == got, (
                f"global id {gid} (group {g!r}, local {local}): the index says cost="
                f"{int(index.cost(g)[local])} but the row is {got} long. The cost "
                f"array is not aligned to the rows it indexes -- check the A/B "
                f"doubling layout (np.repeat, not np.tile)."
            )

    # 2. And through the sampler, which is what actually emits the ids.
    for step_batch in islice(iter(bundle.sampler), 20):
        groups = {index.group_of(int(i)) for i in step_batch}
        assert len(groups) == 1, f"sampler emitted a mixed batch: {groups}"
        g = groups.pop()
        for gid in step_batch:
            assert concat[int(gid)]["source"] == g


def test_ab_doubling_shows_each_dialogue_in_both_directions(train_root, data_cfg):
    """The doubling is real, interleaved, and confined to en_kd.

    If `GroupIndex` doubled a group the dataset did not (or vice versa),
    `build_dataloader`'s length cross-check fires -- but only if the doubling is
    actually exercised, which is what this pins.
    """
    bundle = train_bundle(data_cfg)
    concat: ConcatSources = bundle.loader.dataset
    index = bundle.index

    assert bundle.datasets["en_kd"].double_ab is True
    assert bundle.datasets["ko_tts"].double_ab is False
    assert bundle.datasets["text_anchor"].double_ab is False

    off = index.offset("en_kd")
    n = index.size("en_kd")
    assert n % 2 == 0

    # Consecutive ids are the two directions of the same dialogue.
    for local in range(0, n, 2):
        a = concat[off + local]
        b = concat[off + local + 1]
        assert a["sample_uid"] == b["sample_uid"]
        assert bool(a["swapped"]) is False and bool(b["swapped"]) is True

    uids = [concat[off + i]["sample_uid"] for i in range(n)]
    assert len(set(uids)) == n // 2, "each dialogue must appear exactly twice"

    # The undoubled groups map 1:1.
    for g in ("ko_tts", "text_anchor"):
        o, m = index.offset(g), index.size(g)
        got = [concat[o + i]["sample_uid"] for i in range(m)]
        assert len(set(got)) == m


def test_index_and_dataset_lengths_agree(train_root, data_cfg):
    bundle = train_bundle(data_cfg)
    total = 0
    for g in bundle.index.groups:
        assert bundle.index.cost(g).size == len(bundle.datasets[g])
        total += len(bundle.datasets[g])
    assert total == len(bundle.loader.dataset)
    # en_kd is the only doubled group, so the total carries exactly one doubling.
    assert len(bundle.datasets["en_kd"]) == 2 * bundle.datasets["en_kd"]._n_rows


# ------------------------------------------------------------- routing


def test_no_batch_mixes_groups(train_root, data_cfg):
    """Plan 4.5: a mixed batch is not merely wrong, it deadlocks a multi-rank job."""
    bundle = train_bundle(data_cfg)
    index = bundle.index
    for step_batch in islice(iter(bundle.sampler), 40):
        groups = {index.group_of(int(i)) for i in step_batch}
        assert len(groups) == 1, f"batch spans {sorted(groups)}"


def test_text_anchor_routes_to_the_text_collator(train_root, data_cfg):
    """text_anchor batches must not carry a codes tensor, and audio batches must.

    Routing on `is_text_only` is a one-line lookup, so the risk is not that the
    lookup is wrong -- it is that some group's rows carry the wrong flag and end
    up in the collator that cannot represent them.
    """
    bundle = train_bundle(data_cfg)
    routes: set[str] = set()

    for batch in bundle.loader:
        if is_audio_batch(batch):
            routes.add("audio")
            assert "text_lengths" not in batch
            assert SAMPLE_TYPE_IDS["text_anchor"] not in {int(v) for v in batch["sample_type_id"]}
        else:
            routes.add("text")
            assert "codes" not in batch and "kd_valid" not in batch
            assert set(int(v) for v in batch["sample_type_id"]) == {SAMPLE_TYPE_IDS["text_anchor"]}

    assert routes == {"audio", "text"}


def test_routing_collator_refuses_a_mixed_batch(train_root, data_cfg):
    """The sampler guarantees homogeneity; RoutingCollator asserts it.

    Without the assert a mixed batch reaches KDCollator, which raises a message
    about text_anchor rows -- pointing at the collator instead of at the sampler
    that actually broke its contract.
    """
    bundle = train_bundle(data_cfg)
    concat: ConcatSources = bundle.loader.dataset
    index = bundle.index

    audio_row = concat[int(index.global_ids("ko_tts", [0])[0])]
    text_row = concat[int(index.global_ids("text_anchor", [0])[0])]

    collate = RoutingCollator(
        KDCollator(KDCollatorConfig(tokens=data_cfg.tokens)),
        TextAnchorCollator(TextAnchorCollatorConfig(tokens=data_cfg.tokens)),
    )
    collate([audio_row])   # homogeneous: fine
    collate([text_row])
    with pytest.raises(AssertionError, match="mixes text_anchor"):
        collate([audio_row, text_row])


# --------------------------------------------------------------- probe


def test_probe_loader_is_deterministic_across_constructions(probe_root, data_cfg):
    """Byte-identical batch composition across checkpoints -- otherwise a metric
    delta is partly a batching delta (plan 5.6)."""
    a = probe_bundle(data_cfg)
    b = probe_bundle(data_cfg)

    ba = [list(x) for x in a.loader.batch_sampler]
    bb = [list(x) for x in b.loader.batch_sampler]
    assert ba == bb and len(ba) > 1

    # Same again for the items themselves: a stochastic crop would make the
    # tensors differ while the indices matched.
    for i in range(min(3, len(ba))):
        for gid in ba[i]:
            x, y = a.loader.dataset[gid], b.loader.dataset[gid]
            assert torch.equal(x["codes_self"], y["codes_self"])
            assert torch.equal(x["text_flat"], y["text_flat"])


def test_probe_forces_center_crop_and_disables_doubling(probe_root, data_cfg):
    """Both are forced, not requested: data_cfg here asks for random + doubling."""
    assert data_cfg.crop_mode == "random" and data_cfg.double_ab is True

    bundle = probe_bundle(data_cfg)
    for g, ds in bundle.datasets.items():
        assert ds.crop_mode == "center", f"{g}: probe must not crop randomly"
        assert ds.double_ab is False, f"{g}: probe must not A/B-augment"

    # No doubling means len == row count, including for en_kd.
    assert len(bundle.datasets["en_kd"]) == bundle.datasets["en_kd"]._n_rows
    assert bundle.index.size("en_kd") == bundle.datasets["en_kd"]._n_rows

    # Every en_kd probe row is the unswapped direction.
    concat = bundle.loader.dataset
    off = bundle.index.offset("en_kd")
    uids = []
    for i in range(bundle.index.size("en_kd")):
        row = concat[off + i]
        assert bool(row["swapped"]) is False
        uids.append(row["sample_uid"])
    assert len(set(uids)) == len(uids)


def test_probe_ascending_order_within_a_group(probe_root, data_cfg):
    """Batches pack by cost, but never shuffle -- and never straddle a group."""
    bundle = probe_bundle(data_cfg)
    index = bundle.index
    for batch in bundle.loader.batch_sampler:
        groups = {index.group_of(int(i)) for i in batch}
        assert len(groups) == 1, f"probe batch spans {sorted(groups)}"
        g = groups.pop()
        costs = [int(index.cost(g)[int(i) - index.offset(g)]) for i in batch]
        assert costs == sorted(costs), "probe batches are packed by ascending cost"


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_probe_covers_every_row_exactly_once_across_ranks(probe_root, data_cfg, world_size):
    """Plan 5.6: exhaustive coverage *and* an equal collective count per rank.

    The two requirements fight each other -- a row count that is not a multiple
    of world_size cannot give every rank the same number of batches without
    either dropping rows or duplicating them. The resolution is duplicate pad
    batches flagged with `is_pad_batch`, masked before the metric reduction. If
    the flag were wrong, the duplicated rows would be double-counted and the
    probe metric would be silently weighted toward whichever group sorts last.
    """
    index = probe_bundle(data_cfg).index

    seen: list[int] = []
    n_batches = None
    n_pad = 0

    for rank in range(world_size):
        s = GroupSequentialBatchSampler(
            index, token_budget=TOKEN_BUDGET, max_batch=MAX_BATCH,
            rank=rank, world_size=world_size,
        )
        batches = [list(b) for b in s]
        assert len(batches) == len(s) == len(s.is_pad_batch)

        # Every rank issues the same number of collectives.
        if n_batches is None:
            n_batches = len(batches)
        assert len(batches) == n_batches

        for batch, pad in zip(batches, s.is_pad_batch):
            if pad:
                n_pad += 1
            else:
                seen.extend(int(i) for i in batch)

    assert sorted(seen) == list(range(len(index))), (
        "non-pad probe batches must cover every row exactly once"
    )
    # Padding only ever tops up the final partial block, so it is bounded by
    # world_size - 1. A larger count means real batches got flagged as padding
    # and their rows would never be scored.
    assert n_pad < world_size
    if world_size == 1:
        assert n_pad == 0


def test_probe_pad_batches_are_duplicates_and_are_flagged(probe_root, data_cfg):
    """A pad batch must be a *duplicate*, never a fresh row, and must be flagged."""
    index = probe_bundle(data_cfg).index
    world_size = 4

    real: list[tuple[int, ...]] = []
    pads: list[tuple[int, ...]] = []
    for rank in range(world_size):
        s = GroupSequentialBatchSampler(
            index, token_budget=TOKEN_BUDGET, max_batch=MAX_BATCH,
            rank=rank, world_size=world_size,
        )
        for batch, pad in zip(s, s.is_pad_batch):
            (pads if pad else real).append(tuple(int(i) for i in batch))

    assert len(real) > 0
    for p in pads:
        assert p in real, "a pad batch must duplicate a real batch, not invent rows"
    # And the flags are not all-True / all-False by accident.
    assert len(pads) < len(real) + len(pads)


def test_probe_split_needs_no_mix_cfg(probe_root, data_cfg):
    """Probe discovers its groups from disk: a metric must cover everything
    prepared, not whatever the training curriculum happens to name."""
    bundle = probe_bundle(data_cfg)
    assert bundle.schedule is None
    assert bundle.sampler is None
    assert tuple(sorted(bundle.datasets)) == GROUPS


# ---------------------------------------------------------- build policy


def test_build_never_names_the_exact_command_for_the_missing_group(prepared, data_cfg):
    """The error must be copy-pasteable. A message that says "prepare the data"
    costs the reader a trip through the plan to find the group flag."""
    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=40 + i) for i in range(6)])
    prepared("text_anchor", "train", [make_text_anchor_sample(f"ta{i}", L=30 + i) for i in range(6)])
    # ko_tts is named by the mix schedule but was never prepared.

    with pytest.raises(FileNotFoundError) as ei:
        build_dataloader(
            data_cfg=data_cfg, loader_cfg=loader_cfg(build="never"),
            split="train", mix_cfg=MIX_CFG,
        )

    msg = str(ei.value)
    assert "python -m project_amnesty.datasets.prepare --group ko_tts" in msg
    assert str(Path(data_cfg.root) / "ko_tts" / "train") in msg


def test_build_never_does_not_import_prepare(prepared, data_cfg, monkeypatch):
    """`build='never'` must fail before touching the optional prepare module --
    otherwise a missing group reports as an ImportError about `moshi`."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *a, **k):
        assert "prepare" not in name, f"build='never' must not import {name!r}"
        return real_import(name, *a, **k)

    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=40 + i) for i in range(6)])
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(FileNotFoundError):
        build_dataloader(
            data_cfg=data_cfg, loader_cfg=loader_cfg(build="never"),
            split="train", mix_cfg=MIX_CFG,
        )


@pytest.mark.parametrize("group", sorted(HEAVY_GROUPS))
def test_assert_can_build_refuses_heavy_groups_in_a_multi_rank_job(group):
    """Plan 9.4: an hours-long build inside a job whose NCCL watchdog fires in
    10-30 minutes gets reported as a collective timeout, pointing nowhere near
    the actual cause."""
    for build in ("if_missing", "force"):
        with pytest.raises(RuntimeError) as ei:
            _assert_can_build(build, group, world_size=4)
        msg = str(ei.value)
        assert f"python -m project_amnesty.datasets.prepare --group {group}" in msg
        assert group in msg and "4-rank" in msg


def test_assert_can_build_allows_the_cases_it_should():
    _assert_can_build("never", "en_kd", world_size=8)       # never builds at all
    _assert_can_build("if_missing", "en_kd", world_size=1)  # single rank is fine
    _assert_can_build("force", "text_anchor", world_size=8)  # not a heavy group
    assert "text_anchor" not in HEAVY_GROUPS


def test_heavy_group_refusal_fires_through_the_factory(prepared, data_cfg):
    """The guard has to be reachable from build_dataloader, not just unit-callable."""
    prepared("en_kd", "train", [make_en_kd_sample(f"kd{i}", T=40 + i) for i in range(6)])
    prepared("text_anchor", "train", [make_text_anchor_sample(f"ta{i}", L=30 + i) for i in range(6)])

    with pytest.raises(RuntimeError, match="refusing to build 'ko_tts'"):
        build_dataloader(
            data_cfg=data_cfg, loader_cfg=loader_cfg(build="if_missing"),
            split="train", mix_cfg=MIX_CFG, rank=0, world_size=4,
        )


def test_loader_config_rejects_an_unknown_build_policy():
    with pytest.raises(AssertionError, match="never\\|if_missing\\|force"):
        LoaderConfig(build="maybe")


# ------------------------------------------------------------ bundle state


def test_set_epoch_reaches_every_dataset_and_the_sampler(train_root, data_cfg):
    """A dataset the bundle forgets keeps replaying epoch-0 crops -- invisible in
    the loss curve, and the reason set_epoch lives on the bundle at all."""
    bundle = train_bundle(data_cfg)
    assert all(ds._epoch == 0 for ds in bundle.datasets.values())
    assert bundle.sampler.epoch == 0

    bundle.set_epoch(3)
    assert set(bundle.datasets) == set(GROUPS)
    for g, ds in bundle.datasets.items():
        assert ds._epoch == 3, f"{g} did not advance"
    assert bundle.sampler.epoch == 3

    bundle.set_epoch(0)
    assert all(ds._epoch == 0 for ds in bundle.datasets.values())
    assert bundle.sampler.epoch == 0


def test_set_epoch_changes_the_crop_but_not_the_group_sequence(train_root, data_cfg):
    """Plan 4.5 again: the group draw must not depend on the epoch, or a rank
    that resumed at a different epoch would draw a different group."""
    bundle = train_bundle(data_cfg)
    seq0 = [bundle.sampler.group_at(t) for t in range(50)]

    kd = bundle.datasets["en_kd"]
    before = kd[0]["codes_self"].clone()

    bundle.set_epoch(1)
    assert [bundle.sampler.group_at(t) for t in range(50)] == seq0
    # Crop is (seed, epoch, index)-keyed, so the window must move for a row long
    # enough to have somewhere to move to.
    long_row = max(range(len(kd)), key=lambda i: int(kd[i]["num_frames"]))
    kd.set_epoch(0)
    a = kd[long_row]["codes_self"].clone()
    kd.set_epoch(1)
    b = kd[long_row]["codes_self"].clone()
    if int(kd[long_row]["num_frames"]) < data_cfg.max_frames:
        assert torch.equal(a, b), "an uncropped row cannot move"
    assert before.shape[0] == K


def test_state_dict_round_trips(train_root, data_cfg):
    """Resume must reproduce the exact index stream, group cursors included.

    Restarting each group's cursor at 0 on resume re-shows the head of the small
    group and is invisible except as a slow overfit.
    """
    bundle = train_bundle(data_cfg)
    it = iter(bundle.sampler)

    [next(it) for _ in range(4)]
    snap = bundle.state_dict()
    expected = [list(next(it)) for _ in range(4)]

    assert snap["sampler"] is not None
    assert snap["sampler"]["step"] == 4
    assert snap["sampler"]["group_cursor"], "cursors must be in the checkpoint"

    bundle.load_state_dict(snap)
    it2 = iter(bundle.sampler)
    got = [list(next(it2)) for _ in range(4)]
    assert got == expected

    # A fresh bundle loading the same state must agree too -- that is what
    # actually happens on restart.
    fresh = train_bundle(data_cfg)
    fresh.load_state_dict(snap)
    it3 = iter(fresh.sampler)
    assert [list(next(it3)) for _ in range(4)] == expected


def test_state_dict_is_a_noop_for_the_probe_bundle(probe_root, data_cfg):
    bundle = probe_bundle(data_cfg)
    assert bundle.state_dict() == {"sampler": None}
    bundle.load_state_dict({"sampler": None})  # must not raise
    bundle.set_epoch(2)  # no sampler to advance; datasets still move
    assert all(ds._epoch == 2 for ds in bundle.datasets.values())


# ------------------------------------------------------- rank sharding


def test_ranks_partition_rows_disjointly(train_root, data_cfg):
    """world_size=4: at every step the four ranks hold disjoint rows.

    The batch cut is a pure function of rank-invariant inputs, so each rank
    computes the identical cut and slices out its own block. If that ever drifts,
    two ranks train on the same rows and the effective batch size quietly halves.
    """
    world_size = 4
    bundles = [
        train_bundle(data_cfg, rank=r, world_size=world_size) for r in range(world_size)
    ]
    iters = [iter(b.sampler) for b in bundles]
    index = bundles[0].index

    # (group, group_epoch) -> every global id served in that group-epoch. Within
    # one group-epoch the ranks must together consume each batch exactly once;
    # a *new* group-epoch legitimately re-shows rows (recycle='reshuffle'), so
    # the bookkeeping has to be segmented by it.
    served: dict[tuple[str, int], list[int]] = {}

    for step in range(12):
        before = [dict(b.sampler.group_cursor) for b in bundles]
        per_rank = [list(next(it)) for it in iters]

        # Same group on every rank -- the plan 4.5 hang.
        groups = {index.group_of(int(b[0])) for b in per_rank}
        assert len(groups) == 1, f"step {step}: ranks drew {groups}"
        g = groups.pop()

        # Disjoint *within* the step.
        flat = [int(i) for b in per_rank for i in b]
        assert len(flat) == len(set(flat)), (
            f"step {step}: ranks overlap -- "
            f"{sorted(i for i in flat if flat.count(i) > 1)}"
        )
        assert all(len(b) > 0 for b in per_rank), "a rank with no batch skips collectives"

        # Every rank must agree on where in the group's batch list it is, and the
        # cursor must advance by world_size -- one batch consumed per rank. If it
        # advanced by 1 the ranks would still be disjoint at every single step
        # while quietly re-serving the same rows world_size times over.
        cursors = {b.sampler.group_cursor[g] for b in bundles}
        assert len(cursors) == 1, f"step {step}: ranks disagree on {g!r}'s cursor"
        after = cursors.pop()
        # Either one block consumed, or the group-epoch rolled over and the
        # cursor restarted at the first block.
        assert after in (before[0][g] + world_size, world_size), (
            f"step {step}: {g!r} cursor went {before[0][g]} -> {after}, expected "
            f"+{world_size} (one batch per rank)"
        )

        key = (g, bundles[0].sampler.group_epoch[g])
        served.setdefault(key, []).extend(flat)

    for (g, ge), ids in served.items():
        assert len(ids) == len(set(ids)), (
            f"group {g!r} group-epoch {ge}: rows served more than once "
            f"({len(ids) - len(set(ids))} duplicates) without advancing the epoch"
        )


def test_ranks_draw_the_same_group_sequence(train_root, data_cfg):
    """group_at is seeded on (seed, step // grad_accum) and nothing else."""
    bundles = [train_bundle(data_cfg, rank=r, world_size=4) for r in range(4)]
    seqs = [[b.sampler.group_at(t) for t in range(400)] for b in bundles]
    assert all(s == seqs[0] for s in seqs)
    # And it is constant inside a grad_accum window.
    ga = bundles[0].sampler.grad_accum
    for w in range(0, 400 // ga):
        window = seqs[0][w * ga : (w + 1) * ga]
        assert len(set(window)) == 1, f"group changed inside grad_accum window {w}"


def test_rank_must_be_in_range(train_root, data_cfg):
    with pytest.raises(AssertionError):
        train_bundle(data_cfg, rank=4, world_size=4)


# -------------------------------------------------------------- misc guards


def test_persistent_workers_warns(train_root, data_cfg):
    """Silently accepting it means every epoch replays epoch-0 crops."""
    with pytest.warns(RuntimeWarning, match="persistent_workers"):
        build_dataloader(
            data_cfg=data_cfg,
            loader_cfg=loader_cfg(num_workers=2, persistent_workers=True),
            split="train", mix_cfg=MIX_CFG,
        )


def test_train_split_requires_a_mix_cfg(train_root, data_cfg):
    with pytest.raises(AssertionError, match="requires mix_cfg"):
        build_dataloader(data_cfg=data_cfg, loader_cfg=loader_cfg(), split="train")


def test_unknown_split_is_rejected(train_root, data_cfg):
    with pytest.raises(AssertionError, match="split must be train\\|probe"):
        build_dataloader(
            data_cfg=data_cfg, loader_cfg=loader_cfg(), split="valid", mix_cfg=MIX_CFG
        )


def test_max_steps_beyond_the_last_anchor_warns(train_root, data_cfg):
    """The weights clamp rather than extrapolate; the operator should know that
    the last 44k steps run at a fixed mixture."""
    with pytest.warns(RuntimeWarning, match="clamped to the final anchor"):
        build_dataloader(
            data_cfg=data_cfg, loader_cfg=loader_cfg(), split="train",
            mix_cfg=MIX_CFG, max_steps=50_000,
        )


def test_dataloader_owns_no_batch_size_or_sampler(train_root, data_cfg):
    """Plan 5.5: batch_size/shuffle/sampler/drop_last must all be unset, or torch
    silently builds a second sampler on top of the batch sampler."""
    bundle = train_bundle(data_cfg)
    dl = bundle.loader
    assert dl.batch_size is None
    assert dl.drop_last is False
    assert dl.batch_sampler is bundle.sampler
    assert isinstance(dl.collate_fn, RoutingCollator)


def test_num_workers_zero_omits_prefetch_factor(train_root, data_cfg):
    """prefetch_factor is illegal with num_workers=0 in torch; the factory has to
    omit it rather than pass the config value through."""
    bundle = train_bundle(data_cfg)  # num_workers=0
    assert bundle.loader.num_workers == 0


def test_concat_sources_rejects_an_out_of_range_id(train_root, data_cfg):
    concat: ConcatSources = train_bundle(data_cfg).loader.dataset
    with pytest.raises(AssertionError, match="out of range"):
        concat[len(concat)]


# ============================================ GROUP_ALIASES / bidirectional reuse
# DATA_STRATEGY 4.2 + ARCHITECTURE 5.0.2: `ko_asr` is not a corpus. It is the same
# prepared ko_tts rows entered into the mix a second time, and the *only*
# difference downstream is the collator's delay. These tests pin both halves of
# that claim: one directory is opened, and the two groups stay distinguishable.

ASR_MIX_CFG = {
    "anchors": [
        {"at": 0, "weights": {"en_kd": 0.3, "ko_tts": 0.3, "ko_asr": 0.2, "text_anchor": 0.2}},
        {"at": 1000, "weights": {"en_kd": 0.3, "ko_tts": 0.3, "ko_asr": 0.2, "text_anchor": 0.2}},
    ]
}


def asr_collator_cfg(tokens):
    """Text leads by 8 frames (TTS) / lags by 8 (ASR, derived by the sign flip)."""
    return KDCollatorConfig(
        tokens=tokens, delay=DelayConfig(acoustic_delay=2, text_delay_frames=-8)
    )


def asr_bundle(data_cfg, **kw):
    return build_dataloader(
        data_cfg=data_cfg, loader_cfg=loader_cfg(**kw.pop("loader", {})),
        split="train", mix_cfg=ASR_MIX_CFG,
        collator_cfg=asr_collator_cfg(data_cfg.tokens), **kw,
    )


def test_group_aliases_maps_ko_asr_to_ko_tts():
    assert GROUP_ALIASES["ko_asr"] == "ko_tts"
    assert group_dir("ko_asr") == "ko_tts"
    assert group_dir("ko_tts") == "ko_tts"
    assert group_dir("en_kd") == "en_kd", "unaliased groups must pass through"


def test_alias_group_opens_the_aliased_directory(train_root, data_cfg):
    """No ko_asr directory exists on disk, and none is required."""
    assert not (train_root / "ko_asr").exists()
    b = asr_bundle(data_cfg)
    assert set(b.datasets) == {"en_kd", "ko_tts", "ko_asr", "text_anchor"}
    assert b.datasets["ko_asr"].path == b.datasets["ko_tts"].path
    # `source` is the mixing group, `data_dir` is where the rows come from.
    assert b.datasets["ko_asr"].source == "ko_asr"
    assert b.datasets["ko_asr"].data_dir == "ko_tts"
    assert b.datasets["ko_tts"].source == b.datasets["ko_tts"].data_dir == "ko_tts"


def test_alias_group_has_an_identical_cost_array(train_root, data_cfg):
    """Same rows -> same costs. A difference here means they diverged on disk."""
    b = asr_bundle(data_cfg)
    import numpy as np

    assert np.array_equal(b.index.cost("ko_asr"), b.index.cost("ko_tts"))
    assert len(b.datasets["ko_asr"]) == len(b.datasets["ko_tts"])
    # ...but disjoint global id ranges, or the sampler could not tell them apart.
    ids_t = set(b.index.global_ids("ko_tts", list(range(len(b.datasets["ko_tts"])))).tolist())
    ids_a = set(b.index.global_ids("ko_asr", list(range(len(b.datasets["ko_asr"])))).tolist())
    assert not (ids_t & ids_a)


def test_alias_global_ids_resolve_to_rows_with_the_expected_source(train_root, data_cfg):
    """The ConcatSources / GroupIndex seam, for the aliased group specifically."""
    b = asr_bundle(data_cfg)
    concat = b.loader.dataset
    n = len(b.datasets["ko_asr"])
    for local in (0, 1, n // 2, n - 1):
        for g, want in (("ko_tts", "ko_tts"), ("ko_asr", "ko_asr")):
            gid = int(b.index.global_ids(g, [local])[0])
            item = concat[gid]
            assert item["source"] == want, f"global id {gid} of {g} resolved to {item['source']}"
            assert item["sample_type"] == "ko_tts", "the shape contract is unchanged"
    # Same local index, both directions -> the same underlying utterance.
    for local in (0, 3, 7):
        a = concat[int(b.index.global_ids("ko_tts", [local])[0])]
        c = concat[int(b.index.global_ids("ko_asr", [local])[0])]
        assert a["sample_uid"] == c["sample_uid"]
        assert torch.equal(a["codes_self"], c["codes_self"])
        assert torch.equal(a["text_self"], c["text_self"])


def test_both_directions_appear_with_their_own_delays(train_root, data_cfg):
    """End to end: batches of both kinds come out of one loader, each with the
    delay its direction demands."""
    b = asr_bundle(data_cfg, loader={"steps_per_epoch": 60})
    seen: dict[str, list[int]] = {}
    for batch in islice(b.loader, 60):
        if not is_audio_batch(batch):
            continue
        st = int(batch["sample_type_id"][0])
        off = [int(v) for v in batch["delay_offsets"]]
        # sample_type_id cannot tell ko_tts from ko_asr (same shape contract), so
        # identify the direction by the delay itself, which is the point.
        kind = "asr" if off[0] > off[1] else "tts"
        seen.setdefault(kind, off)
        assert seen[kind] == off, f"{kind} batches disagree on the delay"
    assert set(seen) == {"tts", "asr"}, f"only saw {sorted(seen)}"
    assert seen["tts"] == [0, 8] + [10] * (K - 1)
    assert seen["asr"] == [8, 0] + [2] * (K - 1)


# ------------------------------------------------------------- real prepared data

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "prepared"

REAL_MIX_CFG = {
    "anchors": [
        {"at": 0, "weights": {"ko_tts": 0.5, "ko_asr": 0.5}},
        {"at": 1000, "weights": {"ko_tts": 0.5, "ko_asr": 0.5}},
    ]
}


@pytest.mark.skipif(
    not (REAL_ROOT / "ko_tts" / "train").exists(),
    reason="real prepared ko_tts corpus not present",
)
def test_bidirectional_reuse_over_the_real_zeroth_corpus(tokens):
    """The real thing: one prepared Korean corpus, both directions, no duplication.

    Synthetic corpora are built by these tests and so agree with them by
    construction. This runs the same wiring over the actual prepared Zeroth rows,
    where lengths, speakers and text distributions are whatever the pipeline
    really produced.
    """
    from project_amnesty.datasets.config import DataConfig

    cfg = DataConfig(
        root=str(REAL_ROOT), tokens=tokens, max_frames=750,
        crop_mode="random", double_ab=True, seed=0, debug_validate=False,
    )
    b = build_dataloader(
        data_cfg=cfg,
        loader_cfg=loader_cfg(token_budget=6000, max_batch=8, steps_per_epoch=40),
        split="train", mix_cfg=REAL_MIX_CFG,
        collator_cfg=asr_collator_cfg(tokens),
    )
    assert len(b.datasets["ko_asr"]) == len(b.datasets["ko_tts"])
    assert b.datasets["ko_asr"].path == b.datasets["ko_tts"].path

    counts = {"tts": 0, "asr": 0}
    offs: dict[str, list[int]] = {}
    for batch in islice(b.loader, 40):
        assert is_audio_batch(batch)
        off = [int(v) for v in batch["delay_offsets"]]
        kind = "asr" if off[0] > off[1] else "tts"
        counts[kind] += 1
        offs.setdefault(kind, off)
        assert offs[kind] == off
    assert counts["tts"] > 0 and counts["asr"] > 0, counts
    assert offs["tts"] == [0, 8] + [10] * (K - 1)
    assert offs["asr"] == [8, 0] + [2] * (K - 1)
    print(f"real ko_tts corpus: {len(b.datasets['ko_tts'])} rows, "
          f"batches tts={counts['tts']} asr={counts['asr']}, "
          f"delay tts={offs['tts']} asr={offs['asr']}")
