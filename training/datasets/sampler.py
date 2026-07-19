"""GroupIndex + MixingBatchSampler -- group mixing, token budgeting, rank sharding.

Plan §4. This is a **batch** sampler: it yields `list[int]` and is handed to
`DataLoader(batch_sampler=...)`. A `Sampler[int]` cannot express either of the two
hard requirements -- "this whole batch is one group" (§4.5) and "fill until the
token budget is hit" (§4.4) -- so the batch-level object is not a convenience.

The two axes that must not be confused:

  * **Which group** -- drawn from an RNG seeded on `(seed, step // grad_accum)`
    and *nothing else*. Not the rank, not the epoch. Every rank therefore draws
    the same group at the same step with zero communication. This is the property
    the regression test pins: a `text_anchor` batch never invokes the Depth
    Transformer or `linears.0..7`, so if rank 0 draws `text_anchor` while rank 1
    draws `en_kd`, rank 1 issues an all_gather rank 0 never issues and the job
    **hangs at 100% GPU with no error** until a watchdog kills it 10-30 min later
    pointing at an unrelated line. It is also intermittent -- only steps where the
    multinomial splits -- so a 100-step smoke test passes.
  * **Which rows inside the group** -- sharded by rank. The batch cutting is a
    pure function of (permutation, cost array, budget), all of which are
    rank-invariant, so every rank computes the identical cut and takes its own
    slice. No collective needed.

Per group-epoch the pipeline is: shuffle -> pool of `pool_multiplier*world_size`
batches -> sort by `cost // bucket_width` -> greedy budget cut -> **reshuffle the
batch list** -> deal batch `k` to rank `k % world_size`. The reshuffle matters:
without it a whole grad_accum window can land on the long tail, which is exactly
where OOM risk and gradient-norm outliers coincide, and batch length becomes
correlated with step.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from .schedule import MixSchedule

__all__ = ["GroupIndex", "MixingBatchSampler", "assert_group_sync"]

# Distinct spawn-key namespaces so the group draw, the row permutation and the
# batch-list reshuffle can never alias onto the same stream.
_NS_GROUP = 1
_NS_PERM = 2
_NS_RESHUFFLE = 3

RECYCLE_MODES = ("reshuffle", "with_replacement")


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(key)))


def _gkey(group: str) -> int:
    """Stable per-group integer for spawn keys (PYTHONHASHSEED-independent)."""
    import zlib

    return zlib.crc32(group.encode()) & 0x7FFF_FFFF


# ============================================================== GroupIndex ===
@dataclass(frozen=True)
class GroupIndex:
    """Per-group global id ranges + the `cost` array used for budget batching.

    `cost`, not `num_frames`: `text_anchor` rows have `num_frames == 0` and carry
    an unaligned text sequence instead, so the budget rule needs
    `cost = num_frames if num_frames > 0 else len(text_tokens_a)`. Indexing on
    `num_frames` would give every anchor row cost 0 and pack the whole group into
    one `max_batch` batch.
    """

    groups: tuple[str, ...]
    costs: tuple[np.ndarray, ...]     # int64, aligned to groups
    offsets: tuple[int, ...]          # global id of local index 0, aligned to groups

    # ------------------------------------------------------------- build ----
    @staticmethod
    def from_costs(costs: Mapping[str, Sequence[int]]) -> "GroupIndex":
        groups = tuple(costs.keys())
        arrs, offs, run = [], [], 0
        for g in groups:
            a = np.asarray(costs[g], dtype=np.int64)
            assert a.ndim == 1 and a.size > 0, f"group {g!r} has an empty cost array"
            assert (a > 0).all(), f"group {g!r} has non-positive costs"
            arrs.append(a)
            offs.append(run)
            run += int(a.size)
        return GroupIndex(groups=groups, costs=tuple(arrs), offsets=tuple(offs))

    @staticmethod
    def from_prepared(
        root: str | Path,
        groups: Sequence[str],
        split: str = "train",
        *,
        double_ab: Sequence[str] = (),
        dirs: Mapping[str, str] | None = None,
    ) -> "GroupIndex":
        """Read only the cheap columns out of `data/prepared/{group}/{split}`.

        Never touches `codes_*` or `teacher_*`: those are the dominant tensors and
        materializing them here would read the entire corpus to build an index.
        `num_frames` is a scalar column and the anchor fallback uses
        `list_value_length`, which reads offsets only.

        `dirs` maps a mixing group to the directory it reads. The group name is
        normally the directory name, but an aliased group (loader.GROUP_ALIASES:
        `ko_asr` -> `ko_tts`, bidirectional reuse of one prepared corpus) reads a
        directory it does not own. Both groups then get identical cost arrays,
        which is correct -- they are the same rows -- and the offsets still make
        the two id ranges disjoint.
        """
        import pyarrow.compute as pc
        from datasets import load_from_disk

        dbl = set(double_ab)
        dirs = dict(dirs or {})
        out: dict[str, np.ndarray] = {}
        for g in groups:
            d = dirs.get(g, g)
            path = Path(root) / d / split
            assert path.exists(), (
                f"prepared group missing: {path}\n"
                f"run: python -m training.datasets.prepare --group {d}"
            )
            ds = load_from_disk(str(path))
            tbl = getattr(ds.data, "table", ds.data)
            nf = np.asarray(tbl.column("num_frames").to_numpy(zero_copy_only=False), dtype=np.int64)
            tl = np.asarray(
                pc.list_value_length(tbl.column("text_tokens_a")).to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            cost = np.where(nf > 0, nf, tl)
            assert cost.size > 0, f"prepared group {g!r} is empty at {path}"
            if g in dbl:
                # Dataset-side A/B index doubling: global id i maps to row i // 2.
                cost = np.repeat(cost, 2)
            out[g] = cost
        return GroupIndex.from_costs(out)

    # -------------------------------------------------------------- read ----
    def __len__(self) -> int:
        return sum(int(c.size) for c in self.costs)

    def size(self, group: str) -> int:
        return int(self.costs[self.groups.index(group)].size)

    def cost(self, group: str) -> np.ndarray:
        return self.costs[self.groups.index(group)]

    def offset(self, group: str) -> int:
        return self.offsets[self.groups.index(group)]

    def global_ids(self, group: str, local: np.ndarray | Sequence[int]) -> np.ndarray:
        """Local row indices within `group` -> ConcatDataset global indices."""
        i = self.groups.index(group)
        loc = np.asarray(local, dtype=np.int64)
        assert loc.size == 0 or (loc.min() >= 0 and loc.max() < self.costs[i].size), (
            f"local index out of range for group {group!r}"
        )
        return loc + self.offsets[i]

    def group_of(self, global_id: int) -> str:
        """Inverse of `global_ids` -- which group a global index belongs to."""
        for g, off, c in zip(self.groups, self.offsets, self.costs):
            if off <= global_id < off + c.size:
                return g
        raise IndexError(f"global id {global_id} out of range")


# ================================================= distributed sync helper ===
def assert_group_sync(group: str, step: int, groups: Sequence[str], device=None) -> None:
    """All-gather (group, step) and assert every rank agrees. No-op if not dist.

    Must be called *unconditionally* on every rank that reaches it -- it is itself
    a collective, so hiding it behind `if step < 200` on some ranks only would
    deadlock in the guard meant to prevent a deadlock. Callers gate on the step
    number, which is rank-invariant.

    NOT WIRED YET. There is no trainer in this repo, so nothing calls this in
    production and the tests only cover the not-initialized no-op. Plan section
    4.5 calls rank group agreement the core constraint and this is the only thing
    that detects a violation, so the training loop must call it every step (the
    plan suggests the first 200 steps then every 1000). Until then the guard
    exists but is not guarding: a desync surfaces as 100% GPU utilization with no
    error, and the NCCL watchdog kills the job 10-30 minutes later pointing at an
    unrelated line.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    t = torch.tensor([groups.index(group), int(step)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    for r, g in enumerate(gathered):
        assert torch.equal(g, t), (
            f"rank group desync at step {step}: rank {dist.get_rank()} has "
            f"({group}, {step}), rank {r} has ({groups[int(g[0])]}, {int(g[1])}). "
            f"See plan §4.5 -- this would otherwise hang FSDP2 with no error."
        )


# ======================================================= MixingBatchSampler ==
class MixingBatchSampler(Sampler[list[int]]):
    """step -> group -> token-budget batch of global indices for this rank."""

    def __init__(
        self,
        index: GroupIndex,
        schedule: MixSchedule,
        *,
        steps_per_epoch: int,
        token_budget: int = 6000,
        max_batch: int = 16,
        bucket_width: int = 100,
        grad_accum: int = 4,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        pool_multiplier: int = 64,
        recycle: str = "reshuffle",
        max_repeat_factor: float = 8.0,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        assert 0 <= rank < world_size, f"bad rank/world_size: {rank}/{world_size}"
        assert steps_per_epoch > 0 and token_budget > 0 and max_batch > 0
        assert bucket_width > 0 and grad_accum > 0 and pool_multiplier > 0
        assert recycle in RECYCLE_MODES, f"recycle must be one of {RECYCLE_MODES}"
        missing = [g for g in schedule.groups if g not in index.groups]
        assert not missing, (
            f"mix schedule names groups with no prepared data: {missing}. "
            f"Available: {index.groups}"
        )

        self.index = index
        self.schedule = schedule
        self.steps_per_epoch = int(steps_per_epoch)
        self.token_budget = int(token_budget)
        self.max_batch = int(max_batch)
        self.bucket_width = int(bucket_width)
        self.grad_accum = int(grad_accum)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.pool_multiplier = int(pool_multiplier)
        self.recycle = recycle
        self.max_repeat_factor = float(max_repeat_factor)

        self._epoch = int(epoch)
        self._step = int(step)
        self._groups = tuple(schedule.groups)

        self.group_cursor: dict[str, int] = {g: 0 for g in self._groups}
        self.group_epoch: dict[str, int] = {g: 0 for g in self._groups}
        self._rows_seen: dict[str, int] = {g: 0 for g in self._groups}
        self._step_counts: dict[str, int] = {g: 0 for g in self._groups}
        self._frame_counts: dict[str, int] = {g: 0 for g in self._groups}
        self._warned: set[str] = set()
        self._cache: dict[str, tuple[tuple, list[np.ndarray]]] = {}

    # ------------------------------------------------------------ length ----
    def __len__(self) -> int:
        """Length-defined epoch (§4.2). There is no meaningful joint epoch --
        ko_tts is an order of magnitude larger than en_kd, so 'ConcatDataset
        exhausted' would let corpus size, not the curriculum, set the mixture."""
        return self.steps_per_epoch

    # ------------------------------------------------------ group choice ----
    def window_of(self, step: int) -> int:
        return int(step) // self.grad_accum

    def group_at(self, step: int) -> str:
        """Which group step `step` draws. Pure: no state, no rank, no epoch.

        Seeded on `(seed, step // grad_accum)` only -- see §4.5. Constant across a
        gradient-accumulation window so one optimizer step has a fixed loss
        composition. Inverse-CDF rather than `Generator.choice` so the draw does
        not depend on numpy's internal choice algorithm.
        """
        w = self.window_of(step)
        p = self.schedule.probs_at(w * self.grad_accum)
        u = float(_rng(self.seed, _NS_GROUP, w).random())
        return self._groups[int(np.searchsorted(np.cumsum(p), u, side="right").clip(0, len(p) - 1))]

    # ------------------------------------------------- per-group batching ----
    def _cut(self, order: np.ndarray, cost: np.ndarray) -> list[np.ndarray]:
        """Greedy token-budget cut over an already-ordered row list.

        Accept row i iff `(len(batch)+1) * max(T_max_so_far, T_i) <= budget` and
        `len+1 <= max_batch`. The padded cost `B*T_max`, not `sum(T_i)` -- that is
        what actually pins activation memory. Bucketing keeps the two close.
        """
        out: list[np.ndarray] = []
        cur: list[int] = []
        tmax = 0
        for i in order:
            c = int(cost[i])
            nt = c if c > tmax else tmax
            if cur and ((len(cur) + 1) * nt > self.token_budget or len(cur) + 1 > self.max_batch):
                out.append(np.asarray(cur, dtype=np.int64))
                cur, tmax = [int(i)], c
            else:
                cur.append(int(i))
                tmax = nt
        if cur:
            out.append(np.asarray(cur, dtype=np.int64))
        return out

    def _build(self, group: str, group_epoch: int) -> list[np.ndarray]:
        """All batches for one group-epoch, dealt-ready (rank k takes index k)."""
        cost = self.index.cost(group)
        n = int(cost.size)
        gk = _gkey(group)
        perm_rng = _rng(self.seed, _NS_PERM, gk, self._epoch, group_epoch)

        if self.recycle == "reshuffle":
            perm = perm_rng.permutation(n)
        else:  # with_replacement: exposure becomes a Poisson variable, on purpose
            perm = perm_rng.integers(0, n, size=n)

        # Pool sizing: enough rows to cut ~pool_multiplier*world_size batches, so
        # bucketing has something to sort but length never correlates globally.
        est = max(1, min(self.max_batch, self.token_budget // max(int(cost.mean()), 1)))
        pool_rows = max(int(self.pool_multiplier * self.world_size * est), self.max_batch)

        batches: list[np.ndarray] = []
        for s in range(0, n, pool_rows):
            pool = perm[s : s + pool_rows]
            # Bucket sort, stable so the shuffle still breaks ties inside a bucket.
            keys = cost[pool] // self.bucket_width
            batches.extend(self._cut(pool[np.argsort(keys, kind="stable")], cost))

        # Reshuffle the batch list: otherwise a grad_accum window can sit entirely
        # on the long tail, and batch length becomes a function of step.
        rs = _rng(self.seed, _NS_RESHUFFLE, gk, self._epoch, group_epoch)
        batches = [batches[i] for i in rs.permutation(len(batches))]

        # Emit in whole blocks of world_size; drop the trailing partial block so
        # every rank always has a batch for every step (a rank with no batch would
        # skip the step's collectives -- the §4.5 hang again).
        usable = (len(batches) // self.world_size) * self.world_size
        assert usable > 0, (
            f"group {group!r} yields {len(batches)} batches < world_size="
            f"{self.world_size}; lower token_budget/max_batch or drop the group"
        )
        return batches[:usable]

    def _batches(self, group: str) -> list[np.ndarray]:
        ge = self.group_epoch[group]
        key = (self._epoch, ge)
        hit = self._cache.get(group)
        if hit is None or hit[0] != key:
            self._cache[group] = (key, self._build(group, ge))
        return self._cache[group][1]

    def _next(self, group: str) -> np.ndarray:
        batches = self._batches(group)
        if self.group_cursor[group] + self.world_size > len(batches):
            self.group_epoch[group] += 1
            self.group_cursor[group] = 0
            batches = self._batches(group)
        cur = self.group_cursor[group]
        local = batches[cur + self.rank]
        self.group_cursor[group] = cur + self.world_size
        self._rows_seen[group] += int(sum(len(b) for b in batches[cur : cur + self.world_size]))
        self._check_repeat(group)
        return local

    def _check_repeat(self, group: str) -> None:
        if group in self._warned:
            return
        if self.repeat_factor(group) > self.max_repeat_factor:
            self._warned.add(group)
            # Warn, never raise: this must not kill a job at 3am. But the number
            # has to land in the logs next to the held-out English curve, because
            # that is what separates "English got worse" from "en_kd was memorized".
            warnings.warn(
                f"sampler: group {group!r} passed max_repeat_factor "
                f"({self.repeat_factor(group):.2f} > {self.max_repeat_factor}). "
                f"Read it with sampler/epochs/{group} and the probe-split "
                f"train/probe gap; note that en_solo is re-cropped en_kd, not "
                f"independent data.",
                RuntimeWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------ iterate ----
    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            step = self._step
            g = self.group_at(step)
            local = self._next(g)
            self._step_counts[g] += 1
            cost = self.index.cost(g)
            self._frame_counts[g] += int(len(local) * cost[local].max())
            self._step += 1
            yield self.index.global_ids(g, local).tolist()

    # -------------------------------------------------------------- state ----
    def set_epoch(self, epoch: int) -> None:
        """Reseed the per-group row permutations. Does **not** touch the group
        choice RNG -- if it did, resuming mid-epoch would replay a different
        mixture than the original run. Cursors are preserved for the same reason:
        an epoch boundary is bookkeeping, not a reason to re-show a group's head.
        """
        if int(epoch) != self._epoch:
            self._epoch = int(epoch)
            self._cache.clear()

    def set_step(self, step: int) -> None:
        self._step = int(step)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def repeat_factor(self, group: str) -> float:
        """Mean number of times each row of `group` has been shown so far."""
        return self._rows_seen[group] / max(self.index.size(group), 1)

    def group_epochs(self) -> dict[str, int]:
        return dict(self.group_epoch)

    def repeat_factors(self) -> dict[str, float]:
        return {g: self.repeat_factor(g) for g in self._groups}

    def realized_ratios(self) -> dict[str, dict[str, float]]:
        """Observed occupancy since construction, in both spaces.

        The schedule's weights are *step* ratios. Token-budget batching makes step
        ratio ~ frame ratio, but sample ratio is a different number entirely; both
        are reported so that gap is an observation rather than a surprise.
        """
        def norm(d: dict[str, int]) -> dict[str, float]:
            tot = sum(d.values())
            return {g: (d[g] / tot if tot else 0.0) for g in self._groups}

        return {"steps": norm(self._step_counts), "frames": norm(self._frame_counts)}

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "epoch": self._epoch,
            "seed": self.seed,
            # Without these two, resume silently restarts every group's stream at
            # 0 and re-shows the head of the small group (§4.6).
            "group_cursor": dict(self.group_cursor),
            "group_epoch": dict(self.group_epoch),
            "rows_seen": dict(self._rows_seen),
            "step_counts": dict(self._step_counts),
            "frame_counts": dict(self._frame_counts),
            "warned": sorted(self._warned),
            "recycle": self.recycle,
        }

    def load_state_dict(self, state: Mapping) -> None:
        assert int(state["seed"]) == self.seed, (
            f"checkpoint sampler seed {state['seed']} != {self.seed}; the group "
            f"sequence would not resume"
        )
        assert state.get("recycle", self.recycle) == self.recycle, "recycle mode changed"
        self._step = int(state["step"])
        self._epoch = int(state["epoch"])
        self.group_cursor = {g: int(state["group_cursor"].get(g, 0)) for g in self._groups}
        self.group_epoch = {g: int(state["group_epoch"].get(g, 0)) for g in self._groups}
        self._rows_seen = {g: int(state.get("rows_seen", {}).get(g, 0)) for g in self._groups}
        self._step_counts = {g: int(state.get("step_counts", {}).get(g, 0)) for g in self._groups}
        self._frame_counts = {g: int(state.get("frame_counts", {}).get(g, 0)) for g in self._groups}
        self._warned = set(state.get("warned", ()))
        self._cache.clear()
