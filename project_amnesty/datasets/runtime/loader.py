"""Wire prepared Arrow -> mixed dataset -> sampler -> collator -> DataLoader.

Everything below fails loudly rather than degrading: a missing group, a
group/index length mismatch, or an unset token id stops the run here instead of
producing a batch that trains to a quietly worse model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from .collator import KDCollator, KDCollatorConfig
from .config import TOKENS_YAML, DataConfig, TokenConfig, _load_yaml
from .dataset import MoshiKDDataset
from .item import KDSample
from .sampler import GroupIndex, MixingBatchSampler
from .schedule import MixSchedule
from .text_collator import TextAnchorCollator, TextAnchorCollatorConfig

# Groups whose preparation runs Mimi/Moshi on the GPU. Building these inside a
# multi-rank training job is refused outright -- see _assert_can_build.
HEAVY_GROUPS = frozenset({"en_kd", "en_solo", "ko_tts"})

# Sources that carry a B stream and therefore get A/B index doubling. Must match
# what MoshiKDDataset decides, or global ids point at the wrong rows.
DOUBLED_SOURCES = ("en_kd",)

# Mixing group -> prepared directory, for groups that are a *second reading* of
# another group's rows rather than a corpus of their own.
#
# `ko_asr` is DATA_STRATEGY section 4.2's bidirectional reuse: isolated-utterance
# TTS data teaches only text -> audio, so the listening circuit never sees Korean.
# The fix is to use the same (text, audio) pair twice -- once with text leading
# (TTS) and once with text lagging (ASR). ARCHITECTURE section 5.0.2 is what makes
# this a one-line alias instead of a second pipeline: "a single delay
# hyper-parameter allows for switching from an ASR to a TTS model with no changes
# in the loss, architecture, or training data". So ko_asr is NOT a copy on disk
# and NOT a variant of the rows -- it is literally `data/prepared/ko_tts/{split}`
# read again, with only the collator's delay differing. Being the same rows, the
# deterministic is_holdout split already applies unchanged.
#
# It is a separate mixing group (rather than a per-batch tag) so the ASR ratio is
# expressible in the existing MixSchedule YAML instead of needing a second mixing
# axis alongside it.
GROUP_ALIASES: dict[str, str] = {"ko_asr": "ko_tts"}


def group_dir(group: str) -> str:
    """Prepared directory name for a mixing group."""
    return GROUP_ALIASES.get(group, group)


# --------------------------------------------------------------------- config


@dataclass
class LoaderConfig:
    """Sampler + DataLoader knobs. Dataset/collator configs are separate."""

    # sampler
    token_budget: int = 6000
    max_batch: int = 16
    bucket_width: int = 100
    pool_multiplier: int = 64
    steps_per_epoch: int = 2000
    grad_accum: int = 4
    recycle: str = "reshuffle"
    max_repeat_factor: float = 8.0
    seed: int = 1234

    # dataloader
    num_workers: int = 6
    prefetch_factor: int = 2
    pin_memory: bool = True
    # False by default and deliberately: with persistent workers the dataset's
    # _epoch is frozen at fork time, so every epoch would replay epoch-0 crops --
    # invisible in the loss curve.
    persistent_workers: bool = False

    # build policy: never | if_missing | force
    build: str = "never"

    def __post_init__(self) -> None:
        assert self.build in ("never", "if_missing", "force"), (
            f"build must be never|if_missing|force, got {self.build!r}"
        )
        assert self.num_workers >= 0 and self.token_budget > 0


@dataclass
class LoaderBundle:
    """Everything the trainer needs to keep in sync across epochs and resumes."""

    loader: DataLoader
    sampler: MixingBatchSampler | None
    datasets: dict[str, MoshiKDDataset]
    index: GroupIndex
    schedule: MixSchedule | None

    def set_epoch(self, epoch: int) -> None:
        """Advance every RNG stream. Call BEFORE iter(loader) each epoch."""
        for ds in self.datasets.values():
            ds.set_epoch(epoch)
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def state_dict(self) -> dict:
        return {"sampler": self.sampler.state_dict() if self.sampler else None}

    def load_state_dict(self, state: dict) -> None:
        if self.sampler is not None and state.get("sampler") is not None:
            self.sampler.load_state_dict(state["sampler"])


# ------------------------------------------------------------------- dataset


class ConcatSources(Dataset):
    """Global index -> (group, local index), laid out to match GroupIndex.

    Deliberately not torch's ConcatDataset: the offsets must agree with
    GroupIndex exactly, so both are derived from one ordering and cross-checked
    in build_dataloader.
    """

    def __init__(self, datasets: dict[str, MoshiKDDataset], order: Sequence[str]) -> None:
        self.order = tuple(order)
        self.datasets = datasets
        self.sizes = np.array([len(datasets[g]) for g in self.order], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> KDSample:
        gi = int(np.searchsorted(self.offsets, index, side="right")) - 1
        assert 0 <= gi < len(self.order), f"global index {index} out of range"
        return self.datasets[self.order[gi]][index - int(self.offsets[gi])]


# ------------------------------------------------------------------ collator


class RoutingCollator:
    """Dispatch a homogeneous batch to the audio or the text-anchor collator.

    The sampler guarantees a batch never mixes groups, so routing is a lookup,
    not a merge. The assert is what turns a broken guarantee into a crash rather
    than a batch where half the rows are all-padding.
    """

    def __init__(self, kd: KDCollator, text: TextAnchorCollator) -> None:
        self.kd = kd
        self.text = text

    def __call__(self, rows: list[KDSample]) -> dict:
        flags = {bool(r["is_text_only"]) for r in rows}
        assert len(flags) == 1, (
            "batch mixes text_anchor with frame samples; the sampler must emit "
            "homogeneous batches (plan section 5.4)"
        )
        text_only = flags.pop()
        batch = self.text(rows) if text_only else self.kd(rows)
        # The two paths share key names whose meaning differs -- `text_tokens` is
        # frame-aligned (B, T_frames) in an audio batch and unaligned
        # (B, L_tokens) here. Branching on a shared key would silently do the
        # wrong thing on one path, so state the kind explicitly.
        batch["batch_kind"] = "text_anchor" if text_only else "audio"
        return batch


# ------------------------------------------------------------- probe sampler


class GroupSequentialBatchSampler(Sampler[list[int]]):
    """Deterministic, exhaustive, per-group batches for evaluation.

    Differences from the training sampler, each for a reason (plan section 5.6):
    no mixing (metrics must be per group), ascending order (batch composition
    identical across checkpoints, so a metric delta is a model delta), and
    rank blocks padded with duplicates rather than dropped -- every probe row is
    evaluated exactly once while each rank still issues the same number of
    collectives.
    """

    def __init__(
        self,
        index: GroupIndex,
        *,
        token_budget: int = 6000,
        max_batch: int = 16,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        assert 0 <= rank < world_size
        self.rank, self.world_size = int(rank), int(world_size)
        self._batches: list[tuple[list[int], bool]] = []

        for g in index.groups:
            cost = index.cost(g)
            order = np.argsort(cost, kind="stable")  # pack by length; no shuffle
            cur: list[int] = []
            cur_max = 0
            for local in order:
                c = int(cost[local])
                nmax = max(cur_max, c)
                if cur and ((len(cur) + 1) * nmax > token_budget or len(cur) + 1 > max_batch):
                    self._batches.append((cur, False))
                    cur, cur_max = [], 0
                    nmax = c
                cur.append(int(index.global_ids(g, [int(local)])[0]))
                cur_max = nmax
            if cur:
                self._batches.append((cur, False))

        # Pad to a whole multiple of world_size with duplicates flagged as such.
        while len(self._batches) % self.world_size:
            self._batches.append((list(self._batches[-1][0]), True))

        self.is_pad_batch = [p for _, p in self._batches[self.rank::self.world_size]]

    def __len__(self) -> int:
        return len(self._batches) // self.world_size

    def __iter__(self) -> Iterator[list[int]]:
        for batch, _ in self._batches[self.rank::self.world_size]:
            yield batch


# -------------------------------------------------------------------- factory


def _assert_can_build(build: str, source: str, world_size: int) -> None:
    """Refuse to run a heavy build inside a multi-rank job.

    rank0-builds + barrier is the usual pattern, but en_kd generation runs for
    hours while the NCCL watchdog fires in 10-30 minutes -- the other ranks get
    killed and it is reported as a collective timeout, pointing nowhere near the
    cause.
    """
    if build == "never" or world_size <= 1 or source not in HEAVY_GROUPS:
        return
    raise RuntimeError(
        f"refusing to build {source!r} inside a {world_size}-rank job (plan section 9.4).\n"
        f"    run first: python -m project_amnesty.datasets.prepare --group {source}"
    )


def build_dataloader(
    *,
    data_cfg: DataConfig,
    loader_cfg: LoaderConfig,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    mix_cfg: dict | None = None,
    collator_cfg: KDCollatorConfig | None = None,
    max_steps: int | None = None,
) -> LoaderBundle:
    """Assemble the data stack for one split.

    Returns a bundle rather than a bare DataLoader because the trainer must call
    set_epoch/state_dict on the sampler, which is unreachable through
    DataLoader.batch_sampler once workers are running.
    """
    assert split in ("train", "probe"), f"split must be train|probe, got {split!r}"
    root = Path(data_cfg.root)

    # Every token id the whole stack will need, checked before any I/O. The
    # collator would catch the rest at step 5, but by then each group has been
    # opened twice -- so with today's all-null tokens.yaml every invocation pays
    # full index I/O just to be told the ids are unset.
    data_cfg.tokens.require(
        "text_pad_id", "text_epad_id", "batch_pad_id",
        "audio_init_id", "silence_bank", "mimi_ckpt_id",
    )

    # --- 1. schedule first: it names the groups, and validating it is free ---
    schedule = None
    if split == "train":
        assert mix_cfg is not None, "split='train' requires mix_cfg (the ratio schedule)"
        schedule = MixSchedule.from_cfg(mix_cfg)
        groups = list(schedule.groups)
    else:
        groups = sorted(p.name for p in root.iterdir() if (p / split).exists()) if root.exists() else []
        assert groups, f"no prepared groups under {root} for split={split!r}"

    # --- 2. build policy, before any I/O-shaped work ---
    # Aliased groups share a directory, so build/open it once even when both
    # directions are in the mix.
    dirs = {g: group_dir(g) for g in groups}
    for d in dict.fromkeys(dirs.values()):
        if not (root / d / split).exists():
            _assert_can_build(loader_cfg.build, d, world_size)
            if loader_cfg.build == "never":
                raise FileNotFoundError(
                    f"prepared data missing: {root / d / split}\n"
                    f"    run: python -m project_amnesty.datasets.prepare --group {d}"
                )
            from project_amnesty.datasets.prepare import ensure_prepared  # local import: optional dependency

            ensure_prepared(d, root=root, force=(loader_cfg.build == "force"))

    # --- 3. datasets, in a fixed order ---
    if split == "probe":
        # Probe must never be stochastic: a random crop makes the metric jitter
        # for reasons unrelated to the model. The dataset also hard-forces
        # center cropping for probe, so this is belt and braces.
        data_cfg = _replace_probe(data_cfg)

    datasets = {
        g: MoshiKDDataset(root, g, split, cfg=data_cfg, seed=loader_cfg.seed, data_dir=dirs[g])
        for g in groups
    }
    concat = ConcatSources(datasets, groups)

    # --- 4. index, cross-checked against the datasets ---
    doubled = tuple(g for g in groups if getattr(datasets[g], "double_ab", False))
    index = GroupIndex.from_prepared(root, groups, split, double_ab=doubled, dirs=dirs)
    for g in groups:
        n_idx, n_ds = int(index.cost(g).size), len(datasets[g])
        assert n_idx == n_ds, (
            f"group {g!r}: index has {n_idx} entries but the dataset has {n_ds}. "
            f"Global ids would point at the wrong rows. Check A/B doubling "
            f"(doubled={doubled})."
        )
    assert index.groups == tuple(groups), (
        f"GroupIndex order {index.groups} != dataset order {tuple(groups)}; "
        f"global-id offsets would not line up"
    )

    # --- 5. collators ---
    ccfg = collator_cfg or KDCollatorConfig(tokens=data_cfg.tokens)
    collate = RoutingCollator(
        KDCollator(ccfg),
        TextAnchorCollator(TextAnchorCollatorConfig(
            tokens=data_cfg.tokens, max_text_len=data_cfg.max_text_len
        )),
    )

    # --- 6. sampler ---
    sampler: MixingBatchSampler | None
    if split == "train":
        sampler = MixingBatchSampler(
            index, schedule,
            steps_per_epoch=loader_cfg.steps_per_epoch,
            token_budget=loader_cfg.token_budget,
            max_batch=loader_cfg.max_batch,
            bucket_width=loader_cfg.bucket_width,
            grad_accum=loader_cfg.grad_accum,
            rank=rank, world_size=world_size,
            seed=loader_cfg.seed,
            pool_multiplier=loader_cfg.pool_multiplier,
            recycle=loader_cfg.recycle,
            max_repeat_factor=loader_cfg.max_repeat_factor,
        )
        batch_sampler: Sampler[list[int]] = sampler
        if max_steps is not None:
            _validate_schedule_span(schedule, max_steps)
    else:
        sampler = None
        batch_sampler = GroupSequentialBatchSampler(
            index, token_budget=loader_cfg.token_budget,
            max_batch=loader_cfg.max_batch, rank=rank, world_size=world_size,
        )

    if loader_cfg.persistent_workers and split == "train":
        warnings.warn(
            "persistent_workers=True freezes the dataset's epoch at fork time, so "
            "set_epoch() becomes a no-op and every epoch replays epoch-0 crops. "
            "This is invisible in the loss curve.",
            RuntimeWarning, stacklevel=2,
        )

    kwargs = dict(
        batch_sampler=batch_sampler,
        collate_fn=collate,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    if loader_cfg.num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.prefetch_factor
        kwargs["persistent_workers"] = loader_cfg.persistent_workers

    return LoaderBundle(
        loader=DataLoader(concat, **kwargs),
        sampler=sampler, datasets=datasets, index=index, schedule=schedule,
    )


def _replace_probe(cfg: DataConfig) -> DataConfig:
    """Probe overrides: deterministic crop, no A/B augmentation."""
    from dataclasses import replace

    return replace(cfg, crop_mode="center", double_ab=False)


def _validate_schedule_span(schedule: MixSchedule, max_steps: int) -> None:
    last = schedule.steps[-1] if len(schedule.steps) else 0
    if last < max_steps:
        warnings.warn(
            f"mix schedule's last anchor is at step {last} but max_steps={max_steps}; "
            f"weights are clamped to the final anchor for the remaining "
            f"{max_steps - last} steps.",
            RuntimeWarning, stacklevel=2,
        )


def load_configs(
    loader_yaml: str | Path = "configs/data/loader.yaml",
    tokens_yaml: str | Path = TOKENS_YAML,
) -> tuple[DataConfig, LoaderConfig, dict]:
    """configs/data/loader.yaml -> (DataConfig, LoaderConfig, mix_cfg)."""
    raw = _load_yaml(loader_yaml)
    tokens = TokenConfig(**_load_yaml(tokens_yaml))
    data = DataConfig(root=raw.get("root", "data/prepared"), tokens=tokens,
                      **(raw.get("dataset") or {}))
    loader = LoaderConfig(**{**(raw.get("sampler") or {}), **(raw.get("dataloader") or {})})
    return data, loader, raw.get("mix") or {}
