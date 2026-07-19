"""All sources → unified-schema Arrow dataset.

Inputs are the ~~Dataset classes in data_pipeline.datasets — for each entry under
the config's `datasets` key an instance is built, its iter_samples() collected,
and the rows stored per the class's source group (en_kd / en_solo / ko_tts /
text_anchor). mixing_sampler draws at this group granularity, so group names must
match the training configs' sources. Multiple datasets in the same group
(e.g. kss + css10_ko → ko_tts) are merged into one.

Output:
  data/prepared/{source}/{split}/   (stored per group — mixing_sampler draws per group)
  split ∈ {train, probe}            probe = CP/A-B probing + held-out (never used for training)

Hold-out is decided by a hash of sample_uid → identical split across re-runs.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import chain
from pathlib import Path
from typing import Iterator

import yaml
from datasets import Dataset

from .datasets import REGISTRY, BaseDataset, build_dataset
from .schema import arrow_features

DEFAULT_HOLDOUT_RATIO = 0.02
SPLITS = ("train", "probe")


def is_holdout(uid: str, ratio: float) -> bool:
    h = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < ratio * 10_000


def write(rows: Iterator[dict], out_root: Path, source: str,
          holdout_ratio: float) -> dict[str, int]:
    """Rows -> {out_root}/{source}/{split}. Returns per-split row counts.

    Splits with no rows are not written at all, and are absent from the returned
    counts -- the caller records exactly what exists on disk.
    """
    buf: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for row in rows:
        split = "probe" if is_holdout(row["sample_uid"], holdout_ratio) else "train"
        buf[split].append(row)

    feats = arrow_features()
    counts: dict[str, int] = {}
    for split, rows_ in buf.items():
        if not rows_:
            continue
        ds = Dataset.from_list(rows_, features=feats)
        path = out_root / source / split
        ds.save_to_disk(str(path))
        counts[split] = len(ds)
        print(f"[prepare] {source}/{split}: {len(ds)} samples → {path}")
    return counts


# ------------------------------------------------------------- group assembly


def builders_for_group(group: str, cfg: dict | None = None) -> list[BaseDataset]:
    """Every registered builder whose `source` is `group`, instantiated.

    N:1 is the constraint that shapes this whole layer: kss + css10_ko +
    zeroth_ko + common_voice_ko all feed ko_tts (plan section 9.1). The mapping is
    derived from REGISTRY rather than written down, so registering a new builder
    with `source = "ko_tts"` is the only step needed to add a corpus.

    Per-builder kwargs come from cfg["datasets"][<name>]; a builder absent from
    that mapping is still built, with its own defaults.
    """
    ds_cfg = (cfg or {}).get("datasets") or {}
    names = sorted(n for n, cls in REGISTRY.items() if cls.source == group)
    if not names:
        raise KeyError(
            f"no registered builder has source={group!r}. "
            f"Known groups: {sorted({c.source for c in REGISTRY.values() if c.source})}"
        )
    return [build_dataset(n, **(ds_cfg.get(n) or {})) for n in names]


def collect_groups(cfg: dict) -> dict[str, list[BaseDataset]]:
    """cfg["datasets"] -> {source group: [builder, ...]}. Config-driven selection."""
    groups: dict[str, list[BaseDataset]] = {}
    for name, kwargs in (cfg.get("datasets") or {}).items():
        ds = build_dataset(name, **(kwargs or {}))
        assert ds.source, f"'{name}' is not a training source (empty source attribute)"
        groups.setdefault(ds.source, []).append(ds)
    return groups


def build_group(group: str, builders: list[BaseDataset], out_root: Path,
                holdout_ratio: float = DEFAULT_HOLDOUT_RATIO) -> dict[str, int]:
    """Builders -> iter_samples() -> unified Arrow under {out_root}/{group}/.

    Note there is no sentinel here: this writes, nothing more. Idempotency and
    the concurrency gate live one level up in training/data/prepare.py, so that
    the plain config-driven CLI and the ensure-prepared path share exactly this
    body and cannot drift.
    """
    rows = (s.to_row() for s in chain.from_iterable(b.iter_samples() for b in builders))
    return write(rows, Path(out_root), group, holdout_ratio)


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_root"])
    hr = cfg.get("holdout_ratio", DEFAULT_HOLDOUT_RATIO)

    for source, dss in sorted(collect_groups(cfg).items()):
        build_group(source, dss, out, hr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/prepare.yaml")
    main(p.parse_args().config)
