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

from .datasets import BaseDataset, build_dataset
from .schema import arrow_features


def is_holdout(uid: str, ratio: float) -> bool:
    h = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < ratio * 10_000


def write(rows: Iterator[dict], out_root: Path, source: str,
          holdout_ratio: float) -> None:
    buf = {"train": [], "probe": []}
    for row in rows:
        split = "probe" if is_holdout(row["sample_uid"], holdout_ratio) else "train"
        buf[split].append(row)

    feats = arrow_features()
    for split, rows_ in buf.items():
        if not rows_:
            continue
        ds = Dataset.from_list(rows_, features=feats)
        path = out_root / source / split
        ds.save_to_disk(str(path))
        print(f"[prepare] {source}/{split}: {len(ds)} samples → {path}")


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_root"])
    hr = cfg.get("holdout_ratio", 0.02)

    # collect dataset instances per source group
    groups: dict[str, list[BaseDataset]] = {}
    for name, kwargs in (cfg.get("datasets") or {}).items():
        ds = build_dataset(name, **(kwargs or {}))
        assert ds.source, f"'{name}' is not a training source (empty source attribute)"
        groups.setdefault(ds.source, []).append(ds)

    for source, dss in sorted(groups.items()):
        rows = (s.to_row() for s in chain.from_iterable(ds.iter_samples() for ds in dss))
        write(rows, out, source, hr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/prepare.yaml")
    main(p.parse_args().config)
