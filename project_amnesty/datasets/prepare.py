"""Ensure-materialized gate over the offline builders -- and the prepare-only CLI.

    python -m project_amnesty.datasets.prepare --group en_kd

is the command every "prepared data missing" error in dataset.py / sampler.py /
loader.py tells the user to run, so it is the one entry point that has to exist
and work.

The idiom is torchvision's ``MNIST(root, download=True)``: ask for a group, get a
directory that is guaranteed complete. Three things make that guarantee real.

1. **A sentinel, written last and atomically.** Completeness is *not* judged by
   HF's ``dataset_info.json``: an interrupted ``save_to_disk`` leaves that behind
   next to a half-written arrow file, and a build that resumes on it silently
   trains on truncated data. ``_SUCCESS.json`` is written to a temp file and
   ``os.replace``-d into place only after every split has landed.

2. **A schema version inside the sentinel.** ``arrow_features()`` will grow
   ``zone_a_frames`` (plan section 3.7); when ``SCHEMA_VERSION`` bumps, every
   prepared group is stale and rebuilds itself rather than being read back
   through a schema it was not written with.

3. **A file lock, with the sentinel re-checked after acquiring it.** Two ranks or
   two shells hitting the same group must not both spend GPU-hours on it. The
   re-check matters as much as the lock: the process that waited is, by
   definition, waiting on someone who is about to finish.

Group selection is N:1 and derived from ``REGISTRY``, never hardcoded -- see
``project_amnesty.datasets.prepare_dataset.builders_for_group``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Advisory locking is platform-split: fcntl does not exist on Windows. Production
# builds run on POSIX (flock); the msvcrt path exists so the module imports and
# the sentinel logic stays testable on a Windows dev box.
if os.name == "nt":
    import msvcrt
else:
    import fcntl
from contextlib import contextmanager
from pathlib import Path

import yaml

from project_amnesty.datasets import REGISTRY, BaseDataset
from project_amnesty.datasets.prepare_dataset import (
    DEFAULT_HOLDOUT_RATIO,
    build_group,
    builders_for_group,
)
from project_amnesty.datasets.schema import SCHEMA_VERSION

SENTINEL_NAME = "_SUCCESS.json"
LOCK_DIR_NAME = ".locks"

# Builder-level DAG, not a data dependency between prepared groups: en_solo crops
# solo spans out of the en_kd *artifacts* (plan section 9.1). Preparing en_kd is
# what puts those artifacts on disk, so the ordering is expressed over groups.
GROUP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "en_solo": ("en_kd",),
}


# ----------------------------------------------------------------- sentinel


def sentinel_path(root: Path, group: str) -> Path:
    return Path(root) / group / SENTINEL_NAME


def read_sentinel(root: Path, group: str) -> dict | None:
    """Parsed sentinel, or None if absent/unreadable/not a JSON object.

    A corrupt sentinel is treated exactly like a missing one: the build was not
    proven to finish, so it has to run again.
    """
    p = sentinel_path(root, group)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_prepared(root: Path, group: str) -> bool:
    """True only if the sentinel is present, current-schema, and its splits exist.

    The last check is what catches "sentinel survived, data directory did not"
    (a partial rsync, a stray rm -rf of one split).
    """
    meta = read_sentinel(root, group)
    if meta is None:
        return False
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    counts = meta.get("counts")
    if not isinstance(counts, dict) or not counts:
        return False
    return all((Path(root) / group / split).is_dir() for split in counts)


def write_sentinel(root: Path, group: str, *, counts: dict[str, int],
                   builders: list[str], holdout_ratio: float) -> Path:
    """Atomically stamp the group complete. Must be the last write of a build."""
    path = sentinel_path(root, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "group": group,
        "builders": sorted(builders),
        "counts": dict(counts),
        "holdout_ratio": float(holdout_ratio),
        "timestamp": time.time(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # Same directory as the target: os.replace is only atomic within a filesystem.
    tmp = path.with_name(f"{SENTINEL_NAME}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------- lock


@contextmanager
def group_lock(root: Path, group: str):
    """Exclusive advisory lock for one group, held for the whole build.

    flock, not a lock *file's* existence: the kernel drops it when the holder
    dies, so a crashed build cannot wedge every later run behind a stale marker.
    """
    lock_dir = Path(root) / LOCK_DIR_NAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"{group}.lock"
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if os.name == "nt":
            # msvcrt byte-range lock. LK_LOCK retries ~10x over ~10s then raises
            # instead of blocking forever -- acceptable on a dev box, where the
            # contention flock guards against (two ranks sharing a build) does
            # not occur. Production stays on flock.
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield lock_file
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------- guardrails


def _check_dependencies(group: str, root: Path) -> None:
    for dep in GROUP_DEPENDENCIES.get(group, ()):
        if is_prepared(root, dep):
            continue
        raise FileNotFoundError(
            f"{group!r} is built from {dep!r} artifacts, but {dep!r} is not prepared "
            f"under {root} (plan section 9.1: builders form a DAG, not independent units).\n"
            f"    run first: python -m project_amnesty.datasets.prepare --group {dep}"
        )


EN_KD_MISSING_MSG = """\
en_kd has no dialogues to prepare. Preparing anyway would write an empty group,
which trains fine and teaches nothing, so this stops here instead.

en_kd is not generated in this repo -- generation is teacher inference, not
corpus preparation. Produce the dialogues with the Moshi self-play harness
(kmoshi_ab_selfplay_v2.ipynb), then ingest them:

    python -m project_amnesty.datasets en_kd --stage ingest --root <dialogues_dir> \\
        --text-config configs/data/text_tok.yaml \\
        --filter-config configs/data/filter.yaml

...and re-run this command. Ingest also retokenizes the teacher's Helium text
into the student vocabulary (SeqKD) and runs the quality filter, so check
filter_report.json's pass_rate before assuming a low row count is a bug.

Expected artifacts (none found): {artifact_dir}/*.npz
"""


def _check_en_kd_artifacts(builders: list[BaseDataset]) -> None:
    """Refuse to prepare an empty en_kd rather than produce a silent no-op dataset."""
    for b in builders:
        if getattr(b, "source", "") != "en_kd":
            continue
        out_dir = Path(getattr(b, "out_dir", ""))
        if out_dir.is_dir() and any(out_dir.glob("*.npz")):
            continue
        raise RuntimeError(EN_KD_MISSING_MSG.format(artifact_dir=out_dir))


def _wrap_not_implemented(group: str, exc: NotImplementedError) -> RuntimeError:
    return RuntimeError(
        f"building {group!r} hit an unimplemented code path: {exc}\n\n"
        + EN_KD_MISSING_MSG.format(artifact_dir="data/generated/en_kd")
    )


# ----------------------------------------------------------- ensure_prepared


def ensure_prepared(group: str, *, root: str | Path, cfg: dict | None = None,
                    force: bool = False) -> Path:
    """Guarantee ``{root}/{group}/`` is a complete prepared group. Idempotent.

    Returns the group directory. Cheap on the happy path -- one JSON read and a
    couple of stats -- so callers may invoke it unconditionally.

    force=True rebuilds even when the sentinel is valid; the sentinel is removed
    first so an interrupted forced rebuild cannot leave the old one vouching for
    freshly half-written data.
    """
    root = Path(root)
    if not force and is_prepared(root, group):
        return root / group

    root.mkdir(parents=True, exist_ok=True)
    with group_lock(root, group):
        # Re-check under the lock: whoever we queued behind was, by definition,
        # finishing this exact build.
        if not force and is_prepared(root, group):
            return root / group

        cfg = cfg or {}
        holdout_ratio = float(cfg.get("holdout_ratio", DEFAULT_HOLDOUT_RATIO))
        _check_dependencies(group, root)

        builders = builders_for_group(group, cfg)
        _check_en_kd_artifacts(builders)

        # Invalidate before writing: a crash mid-rebuild must not leave a
        # sentinel that describes the previous build's row counts.
        sentinel_path(root, group).unlink(missing_ok=True)

        try:
            counts = build_group(group, builders, root, holdout_ratio)
        except NotImplementedError as exc:
            raise _wrap_not_implemented(group, exc) from exc

        if not counts:
            raise RuntimeError(
                f"preparing {group!r} produced zero rows from builders "
                f"{[b.name for b in builders]}. Refusing to stamp an empty group; "
                f"check that each builder's artifacts exist under its out_dir."
            )

        return write_sentinel(
            root, group,
            counts=counts,
            builders=[b.name for b in builders],
            holdout_ratio=holdout_ratio,
        ).parent


# ----------------------------------------------------------------------- CLI


def known_groups() -> list[str]:
    return sorted({cls.source for cls in REGISTRY.values() if cls.source})


def ordered_groups(groups: list[str]) -> list[str]:
    """Dependency-respecting order (en_kd before en_solo). Small enough for O(n^2)."""
    out: list[str] = []
    for g in groups:
        for dep in GROUP_DEPENDENCIES.get(g, ()):
            if dep in groups and dep not in out:
                out.append(dep)
        if g not in out:
            out.append(g)
    return out


def load_cfg(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m project_amnesty.datasets.prepare",
        description="Materialize prepared Arrow groups under --root (idempotent).",
    )
    p.add_argument("--group", action="append", default=None,
                   help="group to prepare; repeatable. Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="prepare every group registered in project_amnesty.datasets.REGISTRY")
    p.add_argument("--root", default=None,
                   help="prepared-data root (default: config out_root, else data/prepared)")
    p.add_argument("--config", default=None,
                   help="YAML with out_root / holdout_ratio / per-builder datasets kwargs")
    p.add_argument("--force", action="store_true",
                   help="rebuild even when the sentinel says the group is complete")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.group) == bool(args.all):
        build_parser().error("give exactly one of --group (repeatable) or --all")

    cfg = load_cfg(args.config)
    root = Path(args.root or cfg.get("out_root") or "data/prepared")

    groups = known_groups() if args.all else list(dict.fromkeys(args.group))
    unknown = [g for g in groups if g not in known_groups()]
    if unknown:
        build_parser().error(
            f"unknown group(s) {unknown}; registered groups: {known_groups()}"
        )

    for g in ordered_groups(groups):
        print(f"[prepare] === {g} → {root / g} ===")
        try:
            path = ensure_prepared(g, root=root, cfg=cfg, force=args.force)
        except (RuntimeError, FileNotFoundError) as exc:
            # These are the guardrails talking (missing prerequisite, stub engine,
            # empty build). Their messages are the whole point -- a traceback above
            # them only buries the instructions the user needs.
            print(f"\n[prepare] cannot prepare {g!r}:\n\n{exc}\n", file=sys.stderr)
            return 1
        meta = read_sentinel(root, g) or {}
        print(f"[prepare] {g}: ready at {path} counts={meta.get('counts')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
