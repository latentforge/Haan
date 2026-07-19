"""Tests for the data-preparation gate (plan section 9, items 42-46).

Everything here runs on fake builders registered into REGISTRY, never on real
corpora: the point under test is the *gate* -- sentinel, schema version, lock,
dependency order -- not Mimi encoding. The fakes are defined at module scope so
the class bodies execute (and therefore self-register) before any test runs, and
so forked subprocesses inherit an already-populated REGISTRY.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

import pytest

from conftest import make_solo_sample  # tests/ is put on sys.path by conftest
from project_amnesty.datasets import REGISTRY, BaseDataset
from project_amnesty.datasets.prepare_dataset import builders_for_group
from project_amnesty.datasets.schema import SCHEMA_VERSION
from project_amnesty.datasets import prepare as P

FAKE_GROUP = "fake_group"


class _FakeBuilderBase(BaseDataset):
    """No name -> not registered. Holds the shared fake-builder behaviour."""

    source = FAKE_GROUP
    lang = "ko"
    sample_type = "ko_tts"

    n_samples = 5
    trace_dir: Path | None = None   # set per-test; each iter_samples() drops a file
    delay_sec = 0.0

    def build(self, limit: int | None = None) -> dict:
        return {"total": self.n_samples}

    def iter_samples(self):
        cls = type(self)
        if cls.trace_dir is not None:
            cls.trace_dir.mkdir(parents=True, exist_ok=True)
            (cls.trace_dir / f"{os.getpid()}-{uuid.uuid4().hex}").write_text("built")
        if cls.delay_sec:
            time.sleep(cls.delay_sec)
        for i in range(cls.n_samples):
            yield make_solo_sample(f"{cls.name}-{i:04d}", T=8)


class FakeBuilderA(_FakeBuilderBase):
    name = "fake_a"


class FakeBuilderB(_FakeBuilderBase):
    """Second builder in the same group -- the N:1 case (kss + zeroth_ko -> ko_tts)."""

    name = "fake_b"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "prepared"


@pytest.fixture(autouse=True)
def _reset_fakes():
    for cls in (FakeBuilderA, FakeBuilderB):
        cls.trace_dir = None
        cls.delay_sec = 0.0
        cls.n_samples = 5
    yield
    for cls in (FakeBuilderA, FakeBuilderB):
        cls.trace_dir = None
        cls.delay_sec = 0.0
        cls.n_samples = 5


# ------------------------------------------------------- N:1 group discovery


def test_group_is_discovered_from_registry_not_hardcoded():
    """All builders sharing a `source` feed one group; no mapping is written down."""
    builders = builders_for_group(FAKE_GROUP)
    assert sorted(b.name for b in builders) == ["fake_a", "fake_b"]

    # the real N:1 case the plan calls out
    assert sorted(b.name for b in builders_for_group("ko_tts")) == [
        "common_voice_ko", "kss", "zeroth_ko",
    ]


def test_unknown_group_raises():
    with pytest.raises(KeyError, match="no registered builder"):
        builders_for_group("not_a_group")


# ---------------------------------------------------------- happy path + 42


def test_prepare_writes_sentinel_last_with_expected_contents(root: Path):
    out = P.ensure_prepared(FAKE_GROUP, root=root)
    assert out == root / FAKE_GROUP

    meta = json.loads((out / P.SENTINEL_NAME).read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["builders"] == ["fake_a", "fake_b"]          # both contributed
    assert sum(meta["counts"].values()) == 10                # 5 + 5, N:1 merged
    assert set(meta["counts"]) <= {"train", "probe"}
    assert meta["holdout_ratio"] == pytest.approx(0.02)
    assert isinstance(meta["timestamp"], float)

    for split in meta["counts"]:
        assert (out / split / "dataset_info.json").exists()


def test_42_second_call_returns_without_rebuilding(root: Path, monkeypatch):
    P.ensure_prepared(FAKE_GROUP, root=root)

    def _boom(*a, **k):
        raise AssertionError("build_group must not run a second time")

    monkeypatch.setattr(P, "build_group", _boom)
    assert P.ensure_prepared(FAKE_GROUP, root=root) == root / FAKE_GROUP


def test_42b_build_invocation_count(root: Path, tmp_path: Path):
    FakeBuilderA.trace_dir = tmp_path / "trace"
    for _ in range(3):
        P.ensure_prepared(FAKE_GROUP, root=root)
    assert len(list((tmp_path / "trace").iterdir())) == 1


def test_force_rebuilds(root: Path, tmp_path: Path):
    FakeBuilderA.trace_dir = tmp_path / "trace"
    P.ensure_prepared(FAKE_GROUP, root=root)
    P.ensure_prepared(FAKE_GROUP, root=root, force=True)
    assert len(list((tmp_path / "trace").iterdir())) == 2


# -------------------------------------------------- 43: sentinel absent/corrupt


def test_43_partial_output_without_sentinel_rebuilds(root: Path, tmp_path: Path):
    """An interrupted save_to_disk leaves dataset_info.json behind. That is not done."""
    P.ensure_prepared(FAKE_GROUP, root=root)
    (root / FAKE_GROUP / P.SENTINEL_NAME).unlink()
    assert (root / FAKE_GROUP / "train" / "dataset_info.json").exists()
    assert P.is_prepared(root, FAKE_GROUP) is False

    FakeBuilderA.trace_dir = tmp_path / "trace"
    P.ensure_prepared(FAKE_GROUP, root=root)
    assert len(list((tmp_path / "trace").iterdir())) == 1
    assert P.is_prepared(root, FAKE_GROUP) is True


def test_43b_corrupt_sentinel_rebuilds(root: Path, tmp_path: Path):
    P.ensure_prepared(FAKE_GROUP, root=root)
    (root / FAKE_GROUP / P.SENTINEL_NAME).write_text("{not json at all")
    assert P.read_sentinel(root, FAKE_GROUP) is None

    FakeBuilderA.trace_dir = tmp_path / "trace"
    P.ensure_prepared(FAKE_GROUP, root=root)
    assert len(list((tmp_path / "trace").iterdir())) == 1


def test_43c_sentinel_without_its_split_dir_rebuilds(root: Path):
    import shutil

    P.ensure_prepared(FAKE_GROUP, root=root)
    shutil.rmtree(root / FAKE_GROUP / "train")
    assert P.is_prepared(root, FAKE_GROUP) is False


# ------------------------------------------------------ 44: schema_version


def test_44_schema_version_mismatch_rebuilds(root: Path, tmp_path: Path):
    """`zone_a_frames` will be added to arrow_features(); stale data must not survive."""
    P.ensure_prepared(FAKE_GROUP, root=root)
    sent = root / FAKE_GROUP / P.SENTINEL_NAME
    meta = json.loads(sent.read_text())
    meta["schema_version"] = SCHEMA_VERSION - 1
    sent.write_text(json.dumps(meta))
    assert P.is_prepared(root, FAKE_GROUP) is False

    FakeBuilderA.trace_dir = tmp_path / "trace"
    P.ensure_prepared(FAKE_GROUP, root=root)
    assert len(list((tmp_path / "trace").iterdir())) == 1
    assert json.loads(sent.read_text())["schema_version"] == SCHEMA_VERSION


# -------------------------------------------------------- 45: rank guard


def test_45_heavy_group_refused_in_multirank_job():
    from project_amnesty.datasets.loader import HEAVY_GROUPS, _assert_can_build

    assert {"en_kd", "en_solo", "ko_tts"} <= set(HEAVY_GROUPS)

    for group in sorted(HEAVY_GROUPS):
        with pytest.raises(RuntimeError) as ei:
            _assert_can_build("if_missing", group, world_size=4)
        msg = str(ei.value)
        assert f"python -m project_amnesty.datasets.prepare --group {group}" in msg
        assert "4-rank" in msg


def test_45b_rank_guard_allows_the_cases_it_should():
    from project_amnesty.datasets.loader import _assert_can_build

    _assert_can_build("never", "en_kd", world_size=4)        # never build in-job
    _assert_can_build("if_missing", "en_kd", world_size=1)   # single process
    _assert_can_build("force", "text_anchor", world_size=4)  # light group


def test_45c_the_command_the_guard_prints_is_a_real_cli():
    """The message is only useful if the entry point it names actually parses."""
    args = P.build_parser().parse_args(["--group", "en_kd"])
    assert args.group == ["en_kd"]


# ------------------------------------------------------- 46: concurrency


def _worker(root: str, trace: str, barrier, q):
    FakeBuilderA.trace_dir = Path(trace)
    FakeBuilderA.delay_sec = 0.5      # widen the race window
    try:
        barrier.wait(timeout=30)
        q.put(("ok", str(P.ensure_prepared(FAKE_GROUP, root=Path(root)))))
    except Exception as exc:  # pragma: no cover - reported through the queue
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def test_46_two_concurrent_processes_produce_one_set_of_artifacts(root: Path, tmp_path: Path):
    ctx = mp.get_context("fork")   # children inherit REGISTRY with the fakes in it
    trace = tmp_path / "trace"
    barrier, q = ctx.Barrier(2), ctx.Queue()

    procs = [ctx.Process(target=_worker, args=(str(root), str(trace), barrier, q))
             for _ in range(2)]
    for p in procs:
        p.start()
    results = [q.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    assert all(kind == "ok" for kind, _ in results), results
    assert {payload for _, payload in results} == {str(root / FAKE_GROUP)}

    # exactly one build ran, and exactly one sentinel describes it
    assert len(list(trace.iterdir())) == 1
    assert P.is_prepared(root, FAKE_GROUP)
    meta = P.read_sentinel(root, FAKE_GROUP)
    assert sum(meta["counts"].values()) == 10        # not 20 -- no double-append
    assert not list((root / FAKE_GROUP).glob(f"{P.SENTINEL_NAME}.*.tmp"))


# ------------------------------------------------------ dependency ordering


def test_en_solo_requires_en_kd_first(root: Path):
    with pytest.raises(FileNotFoundError) as ei:
        P.ensure_prepared("en_solo", root=root)
    msg = str(ei.value)
    assert "en_kd" in msg
    assert "python -m project_amnesty.datasets.prepare --group en_kd" in msg


def test_ordered_groups_puts_dependencies_first():
    assert P.ordered_groups(["en_solo", "en_kd"]) == ["en_kd", "en_solo"]
    assert P.ordered_groups(["en_solo"]) == ["en_solo"]   # dep not requested: unchanged
    assert P.ordered_groups(["ko_tts"]) == ["ko_tts"]


# --------------------------------------------------- en_kd has no in-repo source


def test_en_kd_failure_points_at_the_ingest_command(root: Path, tmp_path: Path,
                                                    monkeypatch):
    """en_kd dialogues come from the teacher, not from this package, so the only
    actionable instruction is the ingest command. It used to name two paths, one
    of which was implementing an in-repo generator; that generator is gone."""
    monkeypatch.chdir(tmp_path)            # no data/generated/en_kd here
    with pytest.raises(RuntimeError) as ei:
        P.ensure_prepared("en_kd", root=root)
    msg = str(ei.value)
    assert "python -m project_amnesty.datasets en_kd --stage ingest --root" in msg
    assert "--text-config" in msg, "ingest needs text_cfg for SeqKD; say so"
    assert "MoshiSelfTalkEngine" not in msg, "the in-repo generator no longer exists"
    # and it must not have left a group behind
    assert not P.is_prepared(root, "en_kd")


def test_en_kd_builder_has_no_generation_stage():
    """Generation is teacher inference, not corpus preparation. build() exists
    only to satisfy the BaseDataset contract and must fail with a pointer rather
    than half-implement a generator."""
    from project_amnesty.datasets import REGISTRY

    cls = REGISTRY["en_kd"]
    assert not hasattr(cls, "_generate")
    with pytest.raises(NotImplementedError, match="--stage ingest"):
        cls().build()


def test_not_implemented_is_translated(root: Path, monkeypatch):
    def _raise(*a, **k):
        raise NotImplementedError("streaming interface not finalized")

    monkeypatch.setattr(P, "build_group", _raise)
    with pytest.raises(RuntimeError, match="unimplemented code path"):
        P.ensure_prepared(FAKE_GROUP, root=root)
    assert not P.is_prepared(root, FAKE_GROUP)


def test_empty_build_is_not_stamped(root: Path):
    FakeBuilderA.n_samples = 0
    FakeBuilderB.n_samples = 0
    with pytest.raises(RuntimeError, match="zero rows"):
        P.ensure_prepared(FAKE_GROUP, root=root)
    assert not P.is_prepared(root, FAKE_GROUP)


# -------------------------------------------------------------------- CLI


def test_cli_prepares_a_group(root: Path, capsys):
    assert P.main(["--group", FAKE_GROUP, "--root", str(root)]) == 0
    assert P.is_prepared(root, FAKE_GROUP)
    assert "ready at" in capsys.readouterr().out


def test_cli_is_idempotent(root: Path, tmp_path: Path):
    FakeBuilderA.trace_dir = tmp_path / "trace"
    P.main(["--group", FAKE_GROUP, "--root", str(root)])
    P.main(["--group", FAKE_GROUP, "--root", str(root)])
    assert len(list((tmp_path / "trace").iterdir())) == 1


def test_cli_force(root: Path, tmp_path: Path):
    FakeBuilderA.trace_dir = tmp_path / "trace"
    P.main(["--group", FAKE_GROUP, "--root", str(root)])
    P.main(["--group", FAKE_GROUP, "--root", str(root), "--force"])
    assert len(list((tmp_path / "trace").iterdir())) == 2


def test_cli_reads_root_and_holdout_from_config(root: Path, tmp_path: Path):
    cfg = tmp_path / "prepare.yaml"
    cfg.write_text(f"out_root: {root}\nholdout_ratio: 0.5\n")
    P.main(["--group", FAKE_GROUP, "--config", str(cfg)])
    meta = P.read_sentinel(root, FAKE_GROUP)
    assert meta["holdout_ratio"] == 0.5
    assert set(meta["counts"]) == {"train", "probe"}   # 0.5 must actually split


def test_cli_reports_guardrail_failures_without_a_traceback(root: Path, capsys):
    assert P.main(["--group", "en_solo", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "cannot prepare 'en_solo'" in err
    assert "python -m project_amnesty.datasets.prepare --group en_kd" in err


def test_cli_all_covers_every_registered_group(root: Path, monkeypatch):
    """--all must attempt en_kd first and stop cleanly on its stub, not crash."""
    seen: list[str] = []
    monkeypatch.setattr(P, "ensure_prepared",
                        lambda g, **k: (seen.append(g), root / g)[1])
    assert P.main(["--all", "--root", str(root)]) == 0
    assert set(seen) == set(P.known_groups())
    assert seen.index("en_kd") < seen.index("en_solo")


def test_cli_rejects_unknown_group(root: Path):
    with pytest.raises(SystemExit):
        P.main(["--group", "nope", "--root", str(root)])


def test_cli_requires_exactly_one_selector(root: Path):
    with pytest.raises(SystemExit):
        P.main(["--root", str(root)])
    with pytest.raises(SystemExit):
        P.main(["--group", FAKE_GROUP, "--all", "--root", str(root)])


def test_known_groups_covers_the_real_ones():
    assert {"en_kd", "en_solo", "ko_tts", "text_anchor"} <= set(P.known_groups())
    assert set(P.known_groups()) == {c.source for c in REGISTRY.values() if c.source}
