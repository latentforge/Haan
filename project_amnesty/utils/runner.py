"""runner.py — Phase manager.  [fixed location/name]

Drives the TRAINING_CURRICULUM §2 curriculum in order. For each phase it selects
the matching JSON config, runs the train.py entry point with it, resumes from the
previous phase's checkpoint, and — when a phase finishes — runs an evaluate.py gate
that judges checkpoints A/B/C before allowing the next phase to start.

  Phase 0    Preparation (warm-start, tokenizer, template, text probe, Mimi round-trip)
             — happens before training. Not driven as code here; it only fixes the
             "starting state" (delay/init/tokenizer, etc.).
  Phase 1    English-only warmup (semantic KD only, ~5-10% of total steps)  -> checkpoint A
  Phase 2    Early joint (English KD + Korean single-turn + voice-cloning, Korean ratio
             ramp-up)                                                       -> checkpoint B
  Phase 3    Main training (joint continues, A/B/C 3-axis tracking)         -> checkpoint C
  Phase 3.5  Acoustic prosody graft (optional; guard early-stop, ARCHITECTURE §5.3)
  Phase 4    Japanese transfer sweep (0/1/10/100h, separate branch, LoRA/QLoRA)
  Phase 5    Ablation & final validation (curriculum timing, KD-scope, init, seq vs parallel)

Key phase-transition switches (fixed in Phase 0, ARCHITECTURE §5.0.2 / §5.4.2 / §5.3):
  - delay   : pre-training uses acoustic 2 / text +-0.6; post-training (Phase 1+) uses
              acoustic 1 / text 0. Conversation-mode text delay is pinned to 0.
              -> swapped purely via KDCollator.set_delay(acoustic, text) with no weight
                 rebuild (delay is a training hyperparameter, not an inference switch — §5.0.2).
  - mix     : Korean data ratio. Ramped 10%->30%->50% in Phase 2, held at target in Phase 3.
  - graft   : acoustic prosody graft on/off. On only in Phase 3.5 (low weight / low LR + add,
              turn-event local, guard early-stop). Off in every other phase.

These switches reach the training loop through each phase's config file (configs/<name>.json),
which train.py loads via `--config`. The runner never rebuilds weights; it only selects the
config and threads the resume checkpoint from phase to phase.

Resume threading (TRAINING_CHECKPOINT_RESUME_PLAN §6):
  - train.save_checkpoint writes to `<out_dir>/<phase>/step_<step>`, so the loadable checkpoint
    for `--resume` is the latest `step_*` sub-directory (it holds `state.pt`), not the bare phase
    directory. This runner resolves that latest-step directory in both directions:
      * interrupted mid-phase -> the phase re-enters *itself* from its own latest step_* checkpoint,
      * completed phase       -> the next phase is seeded from the completed phase's final/gate
                                 checkpoint.
  - Phase-boundary provenance: at a transition where the trainable set changes (Full-FT <-> LoRA in
    Phase 4, freeze windows) the prior phase's optimizer/param-group state will not line up. The
    runner then prefers the gate/full-export checkpoint as the resume target and expects a tolerant
    optimizer load. It only *chooses and threads* the resume path — checkpoint I/O (the tolerant
    optimizer load + provenance check) lives entirely in train.load_checkpoint and is not touched
    here.

Run: `python -m project_amnesty.utils.runner --phases 1,2,3 --resume <ckpt>`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

# NOTE: `train`/`evaluate` are imported lazily (see _train_module/_evaluate_module below) rather
# than at module top level. train.py imports torch/transformers at top level, so a top-level
# import here would drag those heavy deps into every `import runner` — breaking plain-python
# import / py_compile / the CPU smoke path. The runner only needs them at dispatch time, so the
# imports are deferred into the two functions that actually call into them.

# phase name -> configs/<name>.json (configs are JSON, no yaml — fixed filemap rule)
PHASES: tuple[str, ...] = (
    "phase1_warmup",
    "phase2_joint",
    "phase3_main",
    "phase3_5_graft",
    "phase4_ja",
    "phase5_ablation",
)

# Config file each phase consumes. Passed straight to train.main(--config <path>).
PHASE_CONFIG: dict[str, str] = {name: f"configs/{name}.json" for name in PHASES}

# The evaluate gate to run right after each phase (TRAINING_CURRICULUM §2, evaluate.py).
#   A: Korean-prefix turn-taking activation probing (early signal)
#   B: English held-out interference check + Korean probing re-measure (rise vs Phase 1)
#   C: first-pass Korean multi-turn emergence judgment (mechanism vs content split)
# Phase 3.5/4/5 have their own judging logic (graft guard, transfer curve, ablation
# comparison), so they carry no A/B/C gate.
CHECKPOINT_GATE: dict[str, str | None] = {
    "phase1_warmup": "A",
    "phase2_joint": "B",
    "phase3_main": "C",
    "phase3_5_graft": None,
    "phase4_ja": None,
    "phase5_ablation": None,
}

# Phases that fork onto their own branch instead of the main (Korean) lineage.
# Phase 4 (Japanese transfer) forks from the Phase 3 checkpoint and runs in a separate
# branch directory so it cannot contaminate the main track (TRAINING_CURRICULUM §2 Phase 4).
# Its output must NOT become the resume pointer for subsequent phases.
_BRANCH_PHASES: dict[str, str] = {
    "phase4_ja": "branch_ja",
}

# Phases that run on the main-track model but must NOT become the fork/resume base for the phases
# that follow. Phase 3.5 (acoustic prosody graft) is grafted in place onto the mature Phase 3 model
# (it still executes, resuming from the current main-track checkpoint), yet TRAINING_CURRICULUM §2
# pins the fork base of Phase 4 (JA sweep) and Phase 5 (ablation) to the *pre-graft* Phase 3
# checkpoint ("Phase 3 완료 체크포인트를 베이스로 분기") so the isolated acoustic graft cannot
# contaminate those forks. Such a phase therefore leaves `resume`/`prev_phase` untouched — the
# main-track lineage stays pinned to phase3_main. (Branch phases above are likewise non-advancing.)
_NON_LINEAGE_PHASES: frozenset[str] = frozenset({"phase3_5_graft"})

# train.save_checkpoint names each checkpoint dir `step_<global_step>` under the phase dir.
_STEP_PREFIX = "step_"

# Optional portable full-export checkpoint a gate may drop next to the step_* dirs. Preferred as
# the resume target across a param-group change (§3 format policy: gate = full-export for
# portability/resharding, §6). Absent for ordinary periodic (sharded) checkpoints.
_FULL_EXPORT_DIRNAME = "full_export"

# Phases whose entry flips the trainable set to LoRA/QLoRA (full_ft=False, train.py TrainArgs),
# so the prior Full-FT phase's optimizer/param-group state will not line up. Used ONLY as a static
# fallback when the phase config JSON is unavailable to read `full_ft` directly (see
# _param_group_change). Keeping it a set (not just "phase4_ja") lets the check stay symmetric.
_LORA_PHASES: frozenset[str] = frozenset({"phase4_ja"})


def _train_module():
    """Lazily import the train entry point (drags torch/transformers only at dispatch time)."""
    from project_amnesty.utils import train

    return train


def _evaluate_module():
    """Lazily import the evaluate entry point (heavy deps pulled only when a gate actually runs)."""
    from project_amnesty.utils import evaluate

    return evaluate


@dataclass
class RunPlan:
    """Execution plan: which phases, where to resume from, where to stack outputs.

    phases      : ordered tuple of phase names to drive (defaults to the full PHASES).
    resume      : explicit resume checkpoint for the first phase. None -> start from the Phase 0
                  output (warm-start init), unless `auto_resume` discovers an interrupted run.
                  May point at a concrete `step_*` checkpoint dir, a `state.pt`, or a phase-level
                  dir (resolved to its latest `step_*` via _resolve_resume_target).
    out_dir     : checkpoint root. Each phase's output is stacked underneath it.
    auto_resume : when True (default) and no explicit `resume` is given, the first phase driven
                  continues from its own latest on-disk checkpoint if one exists (interrupted-run
                  recovery). Set False to force a fresh start from Phase 0 init.
    """

    phases: tuple[str, ...] = PHASES
    resume: str | None = None
    out_dir: str = "checkpoints"
    auto_resume: bool = True


def _phase_out_dir(root: str, name: str) -> str:
    """Resolve the output root for a phase.

    Branch phases (e.g. Phase 4 Japanese) get an isolated subdirectory so they do not
    contaminate the main lineage; every other phase writes directly under `root`.
    """
    branch = _BRANCH_PHASES.get(name)
    if branch is not None:
        return os.path.join(root, branch)
    return root


def _checkpoint_path(out_dir: str, name: str) -> str:
    """The phase-level checkpoint directory train.py produces for a phase.

    train.save_checkpoint writes to out_dir/{phase}/step_{step} (see train.py), so the phase-level
    directory is out_dir/<name>. The *loadable* checkpoint the next phase resumes from is the
    latest step_* child of this directory (see _latest_step_ckpt).
    """
    return os.path.join(out_dir, name)


def _latest_step_ckpt(phase_dir: str) -> str | None:
    """Return the newest `step_<N>` checkpoint directory under a phase output dir.

    train.save_checkpoint writes each checkpoint to `<phase_dir>/step_<global_step>` (train.py),
    so the directory the next run must hand to `--resume` is the highest-numbered `step_*` child
    (it holds `state.pt`), not the bare phase dir. Returns None when the phase dir does not exist
    yet or holds no `step_*` checkpoints (a fresh phase). Never raises.
    """
    if not os.path.isdir(phase_dir):
        return None
    best_step = -1
    best_path: str | None = None
    try:
        entries = os.listdir(phase_dir)
    except OSError:
        return None
    for entry in entries:
        if not entry.startswith(_STEP_PREFIX):
            continue
        full = os.path.join(phase_dir, entry)
        if not os.path.isdir(full):
            continue
        try:
            step = int(entry[len(_STEP_PREFIX):])
        except ValueError:
            continue  # ignore non-numeric step_* directories
        if step > best_step:
            best_step, best_path = step, full
    return best_path


def _resolve_resume_target(path: str | None) -> str | None:
    """Normalize a plan/user-supplied resume path to something train.load_checkpoint can load.

    train.load_checkpoint accepts a directory holding `state.pt` or a direct `state.pt`. This
    resolves the three shapes a caller might pass:
      - a concrete `step_<N>` checkpoint dir (has `state.pt`)   -> used as-is,
      - a phase-level dir holding `step_*` children             -> descend to the latest step_*,
      - a direct `state.pt` file, or an as-yet-nonexistent path -> passed through unchanged.
    None -> None (start from the Phase 0 warm-start init).
    """
    if not path:
        return None
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "state.pt")):
            return path  # already a concrete checkpoint dir
        latest = _latest_step_ckpt(path)
        if latest is not None:
            return latest  # a phase dir -> its latest step_* checkpoint
    return path  # a state.pt path (or not-yet-created) -> let train.load_checkpoint handle it


def _phase_resume_target(phase_dir: str, *, prefer_full_export: bool) -> str | None:
    """Pick which checkpoint of a completed phase seeds the next phase.

    At a phase boundary where the trainable set changes (Full-FT <-> LoRA, freeze windows) the
    provenance policy prefers a portable full-export checkpoint when the phase produced one
    (`<phase_dir>/full_export`): it carries an unsharded/portable state that survives a param-group
    change (§3 format policy, §6). Otherwise — and for ordinary same-lineage transitions — it falls
    back to the latest `step_*` checkpoint.
    """
    if prefer_full_export:
        full_export = os.path.join(phase_dir, _FULL_EXPORT_DIRNAME)
        if os.path.isdir(full_export):
            return full_export
    return _latest_step_ckpt(phase_dir)


def _phase_full_ft(name: str) -> bool | None:
    """Best-effort read of a phase config's `full_ft` flag from configs/<name>.json (JSON only).

    Returns the boolean when the config exists and declares `full_ft`, else None (config absent or
    silent on the flag). Kept config-driven per the JSON-only rule (no yaml); never raises.
    """
    path = PHASE_CONFIG.get(name)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    if isinstance(cfg, dict) and "full_ft" in cfg:
        return bool(cfg["full_ft"])
    return None


def _param_group_change(prev_name: str, next_name: str) -> bool:
    """Whether the trainable param set changes across the prev->next phase boundary.

    A change (Full-FT <-> LoRA — most importantly the Phase 4 LoRA fork) means the prior phase's
    optimizer/param-group state will not line up, so the optimizer state must be loaded tolerantly
    (train.load_checkpoint already does this via its try/except + provenance check) and the
    gate/full-export checkpoint is preferred as the resume target.

    Primary signal: the two phases' `full_ft` flags differ (read from the JSON config). When the
    configs are unreadable (not shipped / silent on the flag), fall back to the static
    `_LORA_PHASES` table so the Phase 4 boundary is still detected.
    """
    prev_ft = _phase_full_ft(prev_name)
    next_ft = _phase_full_ft(next_name)
    if prev_ft is not None and next_ft is not None:
        return prev_ft != next_ft
    return (next_name in _LORA_PHASES) != (prev_name in _LORA_PHASES)


def run_phase(name: str, resume: str | None, out_dir: str = "checkpoints") -> str:
    """Drive a single phase -> return the produced (loadable) checkpoint path.

    Loads configs/<name>.json (PHASE_CONFIG[name]) into train.py's entry point and, when training
    finishes, returns the newly written checkpoint directory. If `resume` is given training
    continues from that point (threaded straight through as the `--resume` flag).

    Dispatch contract with train.py:
      - train exposes only `main()`, which parses `sys.argv[1:]` (no argv argument). We build
        the argv and temporarily install it on `sys.argv` around the call, then restore it.
      - `--config <path>` tells train's HfArgumentParser to fill (ModelArgs, DataArgs,
        TrainArgs) from that JSON file (train.parse_args()).
      - out_dir / phase / resume are forwarded as the TrainArgs flags `--out_dir` / `--phase` /
        `--resume`, which train.parse_args overlays on top of the JSON config. This forces train's
        `args.phase` to the runner's phase name, so train.save_checkpoint writes to
        `<out_dir>/<name>` — the config's own `phase` default ('phase2_joint') would otherwise
        diverge from where this runner probes for the produced checkpoint. (TrainArgs defines all
        three fields; runner.py injects them at phase-transition time. `resume` is the concrete,
        loadable `step_*` checkpoint directory chosen by run(), never the bare phase dir.)

    Phase-transition switches (delay/mix/graft) all arrive through the JSON config selected
    here, so no model/loss rebuild happens in the runner (delay is swapped in the collator via
    KDCollator.set_delay(); ARCHITECTURE §5.0.2). NOTE: project_amnesty.datasets/.models stay
    empty and are off-limits — the runner never imports them; it only orchestrates train/evaluate.
    """
    config_path = PHASE_CONFIG[name]
    # Thread out_dir + phase (and resume when present) as explicit CLI flags. train.parse_args
    # parses the JSON config as the base, then OVERLAYS these argv flags on top of it. Forcing
    # --phase to the runner's phase name pins train's args.phase (whose config default is
    # 'phase2_joint') so train.save_checkpoint writes to <out_dir>/<name>/step_* — exactly the path
    # this runner probes for the produced checkpoint below, keeping write path and probe path in
    # agreement regardless of what `phase` the config JSON declares.
    argv: list[str] = ["--config", config_path, "--out_dir", out_dir, "--phase", name]
    if resume:
        argv += ["--resume", resume]

    # train.main() has no argv parameter and reads sys.argv directly, so we swap sys.argv for
    # the duration of the call. This is the thin, real dispatch into the training entry point.
    train = _train_module()
    saved_argv = sys.argv
    sys.argv = ["project_amnesty.utils.train", *argv]
    try:
        train.main()
    finally:
        sys.argv = saved_argv

    # train.py persists to out_dir/{phase}/step_{step}; the phase-level directory is out_dir/<name>
    # and its latest step_* child is the loadable checkpoint the next phase resumes from. Fall back
    # to the phase dir only if nothing was written (e.g. a dry/smoke dispatch that skipped saving).
    phase_dir = _checkpoint_path(out_dir, name)
    return _latest_step_ckpt(phase_dir) or phase_dir


def gate_checkpoint(name: str, ckpt: str) -> dict:
    """Post-phase checkpoint A/B/C gate. Delegates to the evaluate entry point.

    If CHECKPOINT_GATE[name] is a tag (A/B/C), run evaluate.run_checkpoint(ckpt, tag)
    (which keeps mechanism vs content strictly separated) and wrap its report with a pass/fail
    verdict deciding whether the next phase may start. If the phase has no gate (None), pass.

    Returns a dict that always carries `passed` (bool), `phase`, and `gate`; the raw evaluate
    report — when produced — is preserved under `report` without collapsing mechanism/content
    metrics into a single scalar (RISKS §4).
    """
    tag = CHECKPOINT_GATE.get(name)
    if tag is None:
        return {"phase": name, "gate": None, "passed": True}

    evaluate = _evaluate_module()
    report = evaluate.run_checkpoint(ckpt, tag)
    # evaluate.run_checkpoint returns a judgment bundle; treat a missing/true `passed` as a
    # pass so the loop can advance, and an explicit False as a stop.
    passed = bool(report.get("passed", True)) if isinstance(report, dict) else True
    return {"phase": name, "gate": tag, "ckpt": ckpt, "passed": passed, "report": report}


def run(plan: RunPlan) -> list[dict]:
    """Drive plan.phases in order, gating each phase's checkpoint and threading the produced
    checkpoint into the next phase's resume.

    Resume threading (TRAINING_CHECKPOINT_RESUME_PLAN §6):
        resume = _resolve_resume_target(plan.resume)   # normalize explicit path to a loadable dir
        prev_phase = None                              # last main-lineage phase behind `resume`
        for name in plan.phases:
            out_dir = <branch dir for Phase 4, else plan.out_dir>
            # interrupted mid-phase: the first phase re-enters itself from its own latest step_*
            if first phase and no explicit resume and auto_resume:
                resume = latest step_* of this phase, if any
            # phase-boundary provenance: trainable set changed vs the phase behind `resume`
            if param-group change (Full-FT<->LoRA / freeze): prefer prev phase gate/full-export
            ckpt   = run_phase(name, resume, out_dir)   # delay/mix/graft flow in via config
            report = gate_checkpoint(name, ckpt)        # A/B/C judgment (pass if no gate)
            if not report.passed: stop + log            # interference drop / emergence failure
            # Branch phases (Phase 4) AND non-lineage phases (Phase 3.5 graft) don't advance the
            # fork base, so Phase 4/5 keep forking from the pre-graft phase3_main checkpoint.
            if advances_lineage: resume = ckpt; prev_phase = name

    Returns the per-phase result records (checkpoint path + gate report) collected along the way.
    """
    results: list[dict] = []
    # Normalize an explicit resume path (may be a phase dir / state.pt) to a loadable step_* dir.
    resume = _resolve_resume_target(plan.resume)
    prev_phase: str | None = None  # last main-lineage phase whose ckpt `resume` points at

    for index, name in enumerate(plan.phases):
        out_dir = _phase_out_dir(plan.out_dir, name)
        phase_dir = _checkpoint_path(out_dir, name)
        is_branch = name in _BRANCH_PHASES
        # A phase advances the main-track lineage (becomes the next resume / provenance baseline)
        # unless it forks onto its own branch (Phase 4) or is a non-lineage graft (Phase 3.5). Both
        # kinds still run on the current main-track checkpoint; they just don't move the fork base.
        advances_lineage = not is_branch and name not in _NON_LINEAGE_PHASES

        # Interrupted-resume: the first phase we drive continues from its own latest checkpoint
        # when no explicit resume was supplied but the phase already wrote checkpoints on disk.
        # (A completed phase seeds the *next* phase; an interrupted one re-enters *itself*.)
        if index == 0 and resume is None and plan.auto_resume:
            interrupted = _latest_step_ckpt(phase_dir)
            if interrupted is not None:
                resume = interrupted
                print(f"[runner] phase={name} interrupted-resume from {resume}")

        # Phase-boundary provenance: when the trainable set changes vs the phase that produced
        # `resume` (Full-FT <-> LoRA / freeze windows), prefer the prior phase's gate/full-export
        # checkpoint and expect a tolerant optimizer load (train.load_checkpoint absorbs the
        # param-group mismatch; §6). Only the resume *target* is chosen here — no checkpoint I/O.
        if prev_phase is not None and resume is not None and _param_group_change(prev_phase, name):
            prev_dir = _checkpoint_path(_phase_out_dir(plan.out_dir, prev_phase), prev_phase)
            preferred = _phase_resume_target(prev_dir, prefer_full_export=True)
            if preferred is not None:
                resume = preferred
            print(
                f"[runner] phase={name} param-group change vs {prev_phase} "
                f"-> tolerant optimizer load, resume={resume}"
            )

        print(f"[runner] phase={name} resume={resume} out_dir={out_dir} branch={is_branch}")
        ckpt = run_phase(name, resume, out_dir=out_dir)
        report = gate_checkpoint(name, ckpt)
        results.append({"phase": name, "ckpt": ckpt, "gate": report})

        gate_tag = report.get("gate")
        if report.get("passed", True):
            print(f"[runner] phase={name} ckpt={ckpt} gate={gate_tag} -> PASS")
        else:
            # Stop the curriculum: a failed gate means the downstream phases are not meaningful
            # (e.g. English interference collapse at B, or no Korean emergence at C).
            print(f"[runner] phase={name} ckpt={ckpt} gate={gate_tag} -> FAIL, stopping run")
            break

        # Branch phases (Phase 4) fork off the current lineage, and the Phase 3.5 graft runs on the
        # main model without becoming its base; neither may become the resume pointer (nor the
        # provenance baseline) for later phases, so Phase 4/5 keep forking from phase3_main. Only
        # lineage-advancing phases move both the resume pointer and the provenance baseline forward.
        if advances_lineage:
            resume = ckpt
            prev_phase = name

    return results


def _resolve_phases(spec: str | None) -> tuple[str, ...]:
    """Resolve the --phases spec into an ordered tuple of phase names.

    Accepts a comma-separated list where each token is either a 1-based phase number
    ("1" -> PHASES[0]) or a phase name ("phase2_joint"). None -> the full PHASES tuple.
    """
    if spec is None:
        return PHASES

    resolved: list[str] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token) - 1
            if not (0 <= idx < len(PHASES)):
                raise ValueError(
                    f"phase number out of range: {token} (valid 1..{len(PHASES)})"
                )
            resolved.append(PHASES[idx])
        elif token in PHASES:
            resolved.append(token)
        else:
            raise ValueError(
                f"unknown phase: {token!r} (expected a number 1..{len(PHASES)} or one of {PHASES})"
            )
    if not resolved:
        raise ValueError("--phases resolved to an empty list")
    return tuple(resolved)


def main() -> None:
    """CLI: --phases (comma-separated phase numbers or names) / --resume (first-phase resume
    checkpoint) / --out-dir / --no-auto-resume -> build a RunPlan -> run().

    --phases takes numbers like "1,2,3" (mapped to PHASES names) or names directly; if omitted,
    the full PHASES tuple is driven in order. --resume seeds the first phase explicitly; with no
    --resume the first phase auto-continues from its own latest on-disk checkpoint (interrupted-run
    recovery) unless --no-auto-resume is given.
    """
    parser = argparse.ArgumentParser(
        description="Haan phase manager (TRAINING_CURRICULUM §2)."
    )
    parser.add_argument(
        "--phases",
        default=None,
        help='phases to drive (e.g. "1,2,3" or "phase2_joint,phase3_main"). Omit = all.',
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume checkpoint for the first phase. Omit = start from Phase 0 warm-start init "
        "(or auto-continue from the phase's latest on-disk checkpoint if one exists).",
    )
    parser.add_argument("--out-dir", default="checkpoints", help="checkpoint root")
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="disable interrupted-run recovery: force a fresh start from Phase 0 init even when "
        "the first phase already has on-disk checkpoints.",
    )
    args = parser.parse_args()

    plan = RunPlan(
        phases=_resolve_phases(args.phases),
        resume=args.resume,
        out_dir=args.out_dir,
        auto_resume=args.auto_resume,
    )
    run(plan)


if __name__ == "__main__":
    main()
