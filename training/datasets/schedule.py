"""MixSchedule -- the step-keyed group mixing ramp (plan §5.1).

A curriculum over *unnormalized* group weights, piecewise linear in global_step,
normalized only at read time. Two properties carry the design:

  * **Unnormalized anchors.** Authors write `{en_kd: 0.85, ko_tts: 0.05, ...}`
    per anchor without having to keep the row summing to 1 while editing a ramp.
    Normalization at read time means a typo changes a ratio, never the contract.
  * **Loud construction-time validation.** Every failure mode here is one that
    otherwise surfaces at step 45000: a missing key silently becoming 0 (which is
    exactly how a phase loses `text_anchor` and makes Phase 4 meaningless),
    anchors out of order, or a constraint that holds at every anchor but is
    violated *between* them. So constraints are checked on a dense sweep, not on
    the anchor set.

Outside the anchor range the schedule clamps -- an overrun job must not
extrapolate to 80% Korean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = ["MixSchedule"]

_EPS = 1e-9


@dataclass(frozen=True)
class MixSchedule:
    """Piecewise-linear, step-keyed mixing weights.

    Attributes are flat tuples rather than a list of dicts so the dataclass can be
    frozen/hashable and so `weights_at` is a couple of numpy ops on a fixed layout.
    `values[i]` is aligned to `groups`.
    """

    groups: tuple[str, ...]
    steps: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    interp: str = "linear"
    unit: str = "global_step"
    # (group, min, max) on the *normalized* weight.
    bounds: tuple[tuple[str, float, float], ...] = ()
    require_groups: tuple[str, ...] = ()
    sweep_stride: int = 100

    _order: dict = field(default_factory=dict, repr=False, compare=False)

    # ---------------------------------------------------------------- build ---
    def __post_init__(self) -> None:
        object.__setattr__(self, "_order", {g: i for i, g in enumerate(self.groups)})
        self._validate()

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> "MixSchedule":
        """Build from the `data.mix` mapping (see plan §5.1 for the YAML shape)."""
        anchors = cfg.get("anchors")
        assert anchors, "mix.anchors is empty -- a schedule needs at least one anchor"

        parsed: list[tuple[int, Mapping[str, float]]] = []
        for i, a in enumerate(anchors):
            assert "at" in a, f"mix.anchors[{i}] has no 'at' key"
            w = a.get("weights")
            assert w, f"mix.anchors[{i}] (at={a['at']}) has no weights"
            parsed.append((int(a["at"]), w))

        # Group set is taken from the first anchor; _validate rejects any anchor
        # that disagrees. Sorted for a stable, config-order-independent layout.
        groups = tuple(sorted(parsed[0][1].keys()))
        for i, (at, w) in enumerate(parsed):
            got = tuple(sorted(w.keys()))
            assert got == groups, (
                f"mix.anchors[{i}] (at={at}) names groups {got}, but anchor 0 names "
                f"{groups}. A missing key is an error, not an implicit 0 -- that is "
                f"precisely how a phase silently drops a group."
            )

        constraints = dict(cfg.get("constraints") or {})
        require = tuple(constraints.pop("require_groups", ()) or ())
        bounds = tuple(
            (g, float(b.get("min", 0.0)), float(b.get("max", 1.0)))
            for g, b in constraints.items()
        )

        return cls(
            groups=groups,
            steps=tuple(int(at) for at, _ in parsed),
            values=tuple(tuple(float(w[g]) for g in groups) for _, w in parsed),
            interp=str(cfg.get("interp", "linear")),
            unit=str(cfg.get("unit", "global_step")),
            bounds=bounds,
            require_groups=require,
            sweep_stride=int(cfg.get("sweep_stride", 100)),
        )

    # ----------------------------------------------------------- validation ---
    def _validate(self) -> None:
        assert self.interp == "linear", f"interp={self.interp!r} unsupported (linear only)"
        assert self.unit == "global_step", f"unit={self.unit!r} unsupported"
        assert self.groups, "schedule has no groups"
        assert len(self.steps) == len(self.values) >= 1, "steps/values length mismatch"
        assert self.sweep_stride > 0

        assert self.steps[0] == 0, (
            f"first mix anchor must be at step 0, got at={self.steps[0]}. Without it "
            f"the weights before the first anchor are a clamp to an unstated value."
        )
        for a, b in zip(self.steps, self.steps[1:]):
            assert b > a, f"mix anchors must be strictly increasing, got {a} then {b}"

        for at, row in zip(self.steps, self.values):
            assert len(row) == len(self.groups)
            assert all(v >= 0.0 for v in row), f"negative weight at anchor {at}: {row}"
            assert sum(row) > _EPS, f"all weights are zero at anchor {at}"

        for g in self.require_groups:
            assert g in self._order, (
                f"constraints.require_groups names {g!r}, which no anchor declares. "
                f"Declared groups: {self.groups}"
            )
        for g, lo, hi in self.bounds:
            assert g in self._order, f"constraints names unknown group {g!r}"
            assert 0.0 <= lo <= hi <= 1.0, f"constraints[{g}]: bad range [{lo}, {hi}]"

        # Dense sweep: anchors alone are not enough. Two groups ramping in opposite
        # directions can push a third group's *normalized* share out of its band
        # strictly between anchors while every anchor looks fine.
        for s in self._sweep_steps():
            w = self.weights_at(s)
            for g in self.require_groups:
                assert w[g] > _EPS, (
                    f"required group {g!r} has weight ~0 at step {s} "
                    f"(constraints.require_groups)"
                )
            for g, lo, hi in self.bounds:
                assert lo - 1e-9 <= w[g] <= hi + 1e-9, (
                    f"constraint violated at step {s}: {g}={w[g]:.4f} outside "
                    f"[{lo}, {hi}] (checked on a dense sweep, not just at anchors)"
                )

    def _sweep_steps(self) -> list[int]:
        last = self.steps[-1]
        pts = set(range(0, last + 1, self.sweep_stride))
        pts.update(self.steps)
        pts.add(last)
        return sorted(pts)

    # ---------------------------------------------------------------- read ----
    def _unnormalized(self, step: int) -> np.ndarray:
        steps = np.asarray(self.steps, dtype=np.int64)
        vals = np.asarray(self.values, dtype=np.float64)
        s = int(step)
        if s <= steps[0]:
            return vals[0].copy()
        if s >= steps[-1]:
            return vals[-1].copy()
        j = int(np.searchsorted(steps, s, side="right"))  # steps[j-1] <= s < steps[j]
        lo, hi = steps[j - 1], steps[j]
        t = (s - lo) / float(hi - lo)
        return vals[j - 1] * (1.0 - t) + vals[j] * t

    def weights_at(self, step: int) -> dict[str, float]:
        """Normalized weights at `step`. Sums to 1; clamped outside the anchors."""
        w = self._unnormalized(step)
        total = float(w.sum())
        assert total > _EPS, f"degenerate schedule at step {step}"
        return {g: float(v) for g, v in zip(self.groups, w / total)}

    def mean_weights_over(self, start: int, end: int) -> dict[str, float]:
        """Mean of `weights_at` over the integer steps in [start, end).

        Inside a ramp the weights move across the window, so comparing a realized
        ratio against the *left edge* (`weights_at(start)`) produces a systematic
        bias that reads like a sampler bug. This is the target the statistical
        ratio test compares against.
        """
        assert end > start, f"empty window [{start}, {end})"
        acc = np.zeros(len(self.groups), dtype=np.float64)
        for s in range(int(start), int(end)):
            w = self._unnormalized(s)
            acc += w / w.sum()
        acc /= end - start
        return {g: float(v) for g, v in zip(self.groups, acc)}

    def probs_at(self, step: int) -> np.ndarray:
        """`weights_at` as an array aligned to `self.groups` (sampler fast path)."""
        w = self._unnormalized(step)
        return w / w.sum()
