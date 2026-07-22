"""Weight-audit tool at project_amnesty/tools/. Reproduces ARCHITECTURE.md 3.5.1.

Question this answers: in the *original* Moshi, where does the self/user role
distinction actually live, and is it a clean additive offset?

The Role Token design (ARCH 3.3) shares one audio-embedding table across self and
user and tags the role with a single learned additive vector `RoleEmb[role]`. That
is only well-motivated if the original Moshi's role difference is itself close to a
constant offset. This tool checks that empirically by comparing, codebook by
codebook, the self table (`emb.0..7`) against the user table (`emb.8..15`) of a
frozen Moshi checkpoint.

For every codebook k it reports three numbers (the ARCH 3.5.1 table):

1. row cosine(self_i, user_i)  -- mean over codes of the cosine between the self and
   user embedding of the *same* code index. High = the two tables already agree.
2. constant-offset share       -- fraction of the total size of the difference matrix
   D = user - self that a single constant vector c* = mean(D) explains. This is
   exactly "how well does user_i ~ self_i + c approximate the table". Small = the
   role difference is *not* a clean additive offset.
3. per-code residual share     -- the complement (1 - offset share).

Reference result (kyutai/moshiko-pytorch-bf16, ARCH 3.5.1):

    codebook 0 (semantic):  row cosine 0.501 | offset 1.76% | residual 98.24%
    codebook 1 (acoustic):  row cosine 0.751 | offset 7.82% | residual 92.18%

Two conclusions the numbers force (ARCH 3.5.1):
  - the role signal is NOT a clean additive offset (>90% is per-code), so "extract the
    role offset and graft it onto another table" has no basis;
  - role separation concentrates in the *semantic* codebook (0.50 < 0.75), which is why
    the Role Token carries its real load there.

Honest confound (ARCH 3.5.1): acoustic codebooks are loss-downweighted (x0.02), so their
high similarity may reflect *lack of differentiation pressure* rather than role
irrelevance. This tool therefore measures ALL codebooks 0..dep_q-1 and prints whether
the acoustic cosine increases monotonically 1 -> 7, which separates the two readings.

This is a one-time audit, run by hand -- it is not imported by the training loop:

    uv run python -m project_amnesty.tools.inspect_moshi_weights
    uv run python -m project_amnesty.tools.inspect_moshi_weights --ckpt kyutai/moshiko-pytorch-bf16
    uv run python -m project_amnesty.tools.inspect_moshi_weights --ckpt-path /local/model.safetensors
    uv run python -m project_amnesty.tools.inspect_moshi_weights --out configs/data/moshi_role_audit.json
    uv run python -m project_amnesty.tools.inspect_moshi_weights --dry-run   # synthetic, no download

Heavy deps (torch, safetensors, huggingface_hub) are imported lazily inside the loader
so the module stays importable without a CUDA / model environment. numpy does the math.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CodebookAudit",
    "compare_tables",
    "row_cosine",
    "offset_decomposition",
    "acoustic_monotonicity",
    "load_embedding_tables",
    "synthesize_tables",
    "inspect_moshi_weights",
    "write_payload",
]

# -- defaults ---------------------------------------------------------------------
# Moshi audio embeddings are laid out as 2 * dep_q tables: indices [0, dep_q) are the
# self ("Moshi") stream, [dep_q, 2*dep_q) are the user stream (memory: emb.8..15 = user).
# Codebook 0 is the semantic VQ level; 1..dep_q-1 are the acoustic RVQ levels (ARCH 4.1).
DEFAULT_CKPT = "kyutai/moshiko-pytorch-bf16"
DEFAULT_DEP_Q = 8            # self codebooks (== user codebooks); 16 tables total
DEFAULT_AUDIO_CARD = 2048    # real Mimi codes per table; trailing rows are special tokens
EMB_KEY_RE = re.compile(r"(?:^|\.)emb\.(\d+)\.weight$")


@dataclass
class CodebookAudit:
    """Per-codebook comparison of the self vs user embedding table."""

    codebook: int
    kind: str                # "semantic" (k == 0) or "acoustic" (k >= 1)
    row_cosine: float        # mean_i cos(self_i, user_i)
    offset_share: float      # energy of D = user - self explained by the constant mean(D)
    residual_share: float    # 1 - offset_share (per-code, role-specific part)
    n_codes: int             # number of code rows actually compared


# -- core math (numpy, no model needed) -------------------------------------------
def row_cosine(self_tab: np.ndarray, user_tab: np.ndarray, eps: float = 1e-8) -> float:
    """Mean cosine similarity between self and user rows at the same code index."""
    s = self_tab.astype(np.float64)
    u = user_tab.astype(np.float64)
    sn = s / (np.linalg.norm(s, axis=1, keepdims=True) + eps)
    un = u / (np.linalg.norm(u, axis=1, keepdims=True) + eps)
    return float((sn * un).sum(axis=1).mean())


def offset_decomposition(self_tab: np.ndarray, user_tab: np.ndarray) -> tuple[float, float]:
    """Split the self->user difference into a constant offset vs a per-code residual.

    D_i = user_i - self_i. The single constant c that best approximates every D_i (in
    least squares) is the mean c* = mean_i(D_i). We report the fraction of the total
    energy sum_i ||D_i||^2 that c* captures:

        offset_share   = N * ||c*||^2 / sum_i ||D_i||^2      (== 1 - SS_resid / SS_total)
        residual_share = sum_i ||D_i - c*||^2 / sum_i ||D_i||^2

    A small offset_share means "user = self + constant" is a poor model, i.e. the role
    difference is encoded per code, not as one shared bias vector.
    """
    d = user_tab.astype(np.float64) - self_tab.astype(np.float64)
    ss_total = float((d * d).sum())
    if ss_total == 0.0:
        # self == user: there is no role difference to decompose. Returning 1.0 here
        # renders as "a constant offset explains 100% of it" -- the strongest form of
        # this tool's verdict -- when the truth is the opposite: the offset vector is
        # itself zero and there is no role signal at all. Report it as undefined so a
        # degenerate input cannot masquerade as the headline finding.
        return float("nan"), float("nan")
    c = d.mean(axis=0)
    resid = d - c
    ss_resid = float((resid * resid).sum())
    residual_share = ss_resid / ss_total
    offset_share = 1.0 - residual_share
    return offset_share, residual_share


def compare_tables(
    tables: dict[int, np.ndarray],
    dep_q: int = DEFAULT_DEP_Q,
    audio_card: int = DEFAULT_AUDIO_CARD,
    row_offset: int = 0,
) -> list[CodebookAudit]:
    """Run row-cosine + offset decomposition for each codebook k in [0, dep_q)."""
    out: list[CodebookAudit] = []
    for k in range(dep_q):
        self_idx, user_idx = k, dep_q + k
        if self_idx not in tables or user_idx not in tables:
            raise KeyError(
                f"missing embedding table for codebook {k}: need emb.{self_idx} and "
                f"emb.{user_idx}, have indices {sorted(tables)}"
            )
        lo = row_offset
        hi = row_offset + audio_card
        # numpy clamps an out-of-range slice rather than raising, so without this the
        # tool audits a row set nobody asked for. A `row_offset` at or past the end
        # yields an EMPTY slice for both tables, whose zero difference reads back as
        # offset_share=1.0 with a NaN cosine -- the exact inverse of this tool's
        # finding, written out as a legitimate-looking report. A too-large
        # `audio_card` silently folds in the trailing special-token rows instead.
        for side, idx in (("self", self_idx), ("user", user_idx)):
            rows = tables[idx].shape[0]
            if lo < 0 or hi > rows:
                raise ValueError(
                    f"codebook {k}: {side} row window [{lo}:{hi}] is out of range for a "
                    f"{rows}-row table (row_offset={row_offset}, audio_card={audio_card}). "
                    "Slicing would be clamped and the audit would run on the wrong rows."
                )
        s = tables[self_idx][lo:hi]
        u = tables[user_idx][lo:hi]
        if s.shape != u.shape:
            raise ValueError(f"codebook {k}: self {s.shape} vs user {u.shape} shape mismatch")
        offset_share, residual_share = offset_decomposition(s, u)
        out.append(
            CodebookAudit(
                codebook=k,
                kind="semantic" if k == 0 else "acoustic",
                row_cosine=row_cosine(s, u),
                offset_share=offset_share,
                residual_share=residual_share,
                n_codes=int(s.shape[0]),
            )
        )
    return out


def acoustic_monotonicity(results: list[CodebookAudit]) -> dict[str, Any]:
    """Check whether the acoustic (k>=1) row cosine increases monotonically with level.

    This is the ARCH 3.5.1 confound test: if downweighting (not role irrelevance) drives
    acoustic similarity up, cosine should climb as the level gets less supervised.
    """
    acoustic = [r for r in results if r.kind == "acoustic"]
    cosines = [r.row_cosine for r in acoustic]
    # `all()` over an empty pairing is True, so fewer than two acoustic levels used to
    # report the confound test as PASSED with no data behind it (dep_q<=2 gave a
    # confident "yes" off zero comparisons). An undecided verdict is the honest answer.
    if len(cosines) < 2:
        monotonic = None
    else:
        monotonic = bool(all(b >= a for a, b in zip(cosines, cosines[1:])))
    return {
        "levels": [r.codebook for r in acoustic],
        "cosines": cosines,
        "monotonic_increasing": monotonic,
    }


# -- checkpoint loading (lazy heavy imports) --------------------------------------
def load_embedding_tables(
    ckpt: str | None = DEFAULT_CKPT,
    ckpt_path: str | None = None,
    dep_q: int = DEFAULT_DEP_Q,
    revision: str | None = None,
) -> dict[int, np.ndarray]:
    """Load the `emb.<i>.weight` audio tables (indices 0..2*dep_q-1) as float32 arrays.

    Reads only the embedding tensors via safetensors' lazy `safe_open`, so the ~16 GB
    backbone is never materialized. Accepts either a Hugging Face repo id (`ckpt`) or a
    local `.safetensors` file (`ckpt_path`). bf16 tensors are up-cast to float32.
    """
    import torch  # noqa: PLC0415 -- heavy, tool-only
    from safetensors import safe_open  # noqa: PLC0415

    if ckpt_path is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        # moshi checkpoints ship the LM weights as a single model.safetensors.
        ckpt_path = hf_hub_download(repo_id=ckpt, filename="model.safetensors", revision=revision)

    want = set(range(2 * dep_q))
    tables: dict[int, np.ndarray] = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            m = EMB_KEY_RE.search(key)
            if m is None:
                continue
            idx = int(m.group(1))
            if idx not in want:
                continue
            t = f.get_tensor(key)
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            tables[idx] = t.detach().cpu().numpy().astype(np.float32)
    missing = sorted(want - set(tables))
    if missing:
        raise KeyError(
            f"checkpoint at {ckpt_path} is missing emb tables {missing}; "
            f"found indices {sorted(tables)}"
        )
    return tables


# -- synthetic path (for --dry-run: exercises the math with no download) ----------
def synthesize_tables(
    dep_q: int = DEFAULT_DEP_Q,
    audio_card: int = DEFAULT_AUDIO_CARD,
    dim: int = 32,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Fabricate self/user tables whose per-codebook cosine rises with the level.

    Purely to smoke-test the pipeline offline; the numbers are not Moshi's. Codebook 0
    (semantic) is made the most self/user-divergent, higher levels progressively more
    aligned -- mimicking the qualitative ARCH 3.5.1 pattern.
    """
    rng = np.random.default_rng(seed)
    tables: dict[int, np.ndarray] = {}
    for k in range(dep_q):
        base = rng.standard_normal((audio_card, dim)).astype(np.float32)
        # align fraction grows with k: k=0 most divergent, k=dep_q-1 most shared.
        align = 0.3 + 0.6 * (k / max(dep_q - 1, 1))
        noise = rng.standard_normal((audio_card, dim)).astype(np.float32)
        user = align * base + (1.0 - align) * noise
        tables[k] = base
        tables[dep_q + k] = user.astype(np.float32)
    return tables


# -- reporting --------------------------------------------------------------------
def format_report(results: list[CodebookAudit], mono: dict[str, Any]) -> str:
    """Render the ARCH 3.5.1-style table plus the monotonicity verdict."""
    lines = [
        "codebook | kind     | row cosine | offset share | residual share",
        "---------|----------|------------|--------------|---------------",
    ]
    for r in results:
        lines.append(
            f"{r.codebook:>8} | {r.kind:<8} | {r.row_cosine:>10.3f} | "
            f"{r.offset_share * 100:>11.2f}% | {r.residual_share * 100:>13.2f}%"
        )
    decided = mono["monotonic_increasing"]
    verdict = "n/a (needs >= 2 acoustic levels)" if decided is None else ("yes" if decided else "no")
    lines.append("")
    lines.append(
        # "non-decreasing", not "increasing": the test is `>=`, so a flat profile counts.
        f"acoustic cosine non-decreasing 1->{results[-1].codebook}: {verdict} "
        f"(levels {mono['levels']} -> {[round(c, 3) for c in mono['cosines']]})"
    )
    return "\n".join(lines)


def write_payload(results: list[CodebookAudit], mono: dict[str, Any], source: str, out: Path,
                  *, force: bool = False) -> dict[str, Any]:
    """Persist the audit as JSON (per-codebook rows + monotonicity + provenance)."""
    payload = {
        "source": source,
        "audited_by": "project_amnesty/tools/inspect_moshi_weights.py",
        "reference_doc": "docs/contexts/ARCHITECTURE.md#3.5.1",
        "codebooks": [asdict(r) for r in results],
        "acoustic_monotonicity": mono,
    }
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists. A --dry-run audit carries synthetic numbers, so silently "
            "replacing a real audit with one would be worse than refusing. Pass --force to "
            "overwrite deliberately."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: serialize to a sibling temp file and rename, so an interrupted run cannot
    # leave a truncated audit where a valid one used to be.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(out)
    return payload


# -- top-level driver -------------------------------------------------------------
def inspect_moshi_weights(
    ckpt: str | None = DEFAULT_CKPT,
    ckpt_path: str | None = None,
    dep_q: int = DEFAULT_DEP_Q,
    audio_card: int = DEFAULT_AUDIO_CARD,
    row_offset: int = 0,
    revision: str | None = None,
    dry_run: bool = False,
) -> tuple[list[CodebookAudit], dict[str, Any], str]:
    """Load (or synthesize) the tables, compare them, and return (results, mono, source)."""
    if dry_run:
        tables = synthesize_tables(dep_q=dep_q, audio_card=audio_card)
        source = f"synthetic (dry-run, dep_q={dep_q}, audio_card={audio_card})"
    else:
        tables = load_embedding_tables(ckpt=ckpt, ckpt_path=ckpt_path, dep_q=dep_q, revision=revision)
        source = ckpt_path or f"{ckpt}@{revision or 'main'}"
    results = compare_tables(tables, dep_q=dep_q, audio_card=audio_card, row_offset=row_offset)
    mono = acoustic_monotonicity(results)
    return results, mono, source


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m project_amnesty.tools.inspect_moshi_weights",
        description="Audit the original Moshi self/user audio embeddings (ARCHITECTURE 3.5.1).",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ckpt", default=DEFAULT_CKPT, help="Hugging Face repo id of the Moshi checkpoint")
    src.add_argument("--ckpt-path", default=None, help="local path to a model.safetensors instead of downloading")
    p.add_argument("--revision", default=None, help="checkpoint revision / branch (hub only)")
    p.add_argument("--dep-q", type=int, default=DEFAULT_DEP_Q, help="self codebooks; user tables are emb.[dep_q..2*dep_q)")
    p.add_argument("--audio-card", type=int, default=DEFAULT_AUDIO_CARD, help="code rows compared per table")
    # NOT "skip leading special tokens": the real codes are the LEADING rows and the
    # special tokens trail them (see DEFAULT_AUDIO_CARD), so the default 0 is already
    # the start of the real-code block. The old help text pointed the opposite way.
    p.add_argument("--row-offset", type=int, default=0,
                   help="first code row to compare; 0 is the start of the real-code block "
                        "(special tokens are the trailing rows, so this is rarely nonzero)")
    p.add_argument("--out", default=None, help="optional JSON output path for the audit")
    p.add_argument("--force", action="store_true",
                   help="overwrite --out if it already exists (a --dry-run audit is synthetic, so "
                        "replacing a real one must be deliberate)")
    p.add_argument("--dry-run", action="store_true", help="use synthetic tables (no download); smoke-test only")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results, mono, source = inspect_moshi_weights(
        ckpt=args.ckpt,
        ckpt_path=args.ckpt_path,
        dep_q=args.dep_q,
        audio_card=args.audio_card,
        row_offset=args.row_offset,
        revision=args.revision,
        dry_run=args.dry_run,
    )
    print(f"source: {source}")
    print(format_report(results, mono))
    if args.out is not None:
        payload = write_payload(results, mono, source, Path(args.out), force=args.force)
        print(f"\nwrote {len(payload['codebooks'])} codebook audits -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
