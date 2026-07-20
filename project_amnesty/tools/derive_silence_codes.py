"""Data-prep tool at project_amnesty/tools/ (originally from training/tools/). ARCHITECTURE §7.6 (0 is a valid Mimi code, not silence)

Derive the Mimi silence *bank*, once, offline (plan section 2.5).

`TokenConfig.silence_bank` fills the absent user channel of every solo
(`ko_tts` / `en_solo`) sample. Guessing it -- zeros, most obviously -- is not a
benign default: 0 is a perfectly valid Mimi code, so a wrong constant is a
*maximally learnable* signal that correlates perfectly with `lang=ko`. The user
stream CE then solves trivially, the shared embedding table receives a large
gradient on a code that never actually occurs, and the Role Token degenerates
into a "is this channel synthetic?" detector while the loss curve looks healthy.

So the fill is measured, not chosen:

1. synthesize 20 s of digital silence at 24 kHz **and** 20 s of -60 dBFS
   Gaussian room tone (a pure-zero probe alone cannot distinguish "the code for
   silence" from "the code for a numerically degenerate input"),
2. encode both with the frozen Mimi from `kmhf/hf-moshiko` -- the HF conversion
   of the checkpoint the corpus is baked with,
3. discard the first and last `--trim-frames` frames (encoder receptive-field
   warm-up),
4. record the per-codebook mode and its share as a *diagnostic*,
5. write the trimmed frames themselves as a `(K, P)` bank into
   `configs/data/mimi_silence.json`, together with the checkpoint id, so the
   loader can refuse a corpus built with a different codec.

Why a bank and not a constant (this is measured, not a design preference)
------------------------------------------------------------------------
The original plan assumed silence was either a single code per codebook or a
short repeating loop, and gated on modal share > 0.9. Against the real codec,
that gate fires on **all 8** codebooks::

    cb0 mode=1316 share=0.859   cb4 mode=1736 share=0.243
    cb1 mode=1211 share=0.470   cb5 mode=1572 share=0.380
    cb2 mode=783  share=0.248   cb6 mode=825  share=0.435
    cb3 mode=164  share=0.385   cb7 mode=1648 share=0.552

(pooled over both probes; on digital silence alone cb0 reaches 0.943.) Only cb0
is near-constant, which is exactly what you would expect: cb0 is the
WavLM-distilled *semantic* VQ, while cb1..7 are acoustic RVQ residuals encoding
the actual noise floor, which has no reason to quantize to a fixed code.

Nor is there a short loop. Run-length structure differs per codebook: cb2 holds
plateaus of 20-120 frames, cb4/cb6 alternate fast among ~3 values with runs of
1-3. Raising the edge trim 10 -> 25 -> 50 does not change the picture, so this is
structure, not a warm-up transient.

Both branches of the plan's dichotomy are therefore wrong, and the honest fill is
a stretch of the real thing. Hence the bank.

Choosing P
----------
P is the **full trimmed probe** (230 frames for the default 20 s / trim 10), not
a short window. The bank is tiled to cover a crop, so any structure longer than P
becomes a periodic artifact at period P -- and the longest plateau measured (cb2,
124 frames) already exceeds any "short window" worth the name. A P below ~250
would turn cb2's plateaus into a visible sawtooth. Taking the whole trimmed probe
costs ~2k integers in the JSON and removes the question entirely; there is no
budget pressure here that would justify tuning it down.

The dataset tiles the bank with a random phase per sample (`_normalize_solo`), so
a fixed starting frame is not itself a giveaway.

Usage::

    uv run python -m project_amnesty.tools.derive_silence_codes
    uv run python -m project_amnesty.tools.derive_silence_codes --dry-run
    uv run python -m project_amnesty.tools.derive_silence_codes --require-constant

`moshi` and `torch` are imported lazily inside `load_mimi` / `encode_probe`, so
`--help` and the pure helpers work in an environment without the codec
installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Mirrors of data_pipeline.schema, imported rather than restated so a codec swap
# cannot leave this tool writing an 8-entry file for a 16-codebook model.
from data_pipeline.schema import CODEBOOK_SIZE, FRAME_RATE_HZ, NUM_CODEBOOKS, SAMPLE_RATE

__all__ = [
    "ProbeParams",
    "synthesize_probes",
    "trim_edges",
    "codebook_modes",
    "codebook_max_runs",
    "assert_modal_shares",
    "derive_silence_codes",
    "main",
]

# The corpus is baked with this checkpoint (data_pipeline/datasets/mixins.py).
# Deriving silence from a different one is the exact mismatch mimi_ckpt_id exists
# to catch, so it is a flag, not a constant, and it is recorded in the output.
DEFAULT_MIMI_CKPT = "kmhf/hf-moshiko"  # HF conversion; the moshi package is not installed
DEFAULT_OUTPUT = "configs/data/mimi_silence.json"

PROBE_SECONDS = 20.0
ROOM_TONE_DBFS = -60.0
TRIM_FRAMES = 10
MIN_MODAL_SHARE = 0.9

# Which probe's frames become the shipped bank. Digital silence is the literal
# thing a solo sample's user channel represents (there is no microphone), so it
# is the default; room_tone is derived and stored alongside for comparison and
# for anyone who wants to swap the fill for a more lifelike noise floor.
DEFAULT_BANK_PROBE = "digital_silence"


@dataclass(frozen=True)
class ProbeParams:
    """Everything that would change the answer. Written into the JSON verbatim."""

    seconds: float = PROBE_SECONDS
    sample_rate: int = SAMPLE_RATE
    room_tone_dbfs: float = ROOM_TONE_DBFS
    trim_frames: int = TRIM_FRAMES
    min_modal_share: float = MIN_MODAL_SHARE
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.seconds > 0, "probe must be non-empty"
        assert self.sample_rate > 0
        assert self.trim_frames >= 0
        assert 0.0 < self.min_modal_share <= 1.0
        # Trimming both edges must leave something to take a mode over.
        n_frames = int(self.seconds * FRAME_RATE_HZ)
        assert n_frames > 2 * self.trim_frames, (
            f"a {self.seconds}s probe is {n_frames} frames at {FRAME_RATE_HZ} Hz, "
            f"which does not survive trimming {self.trim_frames} frames off each end"
        )


# --------------------------------------------------------------- probe signals


def synthesize_probes(params: ProbeParams) -> dict[str, np.ndarray]:
    """The two probe waveforms, float32 mono in [-1, 1], keyed by name.

    Both are needed. Digital silence is the literal thing solo samples' user
    channel represents, but an all-zero input can hit a degenerate encoder path;
    -60 dBFS room tone is what "silence" sounds like in any real recording. If
    the two disagree the modal share collapses and step 4 fails -- which is the
    intended outcome, not a nuisance.
    """
    n = int(round(params.seconds * params.sample_rate))
    assert n > 0, "probe length rounded to zero samples"

    silence = np.zeros(n, dtype=np.float32)

    # -60 dBFS is an RMS target: amplitude = 10 ** (dbfs / 20).
    rms = 10.0 ** (params.room_tone_dbfs / 20.0)
    rng = np.random.default_rng(params.seed)
    room_tone = (rng.standard_normal(n) * rms).astype(np.float32)
    # Clip is a no-op at -60 dBFS (a ~6-sigma excursion is still ~0.006) but keeps
    # the contract "valid PCM" true for any dbfs a caller passes.
    room_tone = np.clip(room_tone, -1.0, 1.0)

    return {"digital_silence": silence, "room_tone": room_tone}


# ---------------------------------------------------------------- pure helpers


def trim_edges(codes: np.ndarray, trim: int) -> np.ndarray:
    """Drop `trim` frames from each end of a (K, T) code array.

    The encoder's receptive field means the leading frames are still filling with
    zeros of a different kind and the trailing frames are truncated. Including
    them biases the mode toward whatever the transient state emits.
    """
    codes = np.asarray(codes)
    assert codes.ndim == 2, f"expected (K, T) codes, got shape {codes.shape}"
    assert trim >= 0, f"trim must be non-negative, got {trim}"
    K, T = codes.shape
    assert T > 2 * trim, (
        f"cannot trim {trim} frames from each end of a {T}-frame probe "
        f"(codebooks={K}); lengthen the probe or lower --trim-frames"
    )
    return codes[:, trim : T - trim]


def codebook_modes(
    codes: np.ndarray, codebook_size: int = CODEBOOK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """(K, T) codes -> (modes (K,) int64, shares (K,) float64).

    `share` is the fraction of frames equal to the mode -- the number step 4
    thresholds on. Ties go to the lowest code id (argmax on bincount), which is
    irrelevant in the passing case (share > 0.9 cannot tie) and unreached in the
    failing case because the assert fires first.
    """
    codes = np.asarray(codes)
    assert codes.ndim == 2, f"expected (K, T) codes, got shape {codes.shape}"
    K, T = codes.shape
    assert T > 0, "no frames to take a mode over"
    assert codes.min() >= 0 and codes.max() < codebook_size, (
        f"codes out of range [0, {codebook_size}): "
        f"observed [{int(codes.min())}, {int(codes.max())}]"
    )

    modes = np.empty(K, dtype=np.int64)
    shares = np.empty(K, dtype=np.float64)
    for k in range(K):
        counts = np.bincount(codes[k].astype(np.int64), minlength=codebook_size)
        modes[k] = int(np.argmax(counts))
        shares[k] = float(counts[modes[k]]) / float(T)
    return modes, shares


def codebook_max_runs(codes: np.ndarray) -> np.ndarray:
    """(K, T) codes -> (K,) longest run of an identical code, per codebook.

    This is the number that justifies P. A bank shorter than the longest plateau
    chops that plateau and re-injects it at period P, which is precisely the
    periodic artifact the bank exists to avoid. Recorded in the payload so the
    choice of P stays auditable against the data rather than against this
    docstring.
    """
    codes = np.asarray(codes)
    assert codes.ndim == 2, f"expected (K, T) codes, got shape {codes.shape}"
    K, T = codes.shape
    assert T > 0, "no frames to measure runs over"

    out = np.empty(K, dtype=np.int64)
    for k in range(K):
        # Boundaries where the code changes; run lengths are the gaps between them.
        edges = np.flatnonzero(np.diff(codes[k])) + 1
        bounds = np.concatenate(([0], edges, [T]))
        out[k] = int(np.diff(bounds).max())
    return out


def assert_modal_shares(
    shares: np.ndarray, modes: np.ndarray, min_share: float = MIN_MODAL_SHARE
) -> None:
    """Assert every codebook's silence really is a single constant code.

    NOT called by default any more. Against the real codec this fails on all 8
    codebooks (see the module docstring): silence is neither a constant nor a
    short loop, so the tool ships a bank of real frames instead and keeps the
    modal shares as a diagnostic. `--require-constant` restores this as a gate,
    which is useful only for a codec you expect to be genuinely degenerate.
    """
    shares = np.asarray(shares, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.int64)
    assert shares.shape == modes.shape, "shares/modes length mismatch"

    bad = [k for k in range(shares.size) if not shares[k] > min_share]
    if bad:
        detail = ", ".join(
            f"codebook {k}: mode={int(modes[k])} share={shares[k]:.3f}" for k in bad
        )
        raise AssertionError(
            f"modal share <= {min_share} for {len(bad)} codebook(s): {detail}.\n"
            f"Silence is NOT a single code in these codebooks -- it is a plateau/loop "
            f"structure over the noise floor. A constant fill would stamp a learnable "
            f"artifact onto every solo sample's user channel. Drop --require-constant "
            f"to ship the measured (K, P) bank instead, which is the supported answer; "
            f"lowering --min-modal-share is not."
        )


# ---------------------------------------------------------------- model access


def load_mimi(ckpt_id: str = DEFAULT_MIMI_CKPT, device: str = "cpu"):
    """The frozen Mimi, pulled out of the HF Moshi checkpoint as a standalone model.

    NOT the `moshi` package. data_pipeline/datasets/mixins.py reaches for
    `moshi.models.loaders`, but that package is not installed here and that code
    path has never run (there are no artifacts under data/). The repo's working
    codec access -- project_amnesty/datasets/scenarios/scenario_run.py -- goes
    through transformers against `kmhf/hf-moshiko`, which is what is cached.

    Loading `MoshiForConditionalGeneration` would pull ~15 GB to encode a few
    seconds of silence, so instead the `audio_encoder.*` tensors are filtered out
    of the shards into a bare MimiModel. Only the shards that actually contain
    those tensors are fetched.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import MimiConfig, MimiModel

    cfg_path = hf_hub_download(ckpt_id, "config.json")
    audio_cfg = json.load(open(cfg_path)).get("audio_encoder_config")
    assert audio_cfg is not None, (
        f"{ckpt_id}/config.json has no audio_encoder_config; this is not an HF "
        f"Moshi checkpoint. Pass --ckpt-id kmhf/hf-moshiko."
    )
    model = MimiModel(MimiConfig(**audio_cfg)).eval()

    index = json.load(open(hf_hub_download(ckpt_id, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    prefix = "audio_encoder."
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    assert shards, f"no {prefix}* tensors in {ckpt_id}"

    state: dict[str, "torch.Tensor"] = {}
    for shard in shards:
        for k, v in load_file(hf_hub_download(ckpt_id, shard)).items():
            if k.startswith(prefix):
                state[k[len(prefix):]] = v

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Silence codes derived from a partially-loaded codec would be plausible and
    # wrong, so refuse rather than warn.
    assert not missing and not unexpected, (
        f"Mimi state_dict mismatch for {ckpt_id}: "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )
    return model.to(device)


def encode_probe(mimi, wav: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(L,) float32 mono at 24 kHz -> (K, T) int64 Mimi codes."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None, None]
    with torch.inference_mode():
        out = mimi.encode(x.to(device), num_quantizers=NUM_CODEBOOKS)
    codes = out.audio_codes if hasattr(out, "audio_codes") else out
    assert codes.ndim == 3 and codes.shape[0] == 1, (
        f"expected (1, K, T) from Mimi.encode, got {tuple(codes.shape)}"
    )
    assert codes.shape[1] == NUM_CODEBOOKS, (
        f"expected {NUM_CODEBOOKS} codebooks, got {codes.shape[1]}"
    )
    return codes[0].to("cpu").numpy().astype(np.int64)


def _fake_encoder(params: ProbeParams, num_codebooks: int = NUM_CODEBOOKS):
    """A stand-in encoder for --dry-run: constant codes plus a little edge noise.

    Deliberately *not* uniformly constant -- the noise lands in the frames
    `trim_edges` removes, so a dry run that silently stopped trimming would show
    up as a modal share below 1.0 rather than passing regardless.
    """
    def encode(wav: np.ndarray) -> np.ndarray:
        n_frames = int(round(len(wav) / params.sample_rate * FRAME_RATE_HZ))
        base = np.arange(num_codebooks, dtype=np.int64)[:, None] * 7 + 3
        codes = np.broadcast_to(base, (num_codebooks, n_frames)).copy()
        edge = min(params.trim_frames, n_frames // 2)
        if edge:
            codes[:, :edge] = 1
            codes[:, -edge:] = 2
        return codes

    return encode


# ------------------------------------------------------------------- pipeline


def derive_silence_codes(
    *,
    params: ProbeParams | None = None,
    ckpt_id: str = DEFAULT_MIMI_CKPT,
    device: str = "cpu",
    num_codebooks: int = NUM_CODEBOOKS,
    codebook_size: int = CODEBOOK_SIZE,
    bank_probe: str = DEFAULT_BANK_PROBE,
    require_constant: bool = False,
    encode=None,
) -> dict:
    """Run the whole procedure and return the JSON payload.

    `encode` is the injection seam: a callable `(L,) float32 -> (K, T) codes`.
    Left None it loads the real Mimi; tests and --dry-run pass a stub, which is
    why nothing below this line knows what a checkpoint is.

    The shipped artifact is `silence_bank`: the full trimmed frames of the
    `bank_probe` encode, `(K, P)`. Modes and modal shares are computed for every
    probe and pooled, and recorded -- but they are diagnostics. Only
    `require_constant=True` turns the modal-share gate back into a hard failure.
    """
    params = params or ProbeParams()
    probes = synthesize_probes(params)
    assert bank_probe in probes, (
        f"bank_probe={bank_probe!r} is not one of {sorted(probes)}"
    )

    if encode is None:
        mimi = load_mimi(ckpt_id, device=device)
        encode = lambda wav: encode_probe(mimi, wav, device=device)  # noqa: E731

    per_probe: dict[str, dict] = {}
    trimmed: dict[str, np.ndarray] = {}
    for name, wav in probes.items():
        codes = trim_edges(np.asarray(encode(wav)), params.trim_frames)
        assert codes.shape[0] == num_codebooks, (
            f"probe {name!r} encoded to {codes.shape[0]} codebooks, expected "
            f"{num_codebooks}. The checkpoint does not match data_pipeline.schema."
        )
        m, s = codebook_modes(codes, codebook_size)
        per_probe[name] = {
            # Every probe's bank is written out, not just the chosen one: comparing
            # digital silence against room tone after the fact is the only way to
            # tell "this is what silence encodes to" from "this is what a
            # numerically degenerate input encodes to".
            "silence_bank": [[int(v) for v in row] for row in codes],
            "modes": [int(v) for v in m],
            "modal_shares": [round(float(v), 6) for v in s],
            "max_runs": [int(v) for v in codebook_max_runs(codes)],
            "frames_used": int(codes.shape[1]),
        }
        trimmed[name] = codes

    # Pool the two probes before taking the mode. Under the old constant contract
    # this was a gate (the code had to agree across probes); it is now a reported
    # number, and a disagreement shows up as a low pooled share next to two high
    # per-probe shares.
    pooled = np.concatenate([trimmed[n] for n in probes], axis=1)
    modes, shares = codebook_modes(pooled, codebook_size)
    if require_constant:
        assert_modal_shares(shares, modes, params.min_modal_share)

    bank = trimmed[bank_probe]
    return {
        "silence_bank": [[int(v) for v in row] for row in bank],
        "bank_probe": bank_probe,
        "bank_period": int(bank.shape[1]),
        "mimi_ckpt_id": ckpt_id,
        # Diagnostics. Kept because "how constant is silence in this codec?" is the
        # first question anyone re-deriving this will ask, and re-running the probe
        # to answer it costs a model download.
        "modes": [int(v) for v in modes],
        "modal_shares": [round(float(v), 6) for v in shares],
        "max_runs": [int(v) for v in codebook_max_runs(pooled)],
        "require_constant": bool(require_constant),
        "num_codebooks": int(num_codebooks),
        "codebook_size": int(codebook_size),
        "frames_used": int(pooled.shape[1]),
        "probe": asdict(params),
        "per_probe": per_probe,
        "derived_by": "project_amnesty/tools/derive_silence_codes.py",
    }


def write_payload(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


# ------------------------------------------------------------------------ cli


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m project_amnesty.tools.derive_silence_codes",
        description=(
            "Derive Mimi's silence codes by encoding digital silence and room tone "
            "with the frozen codec (plan section 2.5). Run once; commit the output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt-id", default=DEFAULT_MIMI_CKPT,
                   help="HF repo of the codec the corpus was baked with")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="where to write the JSON")
    p.add_argument("--device", default="cpu", help="torch device for the encoder")
    p.add_argument("--seconds", type=float, default=PROBE_SECONDS,
                   help="length of each probe signal")
    p.add_argument("--room-tone-dbfs", type=float, default=ROOM_TONE_DBFS,
                   help="RMS level of the Gaussian room-tone probe")
    p.add_argument("--trim-frames", type=int, default=TRIM_FRAMES,
                   help="frames discarded from each end (encoder warm-up)")
    p.add_argument("--min-modal-share", type=float, default=MIN_MODAL_SHARE,
                   help="threshold for the modal-share diagnostic (see --require-constant)")
    p.add_argument("--bank-probe", default=DEFAULT_BANK_PROBE,
                   choices=["digital_silence", "room_tone"],
                   help="which probe's frames become the shipped bank")
    p.add_argument("--require-constant", action="store_true",
                   help=("fail unless every codebook's silence is a single code above "
                         "--min-modal-share. The pre-bank behaviour; the real codec "
                         "does not satisfy it."))
    p.add_argument("--seed", type=int, default=0, help="room-tone RNG seed")
    p.add_argument("--dry-run", action="store_true",
                   help="use a stub encoder, load no model, and write no file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    params = ProbeParams(
        seconds=args.seconds,
        sample_rate=SAMPLE_RATE,
        room_tone_dbfs=args.room_tone_dbfs,
        trim_frames=args.trim_frames,
        min_modal_share=args.min_modal_share,
        seed=args.seed,
    )

    encode = _fake_encoder(params) if args.dry_run else None
    payload = derive_silence_codes(
        params=params,
        ckpt_id=args.ckpt_id,
        device=args.device,
        bank_probe=args.bank_probe,
        require_constant=args.require_constant,
        encode=encode,
    )

    print(json.dumps(payload, indent=2))
    P = payload["bank_period"]
    print(
        f"bank: {payload['num_codebooks']} x {P} frames from {payload['bank_probe']!r}",
        file=sys.stderr,
    )
    for k, (m, s, r) in enumerate(
        zip(payload["modes"], payload["modal_shares"], payload["max_runs"])
    ):
        head = " ".join(f"{v:4d}" for v in payload["silence_bank"][k][:6])
        print(
            f"  codebook {k}: mode={m:5d} share={s:.3f} max_run={r:4d}  bank[:6]= {head}",
            file=sys.stderr,
        )

    if args.dry_run:
        print(
            f"\n--dry-run: plumbing OK, no model loaded, {args.output} not written.",
            file=sys.stderr,
        )
        return 0

    written = write_payload(payload, args.output)
    print(f"\nwrote {written}", file=sys.stderr)
    print(
        "configs/tokens.yaml points at this file via silence_bank_path and asserts "
        "its mimi_ckpt_id matches before it will build a batch. The bank is NOT "
        "copied into the yaml -- it is ~2k integers.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
