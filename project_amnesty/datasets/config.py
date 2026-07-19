"""Config dataclasses for the training-time data stack.

Follows the house style already used by GenConfig/FilterConfig/TextTokCfg in
project_amnesty: plain dataclasses + from_yaml + loud asserts, no hydra.

Token ids live in exactly one file (configs/tokens.yaml). They are `None` today
because the PAD/EPAD slot assignment is still open, so TokenConfig stays
*loadable* with nulls and each consumer calls `require()` for the fields it
actually needs. That way an unrelated tool can inspect the config while anything
that would silently train on a wrong id fails immediately, naming the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from project_amnesty.datasets.schema import CODEBOOK_SIZE, NUM_CODEBOOKS

TOKENS_YAML = "configs/tokens.yaml"
SILENCE_JSON = "configs/data/mimi_silence.json"


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _as_bank(raw, where: str) -> tuple[tuple[int, ...], ...]:
    """Normalize a nested sequence to an immutable (K, P) tuple-of-tuples.

    Rectangularity is checked here rather than left to numpy: a ragged bank
    silently becomes a 1-D object array, and `bank[:, phase]` on that returns
    garbage instead of raising.
    """
    rows = tuple(tuple(int(c) for c in row) for row in raw)
    assert rows, f"{where}: silence_bank is empty"
    widths = {len(r) for r in rows}
    assert len(widths) == 1, (
        f"{where}: silence_bank is ragged -- codebook lengths {sorted(widths)}. "
        f"It must be a rectangular (K, P) array of consecutive frames."
    )
    assert widths.pop() >= 1, f"{where}: silence_bank has period P=0; need P >= 1"
    return rows


@dataclass
class TokenConfig:
    """Special-token ids and codec constants. Mirrors configs/tokens.yaml."""

    tokenizer_name: str = "Qwen/Qwen3-8B"

    # Stream PAD/EPAD: predicted tokens inside the text stream. PAD is ~65% of
    # English dialogue text tokens and is down-weighted; EPAD is the speech-onset
    # trigger and is NOT down-weighted.
    text_pad_id: int | None = None
    text_epad_id: int | None = None

    # Batch padding: never a prediction target, fully loss-masked. Must differ
    # from text_pad_id -- conflating them makes "ignore this position" and
    # "predict a pause here" the same token.
    batch_pad_id: int | None = None

    # Audio-side initial/pad token. Moshi's audio embedding tables are
    # [CODEBOOK_SIZE + 1, dim]; the extra row carries this token, which fills the
    # head positions vacated by the delay shift.
    audio_init_id: int | None = None

    # A *bank* of real consecutive Mimi silence frames, shape (K, P). Derived by
    # encoding silence with the frozen codec -- see tools/derive_silence_codes.py.
    #
    # Not a single code per codebook: measured against kmhf/hf-moshiko, silence is
    # constant only in cb0 (the WavLM-distilled semantic VQ, share 0.943). The
    # seven acoustic RVQ codebooks encode the actual noise floor and never settle
    # -- modal shares run 0.24-0.55 -- and there is no short loop either: cb2 sits
    # on plateaus of 20-120 frames while cb4/cb6 alternate among ~3 values with
    # runs of 1-3. So neither "a constant" nor "a short cycle" describes it, and
    # the only faithful fill is a stretch of the real thing, tiled with a random
    # phase per sample (dataset.py) so the fill is not a fixed pattern the model
    # can latch onto as "this sample has no user".
    #
    # P == 1 degenerates to the old one-code-per-codebook contract, so nothing
    # that used to be expressible is lost.
    #
    # NOT defaulted to zeros: 0 is a perfectly valid Mimi code, so a wrong fill is
    # a maximally learnable spurious signal rather than benign noise.
    silence_bank: tuple[tuple[int, ...], ...] | None = None
    # Where to read the bank from when `silence_bank` is not given inline. The
    # bank is ~2k integers, which does not belong in a hand-edited yaml.
    silence_bank_path: str | None = None
    mimi_ckpt_id: str | None = None

    codebook_size: int = CODEBOOK_SIZE
    num_codebooks: int = NUM_CODEBOOKS

    def __post_init__(self) -> None:
        if self.silence_bank is None and self.silence_bank_path:
            self._load_silence_bank(self.silence_bank_path)
        if self.silence_bank is not None:
            self.silence_bank = _as_bank(self.silence_bank, TOKENS_YAML)
        # Mirror-of-schema constants: catch drift at load, not at step 40k.
        assert self.codebook_size == CODEBOOK_SIZE, (
            f"{TOKENS_YAML}: codebook_size={self.codebook_size} != schema "
            f"CODEBOOK_SIZE={CODEBOOK_SIZE}"
        )
        assert self.num_codebooks == NUM_CODEBOOKS, (
            f"{TOKENS_YAML}: num_codebooks={self.num_codebooks} != schema "
            f"NUM_CODEBOOKS={NUM_CODEBOOKS}"
        )

    def _load_silence_bank(self, path: str | Path) -> None:
        """Read the (K, P) bank out of a derive_silence_codes.py payload.

        The checkpoint cross-check lives here rather than at first use: a bank
        derived from a different codec than the corpus was baked with produces a
        user channel that is *plausible* silence for the wrong model, which is
        exactly the failure that never surfaces in a loss curve.
        """
        # Unlike TOKENS_YAML, this path is not supplied at the call site -- it comes
        # from inside the yaml -- so a caller launching from a subdirectory has no
        # way to fix a cwd-relative miss. Fall back to the repo root.
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            repo_root = Path(__file__).resolve().parents[2]
            if (repo_root / p).exists():
                p = repo_root / p
        assert p.exists(), (
            f"{TOKENS_YAML}: silence_bank_path={path!r} does not exist (cwd={Path.cwd()}). "
            f"Derive it:\n"
            f"    uv run python -m project_amnesty.tools.derive_silence_codes"
        )
        payload = json.loads(p.read_text())
        bank = payload.get("silence_bank")
        assert bank is not None, (
            f"{p}: no 'silence_bank' key. This looks like the pre-bank "
            f"(one-code-per-codebook) format; re-run derive_silence_codes.py."
        )
        self.silence_bank = _as_bank(bank, str(p))

        derived_ckpt = payload.get("mimi_ckpt_id")
        if derived_ckpt is not None and self.mimi_ckpt_id is not None:
            assert derived_ckpt == self.mimi_ckpt_id, (
                f"codec mismatch: {p} was derived from {derived_ckpt!r} but "
                f"{TOKENS_YAML} says the corpus is baked with {self.mimi_ckpt_id!r}. "
                f"Re-derive against the corpus codec rather than editing one to match."
            )
        elif self.mimi_ckpt_id is None:
            self.mimi_ckpt_id = derived_ckpt

    @classmethod
    def from_yaml(cls, path: str | Path = TOKENS_YAML) -> "TokenConfig":
        return cls(**_load_yaml(path))

    def require(self, *fields: str) -> None:
        """Assert the named ids are set, then check their mutual consistency."""
        for f in fields:
            assert getattr(self, f) is not None, (
                f"{f} is unset ({TOKENS_YAML}) -- finalize the PAD/EPAD assignment "
                f"before running this. A silent default here is a silent bug."
            )
        if self.text_pad_id is not None and self.batch_pad_id is not None:
            assert self.text_pad_id != self.batch_pad_id, (
                f"{TOKENS_YAML}: stream PAD and batch pad must differ. Stream PAD is a "
                f"prediction target (down-weighted x0.3); batch pad is fully loss-masked."
            )
        if self.text_pad_id is not None and self.text_epad_id is not None:
            assert self.text_pad_id != self.text_epad_id, f"{TOKENS_YAML}: PAD == EPAD"
        if self.silence_bank is not None:
            assert len(self.silence_bank) == self.num_codebooks, (
                f"{TOKENS_YAML}: silence_bank has {len(self.silence_bank)} codebooks, "
                f"expected {self.num_codebooks}"
            )
            for k, row in enumerate(self.silence_bank):
                assert all(0 <= c < self.codebook_size for c in row), (
                    f"{TOKENS_YAML}: silence_bank codebook {k} out of range "
                    f"[0, {self.codebook_size})"
                )

    def silence_bank_array(self) -> np.ndarray:
        """(K, P) int16 silence frames. Call require('silence_bank') first."""
        assert self.silence_bank is not None
        return np.asarray(self.silence_bank, dtype=np.int16)


@dataclass
class DataConfig:
    """Dataset-layer config: what __getitem__ needs and nothing more."""

    root: str = "data/prepared"
    tokens: TokenConfig = field(default_factory=TokenConfig)

    # Crop. Generation caps at 1500 frames (120 s); training on all of it with
    # K=8 soft targets is not memory-viable at a useful batch size.
    max_frames: int = 750
    crop_mode: str = "random"       # random | center | head  (probe forces center)
    max_text_len: int = 2048        # text_anchor's unaligned token window

    # A/B index-doubling. Deterministic, so "2x data" means every dialogue is
    # seen in both directions exactly once per epoch, independent of worker count.
    double_ab: bool = True

    # --- Zone B voice prompt (ARCHITECTURE section 7.2/7.4) -------------------
    # Master switch. False makes every item carry empty_ref(), which is the
    # ablation arm against "reference audio in context" and also the escape hatch
    # for a corpus whose `speaker` column is empty (SCHEMA_VERSION < 2).
    voice_prompt: bool = True
    # Frames of reference audio prepended as Zone B. 48 frames = 3.84 s at 12.5 Hz,
    # matching REF_FRAMES in project_amnesty/datasets/scenarios/prep_data.py -- the
    # one voice-cloning setup in this repo that was actually run end to end.
    # A *cap*: a shorter reference row is used whole. Not free to raise -- at 100
    # (8 s) Zone B was as long as Zone C itself on the real Zeroth corpus, so the
    # prompt ate half the context budget.
    voice_prompt_frames: int = 48

    seed: int = 0
    debug_validate: bool = False    # run validate_item() on every __getitem__

    def __post_init__(self) -> None:
        if isinstance(self.tokens, dict):
            self.tokens = TokenConfig(**self.tokens)
        assert self.crop_mode in ("random", "center", "head"), (
            f"crop_mode must be random|center|head, got {self.crop_mode!r}"
        )
        assert self.max_frames > 0 and self.max_text_len > 0
        assert self.voice_prompt_frames > 0, (
            f"voice_prompt_frames must be > 0, got {self.voice_prompt_frames}. "
            f"Use voice_prompt=False to disable Zone B rather than a zero-length prompt."
        )
        # The dataset layer fills the absent user channel and pads text_other,
        # so it needs these two ids up front.
        self.tokens.require("silence_bank", "text_pad_id")

    @classmethod
    def from_yaml(cls, path: str | Path, *, tokens_path: str | Path = TOKENS_YAML) -> "DataConfig":
        cfg = _load_yaml(path)
        cfg.setdefault("tokens", _load_yaml(tokens_path))
        return cls(**cfg)
