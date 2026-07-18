"""Unified sample schema.

Three sample_types share a single Arrow schema:
  - "en_kd":       dual-Moshi self-talk generated dialogues. Both A/B streams +
                   teacher top-k logits for both streams.
  - "ko_tts":      Korean singleton TTS (Zeroth/KSS/CSS10/CommonVoice). Only codes_a
                   used (mono), ground-truth CE.
  - "text_anchor": plain text. Only text_tokens_a used (not frame-aligned; plain
                   token sequence).

Design decisions:
  * A/B are symmetric. "Who is the user" is not stored; the collator decides
    (used in both directions → 2x data).
  * Teacher logits are the top-k of the raw logits right before sampling. Temperature
    is a loss hyperparameter, so it is not baked in here.
  * The empty user channel (ko_tts) is not stored. The collator fills it with silence codes.
  * PAD/EPAD ids are not hardcoded in the schema (undetermined on the model side)
    → injected via configs/.
"""

from dataclasses import dataclass, field

import numpy as np
from datasets import Features, Sequence, Value

# ---- constants (fixed by Mimi; Mimi is frozen and never changes) ----
FRAME_RATE_HZ = 12.5
NUM_CODEBOOKS = 8          # K
CODEBOOK_SIZE = 2048
SAMPLE_RATE = 24_000

SAMPLE_TYPES = ("en_kd", "ko_tts", "text_anchor")


def arrow_features() -> Features:
    """Arrow Features used with save_to_disk.

    To support variable length T, code arrays are stored flat 1D and (K, T) is
    restored via num_frames. (Multi-dimensional Sequence in `datasets` supports
    fixed lengths only, hence the flat + reshape convention.) The schema is
    independent of topk — the per-row "topk" field carries the actual value.
    """
    return Features(
        {
            "sample_type": Value("string"),
            "lang": Value("string"),                      # "en" | "ko"
            "num_frames": Value("int32"),                 # T
            # audio codes (flat: K*T, row-major [K, T])
            "codes_a": Sequence(Value("int16")),
            "codes_b": Sequence(Value("int16")),          # empty list for ko_tts/text_anchor
            # frame-aligned text stream (T,) / for text_anchor: unaligned token sequence
            "text_tokens_a": Sequence(Value("int32")),
            "text_tokens_b": Sequence(Value("int32")),
            # teacher top-k (flat: K*T*topk). Filled only for en_kd.
            "teacher_topk_val_a": Sequence(Value("float32")),  # cast to fp16 after loading
            "teacher_topk_idx_a": Sequence(Value("int16")),
            "teacher_topk_val_b": Sequence(Value("float32")),
            "teacher_topk_idx_b": Sequence(Value("int16")),
            "topk": Value("int16"),
            # generation metadata for reproducibility/filtering/analysis
            "gen_meta": {
                "seed": Value("int64"),
                "gen_temperature": Value("float32"),
                "gen_top_k": Value("int32"),
                "seed_prompt_id": Value("string"),
            },
            # stable hash for hold-out split (based on source file path or generation seed)
            "sample_uid": Value("string"),
        }
    )


@dataclass
class Sample:
    """In-memory representation used inside the pipeline. Convert to an Arrow row with to_row()."""

    sample_type: str
    lang: str
    codes_a: np.ndarray                     # (K, T) int16
    text_tokens_a: np.ndarray               # (T,) or (L,) int32
    codes_b: np.ndarray | None = None       # (K, T) int16
    text_tokens_b: np.ndarray | None = None
    teacher_topk_val_a: np.ndarray | None = None  # (K, T, topk) float
    teacher_topk_idx_a: np.ndarray | None = None  # (K, T, topk) int16
    teacher_topk_val_b: np.ndarray | None = None
    teacher_topk_idx_b: np.ndarray | None = None
    gen_meta: dict = field(default_factory=dict)
    sample_uid: str = ""

    @property
    def num_frames(self) -> int:
        return int(self.codes_a.shape[-1]) if self.codes_a.size else 0

    def to_row(self) -> dict:
        assert self.sample_type in SAMPLE_TYPES
        topk = 0
        if self.teacher_topk_val_a is not None:
            topk = int(self.teacher_topk_val_a.shape[-1])

        def flat(x, dtype):
            return [] if x is None else np.ascontiguousarray(x, dtype=dtype).ravel().tolist()

        return {
            "sample_type": self.sample_type,
            "lang": self.lang,
            "num_frames": self.num_frames,
            "codes_a": flat(self.codes_a, np.int16),
            "codes_b": flat(self.codes_b, np.int16),
            "text_tokens_a": flat(self.text_tokens_a, np.int32),
            "text_tokens_b": flat(self.text_tokens_b, np.int32),
            "teacher_topk_val_a": flat(self.teacher_topk_val_a, np.float32),
            "teacher_topk_idx_a": flat(self.teacher_topk_idx_a, np.int16),
            "teacher_topk_val_b": flat(self.teacher_topk_val_b, np.float32),
            "teacher_topk_idx_b": flat(self.teacher_topk_idx_b, np.int16),
            "topk": topk,
            "gen_meta": {
                "seed": int(self.gen_meta.get("seed", -1)),
                "gen_temperature": float(self.gen_meta.get("gen_temperature", 0.0)),
                "gen_top_k": int(self.gen_meta.get("gen_top_k", 0)),
                "seed_prompt_id": str(self.gen_meta.get("seed_prompt_id", "")),
            },
            "sample_uid": self.sample_uid,
        }


def row_to_arrays(row: dict) -> dict:
    """Arrow row → restore original shapes such as (K, T). Used by training/data/dataset.py."""
    T = int(row["num_frames"])
    K = NUM_CODEBOOKS
    topk = int(row["topk"])

    def resh(x, shape, dtype):
        arr = np.asarray(x, dtype=dtype)
        return arr.reshape(shape) if arr.size else None

    out = {
        "sample_type": row["sample_type"],
        "lang": row["lang"],
        "num_frames": T,
        "codes_a": resh(row["codes_a"], (K, T), np.int16),
        "codes_b": resh(row["codes_b"], (K, T), np.int16),
        "text_tokens_a": np.asarray(row["text_tokens_a"], dtype=np.int32),
        "text_tokens_b": np.asarray(row["text_tokens_b"], dtype=np.int32),
        "gen_meta": row.get("gen_meta", {}),
        "sample_uid": row.get("sample_uid", ""),
    }
    if topk > 0:
        for side in ("a", "b"):
            out[f"teacher_topk_val_{side}"] = resh(
                row[f"teacher_topk_val_{side}"], (K, T, topk), np.float16
            )
            out[f"teacher_topk_idx_{side}"] = resh(
                row[f"teacher_topk_idx_{side}"], (K, T, topk), np.int16
            )
    else:
        for side in ("a", "b"):
            out[f"teacher_topk_val_{side}"] = None
            out[f"teacher_topk_idx_{side}"] = None
    return out
