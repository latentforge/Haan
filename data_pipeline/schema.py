"""Unified sample schema.

Three sample_types share a single Arrow schema:
  - "en_kd":       dual-Moshi self-talk generated dialogues. Both A/B streams +
                   teacher top-k logits for both streams.
  - "ko_tts":      Korean singleton TTS (Zeroth/KSS/CommonVoice). Only codes_a
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

# ---- prepared-artifact schema version ----
# Bump on ANY change to arrow_features(): added/removed/retyped field, or a change
# to the flat-store convention. training/datasets/prepare.py stamps this into each
# group's _SUCCESS.json sentinel and rebuilds automatically when it no longer
# matches, so stale prepared data can never be read back through a newer schema.
# (Planned bump: adding `zone_a_frames` per plan section 3.7.)
#
# v2: added `speaker`. Voice-prompt conditioning needs the reference audio for a
#     ko_tts sample to be *the same speaker's other utterance* -- i.e. another row
#     of the same dataset -- so no reference codes are stored, but the training
#     Dataset has to be able to find same-speaker rows. Every dataset already set
#     RawEntry.speaker and base.py wrote it into the per-sample .json; it was then
#     dropped at the Arrow boundary. Corpora prepared before this bump have no
#     speaker column and must be rebuilt rather than read back with an empty one.
#
# v3: teacher_topk_val_* retyped float32 -> float16, per plan section 2.8. The
#     generator already dumps fp16 and the Dataset already casts back to fp16 on
#     load, so f32 on disk was pure round-trip cost on the single largest tensor
#     in the corpus (K*T*topk per stream) with no precision to show for it.
#     pyarrow halffloat still reads zero-copy, so the fast path is unaffected.
SCHEMA_VERSION = 3


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
            "teacher_topk_val_a": Sequence(Value("float16")),
            "teacher_topk_idx_a": Sequence(Value("int16")),
            "teacher_topk_val_b": Sequence(Value("float16")),
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
            # speaker identity, source-local (only comparable within one dataset).
            # "" when the source genuinely has no notion of a speaker (text_anchor)
            # or when a row carries two of them (en_kd holds both A and B).
            # Used to pick a same-speaker row as the voice prompt.
            "speaker": Value("string"),
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
    speaker: str = ""                       # source-local speaker id; "" when unknown

    @property
    def num_frames(self) -> int:
        return int(self.codes_a.shape[-1]) if self.codes_a.size else 0

    def to_row(self) -> dict:
        assert self.sample_type in SAMPLE_TYPES
        topk = 0
        if self.teacher_topk_val_a is not None:
            topk = int(self.teacher_topk_val_a.shape[-1])

        def flat(x, dtype):
            """Flat 1D **numpy**, not a Python list.

            `.tolist()` here cost ~18x: a 2-byte int16 becomes a 28-byte Python
            int, and pyarrow immediately re-packs it back into the same 2 bytes.
            Invisible at 150 rows, fatal at corpus scale -- the teacher top-k is
            (K, T, topk) per stream, so 10k en_kd dialogues is ~30 GB as arrays
            and ~430 GB as lists. pyarrow accepts numpy for Sequence(Value(...))
            directly, and row_to_arrays already goes through np.asarray on read.
            """
            return (
                np.empty(0, dtype=dtype) if x is None
                else np.ascontiguousarray(x, dtype=dtype).ravel()
            )

        return {
            "sample_type": self.sample_type,
            "lang": self.lang,
            "num_frames": self.num_frames,
            "codes_a": flat(self.codes_a, np.int16),
            "codes_b": flat(self.codes_b, np.int16),
            "text_tokens_a": flat(self.text_tokens_a, np.int32),
            "text_tokens_b": flat(self.text_tokens_b, np.int32),
            "teacher_topk_val_a": flat(self.teacher_topk_val_a, np.float16),
            "teacher_topk_idx_a": flat(self.teacher_topk_idx_a, np.int16),
            "teacher_topk_val_b": flat(self.teacher_topk_val_b, np.float16),
            "teacher_topk_idx_b": flat(self.teacher_topk_idx_b, np.int16),
            "topk": topk,
            "gen_meta": {
                "seed": int(self.gen_meta.get("seed", -1)),
                "gen_temperature": float(self.gen_meta.get("gen_temperature", 0.0)),
                "gen_top_k": int(self.gen_meta.get("gen_top_k", 0)),
                "seed_prompt_id": str(self.gen_meta.get("seed_prompt_id", "")),
            },
            "sample_uid": self.sample_uid,
            "speaker": self.speaker,
        }


def row_to_arrays(row: dict) -> dict:
    """Arrow row → restore original shapes such as (K, T). Used by training/datasets/dataset.py."""
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
        "speaker": row.get("speaker", "") or "",
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
