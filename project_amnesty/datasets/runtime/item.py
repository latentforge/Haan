"""The per-sample contract between MoshiKDDataset and the collators.

Single source of truth for what __getitem__ returns. No I/O lives here so that
both the dataset and the collator can depend on it without a cycle.

Every tensor is CPU, unbatched, **delay-free**, and **unpadded**:
  * delay is a model hyperparameter applied by the collator (storage is
    delay-free by design; applying it twice is the bug project_amnesty's
    en_kd ingest was written to prevent)
  * padding needs the batch max-T, which a single item cannot know

Roles are already resolved. The A/B axis of the Arrow schema is symmetric and
"who is the user" is decided by index-doubling in the dataset, so downstream
code sees `self`/`other` and must never re-swap.
"""

from __future__ import annotations

from typing import TypedDict

import torch

from project_amnesty.datasets.shared.schema import NUM_CODEBOOKS

# Sample types, mirroring project_amnesty.datasets.shared.schema.SAMPLE_TYPES. Note this is the
# *shape* contract, not the mixing group: en_solo rows carry sample_type
# "ko_tts" while their source is "en_solo".
SAMPLE_TYPE_IDS: dict[str, int] = {"en_kd": 0, "ko_tts": 1, "text_anchor": 2}
LANG_IDS: dict[str, int] = {"en": 0, "ko": 1, "ja": 2}


class KDSample(TypedDict):
    """One training sample. See module docstring for the invariants."""

    # --- identity / routing (python scalars: cheap, and str cannot be a tensor) ---
    sample_uid: str
    # ko_asr is the ASR direction of the ko_tts rows -- same rows, same shapes,
    # only the collator's delay differs (loader.GROUP_ALIASES).
    source: str        # mixing group: en_kd | en_solo | ko_tts | ko_asr | text_anchor
    sample_type: str   # shape contract: en_kd | ko_tts | text_anchor
    lang: str          # en | ko | ja
    # Speaker identity, "" when unknown (en_kd rows carry two voices, so no single
    # id describes them). This is the one thing a voice prompt needs that cannot be
    # recomputed at load time: the reference *audio* is just another row from the
    # same speaker, so it is never stored, but knowing WHICH rows share a voice is
    # not derivable from the codes without a speaker model.
    speaker: str

    # --- per-sample flags (0-dim tensors so default_collate stacks them) ---
    swapped: torch.Tensor       # bool ()   A/B were exchanged (logging/repro only)
    is_text_only: torch.Tensor  # bool ()   text_anchor
    has_teacher: torch.Tensor   # bool ()   teacher top-k present
    num_frames: torch.Tensor    # int32 ()  == T (0 for text_anchor)
    topk: torch.Tensor          # int32 ()  0 when has_teacher is False

    # --- frame-aligned streams, canonical roles ---
    codes_self: torch.Tensor    # int16 (K, T)
    codes_other: torch.Tensor   # int16 (K, T)  silence-filled when absent
    text_self: torch.Tensor     # int32 (T,)
    text_other: torch.Tensor    # int32 (T,)    PAD-filled when absent

    # --- unaligned text (text_anchor only; (0,) elsewhere) ---
    text_flat: torch.Tensor     # int32 (L,)

    # --- Zone B voice prompt: a reference utterance by the SAME speaker ---
    # Not stored anywhere: the reference is just another row, so the dataset looks
    # one up by `speaker` at __getitem__ time. Varying it per epoch is what Phase 2
    # means by "per-sample varied reference voices"; freezing it into the corpus
    # would both bloat storage and remove the augmentation.
    # Empty (K,0)/(0,) when no reference is available (unknown speaker, singleton
    # speaker, or voice prompting disabled).
    ref_codes: torch.Tensor     # int16 (K, R)  reference audio, agent channel
    ref_text: torch.Tensor      # int32 (R,)    its frame-aligned transcript
    has_ref: torch.Tensor       # bool  ()

    # --- KD soft labels, self side only (en_kd only; (0,0,0) elsewhere) ---
    teacher_val: torch.Tensor   # fp16  (K, T, topk)  RAW logits, no temperature
    teacher_idx: torch.Tensor   # int16 (K, T, topk)

    # --- supervision routing ---
    use_kd: torch.Tensor        # bool ()
    use_ce_audio: torch.Tensor  # bool ()
    use_ce_text: torch.Tensor   # bool ()


ITEM_KEYS: tuple[str, ...] = tuple(KDSample.__annotations__)

_STR_KEYS = ("sample_uid", "source", "sample_type", "lang", "speaker")
_BOOL_KEYS = ("swapped", "is_text_only", "has_teacher", "has_ref",
              "use_kd", "use_ce_audio", "use_ce_text")
_I32_SCALARS = ("num_frames", "topk")


def empty_like_streams(K: int = NUM_CODEBOOKS) -> dict[str, torch.Tensor]:
    """Correctly-typed empty frame tensors, for text_anchor's T=0 degenerate case.

    The alternative -- fabricating L frames of silence so the shapes look uniform --
    was rejected: it invents 12.5 Hz timing for text that has none and feeds
    fabricated audio into the CE loss.
    """
    return {
        "codes_self": torch.empty((K, 0), dtype=torch.int16),
        "codes_other": torch.empty((K, 0), dtype=torch.int16),
        "text_self": torch.empty((0,), dtype=torch.int32),
        "text_other": torch.empty((0,), dtype=torch.int32),
        "teacher_val": torch.empty((0, 0, 0), dtype=torch.float16),
        "teacher_idx": torch.empty((0, 0, 0), dtype=torch.int16),
    }


def empty_ref(K: int = NUM_CODEBOOKS) -> dict[str, torch.Tensor]:
    """No voice prompt: unknown speaker, singleton speaker, or prompting disabled."""
    return {
        "ref_codes": torch.empty((K, 0), dtype=torch.int16),
        "ref_text": torch.empty((0,), dtype=torch.int32),
        "has_ref": torch.zeros((), dtype=torch.bool),
    }


def validate_item(item: KDSample, *, K: int = NUM_CODEBOOKS) -> None:
    """Assert the full contract. Debug/test only -- not on the hot path."""
    missing = set(ITEM_KEYS) - set(item)
    extra = set(item) - set(ITEM_KEYS)
    assert not missing, f"KDSample missing keys: {sorted(missing)}"
    assert not extra, f"KDSample has unexpected keys: {sorted(extra)}"

    for k in _STR_KEYS:
        assert isinstance(item[k], str), f"{k} must be str, got {type(item[k])}"
    assert item["sample_type"] in SAMPLE_TYPE_IDS, f"bad sample_type {item['sample_type']!r}"
    assert item["lang"] in LANG_IDS, f"bad lang {item['lang']!r}"

    for k in _BOOL_KEYS:
        t = item[k]
        assert t.dtype is torch.bool and t.ndim == 0, f"{k} must be a 0-dim bool tensor"
    for k in _I32_SCALARS:
        t = item[k]
        assert t.dtype is torch.int32 and t.ndim == 0, f"{k} must be a 0-dim int32 tensor"

    T = int(item["num_frames"])
    topk = int(item["topk"])

    assert item["codes_self"].dtype is torch.int16
    assert item["codes_other"].dtype is torch.int16
    assert item["codes_self"].shape == (K, T), f"codes_self {tuple(item['codes_self'].shape)} != {(K, T)}"
    assert item["codes_other"].shape == (K, T), f"codes_other {tuple(item['codes_other'].shape)} != {(K, T)}"
    assert item["text_self"].dtype is torch.int32 and item["text_self"].shape == (T,)
    assert item["text_other"].dtype is torch.int32 and item["text_other"].shape == (T,)
    assert item["text_flat"].dtype is torch.int32 and item["text_flat"].ndim == 1

    assert item["ref_codes"].dtype is torch.int16 and item["ref_codes"].ndim == 2
    assert item["ref_text"].dtype is torch.int32 and item["ref_text"].ndim == 1
    R = int(item["ref_codes"].shape[1])
    assert item["ref_codes"].shape[0] == K, f"ref_codes must be (K, R), got {tuple(item['ref_codes'].shape)}"
    assert item["ref_text"].shape == (R,), "ref_text must be frame-aligned to ref_codes"
    assert bool(item["has_ref"]) == (R > 0), "has_ref must agree with the reference length"

    assert item["teacher_val"].dtype is torch.float16
    assert item["teacher_idx"].dtype is torch.int16
    if bool(item["has_teacher"]):
        assert topk > 0, "has_teacher but topk == 0"
        assert item["teacher_val"].shape == (K, T, topk)
        assert item["teacher_idx"].shape == (K, T, topk)
        # Contiguity is the observable proof that cropping actually materialized a
        # window instead of retaining a view onto the whole mmapped row.
        assert item["teacher_val"].is_contiguous(), "teacher_val must be contiguous"
    else:
        assert topk == 0, "topk must be 0 when has_teacher is False"
        assert item["teacher_val"].numel() == 0
        assert item["teacher_idx"].numel() == 0

    if bool(item["is_text_only"]):
        assert T == 0, "text_anchor must have T == 0"
        assert item["text_flat"].numel() > 0, "text_anchor must carry text_flat"
        assert not bool(item["use_kd"]) and not bool(item["use_ce_audio"])
    else:
        assert T > 0, "frame samples must have T > 0"
        assert item["text_flat"].numel() == 0, "text_flat is text_anchor-only"
