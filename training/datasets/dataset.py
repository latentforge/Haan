"""MoshiKDDataset -- prepared Arrow rows -> KDSample.

One instance covers exactly one (source, split). It knows nothing about mixing
ratios, batch composition, or delay; `__getitem__` is a pure function of the
index (plus the epoch set by `set_epoch`).

Three things in here are load-bearing and easy to get subtly wrong:

1. **A/B direction is index-doubling, not a collator coin flip.** len() doubles
   and `_resolve` interleaves, so every dialogue is seen in both directions
   exactly once per epoch regardless of num_workers.
2. **On swap, only the self-side teacher survives.** KD is a loss on the
   modeled speaker's codebook-0 logits; the other side is input, not target.
   Carrying the A-side teacher through a swap does not crash -- it just quietly
   trains against the wrong speaker.
3. **Crop before materializing teacher top-k**, and never touch `hf_ds[i]` or
   `row_to_arrays` on the hot path: both materialize every column (both
   teachers) into Python objects.

`row_to_arrays` stays the correctness oracle for the tests.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from data_pipeline.schema import NUM_CODEBOOKS
from training.datasets.config import DataConfig
from training.datasets.crop import Window, apply_window_kt, apply_window_ktk, apply_window_t, choose_window
from training.datasets.item import KDSample, empty_ref, empty_like_streams, validate_item

_SOURCES = ("en_kd", "en_solo", "ko_tts", "text_anchor")

# Columns read through the zero-copy list path. Scalars are read per-row off the
# ChunkedArray, which is already cheap.
_LIST_COLUMNS = (
    "codes_a",
    "codes_b",
    "text_tokens_a",
    "text_tokens_b",
    "teacher_topk_val_a",
    "teacher_topk_idx_a",
    "teacher_topk_val_b",
    "teacher_topk_idx_b",
)


def _bool(x: bool) -> torch.Tensor:
    return torch.tensor(bool(x), dtype=torch.bool)


def _i32(x: int) -> torch.Tensor:
    return torch.tensor(int(x), dtype=torch.int32)


class _ListColumn:
    """Zero-copy row accessor for a pyarrow list<primitive> column.

    Holds one flat numpy view per chunk plus that chunk's offsets, so a row read
    is two integer lookups and a slice -- no pyarrow scalar objects, no Python
    lists, no copy until the caller crops and calls ascontiguousarray.

    Verified against pyarrow 25.0.0 / datasets 5.0.0: `ListArray.values` returns
    the *full* child array while `ListArray.offsets` is already slice-adjusted,
    so `values[offsets[j]:offsets[j+1]]` is correct even for a sliced chunk.
    `to_numpy(zero_copy_only=True)` is passed explicitly so that a schema change
    (nullable child, a type needing conversion) fails loudly here instead of
    silently reverting to a full copy of every row.
    """

    def __init__(self, chunked, name: str):
        self.name = name
        self._flat: list[np.ndarray] = []
        self._offsets: list[np.ndarray] = []
        bounds = [0]
        for chunk in chunked.chunks:
            if len(chunk) == 0:
                self._flat.append(np.empty(0, dtype=np.int8))
                self._offsets.append(np.zeros(1, dtype=np.int64))
            else:
                self._flat.append(self._child_numpy(chunk, name))
                self._offsets.append(chunk.offsets.to_numpy(zero_copy_only=False))
            bounds.append(bounds[-1] + len(chunk))
        self._bounds = np.asarray(bounds, dtype=np.int64)

    @staticmethod
    def _child_numpy(chunk, name: str) -> np.ndarray:
        try:
            return chunk.values.to_numpy(zero_copy_only=True)
        except Exception as exc:  # pragma: no cover - schema drift guard
            # Deliberately not silent: the whole point of zero_copy_only=True is
            # that a copy here would reintroduce the per-row materialization cost
            # the fast path exists to avoid. We still return a correct array --
            # correctness beats the optimization -- but say so once, loudly.
            warnings.warn(
                f"zero-copy read unavailable for column {name!r} ({type(exc).__name__}: {exc}); "
                f"falling back to a copying read. Check for nulls or a changed child type "
                f"in data_pipeline.schema.arrow_features().",
                RuntimeWarning,
                stacklevel=2,
            )
            return chunk.values.to_numpy(zero_copy_only=False)

    def __call__(self, row: int) -> np.ndarray:
        ci = int(np.searchsorted(self._bounds, row, side="right")) - 1
        j = row - int(self._bounds[ci])
        off = self._offsets[ci]
        return self._flat[ci][off[j] : off[j + 1]]


class MoshiKDDataset(Dataset):
    """One (source, split) -> KDSample. Ratios and mixing live in the sampler."""

    def __init__(
        self,
        root: str | Path,
        source: str,
        split: str = "train",
        *,
        cfg: DataConfig,
        double_ab: bool | None = None,
        seed: int = 0,
        data_dir: str | None = None,
    ) -> None:
        self.root = Path(root)
        # `source` is the *mixing group* name; `data_dir` is where the rows live.
        # They differ only for bidirectional reuse (loader.GROUP_ALIASES): ko_asr
        # is the same prepared ko_tts rows read a second time, so item["source"]
        # must say "ko_asr" (that is what the collator picks the delay from) while
        # the reader still opens data/prepared/ko_tts.
        self.data_dir = source if data_dir is None else str(data_dir)
        self.source = source
        self.split = split
        self.cfg = cfg
        self.path = self.root / self.data_dir / split

        self._seed = int(seed)
        self._epoch = 0
        self._hf_ds = None
        self._hf_pid: int | None = None
        self._cols: dict[str, _ListColumn] | None = None
        self._cols_pid: int | None = None
        self._spk: dict[str, list[int]] | None = None
        self._spk_pid: int | None = None

        assert self.path.exists(), (
            f"prepared data missing at {self.path}. Build it first:\n"
            f"    python -m training.datasets.prepare --group {self.data_dir}"
        )

        # Open once to learn len() and the shape contract, then drop the handle so
        # nothing can accidentally capture an Arrow table in a pickle or a fork.
        hf = self._hf()
        self._n_rows = len(hf)
        assert self._n_rows > 0, f"{self.path} is empty"
        table = self._table()
        self.sample_type = str(table.column("sample_type")[0].as_py())
        self.lang = str(table.column("lang")[0].as_py())
        self._hf_ds = None
        self._hf_pid = None
        self._cols = None
        self._cols_pid = None
        self._spk = None
        self._spk_pid = None

        # source != sample_type. en_solo rows carry sample_type "ko_tts"; behavior
        # branches on sample_type (the shape contract), mixing/logging on source.
        assert self.sample_type in ("en_kd", "ko_tts", "text_anchor"), (
            f"unknown sample_type {self.sample_type!r} in {self.path}"
        )

        # probe must not be stochastic: a random crop makes the eval metric move
        # for reasons that have nothing to do with the checkpoint. Hard-forced
        # here rather than trusted from config.
        self.crop_mode = "center" if split == "probe" else cfg.crop_mode

        want = cfg.double_ab if double_ab is None else bool(double_ab)
        # Only en_kd has a B stream; doubling anything else would just serve every
        # row twice with swapped=True and no swap actually performed.
        self.double_ab = bool(want) and self.sample_type == "en_kd"

    # --- lazy, pid-keyed Arrow access ----------------------------------------

    def _hf(self):
        pid = os.getpid()
        if self._hf_ds is None or self._hf_pid != pid:
            from datasets import load_from_disk

            self._hf_ds = load_from_disk(str(self.path))
            self._hf_pid = pid
            self._cols = None
            self._cols_pid = None
            self._spk = None
            self._spk_pid = None
        return self._hf_ds

    def _table(self):
        data = self._hf().data
        # datasets wraps pyarrow in MemoryMappedTable/InMemoryTable.
        return getattr(data, "table", data)

    def _columns(self) -> dict[str, _ListColumn]:
        pid = os.getpid()
        if self._cols is None or self._cols_pid != pid:
            hf = self._hf()
            # A shuffled/selected split would make row i of the table != row i of
            # the dataset. We load straight from disk, so this must hold.
            assert getattr(hf, "_indices", None) is None, (
                f"{self.path} carries an indices mapping; the column fast path assumes "
                f"identity row order"
            )
            table = self._table()
            self._cols = {n: _ListColumn(table.column(n), n) for n in _LIST_COLUMNS}
            self._cols_pid = pid
        return self._cols

    def _speaker_index(self) -> dict[str, list[int]]:
        """speaker -> ascending row indices, built lazily per worker.

        Reads the `speaker` column and *nothing else*. It is a small string
        column (147 rows here, ~1e6 at corpus scale); the list columns next to it
        carry megabytes of codes and teacher logits per row, so touching
        `_columns()` -- or worse `hf_ds[i]` -- to learn who is speaking would pay
        the entire materialization cost the fast path exists to avoid.

        Rows whose speaker is "" are excluded rather than grouped under a shared
        "unknown" key: they are not the same voice, and pooling them would hand
        the model a reference from an arbitrary stranger.
        """
        pid = os.getpid()
        if self._spk is None or self._spk_pid != pid:
            table = self._table()
            index: dict[str, list[int]] = {}
            if "speaker" in table.column_names:
                for row, spk in enumerate(table.column("speaker").to_pylist()):
                    if spk:
                        index.setdefault(str(spk), []).append(row)
            self._spk = index
            self._spk_pid = pid
        return self._spk

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_hf_ds"] = None
        state["_hf_pid"] = None
        state["_cols"] = None
        state["_cols_pid"] = None
        state["_spk"] = None
        state["_spk_pid"] = None
        return state

    # --- index space ----------------------------------------------------------

    def __len__(self) -> int:
        return self._n_rows * (2 if self.double_ab else 1)

    def _resolve(self, index: int) -> tuple[int, bool]:
        """Doubled index -> (row, swapped). Interleaved, so consecutive indices are
        the two directions of the same dialogue."""
        if not self.double_ab:
            return index, False
        return index // 2, bool(index % 2)

    def _rng_for(self, index: int) -> np.random.Generator:
        """Per-(epoch, index) generator. Never mutates shared or global state, so
        the crop for sample i at epoch e is identical for any worker count,
        batch size, or call order."""
        return np.random.default_rng(
            np.random.SeedSequence(entropy=self._seed, spawn_key=(self._epoch, index))
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if get_worker_info() is not None:
            warnings.warn(
                "set_epoch() called inside a DataLoader worker: the main-process copy of "
                "this dataset is unaffected. With persistent_workers=True the workers keep "
                "the _epoch they were forked with, so every epoch replays the same crops -- "
                "a failure invisible in the loss curve. Use persistent_workers=False and "
                "rebuild the loader after set_epoch().",
                RuntimeWarning,
                stacklevel=2,
            )

    # --- item construction ----------------------------------------------------

    def __getitem__(self, index: int) -> KDSample:
        n = len(self)
        if index < 0:
            index += n
        assert 0 <= index < n, f"index {index} out of range for {n} items"

        row, swapped = self._resolve(index)
        rng = self._rng_for(index)

        if self.sample_type == "en_kd":
            item = self._normalize_dual(row, swapped, rng)
        elif self.sample_type == "ko_tts":
            item = self._normalize_solo(row, rng)
        else:
            item = self._normalize_text_only(row, rng)

        # Drawn LAST, from the same generator, so that adding Zone B cannot move
        # the crop window of the sample itself: every prior draw has already been
        # consumed by choose_window (and the solo silence phase). Flipping
        # cfg.voice_prompt therefore changes only ref_*, which is what makes the
        # prompted / unprompted ablation a controlled comparison instead of two
        # different croppings of the corpus.
        item.update(self._pick_ref(item, row, rng))

        if self.cfg.debug_validate:
            validate_item(item)
        return item

    def _meta(self, row: int) -> tuple[str, str, str]:
        table = self._table()
        uid = str(table.column("sample_uid")[row].as_py())
        lang = str(table.column("lang")[row].as_py())
        # Additive column (SCHEMA_VERSION 2). Tolerate older prepared data rather
        # than crash: "" is the same thing an en_kd row legitimately carries.
        speaker = (str(table.column("speaker")[row].as_py() or "")
                   if "speaker" in table.column_names else "")
        return uid, lang, speaker

    def _shape_meta(self, row: int) -> tuple[int, int]:
        table = self._table()
        T = int(table.column("num_frames")[row].as_py())
        topk = int(table.column("topk")[row].as_py())
        return T, topk

    def _normalize_dual(self, row: int, swapped: bool, rng: np.random.Generator) -> KDSample:
        K = NUM_CODEBOOKS
        col = self._columns()
        uid, lang, speaker = self._meta(row)
        T, topk = self._shape_meta(row)
        assert T > 0, f"{uid}: en_kd row with num_frames=0"

        # self side first: on a swap the B stream becomes the modeled speaker.
        lo, hi = ("b", "a") if swapped else ("a", "b")

        w = choose_window(T, self.cfg.max_frames, rng, self.crop_mode)

        codes_self = self._codes(col, lo, row, K, T, w, uid)
        codes_other = self._codes(col, hi, row, K, T, w, uid)
        text_self = self._text(col, lo, row, T, w, uid)
        text_other = self._text(col, hi, row, T, w, uid)

        has_teacher = topk > 0
        if has_teacher:
            # Only the self side survives. The `hi` teacher is never even reshaped:
            # dropping it here halves the per-sample IPC payload and is the one line
            # that keeps KD pointed at the speaker actually being modeled.
            val_flat = col[f"teacher_topk_val_{lo}"](row)
            idx_flat = col[f"teacher_topk_idx_{lo}"](row)
            assert val_flat.size == K * T * topk, (
                f"{uid}: teacher_topk_val_{lo} has {val_flat.size} elems, expected {K * T * topk}"
            )
            # dtype is pinned rather than inherited: storage is already fp16
            # (SCHEMA_VERSION 3), so this is a no-op copy on the cropped window,
            # but it keeps the item contract fp16 regardless of what the Arrow
            # column happens to hold.
            teacher_val = apply_window_ktk(val_flat.reshape(K, T, topk), w, dtype=np.float16)
            teacher_idx = apply_window_ktk(idx_flat.reshape(K, T, topk), w)
            if self.cfg.debug_validate:
                assert np.isfinite(teacher_val).all(), (
                    f"{uid}: non-finite teacher logits -- corrupted upstream dump. "
                    f"Unchecked, this turns the KD loss into a silent NaN."
                )
        else:
            teacher_val = np.empty((0, 0, 0), dtype=np.float16)
            teacher_idx = np.empty((0, 0, 0), dtype=np.int16)
            topk = 0

        # NOTE: teacher_val is RAW pre-softmax logits. No temperature, no softmax --
        # tau is a swept loss hyperparameter, softmax over a top-k *subset* is not
        # the tempered distribution anyway, and gen_meta.gen_temperature is the
        # *sampling* temperature, unrelated to the KD tau.
        return self._pack(
            speaker=speaker,
            uid=uid,
            lang=lang,
            swapped=swapped,
            num_frames=len(w),
            codes_self=codes_self,
            codes_other=codes_other,
            text_self=text_self,
            text_other=text_other,
            text_flat=np.empty((0,), dtype=np.int32),
            has_teacher=has_teacher,
            topk=topk,
            teacher_val=teacher_val,
            teacher_idx=teacher_idx,
            is_text_only=False,
            use_kd=has_teacher,
            use_ce_audio=True,
            use_ce_text=True,
        )

    def _normalize_solo(self, row: int, rng: np.random.Generator) -> KDSample:
        """ko_tts and en_solo: mono, no B stream, no teacher.

        The absent user channel is filled here rather than in the collator: it is
        a per-sample op with no batch coupling, and doing it during collation
        would force a sample_type branch inside the stack and break the uniformity
        of the item schema.
        """
        K = NUM_CODEBOOKS
        col = self._columns()
        uid, lang, speaker = self._meta(row)
        T, _ = self._shape_meta(row)
        assert T > 0, f"{uid}: solo row with num_frames=0"

        w = choose_window(T, self.cfg.max_frames, rng, self.crop_mode)
        Tw = len(w)

        codes_self = self._codes(col, "a", row, K, T, w, uid)
        text_self = self._text(col, "a", row, T, w, uid)

        # Config-injected, never defaulted to zeros: 0 is a valid Mimi code, so a
        # guessed constant would be a maximally learnable spurious signal that
        # correlates perfectly with lang=ko.
        #
        # The fill is a (K, P) bank of *real* consecutive silence frames, tiled
        # with a per-sample phase. A constant fill would be trivially separable
        # from real Mimi silence (which is constant only in cb0) and hand the model
        # a free "this sample has no user" shortcut; rotating the bank makes every
        # solo sample's user channel start somewhere else in the noise floor.
        #
        # The phase is drawn from the same _rng_for(index) generator as the crop,
        # so it stays a pure function of (seed, epoch, index) -- identical for any
        # num_workers, batch size or call order -- and is drawn *after*
        # choose_window so the crop distribution is untouched.
        bank = self.cfg.tokens.silence_bank_array()
        assert bank.ndim == 2 and bank.shape[0] == K, (
            f"silence_bank must be (K, P) with K={K}, got {bank.shape}"
        )
        P = bank.shape[1]
        assert P >= 1, "silence_bank must have at least one frame"
        phase = int(rng.integers(P))
        codes_other = np.ascontiguousarray(bank[:, (phase + np.arange(Tw)) % P])
        text_other = np.full((Tw,), int(self.cfg.tokens.text_pad_id), dtype=np.int32)

        return self._pack(
            speaker=speaker,
            uid=uid,
            lang=lang,
            swapped=False,
            num_frames=Tw,
            codes_self=codes_self,
            codes_other=codes_other,
            text_self=text_self,
            text_other=text_other,
            text_flat=np.empty((0,), dtype=np.int32),
            has_teacher=False,
            topk=0,
            teacher_val=np.empty((0, 0, 0), dtype=np.float16),
            teacher_idx=np.empty((0, 0, 0), dtype=np.int16),
            is_text_only=False,
            use_kd=False,
            use_ce_audio=True,
            use_ce_text=True,
        )

    def _normalize_text_only(self, row: int, rng: np.random.Generator) -> KDSample:
        """text_anchor: genuinely T=0.

        Fabricating L frames of silent audio to make the shapes uniform was
        rejected -- it invents 12.5 Hz timing for text that has none and feeds
        made-up frames into the audio CE. The tokens go in text_flat and a
        separate collator/loss path consumes them; use_ce_text stays False
        because that flag routes the frame-aligned text stream, which this
        sample does not have.
        """
        col = self._columns()
        uid, lang, speaker = self._meta(row)
        T, _ = self._shape_meta(row)
        assert T == 0, f"{uid}: text_anchor must have num_frames=0, got {T}"

        flat = np.asarray(col["text_tokens_a"](row), dtype=np.int32)
        L = int(flat.size)
        assert L > 0, f"{uid}: text_anchor with no tokens"
        # Same windowing machinery, token axis instead of the frame axis.
        w = choose_window(L, self.cfg.max_text_len, rng, self.crop_mode)
        text_flat = apply_window_t(flat, w)

        streams = empty_like_streams()
        return self._pack(
            speaker=speaker,
            uid=uid,
            lang=lang,
            swapped=False,
            num_frames=0,
            codes_self=streams["codes_self"],
            codes_other=streams["codes_other"],
            text_self=streams["text_self"],
            text_other=streams["text_other"],
            text_flat=text_flat,
            has_teacher=False,
            topk=0,
            teacher_val=streams["teacher_val"],
            teacher_idx=streams["teacher_idx"],
            is_text_only=True,
            use_kd=False,
            use_ce_audio=False,
            use_ce_text=False,
        )

    # --- Zone B voice prompt --------------------------------------------------

    def _pick_ref(self, item: KDSample, row: int, rng: np.random.Generator) -> dict:
        """A reference utterance by the same speaker: ARCHITECTURE section 7.2 Zone B.

        The reference is never stored -- it is just another row -- so it is looked
        up here, per __getitem__. Re-drawing it every epoch is what
        TRAINING_CURRICULUM Phase 2 means by "a varied reference voice per
        sample"; baking one reference per row into the corpus would both bloat
        storage and delete the augmentation.

        It must be a DIFFERENT utterance by the same speaker. Using the sample's
        own audio as its own prompt is the degenerate case section 7.4 exists to
        avoid: the model would learn to copy the prompt rather than to carry a
        voice identity across different content, and the eval would look perfect
        while the capability is absent.

        Returns the ref_* triple, empty when no reference exists.
        """
        K = NUM_CODEBOOKS
        if not self.cfg.voice_prompt:
            return empty_ref(K)

        # text_anchor has no audio at all, so there is nothing to condition on and
        # no frame axis to align a transcript to.
        if self.sample_type == "text_anchor":
            return empty_ref(K)

        speaker = item["speaker"]
        if self.sample_type == "en_kd":
            # An en_kd row carries two voices, so no single id describes it and
            # data_pipeline writes "". Asserted rather than assumed: if a future
            # ingest ever populated it, silently prompting a dialogue with one of
            # its two speakers is exactly the kind of thing that trains fine and
            # is wrong.
            assert speaker == "", (
                f"{item['sample_uid']}: en_kd row carries speaker {speaker!r}, but an "
                f"en_kd row has two voices. Voice-prompt selection has no defined "
                f"meaning here -- fix the ingest or exclude en_kd explicitly."
            )
            return empty_ref(K)

        if not speaker:
            return empty_ref(K)
        rows = self._speaker_index().get(speaker)
        # A singleton speaker has no *other* utterance, so it gets no prompt at
        # all rather than being prompted with itself.
        if rows is None or len(rows) < 2:
            return empty_ref(K)

        # Uniform over the speaker's other rows: draw from n-1 slots and skip over
        # `row`'s own position, which is cheaper and less biased than rejection
        # sampling on a speaker with only two utterances.
        pos = int(np.searchsorted(rows, row))
        assert pos < len(rows) and rows[pos] == row, (
            f"row {row} missing from the index for speaker {speaker!r}"
        )
        j = int(rng.integers(len(rows) - 1))
        ref_row = rows[j] if j < pos else rows[j + 1]
        assert ref_row != row

        col = self._columns()
        ref_uid, _, ref_spk = self._meta(ref_row)
        assert ref_spk == speaker, f"speaker index is stale: {ref_uid} is {ref_spk!r}"
        T_ref, _ = self._shape_meta(ref_row)
        assert T_ref > 0, f"{ref_uid}: reference row with num_frames=0"

        # Probe must not be stochastic for the same reason its crop is not: a
        # moving prompt makes the eval metric drift for reasons unrelated to the
        # checkpoint. Head rather than center so the prompt starts at the
        # utterance onset, which is what a domain opening greeting (section 7.4)
        # actually looks like.
        mode = "random" if self.crop_mode == "random" else "head"
        w = choose_window(T_ref, self.cfg.voice_prompt_frames, rng, mode)

        ref_codes = self._codes(col, "a", ref_row, K, T_ref, w, ref_uid)
        ref_text = self._text(col, "a", ref_row, T_ref, w, ref_uid)
        return {
            "ref_codes": _as_tensor(ref_codes, torch.int16),
            "ref_text": _as_tensor(ref_text, torch.int32),
            "has_ref": _bool(len(w) > 0),
        }

    # --- shared helpers -------------------------------------------------------

    def _codes(self, col, side: str, row: int, K: int, T: int, w: Window, uid: str) -> np.ndarray:
        flat = col[f"codes_{side}"](row)
        assert flat.size == K * T, (
            f"{uid}: codes_{side} has {flat.size} elems, expected K*T={K * T}. "
            f"Corrupted row -- failing loudly beats a silent reshape."
        )
        return apply_window_kt(flat.reshape(K, T), w)

    def _text(self, col, side: str, row: int, T: int, w: Window, uid: str) -> np.ndarray:
        flat = col[f"text_tokens_{side}"](row)
        # Frame alignment is guaranteed by construction; a mismatch means a damaged
        # row, so fail rather than silently truncate.
        assert flat.size == T, (
            f"{uid}: text_tokens_{side} has {flat.size} tokens but num_frames={T}"
        )
        return apply_window_t(np.asarray(flat, dtype=np.int32), w)

    def _pack(self, **kw) -> KDSample:
        return KDSample(
            sample_uid=kw["uid"],
            source=self.source,
            sample_type=self.sample_type,
            lang=kw["lang"],
            speaker=kw.get("speaker", ""),
            swapped=_bool(kw["swapped"]),
            is_text_only=_bool(kw["is_text_only"]),
            has_teacher=_bool(kw["has_teacher"]),
            num_frames=_i32(kw["num_frames"]),
            topk=_i32(kw["topk"]),
            codes_self=_as_tensor(kw["codes_self"], torch.int16),
            codes_other=_as_tensor(kw["codes_other"], torch.int16),
            text_self=_as_tensor(kw["text_self"], torch.int32),
            text_other=_as_tensor(kw["text_other"], torch.int32),
            text_flat=_as_tensor(kw["text_flat"], torch.int32),
            # Placeholder: __getitem__ overwrites this via _pick_ref *after* the
            # crop, so that the reference draw cannot perturb the crop window.
            **empty_ref(NUM_CODEBOOKS),
            teacher_val=_as_tensor(kw["teacher_val"], torch.float16),
            teacher_idx=_as_tensor(kw["teacher_idx"], torch.int16),
            use_kd=_bool(kw["use_kd"]),
            use_ce_audio=_bool(kw["use_ce_audio"]),
            use_ce_text=_bool(kw["use_ce_text"]),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MoshiKDDataset(source={self.source!r}, data_dir={self.data_dir!r}, "
            f"split={self.split!r}, "
            f"sample_type={self.sample_type!r}, rows={self._n_rows}, "
            f"len={len(self)}, double_ab={self.double_ab}, crop={self.crop_mode!r})"
        )


def _as_tensor(arr, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t if t.dtype == dtype else t.to(dtype)


def build_source_datasets(
    cfg: DataConfig,
    split: str = "train",
    seed: int | None = None,
) -> dict[str, MoshiKDDataset]:
    """Open every prepared source found under cfg.root for `split`.

    Keyed by *source* (the mixing group / directory name), not sample_type --
    en_solo and ko_tts are separate mixing groups that share a shape contract,
    and collapsing them would make the mix weights unreachable.
    """
    root = Path(cfg.root)
    assert root.exists(), (
        f"prepared root {root} does not exist. Build it first:\n"
        f"    python -m training.datasets.prepare --group <source>"
    )
    seed = cfg.seed if seed is None else seed

    out: dict[str, MoshiKDDataset] = {}
    for source in _SOURCES:
        if not (root / source / split).exists():
            continue
        out[source] = MoshiKDDataset(root, source, split, cfg=cfg, seed=seed)
    assert out, f"no prepared sources for split={split!r} under {root}"
    return out
