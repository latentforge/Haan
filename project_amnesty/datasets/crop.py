"""Temporal cropping — pure functions over frame windows.

Cropping lives in the dataset and runs *before* teacher top-k is materialized:
a 1500-frame row carries ~1.5 MB of K=8 top-k logits, and cropping in the
collator would serialize all of it across the worker IPC boundary only to throw
80% away. Doing it here costs ~0.3 MB/sample instead.

This module knows nothing about Arrow, torch, or the item contract. It takes a
length and an RNG and returns a half-open window; that is the whole surface.

Delay is deliberately NOT accounted for here. The collator applies delay inside
the cropped window (boundary loss is <=2 frames out of 750); pre-shrinking the
window by tau in the dataset would leak a model hyperparameter into an
index-only pure function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CROP_MODES = ("random", "center", "head")


@dataclass(frozen=True)
class Window:
    """Half-open frame window [start, end)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert 0 <= self.start <= self.end, f"bad window {self.start}:{self.end}"

    def __len__(self) -> int:
        return self.end - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.end)


def choose_window(
    T: int,
    max_frames: int,
    rng: np.random.Generator | None = None,
    mode: str = "random",
) -> Window:
    """Pick a <=max_frames window out of T frames.

    `rng` must be a per-sample generator (see MoshiKDDataset._rng_for): drawing
    from a shared generator makes the crop depend on which worker happened to
    get which index, which is unreproducible across num_workers.

    No minimum-length filtering happens here -- FilterConfig.min_frames=250
    already handled that upstream, and silently dropping rows at training time
    would desync the sampler's index space from len().
    """
    assert mode in CROP_MODES, f"crop mode must be one of {CROP_MODES}, got {mode!r}"
    assert T >= 0 and max_frames > 0

    if T <= max_frames:
        return Window(0, T)

    span = T - max_frames
    if mode == "head":
        start = 0
    elif mode == "center":
        start = span // 2
    else:  # random
        assert rng is not None, "crop_mode='random' needs an rng"
        start = int(rng.integers(0, span + 1))
    return Window(start, start + max_frames)


# --- window application -------------------------------------------------------


def _materialize(view: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """Detach a window from its parent buffer: always a fresh C-contiguous copy.

    NOTE (deviation from plan section 2.6, which prescribes ascontiguousarray):
    `np.ascontiguousarray` is *not* sufficient. When the window covers the whole
    row -- T <= max_frames, i.e. every ko_tts sample and every short en_kd row --
    the slice is already contiguous and ascontiguousarray returns the input
    unchanged. The result is then still a view onto the mmapped Arrow column,
    which is exactly the state section 2.6 exists to prevent: the parent buffer
    stays alive, and torch.from_numpy on a read-only mmap yields a non-writable
    tensor that warns and has undefined write behavior. The uncropped path is the
    one where the bug hides, because the cropped path copies by accident.

    We already need one copy to own the memory, so make it unconditional and fold
    any dtype cast (teacher_val -> fp16) into that same pass.
    """
    out = np.array(view, dtype=dtype, copy=True, order="C")
    assert out.flags.c_contiguous and out.flags.writeable and out.base is None
    return out


def apply_window_kt(arr: np.ndarray, w: Window) -> np.ndarray:
    """(K, T) -> (K, len(w)), owned and contiguous."""
    assert arr.ndim == 2, f"expected (K, T), got {arr.shape}"
    return _materialize(arr[:, w.start : w.end])


def apply_window_t(arr: np.ndarray, w: Window) -> np.ndarray:
    """(T,) -> (len(w),), owned and contiguous."""
    assert arr.ndim == 1, f"expected (T,), got {arr.shape}"
    return _materialize(arr[w.start : w.end])


def apply_window_ktk(arr: np.ndarray, w: Window, dtype: np.dtype | None = None) -> np.ndarray:
    """(K, T, topk) -> (K, len(w), topk), owned and contiguous, optionally cast.

    `dtype` folds the fp16 cast of teacher_val into the copy the crop already has
    to make, instead of paying for two passes over the dominant tensor.
    """
    assert arr.ndim == 3, f"expected (K, T, topk), got {arr.shape}"
    return _materialize(arr[:, w.start : w.end, :], dtype)


def apply_window(arr: np.ndarray, w: Window) -> np.ndarray:
    """ndim-dispatching convenience wrapper: (T,) / (K,T) / (K,T,topk)."""
    if arr.ndim == 1:
        return apply_window_t(arr, w)
    if arr.ndim == 2:
        return apply_window_kt(arr, w)
    if arr.ndim == 3:
        return apply_window_ktk(arr, w)
    raise ValueError(f"apply_window supports 1-3 dims, got {arr.ndim}")
