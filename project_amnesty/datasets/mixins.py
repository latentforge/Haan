"""Shared functionality (mixins). Orthogonal to the dataset hierarchy — classes
that need them compose them freely.

  * MimiEncoderMixin : lazy Mimi load + batched encoding + resampling
  * TextAlignMixin   : lazy tokenizer + 12.5 Hz frame-aligned text stream
  * NpzPairIOMixin   : {uid}.npz(codes) + {uid}.text.npz + {uid}.json save/cache/read

Frame-alignment convention (Moshi inner-monologue):
  - a word's text tokens are placed consecutively starting at the word's
    utterance-start frame
  - remaining frames are PAD, with EPAD on the frame right after an utterance span ends
  - Korean (Qwen tokenizer) includes whitespace in tokens (Ġ), so no no_whitespace
    option is needed. For KO TTS without timestamps (most of it), two modes:
      (a) "uniform": distribute tokens evenly over the utterance
          (hypothesis: sufficient for singletons)
      (b) "aligned": timestamps from an external forced aligner (for ablation)
  - PAD/EPAD ids are undetermined on the model side, so they are config-injected
    (no hardcoding)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from ..schema import SAMPLE_RATE, Sample


# ---------------- text alignment ----------------

@dataclass
class TextTokCfg:
    tokenizer_name: str            # team fork's personaplex tokenizer or Qwen/Qwen3-8B
    text_pad_id: int
    text_epad_id: int
    mode: str = "uniform"          # "uniform" | "aligned"

    @classmethod
    def from_yaml(cls, path: str) -> "TextTokCfg":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def align_uniform(token_ids: list[int], num_frames: int, cfg: TextTokCfg) -> np.ndarray:
    """Distribute tokens evenly across frames. Remaining frames are PAD; EPAD after the last token."""
    stream = np.full(num_frames, cfg.text_pad_id, dtype=np.int32)
    if not token_ids:
        return stream
    n = min(len(token_ids), num_frames)
    # more tokens than frames → truncated. Extremely fast speech; worth logging.
    pos = np.linspace(0, num_frames - 1, n).astype(int)
    # monotonic correction so two tokens never land on the same frame
    for k in range(1, n):
        if pos[k] <= pos[k - 1]:
            pos[k] = pos[k - 1] + 1
    pos = np.clip(pos, 0, num_frames - 1)
    stream[pos] = token_ids[:n]
    last = int(pos[-1])
    if last + 1 < num_frames:
        stream[last + 1] = cfg.text_epad_id
    return stream


def align_timestamps(
    words: list[dict], num_frames: int, tokenizer, cfg: TextTokCfg
) -> np.ndarray:
    """words: [{"word": str, "start": sec, "end": sec}] (forced-aligner output)."""
    stream = np.full(num_frames, cfg.text_pad_id, dtype=np.int32)
    cursor = 0
    for w in words:
        ids = tokenizer.encode(" " + w["word"], add_special_tokens=False)
        start = max(cursor, int(w["start"] * 12.5))
        for j, tid in enumerate(ids):
            if start + j >= num_frames:
                break
            stream[start + j] = tid
        cursor = start + len(ids)
        end_f = int(w["end"] * 12.5)
        if cursor <= end_f < num_frames and stream[end_f] == cfg.text_pad_id:
            stream[end_f] = cfg.text_epad_id
    return stream


class TextAlignMixin:
    """Adds lazy tokenizer loading + alignment to classes that carry a text_cfg (TextTokCfg)."""

    text_cfg: TextTokCfg | None

    def _get_tokenizer(self):
        tok = getattr(self, "_tokenizer", None)
        if tok is None:
            from transformers import AutoTokenizer
            tok = self._tokenizer = AutoTokenizer.from_pretrained(self.text_cfg.tokenizer_name)
        return tok

    def align_text(
        self, text: str, word_timestamps: list[dict] | None, num_frames: int
    ) -> np.ndarray:
        """Transcript → (T,) frame-aligned stream. Uses timestamps when available in aligned mode."""
        assert self.text_cfg.text_pad_id is not None and self.text_cfg.text_epad_id is not None, \
            "text_pad_id/text_epad_id are unset (configs/data/text_tok.yaml) - finalize the PAD/EPAD mapping first"
        if word_timestamps and self.text_cfg.mode == "aligned":
            return align_timestamps(word_timestamps, num_frames, self._get_tokenizer(), self.text_cfg)
        ids = self._get_tokenizer().encode(text, add_special_tokens=False)
        return align_uniform(ids, num_frames, self.text_cfg)


# ---------------- Mimi encoding ----------------

class MimiEncoderMixin:
    """Adds lazy Mimi loading + batched encoding to classes that carry a device attribute.

    Mimi is frozen (slide 25: never modify), so baked tokens are never invalidated.
    """

    device: str

    def _get_mimi(self):
        mimi = getattr(self, "_mimi", None)
        if mimi is None:  # lazy — never loaded if everything is a cache hit
            from moshi.models import loaders
            ckpt = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
            mimi = self._mimi = ckpt.get_mimi(device=self.device)
            mimi.eval()
        return mimi

    def encode_audio(self, wavs: list[np.ndarray]) -> list[np.ndarray]:
        """Batched Mimi encoding. Pad to max length, then return only the valid frames."""
        import torch
        mimi = self._get_mimi()
        lens = [w.shape[-1] for w in wavs]
        x = torch.zeros(len(wavs), 1, max(lens))
        for j, w in enumerate(wavs):
            x[j, :, : w.shape[-1]] = torch.from_numpy(w)
        with torch.inference_mode():
            codes = mimi.encode(x.to(self.device))     # (B, K, T)
        out = []
        for j, L in enumerate(lens):
            t = int(round(L / SAMPLE_RATE * 12.5))
            out.append(codes[j, :, :t].cpu().numpy().astype(np.int16))
        return out

    @staticmethod
    def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(wav.astype(np.float32)), sr, SAMPLE_RATE).numpy()


# ---------------- npz pair IO ----------------

class NpzPairIOMixin:
    """Singleton-shaped ({uid}.npz + {uid}.text.npz + {uid}.json) artifact convention.

    Shared by AudioSourceDataset (build+read) and EnSoloDataset (crop save+read).
    The default iter_samples implementation lives here so prepare_dataset consumes
    every source uniformly.
    """

    out_dir: Path
    lang: str
    sample_type: str

    def is_cached(self, uid: str) -> bool:
        return (self.out_dir / f"{uid}.npz").exists() and \
               (self.out_dir / f"{uid}.text.npz").exists()

    def save_pair(self, uid: str, codes: np.ndarray, text_stream: np.ndarray,
                  meta: dict) -> None:
        np.savez_compressed(self.out_dir / f"{uid}.npz", codes=codes)
        np.savez_compressed(self.out_dir / f"{uid}.text.npz", text_tokens=text_stream)
        (self.out_dir / f"{uid}.json").write_text(json.dumps(
            {"sample_uid": uid, **meta}, ensure_ascii=False))

    def iter_samples(self) -> Iterator[Sample]:
        for meta_path in sorted(self.out_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            if not isinstance(meta, dict) or "sample_uid" not in meta:
                continue   # skip non-sample files such as build_stats.json
            uid = meta["sample_uid"]
            codes = np.load(self.out_dir / f"{uid}.npz")["codes"]
            tp = self.out_dir / f"{uid}.text.npz"
            text = (np.load(tp)["text_tokens"] if tp.exists()
                    else np.load(self.out_dir / f"{uid}.npz")["text_tokens"])
            yield Sample(
                sample_type=self.sample_type, lang=self.lang,
                codes_a=codes, text_tokens_a=text, sample_uid=uid,
            )
