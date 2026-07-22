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

from project_amnesty.datasets.shared.schema import FRAME_RATE_HZ, NUM_CODEBOOKS, SAMPLE_RATE, Sample

# The corpus is baked with this checkpoint. It is the HF conversion of Moshiko:
# the `moshi` package is not installed here, and `kmhf/hf-moshiko` is what the
# repo's working codec code (project_amnesty/.../scenario_run.py) and
# project_amnesty/tools/derive_silence_codes.py both use.
#
# Verified identical to the codec the en_kd generator uses. The Colab generator
# goes through the `moshi` package, which loads
# kyutai/moshiko-pytorch-bf16 :: tokenizer-e351c8d8-checkpoint125.safetensors
# (loaders.MIMI_NAME) -- a different repo and a different tensor naming scheme,
# so "same codec" was an assumption, and ARCHITECTURE.md 5.1's "shared frozen
# codec => identical teacher/student output space" rests on it. Checked directly:
# both hold 96,151,393 audio_encoder parameters, and casting kyutai's fp32
# codebooks to bf16 reproduces kmhf's **bit-for-bit** for the semantic level-0
# codebook and all 7 acoustic levels. Same weights, kmhf is just the bf16 cast.
#
# Caveat worth knowing: quantization is a nearest-neighbour lookup, so the bf16
# rounding (rel ~2e-3 on the codebook vectors) can flip the assigned code for a
# frame sitting on a Voronoi boundary. That never matters here -- no audio is
# encoded through both paths and compared -- but it would if anyone ever tried to
# reproduce en_kd codes locally and expected an exact match.
DEFAULT_MIMI_CKPT = "kmhf/hf-moshiko"


# ---------------- text alignment ----------------

@dataclass
class TextTokCfg:
    tokenizer_name: str            # team fork's personaplex tokenizer or Qwen/Qwen3-8B
    text_pad_id: int
    text_epad_id: int
    mode: str = "uniform"          # "uniform" | "aligned"

    @classmethod
    def from_yaml(cls, path: str) -> "TextTokCfg":
        # utf-8: configs carry Korean comments (cp949 default breaks on Windows).
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


def align_uniform(
    word_token_ids: list[list[int]], num_frames: int, cfg: TextTokCfg
) -> np.ndarray:
    """Spread words evenly across frames; EPAD one frame before each word onset.

    EPAD is a *speech-onset trigger*, not an end-of-utterance marker
    (ARCHITECTURE.md §5.0.1: "다음 단어 시작 한 프레임 전에 삽입", and §7.6 notes it
    is inserted per word, which is why its frequency is unlike a once-per-turn
    token). "Utterance complete" needs no marker of its own: the stream simply
    returns to PAD, which is exactly the remap the instruction template relies on.

    Takes per-word token id lists rather than one flat list because word onsets
    are what EPAD attaches to -- a flat list has no word structure to hang it on.
    """
    stream = np.full(num_frames, cfg.text_pad_id, dtype=np.int32)
    words = [w for w in word_token_ids if w]
    if not words or num_frames <= 0:
        return stream

    # Onsets start at 1 so the first word has a frame for its EPAD, and stop early
    # enough that the final word's tokens still fit.
    hi = max(1, num_frames - len(words[-1]))
    starts = np.linspace(1, hi, len(words)).astype(int)

    cursor = 0  # first frame not yet written
    for ids, s in zip(words, starts):
        # cursor + 1 keeps at least one free frame before the word, both to avoid
        # overwriting the previous word and to leave somewhere for EPAD.
        start = max(int(s), cursor + 1)
        if start >= num_frames:
            break  # ran out of frames: very fast speech, remaining words dropped
        if stream[start - 1] == cfg.text_pad_id:
            stream[start - 1] = cfg.text_epad_id
        n = min(len(ids), num_frames - start)
        stream[start:start + n] = ids[:n]
        cursor = start + n
    return stream


def place_words_at_frames(
    words: list[tuple[str, int]], num_frames: int, tokenizer, cfg: TextTokCfg
) -> np.ndarray:
    """(word, onset_frame) pairs → frame-aligned stream. Onsets are integer frames.

    Split out of align_timestamps so callers that already know the onset *frame*
    (retokenize_frame_aligned) never round-trip through seconds: frame/12.5*12.5
    is not exact in float64 (frame 7 → 6.999...), and int() would truncate to the
    previous frame.

    EPAD goes one frame *before* each word's onset, matching align_uniform and
    ARCHITECTURE.md §5.0.1. Placing it at the word's end frame instead would make
    it a terminator rather than the onset trigger the text channel needs.
    """
    stream = np.full(num_frames, cfg.text_pad_id, dtype=np.int32)
    cursor = 0
    for i, (word, onset) in enumerate(words):
        ids = tokenizer.encode((" " if i else "") + word, add_special_tokens=False)
        if not ids:
            continue
        # cursor + 1 leaves a frame for EPAD without clobbering the previous word.
        start = max(cursor + 1 if cursor else 0, int(onset))
        if start >= num_frames:
            break
        if start - 1 >= 0 and stream[start - 1] == cfg.text_pad_id:
            stream[start - 1] = cfg.text_epad_id
        n = min(len(ids), num_frames - start)
        stream[start:start + n] = ids[:n]
        cursor = start + n
    return stream


def align_timestamps(
    words: list[dict], num_frames: int, tokenizer, cfg: TextTokCfg
) -> np.ndarray:
    """words: [{"word": str, "start": sec, "end": sec}] (forced-aligner output)."""
    return place_words_at_frames(
        [(w["word"], int(w["start"] * FRAME_RATE_HZ)) for w in words],
        num_frames, tokenizer, cfg,
    )


# ---------------- SeqKD retokenization (teacher vocab → student vocab) ----------------

# Moshi's text stream is Helium/SentencePiece (text_card=32000,
# existing_text_padding_id=3 -- moshi/models/loaders.py). The student is Qwen3.
# RISKS_AND_DIAGNOSTICS.md §7.2: logit-level KD on text is impossible across
# different tokenizers, so the teacher's text is decoded and re-tokenized (SeqKD).
MOSHI_TEXT_TOKENIZER = "kmhf/hf-moshiko"   # same repo the Mimi weights come from
MOSHI_TEXT_PAD_ID = 3                      # <pad> in the 32k SentencePiece vocab
_SP_WORD_PREFIX = "▁"                 # SentencePiece word-boundary marker


def words_from_frame_stream(
    stream: np.ndarray, tokenizer, pad_id: int
) -> list[tuple[str, int]]:
    """Frame-aligned source-vocab stream → [(word, onset_frame)].

    Groups consecutive non-PAD tokens into words on the SentencePiece word
    boundary. The onset frame is the frame of a word's *first* token, which is
    what carries the turn-taking timing we are distilling -- so it is preserved
    exactly, and only the sub-word tokenization is allowed to change.
    """
    words: list[tuple[str, int]] = []
    cur_ids: list[int] = []
    cur_start: int | None = None

    def flush():
        nonlocal cur_ids, cur_start
        if cur_ids and cur_start is not None:
            text = tokenizer.decode(cur_ids).strip()
            if text:
                words.append((text, cur_start))
        cur_ids, cur_start = [], None

    for t, raw in enumerate(np.asarray(stream).tolist()):
        tid = int(raw)
        if tid == pad_id or tid < 0:
            continue
        piece = tokenizer.convert_ids_to_tokens(tid)
        if piece is None:
            continue
        # Specials other than PAD (unk/bos/eos) carry no transcript content.
        if piece.startswith("<") and piece.endswith(">"):
            continue
        if piece.startswith(_SP_WORD_PREFIX) and cur_ids:
            flush()
        if cur_start is None:
            cur_start = t
        cur_ids.append(tid)
    flush()
    return words


def retokenize_frame_aligned(
    stream: np.ndarray, src_tokenizer, src_pad_id: int, tgt_tokenizer, cfg: TextTokCfg
) -> np.ndarray:
    """Frame-aligned teacher-vocab stream → frame-aligned student-vocab stream.

    Length is preserved (both are (T,) at FRAME_RATE_HZ). Word onsets are
    preserved; token counts inside a word are not, and a word whose student
    tokenization no longer fits before the next onset is truncated exactly as in
    align_uniform -- timing is the signal being distilled, sub-word identity is not.
    """
    assert cfg.text_pad_id is not None and cfg.text_epad_id is not None, \
        "text_pad_id/text_epad_id are unset (configs/data/text_tok.yaml)"
    words = words_from_frame_stream(stream, src_tokenizer, src_pad_id)
    return place_words_at_frames(words, len(stream), tgt_tokenizer, cfg)


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
        # Tokenize per word, not as one string: EPAD attaches to word onsets, so
        # the word structure has to survive into align_uniform. The leading space
        # is kept for non-initial words because Qwen's BPE is space-prefixed --
        # dropping it would produce different (word-initial) token ids than the
        # same text tokenized whole.
        tok = self._get_tokenizer()
        ids = [
            tok.encode((" " if i else "") + w, add_special_tokens=False)
            for i, w in enumerate(text.split())
        ]
        return align_uniform(ids, num_frames, self.text_cfg)


# ---------------- Mimi encoding ----------------

class MimiEncoderMixin:
    """Adds lazy Mimi loading + batched encoding to classes that carry a device attribute.

    Mimi is frozen (slide 25: never modify), so baked tokens are never invalidated.
    """

    device: str
    mimi_ckpt_id: str = DEFAULT_MIMI_CKPT

    def _get_mimi(self):
        """The frozen Mimi, lazily loaded — never touched if everything is a cache hit.

        NOT the `moshi` package. This used to go through `moshi.models.loaders`,
        which is not installed here, so the path had never actually run. It now
        mirrors project_amnesty/tools/derive_silence_codes.py::load_mimi: pull the
        `audio_encoder.*` tensors out of the HF Moshi checkpoint into a standalone
        transformers MimiModel, which avoids materializing the full ~15 GB model
        just to encode audio. Only the shards holding those tensors are fetched.
        """
        mimi = getattr(self, "_mimi", None)
        if mimi is not None:
            return mimi

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import MimiConfig, MimiModel

        ckpt_id = self.mimi_ckpt_id
        cfg_path = hf_hub_download(ckpt_id, "config.json")
        with open(cfg_path) as f:
            audio_cfg = json.load(f).get("audio_encoder_config")
        assert audio_cfg is not None, (
            f"{ckpt_id}/config.json has no audio_encoder_config; this is not an HF "
            f"Moshi checkpoint. Set mimi_ckpt_id to {DEFAULT_MIMI_CKPT!r}."
        )
        model = MimiModel(MimiConfig(**audio_cfg)).eval()

        with open(hf_hub_download(ckpt_id, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        prefix = "audio_encoder."
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
        assert shards, f"no {prefix}* tensors in {ckpt_id}"

        state = {}
        for shard in shards:
            for k, v in load_file(hf_hub_download(ckpt_id, shard)).items():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = v

        missing, unexpected = model.load_state_dict(state, strict=False)
        # A partially-loaded codec would bake plausible, wrong codes into the whole
        # corpus, so this refuses rather than warns.
        assert not missing and not unexpected, (
            f"Mimi state_dict mismatch for {ckpt_id}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

        mimi = self._mimi = model.to(self.device)
        return mimi

    def encode_audio(self, wavs: list[np.ndarray]) -> list[np.ndarray]:
        """Batched Mimi encoding. Pad to max length, then return only the valid frames.

        wavs: list of (1, S) or (S,) float32 @ 24 kHz -> list of (K, T) int16,
        K = NUM_CODEBOOKS, T = round(S / SAMPLE_RATE * FRAME_RATE_HZ).
        """
        import torch
        mimi = self._get_mimi()
        lens = [w.shape[-1] for w in wavs]
        x = torch.zeros(len(wavs), 1, max(lens))
        for j, w in enumerate(wavs):
            # accepts (S,) and (1, S) alike
            x[j, 0, : w.shape[-1]] = torch.from_numpy(
                np.ascontiguousarray(np.reshape(w, -1), dtype=np.float32))
        with torch.inference_mode():
            out_enc = mimi.encode(x.to(self.device), num_quantizers=NUM_CODEBOOKS)
        # transformers returns a ModelOutput; tolerate a bare tensor too.
        codes = out_enc.audio_codes if hasattr(out_enc, "audio_codes") else out_enc  # (B, K, T)
        assert codes.shape[1] == NUM_CODEBOOKS, (
            f"expected {NUM_CODEBOOKS} codebooks from {self.mimi_ckpt_id}, "
            f"got {codes.shape[1]}"
        )
        codes = codes.to("cpu").numpy().astype(np.int16)
        out = []
        for j, L in enumerate(lens):
            t = int(round(L / SAMPLE_RATE * FRAME_RATE_HZ))
            out.append(codes[j, :, :t])
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
                # written by AudioSourceDataset._process_batch / EnSoloDataset.build;
                # "" for older artifacts that predate the speaker column.
                speaker=str(meta.get("speaker", "")),
            )
