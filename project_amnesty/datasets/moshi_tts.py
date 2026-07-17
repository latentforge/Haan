"""
Clean Moshi generation with transformers (kmhf/hf-moshiko).

Hard-won lessons (verified on this box):
  * `do_sample=True` is REQUIRED. Greedy decoding collapses moshiko to pure silence.
  * moshiko is a DIALOGUE model: it *responds to user audio*. It is NOT a text-to-speech
    reader. Feed a question on the user stream -> it replies (in its own voice) with a
    coherent inner-monologue text + aligned audio.
  * Loudness-normalize any input audio to ~-24 LUFS, else Mimi codes go out of
    distribution and the output degrades to noise.
  * `generate()` REPLAYS the prefill in the output waveform. Take the NEW frames from
    the END: speech = full[-new_frames * FRAME_SIZE:].

Two entry points:
  dialogue(user_wav)  -> moshiko's spoken reply to that audio       (works well)
  say(text)           -> force the text stream to `text` (best-effort "TTS";
                         moshiko was not trained to read, so quality is limited)

    python moshi_tts.py dialogue question.wav reply.wav
    python moshi_tts.py say "Hello, my name is Moshi." out.wav
"""
import os, sys, math
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import (
    MoshiForConditionalGeneration,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
REPO = "kmhf/hf-moshiko"

print("loading Moshi ...")
model = MoshiForConditionalGeneration.from_pretrained(REPO, dtype=DTYPE).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO)
SR = model.config.sampling_rate                                  # 24000
FRAME = int(SR / model.config.audio_encoder_config.frame_rate)   # 1920 (12.5 Hz)
PAD_ID = tokenizer.encode("<pad>")[0]                            # 3
NCB = model.config.num_codebooks
torch.manual_seed(0)


# ----------------------------- audio io --------------------------------------
def normalize_loudness(wav, target_lufs=-24.0):
    rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
    gain_db = target_lufs - (20.0 * torch.log10(rms).item() - 0.691)
    wav = wav * (10.0 ** (gain_db / 20.0))
    peak = wav.abs().max()
    return wav * (0.99 / peak) if peak > 0.99 else wav


def load_wav_24k_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.tensor(wav.mean(axis=1))
    if sr != SR:
        g = math.gcd(int(sr), SR)
        wav = torch.tensor(resample_poly(wav.numpy(), SR // g, sr // g).astype("float32"))
    return normalize_loudness(wav)


def save(path, wav):
    wav = np.asarray(wav, dtype=np.float32)
    p = np.abs(wav).max()
    if p > 1e-6:
        wav = wav / p * 0.9
    sf.write(path, wav, SR)
    print(f"saved {path}  ({len(wav) / SR:.2f}s)")


# ----------------------------- dialogue (works) ------------------------------
@torch.no_grad()
def dialogue(user_wav_path, out_path="reply.wav", max_new_tokens=200):
    """Feed a question on the user stream; moshiko replies in its own voice.

    Note: no StopOnSilence — early stopping mismatches Moshi's precomputed delay-pattern
    mask (crashes in generate's final un-delay). Generate a fixed length instead.
    """
    voice = load_wav_24k_mono(user_wav_path)
    # match all three streams to the actual Mimi code length
    user_codes = model.audio_encoder.encode(
        voice.view(1, 1, -1).to(DEVICE, DTYPE), num_quantizers=NCB)[0]
    L = user_codes.shape[-1]
    moshi_codes = model.audio_encoder.encode(
        torch.zeros(1, 1, L * FRAME, device=DEVICE, dtype=DTYPE), num_quantizers=NCB)[0][:, :, :L]
    ids = torch.full((1, L), PAD_ID, dtype=torch.long, device=DEVICE)

    out = model.generate(
        input_ids=ids, user_audio_codes=user_codes, moshi_audio_codes=moshi_codes,
        max_new_tokens=max_new_tokens, do_sample=True, concat_unconditional_inputs=True,
    )
    new = out.sequences.shape[-1] - L
    full = out.audio_sequences[0, 0].float().cpu().numpy()
    speech = full[-new * FRAME:] if new > 0 else full
    save(out_path, speech)
    print("moshiko said:", tokenizer.decode(out.sequences[0, L:], skip_special_tokens=True)[:300])
    return speech


# ----------------------------- say (best-effort TTS) -------------------------
class _ForceText(LogitsProcessor):
    """Force the text stream to a fixed schedule of token ids, one per frame.

    `schedule[k]` is the text token the model must emit at generated frame k;
    once the schedule is exhausted we force <pad> so the model falls silent.
    """
    def __init__(self, schedule, prefill_len, pad_id):
        self.schedule, self.prefill, self.pad = schedule, prefill_len, pad_id

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prefill
        forced = self.schedule[step] if 0 <= step < len(self.schedule) else self.pad
        mask = torch.full_like(scores, float("-inf"))
        mask[:, forced] = 0.0
        return mask


@torch.no_grad()
def say(text, out_path="say.wav", gap=1, tail_frames=13):
    """Force the text stream to `text` while moshiko renders aligned audio.

    Best-effort "TTS": moshiko was NOT trained to read arbitrary text, so the text
    stream is driven by a LogitsProcessor rather than sampled. Quality is limited
    (see module docstring); the audio stream is still sampled (`do_sample=True`).

    `gap` <pad> frames are inserted after each word token to approximate a natural
    ~12.5 Hz speaking rate; `tail_frames` of trailing silence let the last word ring out.
    """
    word_ids = tokenizer(text, add_special_tokens=False).input_ids
    schedule = []
    for tid in word_ids:
        schedule.append(tid)
        schedule.extend([PAD_ID] * gap)
    n_new = len(schedule) + tail_frames

    # 1-frame silent prefill on both audio streams to seed generation.
    prefill = 1
    zeros = torch.zeros(1, 1, prefill * FRAME, device=DEVICE, dtype=DTYPE)
    codes = model.audio_encoder.encode(zeros, num_quantizers=NCB)[0]
    ids = torch.full((1, prefill), PAD_ID, dtype=torch.long, device=DEVICE)

    out = model.generate(
        input_ids=ids, user_audio_codes=codes, moshi_audio_codes=codes,
        max_new_tokens=n_new, do_sample=True, concat_unconditional_inputs=True,
        logits_processor=LogitsProcessorList([_ForceText(schedule, prefill, PAD_ID)]),
    )
    new = out.sequences.shape[-1] - prefill
    full = out.audio_sequences[0, 0].float().cpu().numpy()
    speech = full[-new * FRAME:] if new > 0 else full
    save(out_path, speech)
    print("forced text:", tokenizer.decode([t for t in schedule if t != PAD_ID])[:200])
    return speech


if __name__ == "__main__":
    # Two entry points (see module docstring):
    #   python moshi_tts.py dialogue question.wav reply.wav   -> reply to a question wav
    #   python moshi_tts.py say "Hello, my name is Moshi." out.wav  -> best-effort TTS
    mode = sys.argv[1] if len(sys.argv) > 1 else "dialogue"
    if mode == "say":
        text = sys.argv[2] if len(sys.argv) > 2 else "Hello, my name is Moshi."
        out_wav = sys.argv[3] if len(sys.argv) > 3 else "say.wav"
        say(text, out_wav)
    else:
        # `mode` is either "dialogue" or a bare wav path (back-compat).
        if mode == "dialogue":
            user_wav = sys.argv[2] if len(sys.argv) > 2 else "question.wav"
            out_wav = sys.argv[3] if len(sys.argv) > 3 else "reply.wav"
        else:
            user_wav = mode
            out_wav = sys.argv[2] if len(sys.argv) > 2 else "reply.wav"
        dialogue(user_wav, out_wav)
