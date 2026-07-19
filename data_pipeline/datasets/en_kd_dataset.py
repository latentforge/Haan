"""Generated family: dual-Moshi self-talk generation + quality filter → en_kd samples.

Core idea (slide 15):
  Moshi A and B share weights, so the model is not loaded twice.
  Run one model with batch=2N, with slots [0:N]=role A and [N:2N]=role B, and every
  frame cross-feed A[i]'s output audio tokens as B[i]'s "other stream" input, and
  vice versa. VRAM holds one 7B copy + 2N KV caches.

Dumped per frame:
  - each role's own-stream audio codes (sampled hard tokens)
  - each role's inner-monologue text tokens (come frame-aligned for free)
  - top-k (val, idx) of the raw logits right before sampling — KD soft labels,
    no temperature applied

Quality filter — typical self-talk collapse modes and how they are detected:
  1) one side permanently silent / both keep talking → PAD ratio of the text stream
  2) audio-token n-gram loops → repetition pattern in the semantic codebook (0)
  3) same sentence repeated → n-gram duplication rate over non-PAD text tokens
  All thresholds are config-injected. The pass rate is fed back into the
  generation hyperparameters.

Integration point (marked TODO):
  moshi.models.LMGen is an interface for single-dialogue streaming, so batched
  cross-feed requires either extending LMGen.step() to batches or calling the inner
  lm (transformer) forward directly. MoshiSelfTalkEngine is that adapter boundary —
  once the team fork's PersonaplexForConditionalGeneration streaming interface is
  finalized, only this class needs replacing.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from ..schema import NUM_CODEBOOKS, Sample
from .base import BaseDataset


# ---------------- configs ----------------

@dataclass
class GenConfig:
    n_dialogues: int = 1000
    batch_pairs: int = 8            # N (concurrent dialogue pairs) → model batch is 2N
    max_frames: int = 1500          # 120 s at 12.5 Hz
    topk_dump: int = 32
    gen_temperature: float = 0.8
    gen_top_k: int = 250
    seed: int = 0
    seed_prompt_dir: str = "data/seed_prompts"   # SeedPromptDataset output
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> "GenConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


@dataclass
class FilterConfig:
    text_pad_id: int                       # must be injected
    min_frames: int = 250                  # discard under 20 s
    silence_ratio_min: float = 0.15        # lower bound on each speaker's PAD ratio (detects both talking nonstop)
    silence_ratio_max: float = 0.97        # upper bound (detects permanent silence)
    audio_ngram: int = 8                   # window for semantic-codebook loop detection
    audio_diversity_min: float = 0.2       # lower bound on unique n-gram ratio (a periodic loop collapses to 1/period)
    text_ngram: int = 4
    text_diversity_min: float = 0.3

    def __post_init__(self):
        # yaml null → None would silently make pad_ratio 0 and reject every
        # dialogue as "never_silent" — fail loudly instead.
        assert self.text_pad_id is not None, \
            "text_pad_id is unset (configs/data/filter.yaml) - finalize the PAD/EPAD mapping first"

    @classmethod
    def from_yaml(cls, path: str) -> "FilterConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


# ---------------- generation engine ----------------

class MoshiSelfTalkEngine:
    """Batched cross-feed adapter over Moshi streaming generation.

    Responsibilities:
      * reset(batch): initialize KV cache / streaming state
      * prime(seed_audio_codes): prime the first M frames with external audio
        (prevents mode collapse)
      * step(other_codes) -> (self_codes, text_tokens, topk_val, topk_idx)
          other_codes: (B, K) the other stream fed in this frame
          self_codes:  (B, K) own stream generated this frame
          topk_*:      (B, K, topk) raw-logit top-k
    """

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        # TODO(integration): temporarily use the kyutai inference code until the team fork is finalized
        # from moshi.models import loaders
        # ckpt = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        # self.mimi = ckpt.get_mimi(device=cfg.device)
        # self.lm = ckpt.get_moshi(device=cfg.device)
        raise NotImplementedError("Connect once the team fork's streaming interface is finalized")

    def reset(self, batch: int) -> None: ...
    def prime(self, seed_audio_codes) -> None: ...
    def step(self, other_codes): ...


# ---------------- quality filter (pure functions) ----------------

def _ngram_diversity(seq: np.ndarray, n: int) -> float:
    """Unique n-grams / total n-grams. ≈1 for normal speech; collapses to 1/period for loops.

    Note: the "most-frequent n-gram share" approach has a blind spot — a period-p loop
    produces p rotated variants that dilute the share to 1/p — hence a diversity metric.
    """
    if len(seq) < n * 2:
        return 1.0
    grams = [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]
    return len(set(grams)) / len(grams)


def check_stream(codes: np.ndarray, text: np.ndarray, cfg: FilterConfig) -> tuple[bool, str]:
    """Check a single speaker stream. codes: (K, T), text: (T,)."""
    T = codes.shape[-1]
    if T < cfg.min_frames:
        return False, "too_short"

    pad_ratio = float((text == cfg.text_pad_id).mean())
    if pad_ratio > cfg.silence_ratio_max:
        return False, "always_silent"
    if pad_ratio < cfg.silence_ratio_min:
        return False, "never_silent"

    if _ngram_diversity(codes[0], cfg.audio_ngram) < cfg.audio_diversity_min:
        return False, "audio_loop"

    spoken = text[text != cfg.text_pad_id]
    if _ngram_diversity(spoken, cfg.text_ngram) < cfg.text_diversity_min:
        return False, "text_repeat"

    return True, "ok"


def check_dialogue(sample_npz: dict, cfg: FilterConfig) -> tuple[bool, dict]:
    """Accept only if both A and B pass. Reasons are kept for tuning generation parameters."""
    reasons = {}
    ok_all = True
    for side in ("a", "b"):
        ok, reason = check_stream(sample_npz[f"codes_{side}"], sample_npz[f"text_tokens_{side}"], cfg)
        reasons[side] = reason
        ok_all &= ok
    return ok_all, reasons


# ---------------- dataset ----------------

class EnKDDialogueDataset(BaseDataset):
    """en_kd: self-talk generation (build) → quality filter (filter) → Sample (iter_samples)."""

    name = "en_kd"
    source = "en_kd"
    lang = "en"
    sample_type = "en_kd"

    def __init__(
        self,
        out_dir: str | Path = "data/generated",
        gen_cfg: GenConfig | None = None,        # needed only for build
        filter_cfg: FilterConfig | None = None,  # needed only for filter
    ):
        super().__init__(out_dir)
        self.gen_cfg = gen_cfg
        self.filter_cfg = filter_cfg

    @classmethod
    def from_cli(cls, args) -> "EnKDDialogueDataset":
        return cls(
            out_dir=args.out_dir,
            gen_cfg=GenConfig.from_yaml(args.gen_config) if args.gen_config else None,
            filter_cfg=FilterConfig.from_yaml(args.filter_config) if args.filter_config else None,
        )

    def build(self, limit: int | None = None) -> dict:
        """Generate, then run the filter too when a filter config is present."""
        stats = self._generate(limit)
        if self.filter_cfg is not None:
            stats["filter"] = self.filter()
        return stats

    def _generate(self, limit: int | None = None) -> dict:
        import torch

        cfg = self.gen_cfg
        assert cfg is not None, "build() requires gen_cfg"
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        engine = MoshiSelfTalkEngine(cfg)
        seed_prompts = sorted(Path(cfg.seed_prompt_dir).glob("*.safetensors"))  # pre-encoded seed codes

        n_dialogues = min(limit, cfg.n_dialogues) if limit else cfg.n_dialogues
        n_batches = (n_dialogues + cfg.batch_pairs - 1) // cfg.batch_pairs
        K, N, TOPK = NUM_CODEBOOKS, cfg.batch_pairs, cfg.topk_dump

        with torch.inference_mode():
            for b in range(n_batches):
                engine.reset(batch=2 * N)  # [0:N]=A, [N:2N]=B

                prompt = seed_prompts[rng.integers(len(seed_prompts))] if seed_prompts else None
                prompt_id = prompt.stem if prompt else "none"
                # TODO: load prompt → engine.prime(...)

                # buffers: (2N, K, T), (2N, T), (2N, K, T, topk)
                codes = torch.zeros(2 * N, K, cfg.max_frames, dtype=torch.int16)
                text = torch.zeros(2 * N, cfg.max_frames, dtype=torch.int32)
                tk_val = torch.zeros(2 * N, K, cfg.max_frames, TOPK, dtype=torch.float16)
                tk_idx = torch.zeros(2 * N, K, cfg.max_frames, TOPK, dtype=torch.int16)

                # other-stream input at t=0: silence (assumes the engine provides silence codes)
                prev = codes[:, :, 0].clone()

                for t in range(cfg.max_frames):
                    # cross-feed: A hears B's previous output, B hears A's previous output
                    other = torch.cat([prev[N:], prev[:N]], dim=0)
                    self_codes, text_t, val_t, idx_t = engine.step(other.to(cfg.device))
                    codes[:, :, t] = self_codes.cpu()
                    text[:, t] = text_t.cpu()
                    tk_val[:, :, t] = val_t.cpu().to(torch.float16)
                    tk_idx[:, :, t] = idx_t.cpu().to(torch.int16)
                    prev = self_codes.cpu()

                # serialize per pair as Sample → jsonl+npz (prepare_dataset converts to Arrow)
                for i in range(N):
                    meta = {
                        "seed": cfg.seed * 100_000 + b * N + i,
                        "gen_temperature": cfg.gen_temperature,
                        "gen_top_k": cfg.gen_top_k,
                        "seed_prompt_id": prompt_id,
                    }
                    s = Sample(
                        sample_type=self.sample_type,
                        lang=self.lang,
                        codes_a=codes[i].numpy(),
                        codes_b=codes[N + i].numpy(),
                        text_tokens_a=text[i].numpy(),
                        text_tokens_b=text[N + i].numpy(),
                        teacher_topk_val_a=tk_val[i].numpy(),
                        teacher_topk_idx_a=tk_idx[i].numpy(),
                        teacher_topk_val_b=tk_val[N + i].numpy(),
                        teacher_topk_idx_b=tk_idx[N + i].numpy(),
                        gen_meta=meta,
                        sample_uid=self._uid(cfg.seed, b * N + i, prompt_id),
                    )
                    np.savez_compressed(self.out_dir / f"{s.sample_uid}.npz", **{
                        k: v for k, v in s.__dict__.items()
                        if isinstance(v, np.ndarray)
                    })
                    with open(self.out_dir / f"{s.sample_uid}.json", "w") as f:
                        json.dump({"sample_type": s.sample_type, "lang": s.lang,
                                   "gen_meta": meta, "sample_uid": s.sample_uid}, f)

                print(f"[{self.name}] batch {b + 1}/{n_batches} done → {self.out_dir}")

        return {"n_dialogues": n_dialogues}

    def ingest_ab_selfplay(self, src_dir: str | Path) -> dict:
        """Convert Colab ab_selfplay outputs (dialogue_*.npz + .meta.json) into
        the en_kd artifact convention.

        The Colab capture stores top-k logits per *internal model step* in the
        delayed joint-sequence layout, while this pipeline stores everything
        delay-free and frame-aligned (the collator re-applies delay at train
        time — storing the delayed layout would double-apply it). Instead of
        reasoning about LMGen internals, the step→frame mapping is solved
        empirically per codebook: find the offset where the captured *sampled*
        tokens exactly match the emitted own-stream tokens, and fail loudly if
        no exact match exists.

        A and B have different lengths (B starts at the seed handoff), so both
        are cropped to B's active window — the seed intro is not dialogue.
        """
        src = Path(src_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stats = {"ingested": 0, "frames": 0}

        for p in sorted(src.glob("dialogue_*.npz")):
            z = np.load(p)
            meta = json.loads(p.with_name(p.stem + ".meta.json").read_text())
            exec_log = np.asarray(meta["exec_mask_log"], dtype=bool)   # (R, 2)

            own = {"a": z["own_audio_tokens_A"], "b": z["own_audio_tokens_B"]}  # (T, K)
            text = {"a": z["text_tokens_A"], "b": z["text_tokens_B"]}
            K = own["a"].shape[1]

            # --- capture (internal steps) → per-frame teacher top-k ---
            topk_v, topk_i = {}, {}
            for item, side in enumerate(("a", "b")):
                rows = np.flatnonzero(exec_log[:, item])   # rows where this item ran
                sampled = z["cap_audio_sampled"][rows, item]           # (Ri, K)
                T = len(own[side])
                offs = []
                for k in range(K):
                    o = next((o for o in range(len(rows) - T + 1)
                              if np.array_equal(sampled[o:o + T, k], own[side][:, k])),
                             None)
                    assert o is not None, \
                        f"{p.name}/{side}: codebook {k} capture↔output alignment failed"
                    offs.append(o)
                topk_v[side] = np.stack(
                    [z["cap_audio_topk_vals"][rows[offs[k]:offs[k] + T], item, k]
                     for k in range(K)])                               # (K, T, topk)
                topk_i[side] = np.stack(
                    [z["cap_audio_topk_idx"][rows[offs[k]:offs[k] + T], item, k]
                     for k in range(K)])

            # --- crop both to B's active window (drop the seed intro) ---
            T_b = len(own["b"])
            start_a = len(own["a"]) - T_b
            assert start_a >= 0

            def crop(side, arr, axis_t):
                return arr[start_a:] if side == "a" and axis_t == 0 else \
                       arr[:, start_a:] if side == "a" else arr

            seed_id = Path(meta.get("seed_wav", "")).stem
            uid = hashlib.sha1(f"ab_selfplay:{p.stem}:{seed_id}".encode()).hexdigest()[:16]
            gen_meta = {
                "seed": -1,   # Colab run — no integer seed recorded
                "gen_temperature": float(meta.get("temp", 0.0)),
                "gen_top_k": int(meta.get("top_k", 0)),
                "seed_prompt_id": seed_id,
            }
            np.savez_compressed(self.out_dir / f"{uid}.npz",
                codes_a=crop("a", own["a"], 0).T.astype(np.int16),
                codes_b=own["b"].T.astype(np.int16),
                text_tokens_a=crop("a", text["a"], 0).astype(np.int32),
                text_tokens_b=text["b"].astype(np.int32),
                teacher_topk_val_a=crop("a", topk_v["a"], 1).astype(np.float16),
                teacher_topk_idx_a=crop("a", topk_i["a"], 1).astype(np.int16),
                teacher_topk_val_b=topk_v["b"].astype(np.float16),
                teacher_topk_idx_b=topk_i["b"].astype(np.int16),
            )
            (self.out_dir / f"{uid}.json").write_text(json.dumps({
                "sample_type": self.sample_type, "lang": self.lang,
                "gen_meta": gen_meta, "sample_uid": uid,
                "src": p.name,
            }))
            stats["ingested"] += 1
            stats["frames"] += T_b

        print(f"[{self.name}] ingested {stats['ingested']} dialogues "
              f"({stats['frames']} frames) → {self.out_dir}")
        if self.filter_cfg is not None:
            stats["filter"] = self.filter()
        return stats

    def filter(self) -> dict:
        """Remove collapsed dialogues → accepted.json + pass-rate report.
        Re-run standalone when tuning thresholds."""
        cfg = self.filter_cfg
        assert cfg is not None, "filter() requires filter_cfg"
        stats: Counter = Counter()
        accepted = []

        for npz_path in sorted(self.out_dir.glob("*.npz")):
            data = dict(np.load(npz_path))
            ok, reasons = check_dialogue(data, cfg)
            stats[f"a:{reasons['a']}"] += 1
            stats[f"b:{reasons['b']}"] += 1
            stats["accept" if ok else "reject"] += 1
            if ok:
                accepted.append(npz_path.stem)

        manifest = self.out_dir / "accepted.json"
        manifest.write_text(json.dumps(accepted, indent=1))
        report = {"config": asdict(cfg), "stats": dict(stats),
                  "pass_rate": stats["accept"] / max(1, stats["accept"] + stats["reject"])}
        (self.out_dir / "filter_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["stats"], indent=2))
        print(f"pass_rate={report['pass_rate']:.3f} → {manifest}")
        return report

    def iter_samples(self) -> Iterator[Sample]:
        """Emit only the dialogues that passed the filter (accepted.json)."""
        accepted = set(json.loads((self.out_dir / "accepted.json").read_text()))
        for uid in sorted(accepted):
            arr = dict(np.load(self.out_dir / f"{uid}.npz"))
            meta = json.loads((self.out_dir / f"{uid}.json").read_text())
            yield Sample(
                sample_type=self.sample_type, lang=self.lang,
                codes_a=arr["codes_a"], codes_b=arr["codes_b"],
                text_tokens_a=arr["text_tokens_a"], text_tokens_b=arr["text_tokens_b"],
                teacher_topk_val_a=arr["teacher_topk_val_a"],
                teacher_topk_idx_a=arr["teacher_topk_idx_a"],
                teacher_topk_val_b=arr["teacher_topk_val_b"],
                teacher_topk_idx_b=arr["teacher_topk_idx_b"],
                gen_meta=meta["gen_meta"], sample_uid=uid,
                # An en_kd row carries BOTH dialogue streams, i.e. two voices, so a
                # single speaker id cannot describe it. The per-stream identity is
                # recoverable from en_solo crops instead (speaker="{uid}:{a|b}").
                speaker="",
            )

    @staticmethod
    def _uid(seed: int, pair_idx: int, prompt_id: str) -> str:
        return hashlib.sha1(f"{seed}:{pair_idx}:{prompt_id}".encode()).hexdigest()[:16]
