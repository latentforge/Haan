"""Generated family: ingest dual-Moshi self-talk dialogues → quality filter → en_kd samples.

**Generation is out of scope for this package.** en_kd dialogues are produced by
running the teacher (a Moshi self-play harness, the Colab notebook
`notebooks/selfplay.ipynb`), which is teacher *inference*, not corpus
preparation. Every other builder here parses an existing corpus; this one used to
also create it, and the mismatch cost real time — the in-repo generator was a
`NotImplementedError` stub whose docstring blamed an unfinished "team fork
streaming interface", and that turned out not to exist as a blocker at all
(`moshi.models.LMGen` has supported batched streaming and per-item exec masks all
along; see plan section 9.5). This module's contract is now the narrow one:

    dialogue_*.npz + .meta.json  ->  filtered, retokenized en_kd Samples

How the dialogues are made, for reading the artifacts:
  Moshi A and B share weights, so the model is not loaded twice. One LMGen runs at
  batch 2 with item 0 = A and item 1 = B, and every frame each item's own-stream
  audio tokens are fed as the other's "other stream" input. Cross-feed is at token
  level (they are already Mimi tokens), so there is no decode/re-encode loss.

Per frame the harness dumps:
  - each role's own-stream audio codes (sampled hard tokens)
  - each role's inner-monologue text tokens, in the **teacher's** Helium vocabulary
  - top-k (val, idx) of the raw logits right before sampling — KD soft labels,
    no temperature applied

Quality filter — typical self-talk collapse modes and how they are detected:
  1) one side permanently silent / both keep talking → PAD ratio of the text stream
  2) audio-token n-gram loops → repetition pattern in the semantic codebook (0)
  3) same sentence repeated → n-gram duplication rate over non-PAD text tokens
  All thresholds are config-injected. The pass rate is fed back into the
  generation hyperparameters.
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

from project_amnesty.datasets.shared.schema import Sample
from ..base import BaseDataset
from ..mixins import (
    MOSHI_TEXT_PAD_ID,
    MOSHI_TEXT_TOKENIZER,
    TextTokCfg,
    retokenize_frame_aligned,
)


# ---------------- configs ----------------

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
        # utf-8: configs carry Korean comments (cp949 default breaks on Windows).
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


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
    """en_kd: ingest teacher self-play dialogues → quality filter → Sample. No generation."""

    name = "en_kd"
    source = "en_kd"
    lang = "en"
    sample_type = "en_kd"

    # Teacher-side text vocabulary. The generator emits Helium/SentencePiece ids;
    # everything downstream (filter, en_solo, collator, loss weights) assumes the
    # student vocabulary, so ingest retokenizes. See mixins.retokenize_frame_aligned.
    src_tokenizer_name = MOSHI_TEXT_TOKENIZER
    src_pad_id = MOSHI_TEXT_PAD_ID

    def __init__(
        self,
        out_dir: str | Path = "data/generated",
        filter_cfg: FilterConfig | None = None,  # needed only for filter
        text_cfg: TextTokCfg | None = None,      # needed only for ingest (SeqKD)
    ):
        super().__init__(out_dir)
        self.filter_cfg = filter_cfg
        self.text_cfg = text_cfg

    def _tokenizers(self):
        """(source, target) text tokenizers, loaded once."""
        pair = getattr(self, "_tok_pair", None)
        if pair is None:
            from transformers import AutoTokenizer
            pair = self._tok_pair = (
                AutoTokenizer.from_pretrained(self.src_tokenizer_name),
                AutoTokenizer.from_pretrained(self.text_cfg.tokenizer_name),
            )
        return pair

    @classmethod
    def from_cli(cls, args) -> "EnKDDialogueDataset":
        return cls(
            out_dir=args.out_dir or "data/generated",
            filter_cfg=FilterConfig.from_yaml(args.filter_config) if args.filter_config else None,
            text_cfg=TextTokCfg.from_yaml(args.text_config) if args.text_config else None,
        )

    def build(self, limit: int | None = None) -> dict:
        """No build stage: en_kd artifacts come from the teacher, not from here.

        Kept to satisfy the BaseDataset contract, and to fail with a pointer
        rather than silently producing nothing. `ensure_prepared` never reaches
        this -- build_group() goes straight to iter_samples() -- so only the CLI
        `--stage build` lands here.
        """
        raise NotImplementedError(
            "en_kd is not generated in-repo. Run the Moshi self-play harness "
            "(notebooks/selfplay.ipynb) to produce dialogue_*.npz + "
            ".meta.json, then ingest them:\n"
            "    python -m project_amnesty.datasets en_kd --stage ingest "
            "--root <dialogues_dir> --text-config configs/data/text_tok.yaml"
        )

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

        Text is retokenized here (SeqKD, RISKS §7.2): the capture is in the
        teacher's Helium vocabulary and every downstream consumer — the quality
        filter, en_solo's activity mask, the collator's PAD/EPAD loss weights —
        compares against the *student's* PAD id. Passing the teacher ids through
        fails silently, not loudly: PAD never matches, so the filter scores every
        dialogue as `never_silent` and rejects the whole corpus.
        """
        assert self.text_cfg is not None, \
            "ingest_ab_selfplay() requires text_cfg (SeqKD retokenization)"
        src = Path(src_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stats = {"ingested": 0, "frames": 0}

        for p in sorted(src.glob("dialogue_*.npz")):
            z = np.load(p)
            meta = json.loads(p.with_name(p.stem + ".meta.json").read_text())
            exec_log = np.asarray(meta["exec_mask_log"], dtype=bool)   # (R, 2)

            own = {"a": z["own_audio_tokens_A"], "b": z["own_audio_tokens_B"]}  # (T, K)
            # SeqKD: the capture is in the teacher's Helium vocabulary; retokenize
            # into the student's before anything downstream reads a PAD id.
            src_tok, tgt_tok = self._tokenizers()
            text = {
                side: retokenize_frame_aligned(
                    z[f"text_tokens_{side.upper()}"],
                    src_tok, self.src_pad_id, tgt_tok, self.text_cfg,
                )
                for side in ("a", "b")
            }
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
