"""train.py -- training entry point (with loop).  [FIXED]

The location/name of this file (`project_amnesty/utils/train.py`) must not change
(`filemap.md` fixed rule). `runner.py` (the Phase manager) drives this entry point
per phase.

On top of the canonical PyTorch training loop
(`train_loop`: forward -> loss -> backward -> optimizer.step -> zero_grad) we layer
the Haan training stack:
  - **Distributed**: FSDP2 (`fully_shard`). Default `reshard_after_forward=False`
    (keep parameters replicated, shard grad/optim state only) -- a 9.15B bf16 copy
    (~16GB per GPU) sits comfortably resident on an A100 80GB, so the parameters
    themselves are not sharded. Fall back to True (ZeRO-3 class) only when VRAM is
    tight. (TRAINING_CURRICULUM 3.3)
  - **Optimizer**: PagedAdamW8bit -- the A100 (Ampere) cannot accelerate FP8 tensor
    cores, so an 8-bit optimizer yields the same memory savings. (TRAINING_CURRICULUM 3.3)
  - **Kernels**: Liger Kernel (fused RMSNorm/RoPE/SwiGLU/CE/FusedLinearCE). Compatible
    with FlashAttention and FSDP. (TRAINING_CURRICULUM 5.1)
  - **Loss**: semantic-KD KL (Mimi level-0 logit) + Korean TTS CE + voice-cloning CE +
    text anchor CE combined. Acoustic codebooks (1~7) are excluded from KD by default
    (timbre carriers). It consumes the per-token weights the collator ships (stream PAD
    text x0.3, non-semantic audio x0.02) and the semantic-KD internal frame weights
    (speech / turn-transition regions) as-is. Zone A (system prompt) regions and batch
    pad are fully masked out of the loss. (ARCHITECTURE 5.1/7.6, RISKS 7.4)
  - **Diagnostic hooks**: grad-norm (per-task gradient dominance watch) - self/user role
    vector cosine separation - Depth batch-2 two-element output-collapse probing.
    (ARCHITECTURE 3.5/5.4, RISKS 2/3)

Run: `python -m project_amnesty.utils.train --config configs/phase2_joint.json`
Config is JSON (no yaml). ModelArgs/DataArgs/TrainArgs are defined at the top of this
file as @dataclass per the HF `run_clm.py` convention and parsed with `HfArgumentParser`.

Note: `project_amnesty/datasets/` is a REAL, fully-implemented data stack and
`project_amnesty/losses/` provides the real semantic-KD loss, so `build_dataloader` and
`build_loss_fn` delegate to them for real (lazy import). `project_amnesty/models/` exposes
the real transformers-based classes (HaanConfig / HaanModel / HaanForConditionalGeneration);
the configs are functional, but the forward and Moshi warm-start (`from_pretrained`) bodies are
still TODO and raise NotImplementedError -- so `build_model` is the only builder that cannot yet
run end to end. Everything else here (FSDP2 wrapping, optimizer construction, the training loop,
checkpoint I/O, diagnostics, argument parsing) is implemented for real and imports cleanly with
no heavy dependencies installed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# This is an FSDP2 training entry point: it only ever runs under `torchrun` with torch present,
# so torch/transformers are ordinary top-level imports (matching torchtitan / pytorch examples).
# The one exception is bitsandbytes, imported lazily in build_optimizer (it probes CUDA at import
# time, which can warn/fail even when torch is fine).
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from transformers import HfArgumentParser


# ======================================================================================
# 1. Argument definitions (HF run_clm.py convention: ModelArgs / DataArgs / TrainArgs)
# ======================================================================================

@dataclass
class ModelArgs:
    """Backbone / audio / Depth / warm-start settings. (ARCHITECTURE 1/3/5.4)"""

    backbone: str = "Qwen/Qwen3-8B"
    moshi_ckpt: str = "kmhf/hf-moshiko"  # warm-start source (emb.8~15, depformer, linears)
    num_codebooks: int = 8  # K (number of audio codebooks in the self stream = dep_q)
    depth_dim: int = 1024  # Depth Transformer internal dim (independent of backbone dim, 5.4.1)
    audio_cardinality: int = 2048  # frozen Mimi shared -> teacher/student identical
    # Audio embedding table is shared across self/user (8 books). Role is distinguished by a
    # learned additive Role Token. (3.3)
    share_scope: str = "semantic+acoustic"  # ablation: "semantic" | "semantic+acoustic" (3.6)
    init_source: str = "user"  # codebook init: "user"(emb.8~15) | "self"(emb.0~7) | "random" (5.4.2)
    # Depth parallel-prediction mode switch: train q16 (self+user, batch 2) / live inference
    # q8 (self, batch 1). (5.4)
    depth_mode: str = "q16"  # "q16"(training/simulation) | "q8"(live conversation)


@dataclass
class DataArgs:
    """Data root / frames / collator wiring settings. (DATA_STRATEGY 4)"""

    # SUPERSEDED: the whole data stack (root/dataset/mix/sampler/dataloader) now comes from
    # `configs/data/loader.yaml` via `datasets.loader.load_configs` (see build_dataloader). The
    # fields below are kept only for back-compat / phase-JSON tolerance (a phase config may still
    # carry them) -- removing them would break `parse_args` on those JSONs. They are NOT read by the
    # real data stack anymore.
    root: str = "data/prepared"
    max_frames: int = 750  # context cap at 12.5Hz (60 seconds)
    double_ab: bool = True  # reuse the same conversation once in each A/B direction (role swap) (DATA 4.2)
    config_json: str = "configs/data.json"  # datasets pipeline config (JSON)
    tokens_json: str = "configs/tokens.json"  # special-token slot assignment for PAD/EPAD etc. (7.6)
    ko_ratio: float = 0.1  # Korean data ratio (ramped up in Phase 2: 0.1->0.3->0.5, Phase 2)
    num_workers: int = 4

    # --- Collator delay (per-phase knob; NOT in loader.yaml by design, ARCH 5.0.2) ---
    # `delay` changes between curriculum phases, so it is supplied here per phase rather than baked
    # into the static YAML. build_dataloader threads these into the KDCollator's DelayConfig.
    acoustic_delay: int = 1        # Phase 1+ conversation default (ARCH 5.0.2 / runner: acoustic 1, text 0).
    text_delay_frames: int = 0     # Phase 0 pre-training override = acoustic 2 / text +-0.6 via per-phase config.


@dataclass
class TrainArgs:
    """Training hyperparameters / distributed / optimizer / loss weights / diagnostic intervals.
    (TRAINING_CURRICULUM 3/5, ARCHITECTURE 7.6)"""

    phase: str = "phase2_joint"  # phase identifier selected by runner.py

    # --- Distributed (FSDP2) ---
    reshard_after_forward: bool = False  # False=ZeRO-2 class (param replicate), True=ZeRO-3 class (3.3)

    # --- Optimizer ---
    optim: str = "paged_adamw_8bit"  # A100 -> 8-bit optimizer for memory savings (3.3)
    lr: float = 2e-5
    audio_param_lr: float = 2e-5  # dedicated lr for audio embedding / Depth heads / Role Token
    #                               (can be lowered early to prevent drift, 7.5)
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    grad_accum: int = 4
    warmup_steps: int = 500
    steps: int = 45000
    epochs: int = 1  # optional epoch cap; the loop stops at whichever of steps/epochs comes first
    lr_scheduler_type: str = "cosine"  # transformers get_scheduler type; cosine decay w/ warmup (fix1)

    # --- Kernels ---
    use_liger_kernel: bool = True  # (5.1)
    liger_config: dict = field(default_factory=lambda: {
        "rope": True, "swiglu": True, "cross_entropy": True,
        "fused_linear_cross_entropy": True, "rms_norm": True,
    })

    # --- Fine-tuning mode ---
    full_ft: bool = True  # Phase 1~3/5 are Full FT (avoid shortcuts, 3.2). Only Phase 4 uses LoRA.
    freeze_audio_emb_steps: int = 0  # if >0, freeze audio embeddings for the first N steps then warm up (RISKS 7.5)
    # Activation (gradient) checkpointing: trade compute for memory. Enabled BEFORE the FSDP2 wrap
    # in main() so re-materialized activations live under the sharded module (fix3, RISKS/3.3).
    gradient_checkpointing: bool = True

    # --- Combined loss weights ---
    kd_weight: float = 1.0  # semantic KD KL term (5.1)
    ce_weight: float = 1.0  # Korean TTS CE + voice-cloning CE term
    text_anchor_weight: float = 0.1  # multilingual text-ability anchor (small, all regions, RISKS 7.6)
    kd_temperature: float = 1.0  # KL distillation temperature
    pad_text_weight: float = 0.3  # down-weight stream PAD text tokens (7.6)
    non_sem_audio_weight: float = 0.02  # down-weight non-semantic (acoustic) audio tokens (7.6)

    # --- KD operating mode ---
    kd_logit_dump: str = ""  # empty = live teacher; if set, path to pre-dumped top-k logits (offline, RISKS 7.3)
    on_policy_ratio: float = 0.0  # if >0, fraction of self-rollout via scheduled sampling (RISKS 7.3)

    # --- Diagnostics / checkpoint / logging ---
    log_interval: int = 20
    probe_interval: int = 500  # role-vector cosine / batch-collapse probing interval (3.5/5.4)
    ckpt_interval: int = 2000
    keep_last_n_ckpts: int = 3  # keep-last-N rotation of periodic step_* checkpoints (fix8); final always kept
    out_dir: str = "checkpoints"
    resume: str = ""  # checkpoint path to resume from (injected by runner.py on phase transition)
    seed: int = 42
    bf16: bool = True

    # Substring keys used to identify the "brand-new" parameters (audio RVQ embeddings,
    # Depth output heads, Role Token) that must always train in full precision (3.1).
    new_param_keys: tuple[str, ...] = (
        "audio_emb", "depth", "role_emb", "linears", "depformer",
    )


# ======================================================================================
# 2. Component builders (datasets/losses are real; only models/'s forward + warm-start bodies are TODO -> NotImplementedError)
# ======================================================================================

def build_model(margs: ModelArgs, targs: TrainArgs) -> "torch.nn.Module":
    """Build Qwen3 backbone + shared audio embeddings (8) + Role Token (2) + shared Depth, then warm-start.

    - Audio input embeddings (8 shared) are copied from Moshi `emb.8~15` (user side) (5.4.2).
    - The Depth body / `linears.0~7` / `depformer_emb` are warm-started from Moshi; `depformer_in`
      (4096->1024) is initialized from Moshi values but retrained (an adapter over the unaligned
      4096 space, 5.4.1).
    - Liger Kernel is injected into the backbone via `use_liger_kernel`/`liger_config` (5.1).
    """
    # models/ provides the real class; construction + Moshi warm-start live inside
    # HaanForConditionalGeneration.from_pretrained, whose body is still TODO (ARCH 5.4.1/5.4.2).
    from project_amnesty.models import HaanForConditionalGeneration

    return HaanForConditionalGeneration.from_pretrained(
        margs.backbone,
        moshi_ckpt=margs.moshi_ckpt,
        init_source=margs.init_source,
        share_scope=margs.share_scope,
        num_codebooks=margs.num_codebooks,
        depth_dim=margs.depth_dim,
        depth_mode=margs.depth_mode,
        audio_cardinality=margs.audio_cardinality,
        use_liger_kernel=targs.use_liger_kernel,
        liger_config=targs.liger_config,
    )


def build_dataloader(dargs: DataArgs, targs: TrainArgs, split: str = "train") -> "LoaderBundle":
    """Assemble the real data stack via datasets.load_configs + build_dataloader -> LoaderBundle.
    Config (root/dataset/mix/sampler/dataloader) comes from configs/data/loader.yaml; only `delay`
    is a per-phase knob supplied here (ARCH 5.0.2 -- not in the static YAML).

    The returned `LoaderBundle` exposes `.loader` (the iterable DataLoader), `.set_epoch(epoch)`
    (advances datasets + sampler RNG together), and `.state_dict()`/`.load_state_dict()` for resume;
    train_loop / save_checkpoint / load_checkpoint drive those. The collator batch is passed to the
    model as-is; it carries `input_ids`, `audio_codes`, the teacher top-k dump, `kd_frame_weight`,
    and the per-token loss weights the loss consumes (ARCH 7.6/7.4).
    """
    # Heavy datasets imports stay LAZY here so the module py_compiles / imports with torch alone.
    from project_amnesty.datasets.runtime.loader import build_dataloader as _build, load_configs
    from project_amnesty.datasets.runtime.collator import KDCollatorConfig, DelayConfig

    # All static config (root/dataset/mix/sampler/dataloader) is read from configs/data/loader.yaml.
    data_cfg, loader_cfg, mix_cfg = load_configs()
    # `delay` is the one per-phase knob not in the YAML: thread DataArgs' phase delays into the collator.
    collator_cfg = KDCollatorConfig(
        tokens=data_cfg.tokens,
        delay=DelayConfig(acoustic_delay=dargs.acoustic_delay, text_delay_frames=dargs.text_delay_frames),
    )
    return _build(
        data_cfg=data_cfg, loader_cfg=loader_cfg, split=split,
        rank=_dist_rank(), world_size=_world_size(),
        mix_cfg=(mix_cfg if split == "train" else None),  # mix_cfg is REQUIRED for split="train"
        collator_cfg=collator_cfg, max_steps=targs.steps,
    )


# --------------------------------------------------------------------------------------
# MODEL I/O + BATCH contract consumed by the loss (and by the on-policy hook).
# `datasets/` already emits this batch for real; the model's forward is the part still TODO.
# This is the documented interface both this file and the sibling `evaluate.py` are written
# against, so they stay aligned once the model forward lands.
# Every access below is guarded (getattr / `in` / `.get`) with a clear error, so a real
# model/collator that follows the contract "just works" and a non-conforming one fails loud.
#
# MODEL OUTPUT (`outputs = model(**batch)`):
#   - `outputs.text_logits`  Float (B, T, V_text)   Inner-Monologue text logits (ARCH 5.0.1).
#                            Falls back to `outputs.logits` when `text_logits` is absent.
#   - `outputs.audio_logits` Float (B, 2, K, T, C)  role r in {0=self, 1=user}, codebook
#                            k in {0..K-1}, C=2048 (frozen Mimi). Semantic level-0 == k==0 (ARCH 5.0/5.1).
#   - `model.generate(batch, mode="simulation", max_new_frames=..., ...)` -> obj/dict with
#                            `codes` (B, 2, K, T_gen) -- used only by the on-policy hook (ARCH 5.0.3/5.4, RISKS 7.3).
#
# BATCH (KDCollator output; keys guarded, teacher keys optional -> KD contributes 0):
#   - `input_ids`         (B, T)          long   text-stream token ids.
#   - `audio_codes`       (B, 2, K, T)    long   ground-truth Mimi codes (self/user, all K books).
#                                                This is the audio-CE supervision TARGET; it is never
#                                                overwritten by the on-policy hook (fix4).
#   - `input_audio_codes` (B, 2, K, T)    long   OPTIONAL scheduled-sampling INPUT conditioning: a
#                                                copy of `audio_codes` whose trailing self-frames may
#                                                be the model's own rollout (RISKS 7.3). Present only
#                                                when on_policy_ratio>0; the model conditions on it
#                                                while the loss still supervises the GT `audio_codes`.
#   - `role_ids`          (B, 2)          long   {self, user} row order (documentation; not read here).
#   - `text_loss_weight`  (B, T)          float  stream-PAD x0.3, EPAD x1, Zone A / batch-pad = 0 (ARCH 7.6).
#   - `audio_loss_weight` (B, 2, K, T)    float  semantic=1, non-sem x0.02, synthetic user ch=0,
#                                                Zone A / batch-pad = 0 (ARCH 7.6).
#   - `teacher_topk_val`  (B, 2, T, topk) float  teacher top-k logits (semantic k=0), real collator key.
#   - `teacher_topk_idx`  (B, 2, T, topk) long   teacher top-k support indices.
#   - `kd_valid`          (B, 2, T)       bool   frames with a valid teacher dump.
#   - `kd_frame_weight`   (B, 2, T)       float  silence/speech imbalance weight (RISKS 7.4).
#   - `target_aligned`    True                   collator tripwire (semantic_kd_loss_from_batch asserts it).
#   - `is_text_only`      (B,)            bool   pure-text rows for the anchor term (RISKS 7.6).
# --------------------------------------------------------------------------------------

_LOSS_EPS = 1e-8  # denominator floor -> a term with zero valid tokens reduces to 0 (never NaN)


def _weighted_token_ce(
    logits: "torch.Tensor", targets: "torch.Tensor", weights: "torch.Tensor"
) -> "torch.Tensor":
    """Per-token weighted cross-entropy, reduced to a scalar weighted mean. (ARCH 7.6)

    `logits` (..., C) float, `targets` (...) long, `weights` (...) float share the same leading
    dims. Returns sum(nll * weights) / sum(weights); when the weights sum to zero (every token
    masked -- Zone A / batch pad, ARCH 7.6) it returns 0 with no NaN (the floor `_LOSS_EPS`
    guards the division). Targets are clamped into [0, C) so masked/pad positions carrying an
    arbitrary id cannot trigger an out-of-range gather; their weight is 0 anyway. Logits are
    upcast to float32 for a numerically stable log-softmax under bf16 training.
    """
    C = logits.shape[-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = targets.clamp(0, C - 1).long().unsqueeze(-1)  # non-inplace -> never mutates the batch
    nll = -logp.gather(-1, tgt).squeeze(-1)
    w = weights.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(_LOSS_EPS)


def build_loss_fn(targs: TrainArgs) -> Callable[[Any, dict], "torch.Tensor"]:
    """Build the combined-loss callable: semantic KD KL + TTS/voice-cloning CE + text anchor CE.

    Returned-callable contract: `loss_fn(outputs, batch) -> (total_loss, metrics: dict[str, float])`.
    Consumes the MODEL I/O + BATCH contract documented just above this function.

      - **KD** (ARCH 5.1, RISKS 7.4): delegated to `losses.semantic_kd_loss_from_batch` -- KL over
        Mimi semantic (level-0, k==0) logits. teacher/student share the frozen Mimi output space (2048)
        so no projection is needed. Teacher = softmax of the collator's top-k dump
        (`teacher_topk_val`/`teacher_topk_idx`) at temperature `kd_temperature`; student = log-softmax
        of `audio_logits[:, 0, 0]` (role 0, cb 0). Per-frame KL is masked by `kd_valid` and weighted by
        `kd_frame_weight` (silence/speech imbalance -- no separate auxiliary term). Scaled by
        `kd_weight`. Contributes exactly 0 when the batch carries no teacher dump. The SAME
        `losses.semantic_kd_loss` serves the on-policy path so the objective cannot drift (Phase 5).
      - **Audio CE** (ARCH 7.6): cross-entropy of `audio_logits` vs `audio_codes` over all K
        codebooks, per-token weighted by `audio_loss_weight` (semantic=1, non-sem x0.02, synthetic
        user channel=0, Zone A / batch pad=0). Scaled by `ce_weight`.
      - **Text CE** (ARCH 5.0.1/7.6): cross-entropy of `text_logits` vs frame-aligned `input_ids`
        (Inner Monologue -- no shift; each frame already carries its text token), per-token weighted
        by `text_loss_weight` (stream PAD x0.3, Zone A / batch pad=0). Scaled by `ce_weight`.
      - **Anchor** (RISKS 7.6): a light pure-text CE restricted to `is_text_only` rows, guarding the
        backbone's multilingual text ability against catastrophic forgetting. Scaled by
        `text_anchor_weight`.

    total = kd + ce_audio + ce_text + anchor. Returned `metrics` are the post-weight contribution of
    each term (they sum to `total`) plus `total` itself -- floats for logging; per-task grad-norm
    dominance is watched separately by train_loop diagnostics (2). torch-only; numerically safe (no
    NaN when any term has zero valid tokens, no -inf materialization).
    """
    # Reference resolved: models/ now defines the output-head / vocab layout the loss reads.
    from project_amnesty.models import HaanForConditionalGeneration  # noqa: F401 -- output-layout reference

    # Semantic KD is delegated to the ONE shared objective (losses/kd.py) so the offline path here
    # and the on-policy path use an identical KL -- no diverging in-file copy (ARCH 5.1, Phase 5).
    from project_amnesty.losses import semantic_kd_loss_from_batch

    kd_weight = float(targs.kd_weight)
    ce_weight = float(targs.ce_weight)
    text_anchor_weight = float(targs.text_anchor_weight)
    temperature = max(float(targs.kd_temperature), _LOSS_EPS)  # guard against a 0/negative temperature

    def loss_fn(outputs: Any, batch: dict) -> tuple["torch.Tensor", dict]:
        if not isinstance(batch, dict):
            raise TypeError(
                "loss_fn expects the KDCollator batch dict (ARCH 7.6/7.4); got "
                f"{type(batch).__name__}."
            )

        # --- resolve model outputs against the documented I/O contract ---
        text_logits = getattr(outputs, "text_logits", None)
        if text_logits is None:
            text_logits = getattr(outputs, "logits", None)  # fallback (ARCH 5.0.1)
        if text_logits is None and isinstance(outputs, dict):
            text_logits = outputs.get("text_logits", outputs.get("logits"))
        audio_logits = getattr(outputs, "audio_logits", None)
        if audio_logits is None and isinstance(outputs, dict):
            audio_logits = outputs.get("audio_logits")
        assert audio_logits is not None, (
            "model outputs.audio_logits (B,2,K,T,C) is required for the audio CE / semantic KD "
            "terms (ARCH 5/5.1/7.6); model does not follow the I/O contract."
        )
        assert text_logits is not None, (
            "model outputs.text_logits (B,T,V) [or .logits] is required for the text CE / anchor "
            "terms (ARCH 5.0.1/7.6)."
        )

        # --- (1) audio CE over all K codebooks, per-token weighted (ARCH 7.6) ---
        assert "audio_codes" in batch and "audio_loss_weight" in batch, (
            "batch must carry audio_codes (B,2,K,T) and audio_loss_weight (B,2,K,T) (ARCH 7.6)."
        )
        ce_audio_raw = _weighted_token_ce(
            audio_logits, batch["audio_codes"], batch["audio_loss_weight"]
        )

        # --- (2) text CE (Inner Monologue), per-token weighted (ARCH 5.0.1/7.6) ---
        assert "input_ids" in batch and "text_loss_weight" in batch, (
            "batch must carry input_ids (B,T) and text_loss_weight (B,T) (ARCH 5.0.1/7.6)."
        )
        input_ids = batch["input_ids"]
        text_w = batch["text_loss_weight"]
        ce_text_raw = _weighted_token_ce(text_logits, input_ids, text_w)

        # --- (3) semantic KD KL on k=0, delegated to losses.semantic_kd_loss (ARCH 5.1, RISKS 7.4).
        #     Uses the REAL collator keys (`teacher_topk_val`/`teacher_topk_idx`, kd_align spec); 0
        #     when the batch carries no teacher dump (all-ko_tts batch). ---
        kd_metrics: dict[str, float] = {}
        if all(k in batch for k in ("teacher_topk_val", "teacher_topk_idx", "kd_valid", "kd_frame_weight")):
            kd_raw, kd_metrics = semantic_kd_loss_from_batch(
                audio_logits[:, 0, 0],  # role 0 (modeled speaker), codebook 0 -> (B, T, C)
                batch,
                tau=temperature,
            )
        else:
            kd_raw = None  # a batch may lack the teacher dump -> KD contributes 0

        # --- (4) text anchor: light pure-text CE on is_text_only rows (RISKS 7.6) ---
        is_text_only = batch.get("is_text_only")
        if is_text_only is not None:
            # Reuse the same per-token text weights but zero out every non-text-only row.
            row = is_text_only.to(text_w.dtype).view(-1, *([1] * (text_w.dim() - 1)))
            anchor_raw = _weighted_token_ce(text_logits, input_ids, text_w * row)
        else:
            anchor_raw = None

        # --- combine (ARCH 5.1/7.6). Skipped terms are exact scalar zeros on the graph dtype. ---
        zero = audio_logits.new_zeros(())
        ce_audio = ce_weight * ce_audio_raw
        ce_text = ce_weight * ce_text_raw
        kd = kd_weight * kd_raw if kd_raw is not None else zero
        anchor = text_anchor_weight * anchor_raw if anchor_raw is not None else zero
        total = kd + ce_audio + ce_text + anchor

        metrics = {
            "kd": float(kd.detach()),
            "ce_audio": float(ce_audio.detach()),
            "ce_text": float(ce_text.detach()),
            "anchor": float(anchor.detach()),
            "total": float(total.detach()),
        }
        metrics.update(kd_metrics)  # losses/kd.py diagnostics (kd/supervised_frac, teacher_support_mass, ...)
        return total, metrics

    return loss_fn


def setup_fsdp2(model: "torch.nn.Module", args: TrainArgs) -> "torch.nn.Module":
    """Wrap the backbone with `fully_shard`. The `reshard_after_forward` flag switches ZeRO-2/3 class. (3.3)

    Default False: keep parameters replicated (a 9.15B bf16 copy resides on an A100 80GB), shard
    only grad/optim state. If VRAM is tight, fall back to True to also shard parameters (ZeRO-3
    class). bf16 mixed precision. Each transformer block is sharded individually first, then the
    top-level module is wrapped -- compatible with Liger Kernel / FlashAttention (5.1).
    """
    # bf16 mixed-precision policy is optional across torch builds; degrade gracefully if absent.
    mp_policy = None
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy  # noqa: PLC0415

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            reduce_dtype=torch.float32,
        )
    except Exception:
        mp_policy = None

    def _shard(module: "torch.nn.Module") -> None:
        kwargs: dict[str, Any] = {"reshard_after_forward": args.reshard_after_forward}
        if mp_policy is not None:
            kwargs["mp_policy"] = mp_policy
        fully_shard(module, **kwargs)

    # Shard the repeated transformer blocks first (block-wise sharding lets FSDP2 overlap
    # comm/compute), then wrap the root. Locate the decoder-layer list generically so we do not
    # depend on the (not-yet-written) concrete model class.
    blocks = _find_transformer_blocks(model)
    for block in blocks:
        _shard(block)
    _shard(model)
    return model


def _find_transformer_blocks(model: "torch.nn.Module") -> list["torch.nn.Module"]:
    """Best-effort discovery of the repeated transformer-block list for block-wise FSDP2 sharding.

    Looks for a `ModuleList` reachable at common HF locations (`model.model.layers`,
    `model.layers`, `model.backbone.layers`, ...) or any `ModuleList` of length >= 2 whose
    children are all the same type. Returns [] if none is found (the root is still wrapped).
    """
    # Common attribute paths for HF-style decoder stacks.
    for path in ("model.layers", "layers", "backbone.layers", "model.model.layers",
                 "transformer.h", "model.decoder.layers"):
        obj: Any = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList) and len(obj) >= 2:
            return list(obj)

    # Fallback: scan submodules for a homogeneous ModuleList.
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 2:
            child_types = {type(c) for c in module}
            if len(child_types) == 1:
                return list(module)
    return []


def build_optimizer(model: "torch.nn.Module", args: TrainArgs):
    """PagedAdamW8bit. Audio embeddings / Depth heads / Role Token (brand-new parameters) always
    train in a full-precision group. (3.1)

    LoRA targets existing weight matrices, so it cannot apply to fully-new parameters like the audio
    RVQ embeddings, Depth output heads, and Role Token (two additive vectors) -> in every Phase
    these live in a separate param group and train in full. They use a separate lr
    (`audio_param_lr`) to curb early drift (RISKS 7.5). When `full_ft=False` (Phase 4) the backbone
    contributes only its LoRA adapters, while this new-parameter group still trains in full.

    "Full precision" here means the optimizer keeps 32-bit optimizer state for these parameters
    (registered via bitsandbytes GlobalOptimManager override) instead of the default 8-bit state,
    which is important for the sensitive new embedding/head parameters.
    """
    # Lazy heavy imports.
    import bitsandbytes as bnb  # noqa: PLC0415  (bnb.optim.PagedAdamW8bit)

    def _is_new_param(name: str) -> bool:
        return any(k in name for k in args.new_param_keys)

    new_params: list[Any] = []
    backbone_params: list[Any] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # e.g. frozen backbone when full_ft=False (only LoRA adapters remain trainable)
        (new_params if _is_new_param(name) else backbone_params).append(p)

    # Force 32-bit (full-precision) optimizer state for the new-parameter modules so the 8-bit
    # quantization does not corrupt their sensitive updates.
    try:
        mng = bnb.optim.GlobalOptimManager.get_instance()
        for module_name, module in model.named_modules():
            if any(k in module_name for k in args.new_param_keys):
                for pname, _ in module.named_parameters(recurse=False):
                    mng.register_module_override(module, pname, {"optim_bits": 32})
    except Exception:
        # If the running bitsandbytes build lacks the manager API, the param-group split below
        # still keeps these params in their own group; we simply skip the 32-bit override.
        pass

    groups: list[dict[str, Any]] = []
    if new_params:
        groups.append({"params": new_params, "lr": args.audio_param_lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr})
    if not groups:
        raise ValueError("build_optimizer: model has no trainable parameters")

    return bnb.optim.PagedAdamW8bit(
        groups,
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )


# ======================================================================================
# 3. Diagnostic hooks (ARCHITECTURE 3.5/5.4, RISKS 2/3)
# ======================================================================================

def run_diagnostics(model: "torch.nn.Module", outputs: Any, batch: dict, step: int) -> dict:
    """Early-warning probing during training. Instrumentation to leave failure mechanisms in an
    "explainable state" (the purpose of the RISKS doc).

    - **grad-norm dominance**: compare per-group gradient norms (new-param vs backbone). If one
      dominates, loss weights / PCGrad may need rebalancing (RISKS 2).
    - **role-vector separation**: cosine similarity of the self/user Role Token (and Depth role
      embedding). Watch that role distinction is not diluted under other loss pressure; role
      differentiation concentrates at the semantic level (3.5.1).
    - **batch-element collapse**: probe that the Depth batch-2 (self/user) elements do not collapse
      to identical output ignoring role; measure divergence of the two batch elements (5.4). On
      collapse, signal to promote the projection to two role-specific ones (split MLP).

    Returns a metrics dict; never raises on missing tensors (best-effort probing).
    """
    metrics: dict[str, float] = {"diag_step": float(step)}

    # --- grad-norm dominance (new-param group vs backbone group) ---
    # Under FSDP2 each rank owns only a shard of every parameter's grad, so a purely local sum is a
    # per-shard partial. Accumulate the per-group squared grad sums locally, then all_reduce(SUM)
    # across ranks BEFORE taking the sqrt/ratio so the reported dominance ratio is GLOBAL, not a
    # shard artifact (fix5). Best-effort: guarded on is_initialized() and never raises.
    new_sq = 0.0
    back_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        # fix7: under FSDP2 `p.grad` is a sharded DTensor. Calling `.pow(2).sum().item()` on it
        # would trigger an implicit redistribution/all-reduce to materialize the global scalar --
        # and the explicit `dist.all_reduce(SUM)` below would then reduce it a SECOND time
        # (double-counting, ~sqrt(world_size)x inflation in the reported norm). Take THIS rank's
        # local shard via `.to_local()` (no implicit comm) so `val` is a pure per-shard partial;
        # the single all_reduce below combines the partials exactly once. Plain (non-DTensor) grads
        # -- CPU/stub, or a non-sharded param -- have no `to_local`, so they pass through unchanged.
        local = g.to_local() if hasattr(g, "to_local") else g
        val = float(local.pow(2).sum().item())
        if any(k in name for k in ("audio_emb", "depth", "role_emb", "linears", "depformer")):
            new_sq += val
        else:
            back_sq += val

    try:
        if dist.is_available() and dist.is_initialized():
            acc = torch.tensor([new_sq, back_sq], dtype=torch.float64)
            dev = _first_param_device(model)
            if dev is not None and dev.type == "cuda":
                acc = acc.to(dev)  # nccl requires CUDA tensors; gloo reduces on CPU
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
            acc = acc.cpu()
            new_sq, back_sq = float(acc[0].item()), float(acc[1].item())
    except Exception:
        pass  # best-effort: fall back to the local partials if the reduction fails

    metrics["grad_norm_new"] = new_sq ** 0.5
    metrics["grad_norm_backbone"] = back_sq ** 0.5
    if back_sq > 0:
        metrics["grad_norm_ratio"] = (new_sq ** 0.5) / (back_sq ** 0.5)

    # --- role-vector separation (cosine similarity of the two role embeddings) ---
    role_emb = _find_role_embedding(model)
    if role_emb is not None and role_emb.shape[0] >= 2:
        with torch.no_grad():
            self_vec = role_emb[0].flatten().float()
            user_vec = role_emb[1].flatten().float()
            cos = torch.nn.functional.cosine_similarity(self_vec, user_vec, dim=0)
            metrics["role_cosine"] = float(cos.item())

    # --- self/user output collapse: compare the ROLE axis of audio_logits (B,2,K,T,C) ---
    # fix8: the ARCH 5.4 Depth batch-2 probe must detect self/user collapsing to identical audio
    # outputs. That contrast lives on the ROLE axis (dim 1: 0=self, 1=user) of `audio_logits`, NOT
    # across two batch ROWS of the text `logits` (comparing batch rows 0/1 measured unrelated
    # examples, not role separation). Compare audio_logits[:, 0] (self) vs audio_logits[:, 1] (user)
    # over the whole batch. Guarded/best-effort: if audio_logits is absent or lacks a role axis we
    # simply skip the metric and never raise.
    audio_logits = getattr(outputs, "audio_logits", None)
    if audio_logits is None and isinstance(outputs, dict):
        audio_logits = outputs.get("audio_logits")
    if (
        audio_logits is not None
        and hasattr(audio_logits, "dim")
        and audio_logits.dim() >= 2
        and audio_logits.shape[1] >= 2
    ):
        with torch.no_grad():
            a = audio_logits[:, 0].flatten().float()  # self role, all rows/codebooks/frames
            b = audio_logits[:, 1].flatten().float()  # user role
            n = min(a.numel(), b.numel())
            if n > 0:
                diff = (a[:n] - b[:n]).norm()
                denom = a[:n].norm() + b[:n].norm() + 1e-8
                metrics["batch_output_divergence"] = float((diff / denom).item())

    return metrics


def _find_role_embedding(model: "torch.nn.Module") -> "torch.Tensor | None":
    """Best-effort lookup of the (2, dim) Role Token / role-embedding weight for probing."""
    for name, p in model.named_parameters():
        if "role_emb" in name or "role_token" in name or "role_embedding" in name:
            if p.dim() >= 2 and p.shape[0] == 2:
                return p.detach()
    return None


def _extract_logits(outputs: Any) -> "torch.Tensor | None":
    """Pull a logits tensor out of a model-output object/tuple/dict for collapse probing."""
    if outputs is None:
        return None
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if hasattr(first, "dim"):
            return first
    if hasattr(outputs, "dim"):  # a bare tensor
        return outputs
    return None


def _first_param_device(model: "torch.nn.Module") -> "torch.device | None":
    """Device of the first parameter, used to place a reduction tensor for nccl. None if empty."""
    for p in model.parameters():
        return p.device
    return None


# ======================================================================================
# 4. Checkpoint I/O
# ======================================================================================

def save_checkpoint(
    model: "torch.nn.Module",
    optimizer: Any,
    scheduler: Any,
    loader: Any,
    global_step: int,
    micro_step: int,
    epoch: int,
    args: TrainArgs,
) -> str:
    """Save the full resume manifest to `out_dir/{phase}/step_{global_step}` (fix4/7/8/12).

    Manifest (see plan §3): global_step/micro_step/epoch, model (DCP full state_dict),
    optimizer (DCP `get_optimizer_state_dict(full_state_dict, cpu_offload)`, fix4), scheduler
    state, sampler state (hasattr-guarded), RNG states (torch/cuda/numpy/python -- saved PER RANK
    as `rng_rank{r}.pt` under distributed so each rank restores its own stream, fix9), and a
    provenance JSON (world_size/grad_accum/seed/phase/schema_version) for resume-compatibility
    checks (fix12).

    Stability (fix8): the payload is written into a sibling `<dir>.tmp` and then `os.rename`d onto
    the final path, so a kill mid-write cannot leave a torn checkpoint. A keep-last-N rotation
    prunes older periodic `step_*` dirs (the just-written one is always retained; gate exports made
    by runner.py under other names are untouched). All ranks participate in the (collective) state
    gathers; rank 0 writes the shared payload (state.pt / provenance / train_args) while each rank
    writes its own `rng_rank{r}.pt` (fix9); every rank meets the closing barrier.
    """
    phase_dir = os.path.join(args.out_dir, args.phase)
    os.makedirs(phase_dir, exist_ok=True)
    final_dir = os.path.join(phase_dir, f"step_{global_step}")
    tmp_dir = final_dir + ".tmp"

    # Collective calls: every rank must enter these so the full gather can complete; only rank 0
    # receives the materialized full dicts (cpu_offload) and writes them out.
    model_sd = _gather_full_state_dict(model)
    optim_sd = _gather_full_optim_state_dict(model, optimizer)

    rank = _dist_rank()
    # fix9: every rank snapshots its OWN RNG (torch/cuda/numpy/python). Saving only rank 0's RNG and
    # restoring it onto every rank (the previous behavior) collapses all ranks to an identical
    # generator after a resume, so per-rank data augmentation / dropout stops diverging. Under real
    # distributed each rank commits its own `rng_rank{r}.pt`; `state.pt` still carries rank 0's RNG
    # for the single-process / backward-compatible path.
    rng_state = _collect_rng_state()

    if rank == 0:
        _rmtree_quiet(tmp_dir)  # clear any stale tmp from a previous crash
        os.makedirs(tmp_dir, exist_ok=True)

    # Barrier so the tmp dir rank 0 just created is visible before other ranks write into it (fix9).
    _dist_barrier()

    # fix9: under genuine distributed (world_size > 1) each rank writes its own RNG file INTO the
    # tmp dir, so the per-rank RNG becomes part of the atomic `tmp -> final` rename below. In the
    # single-process case (world_size == 1) we skip this entirely -- state.pt's rng is authoritative.
    if _dist_world_size() > 1:
        try:
            torch.save(rng_state, os.path.join(tmp_dir, f"rng_rank{rank}.pt"))
        except Exception:
            pass
        _dist_barrier()  # ensure all per-rank rng files land before rank 0 commits the rename

    if rank == 0:
        payload = {
            "global_step": int(global_step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
            "phase": args.phase,
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": scheduler.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None,
            "sampler": _get_sampler_state(loader),
            "rng": rng_state,  # rank 0's RNG: used for single-process / as the resume fallback
        }
        torch.save(payload, os.path.join(tmp_dir, "state.pt"))

        provenance = {
            "world_size": _dist_world_size(),
            "grad_accum": int(args.grad_accum),
            "seed": int(args.seed),
            "phase": args.phase,
            "schema_version": _schema_version(),
            "global_step": int(global_step),
            "micro_step": int(micro_step),
            "epoch": int(epoch),
        }
        with open(os.path.join(tmp_dir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance, f, ensure_ascii=False, indent=2)
        with open(os.path.join(tmp_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(args), f, ensure_ascii=False, indent=2)

        # Atomic commit: replace the final dir by renaming the fully-written tmp dir onto it.
        _rmtree_quiet(final_dir)
        os.rename(tmp_dir, final_dir)
        _rotate_checkpoints(
            phase_dir, keep=max(1, int(getattr(args, "keep_last_n_ckpts", 3))), keep_dir=final_dir
        )

    _dist_barrier()
    return final_dir


def load_checkpoint(
    model: "torch.nn.Module",
    optimizer: Any,
    scheduler: Any,
    loader: Any,
    path: str,
    args: TrainArgs,
) -> tuple[int, int, int]:
    """Restore the full manifest written by `save_checkpoint`; return (global_step, epoch, micro_step).

    Accepts either the checkpoint directory or a direct `state.pt` path. Restores model (DCP
    `set_model_state_dict`), optimizer (DCP `set_optimizer_state_dict`, tolerant on param-group
    change), scheduler, sampler (`load_state_dict` + `set_step(global_step)`, hasattr-guarded) and
    RNG states. Runs a provenance check first (fix12): world_size / grad_accum / schema mismatches
    are warned about (they break the determinism keys of §4) but do not block the resume.

    fix3: `epoch` and `micro_step` are already saved in the manifest but were previously dropped on
    load, so a resume silently restarted the epoch loop at 0 and reset the micro-batch counter.
    They are now restored and RETURNED (as a 3-tuple with `global_step`) so `train_loop` can resume
    the epoch loop and grad-accumulation cursor at the exact position the checkpoint captured.
    fix9: RNG is restored PER RANK from `rng_rank{r}.pt` when present (each rank its own stream),
    falling back to the shared rng embedded in state.pt for single-process / legacy checkpoints.
    """
    state_path = path
    if os.path.isdir(path):
        state_path = os.path.join(path, "state.pt")

    payload = torch.load(state_path, map_location="cpu", weights_only=False)

    # Backward-compatible path: a bare model state_dict (older/legacy checkpoint).
    if not (isinstance(payload, dict) and "model" in payload):
        try:
            model.load_state_dict(payload, strict=False)
        except Exception:
            pass
        return 0, 0, 0

    # Provenance check (fix12) -- warn on determinism-key changes, then proceed tolerantly.
    _check_provenance(os.path.dirname(state_path), args)

    _set_model_state_dict(model, payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        _set_optim_state_dict(model, optimizer, payload["optimizer"])
    if (
        scheduler is not None
        and payload.get("scheduler") is not None
        and hasattr(scheduler, "load_state_dict")
    ):
        try:
            scheduler.load_state_dict(payload["scheduler"])
        except Exception:
            pass

    global_step = int(payload.get("global_step", payload.get("step", 0)))
    # fix3: restore the epoch / micro-batch cursor (already in the manifest) for a faithful resume.
    epoch = int(payload.get("epoch", 0))
    micro_step = int(payload.get("micro_step", 0))
    _restore_sampler(loader, payload.get("sampler"), global_step)  # sampler cursor + set_step
    _restore_rng_for_rank(os.path.dirname(state_path), payload.get("rng"))  # fix9: per-rank rng
    return global_step, epoch, micro_step


# --------------------------------------------------------------------------------------
# Checkpoint state helpers (all guarded for the CPU/stub environment: DCP falls back to plain
# state_dict I/O, and missing sampler / gradient-checkpointing hooks degrade to no-ops).
# --------------------------------------------------------------------------------------

def _warn_if_distributed_partial(what: str, err: Exception) -> None:
    """fix6: warn LOUDLY when a DCP full-gather fails under real distributed (world_size > 1).

    The plain `module.state_dict()` fallback returns only THIS rank's local shard under FSDP2, so
    silently saving it would produce a shard-only, non-resumable checkpoint (each rank clobbering
    the others, or rank 0 alone persisting an incomplete model/optimizer). In a genuine multi-rank
    run we surface a clear WARNING that the persisted state may be incomplete instead of failing
    quietly; the plain fallback stays *silent* only in the single-process / CPU case where the
    "shard" is in fact the whole state (the stub / py_compile smoke path).
    """
    try:
        if dist.is_available() and dist.is_initialized() and _dist_world_size() > 1:
            print(
                f"[warn] {what}: DCP full-gather failed under distributed "
                f"(world_size={_dist_world_size()}, rank={_dist_rank()}): {type(err).__name__}: {err}. "
                "Falling back to a LOCAL state_dict -- the saved checkpoint may be a partial shard "
                "and is NOT guaranteed resumable (fix6).",
                flush=True,
            )
    except Exception:
        pass


def _gather_full_state_dict(model: "torch.nn.Module") -> dict:
    """Return a CPU full model state_dict, materializing FSDP2 sharded params when necessary."""
    try:
        # FSDP2 exposes a get_model_state_dict helper for full/sharded gathering.
        from torch.distributed.checkpoint.state_dict import (  # noqa: PLC0415
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    except Exception as e:
        # fix6: never SILENTLY save a shard-only model under distributed.
        _warn_if_distributed_partial("_gather_full_state_dict", e)
        try:
            return model.state_dict()
        except Exception:
            return {}


def _gather_full_optim_state_dict(model: "torch.nn.Module", optimizer: Any) -> "dict | None":
    """Return a full (unsharded) optimizer state_dict via DCP (fix4), CPU-offloaded on rank 0.

    The sharded optimizer state of an FSDP2 run is not portable on its own; DCP's
    `get_optimizer_state_dict(full_state_dict=True, cpu_offload=True)` gathers the 8-bit/32-bit
    Adam moments into a full dict. Wrapped in try/except (mirroring `_gather_full_state_dict`) so a
    plain optimizer on CPU falls back to `optimizer.state_dict()`.
    """
    if optimizer is None:
        return None
    try:
        from torch.distributed.checkpoint.state_dict import (  # noqa: PLC0415
            StateDictOptions,
            get_optimizer_state_dict,
        )

        return get_optimizer_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    except Exception as e:
        # fix6: never SILENTLY save a shard-only optimizer state under distributed.
        _warn_if_distributed_partial("_gather_full_optim_state_dict", e)
        try:
            return optimizer.state_dict()
        except Exception:
            return None


def _set_model_state_dict(model: "torch.nn.Module", sd: dict) -> None:
    """Load a full model state_dict via DCP `set_model_state_dict`, else plain non-strict load."""
    try:
        from torch.distributed.checkpoint.state_dict import (  # noqa: PLC0415
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            model, sd, options=StateDictOptions(full_state_dict=True, strict=False)
        )
    except Exception:
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass


def _set_optim_state_dict(model: "torch.nn.Module", optimizer: Any, sd: dict) -> None:
    """Load a full optimizer state_dict via DCP `set_optimizer_state_dict`, else plain load.

    Tolerant (fix12): after a param-group change (Full-FT <-> LoRA/freeze phase transition) the
    optimizer state legitimately mismatches; any failure resumes model weights only, exactly like
    the previous implementation.
    """
    try:
        from torch.distributed.checkpoint.state_dict import (  # noqa: PLC0415
            StateDictOptions,
            set_optimizer_state_dict,
        )

        set_optimizer_state_dict(
            model, optimizer, sd, options=StateDictOptions(full_state_dict=True, strict=False)
        )
    except Exception:
        try:
            optimizer.load_state_dict(sd)
        except Exception:
            pass


def _get_sampler_state(loader: Any) -> "dict | None":
    """Best-effort loader/sampler state for resume; None if unavailable.

    Prefer the LoaderBundle's own `state_dict()` (it captures the MixingBatchSampler cursor /
    group_epoch as `{"sampler": ...}`). Fall back to `loader.sampler.state_dict()` for a plain
    DataLoader / stub loader that has no bundle-level `state_dict` (CPU checkpoint smoke).
    """
    if hasattr(loader, "state_dict"):
        try:
            return loader.state_dict()
        except Exception:
            return None
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "state_dict"):
        try:
            return sampler.state_dict()
        except Exception:
            return None
    return None


def _restore_sampler(loader: Any, sampler_state: Any, global_step: int) -> None:
    """Restore the sampler cursor for resume.

    Prefer the LoaderBundle's own `load_state_dict` (round-trips the `{"sampler": ...}` payload the
    matching `_get_sampler_state`/`state_dict` captured, restoring the MixingBatchSampler cursor).
    Fall back to a direct `loader.sampler.load_state_dict` + `set_step(global_step)` for a plain
    DataLoader / stub loader. Every hook is hasattr-guarded, so a loader without a sampler (or
    without these methods) is a silent no-op (plan §7 stub guard).
    """
    if sampler_state is not None and hasattr(loader, "load_state_dict"):
        try:
            loader.load_state_dict(sampler_state)
        except Exception:
            pass
        return
    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        return
    if sampler_state is not None and hasattr(sampler, "load_state_dict"):
        try:
            sampler.load_state_dict(sampler_state)
        except Exception:
            pass
    if hasattr(sampler, "set_step"):
        try:
            sampler.set_step(int(global_step))
        except Exception:
            pass


def _collect_rng_state() -> dict:
    """Snapshot torch / cuda / numpy / python RNG so on-the-fly data augmentation resumes in sync."""
    import random  # noqa: PLC0415

    rng: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    try:
        import numpy as np  # noqa: PLC0415

        rng["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return rng


def _restore_rng_state(rng: Any) -> None:
    """Restore RNG states saved by `_collect_rng_state`; every stream is guarded independently."""
    if not isinstance(rng, dict):
        return
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
    except Exception:
        pass
    try:
        import random  # noqa: PLC0415

        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception:
        pass
    try:
        import numpy as np  # noqa: PLC0415

        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
    except Exception:
        pass
    try:
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass


def _restore_rng_for_rank(ckpt_dir: str, fallback_rng: Any) -> None:
    """fix9: restore THIS rank's own `rng_rank{r}.pt` if present, else the shared rng from state.pt.

    Under distributed, `save_checkpoint` writes one RNG file per rank, so each rank restores exactly
    the generator state IT held at checkpoint time -- combined with fix2's per-rank seeding this
    keeps the ranks' RNG streams decorrelated across a resume instead of collapsing them onto rank
    0's state. Single-process runs and legacy checkpoints have no per-rank file and fall back to the
    rng embedded in state.pt. Best-effort and never raises.
    """
    rank = _dist_rank()
    rank_path = os.path.join(ckpt_dir, f"rng_rank{rank}.pt")
    if os.path.isfile(rank_path):
        try:
            _restore_rng_state(torch.load(rank_path, map_location="cpu", weights_only=False))
            return
        except Exception:
            pass  # fall through to the shared rng
    _restore_rng_state(fallback_rng)


def _schema_version() -> int:
    """Dataset SCHEMA_VERSION from data_pipeline.schema, or 0 when the module is absent (plan §3)."""
    try:
        from data_pipeline.schema import SCHEMA_VERSION  # noqa: PLC0415

        return int(SCHEMA_VERSION)
    except Exception:
        return 0


def _check_provenance(ckpt_dir: str, args: TrainArgs) -> None:
    """Warn (never block) when a resume changes a determinism key vs the saved provenance (fix12).

    `world_size` and `grad_accum` feed the §4 determinism keys (group draw `rng(seed, step)`, step
    cadence), and `schema_version` guards dataset layout; a mismatch means the resumed data stream
    is not byte-identical, so we surface a clear warning and rely on the tolerant optimizer load.
    """
    prov_path = os.path.join(ckpt_dir, "provenance.json")
    if not os.path.isfile(prov_path):
        return
    try:
        with open(prov_path, encoding="utf-8") as f:
            prov = json.load(f)
    except Exception:
        return

    cur_ws = _dist_world_size()
    if prov.get("world_size") not in (None, cur_ws):
        print(
            f"[warn] resume provenance: world_size {prov.get('world_size')} -> {cur_ws}; group-draw "
            "determinism key differs, data stream not guaranteed identical (fix10/12).",
            flush=True,
        )
    if prov.get("grad_accum") not in (None, int(args.grad_accum)):
        print(
            f"[warn] resume provenance: grad_accum {prov.get('grad_accum')} -> {args.grad_accum}; "
            "optimizer-step cadence / determinism key changed (fix10/12).",
            flush=True,
        )
    cur_schema = _schema_version()
    if prov.get("schema_version") not in (None, cur_schema):
        print(
            f"[warn] resume provenance: SCHEMA_VERSION {prov.get('schema_version')} -> {cur_schema}; "
            "dataset schema differs (fix12).",
            flush=True,
        )


def _rmtree_quiet(path: str) -> None:
    """`shutil.rmtree` that swallows errors (missing path / partial dir); used for tmp/rotation."""
    import shutil  # noqa: PLC0415

    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _rotate_checkpoints(phase_dir: str, keep: int, keep_dir: str) -> None:
    """Keep the newest `keep` periodic `step_*` dirs; prune older ones and any stale `*.tmp` (fix8).

    `keep_dir` (the just-written checkpoint) is always retained even if `keep` would drop it. Only
    `step_*` directories are considered, so gate/full exports runner.py writes under other names are
    never rotated away. Best-effort: any error leaves the directory untouched.
    """
    try:
        entries: list[tuple[int, str]] = []
        for name in os.listdir(phase_dir):
            full = os.path.join(phase_dir, name)
            if not os.path.isdir(full):
                continue
            if name.endswith(".tmp"):
                _rmtree_quiet(full)  # stale tmp from a crashed write
                continue
            if name.startswith("step_"):
                try:
                    n = int(name[len("step_"):])
                except ValueError:
                    continue
                entries.append((n, full))
        entries.sort(key=lambda x: x[0])
        if keep > 0 and len(entries) > keep:
            for _n, full in entries[:-keep]:
                if os.path.abspath(full) == os.path.abspath(keep_dir):
                    continue  # never prune the checkpoint we just committed
                _rmtree_quiet(full)
    except Exception:
        pass


def _dist_world_size() -> int:
    """Distributed world size, or the WORLD_SIZE env fallback (1 when single-process)."""
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_rank() -> int:
    """Return the distributed rank, or 0 if distributed is not initialized."""
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _world_size() -> int:
    """Distributed world size for the data stack: `dist.get_world_size()` when a group is up, else 1.

    Unlike `_dist_world_size` (which falls back to the WORLD_SIZE env var for provenance), this
    reports the ACTUAL communicator size that build_dataloader must shard the sampler across, so a
    single-process CPU smoke (no group) correctly gets world_size=1.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _dist_barrier() -> None:
    """Synchronize all ranks if distributed is initialized; otherwise a no-op."""
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _init_distributed() -> bool:
    """Guarded process-group init at `main` entry (fix2). Returns True iff this call started a group.

    Only initializes when the launcher exported WORLD_SIZE/RANK (i.e. torchrun) and no group is up
    yet: on CUDA it pins the local device with `set_device(LOCAL_RANK)` then `init_process_group`s
    NCCL; on CPU it uses Gloo. A plain single-process CPU smoke (no env vars) is a no-op, so import
    / py_compile / CPU runs are unaffected (plan §7 stub guard). Never raises.
    """
    try:
        if not dist.is_available() or dist.is_initialized():
            return False
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1 or "RANK" not in os.environ:
            return False  # not under torchrun -> single-process, skip init entirely
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend)
        return True
    except Exception:
        return False


def _shutdown_distributed(started: bool) -> None:
    """Tear down the process group iff `_init_distributed` started it (fix2). No-op otherwise."""
    if not started:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _assert_optim_precision(model: "torch.nn.Module", optimizer: Any) -> dict:
    """Best-effort check that sensitive new params keep float32 optimizer state (fix9).

    The 8-bit optimizer must NOT quantize the brand-new audio-embedding / Depth-head / Role-Token
    params -- `build_optimizer` registers a 32-bit bitsandbytes override for them. If that override
    evaporated (e.g. the param identity changed across the FSDP2 wrap) their `exp_avg` would come
    back as int8/uint8, a silent precision loss. This confirms `exp_avg` is float32 for those params.

    Called once before the loop, so `optimizer.state` is usually still empty (bnb populates it on
    the first step); this stays best-effort -- it warns and returns a summary rather than raising on
    CPU/stub or when bnb state is absent. The authoritative verification is a real 4xA100 run right
    after step 1 (plan §8).
    """
    sensitive = ("audio_emb", "depth", "role_emb")
    summary = {"checked": 0, "float32": 0, "nonfloat32": 0, "missing_state": 0}
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        return summary
    try:
        for name, p in model.named_parameters():
            if not any(k in name for k in sensitive):
                continue
            summary["checked"] += 1
            st = state.get(p)
            if not st or "exp_avg" not in st or not hasattr(st["exp_avg"], "dtype"):
                summary["missing_state"] += 1
                continue
            if st["exp_avg"].dtype == torch.float32:
                summary["float32"] += 1
            else:
                summary["nonfloat32"] += 1
                print(
                    f"[warn] _assert_optim_precision: {name} exp_avg dtype={st['exp_avg'].dtype} "
                    "(expected float32); 32-bit bnb override may have evaporated (fix9).",
                    flush=True,
                )
    except Exception:
        return summary
    if summary["checked"] and summary["missing_state"] == summary["checked"]:
        # Empty state before the first step, or a non-bnb optimizer on CPU/stub -> can't verify yet.
        print(
            "[warn] _assert_optim_precision: sensitive-param optimizer state absent "
            "(pre-first-step or no bnb state); best-effort skip -- verify on real hardware (plan §8).",
            flush=True,
        )
    return summary


# ======================================================================================
# 5. Training loop (canonical PyTorch: forward -> loss -> backward -> step -> zero_grad)
# ======================================================================================

def _to_device(batch: Any, device: "torch.device") -> Any:
    """Move a (possibly nested) batch of tensors to `device`; leave non-tensors untouched."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_to_device(v, device) for v in batch]
        return type(batch)(moved)
    return batch


def _set_audio_emb_requires_grad(model: "torch.nn.Module", flag: bool) -> None:
    """Freeze/unfreeze the audio input embeddings (used for the optional early-freeze warmup, 7.5)."""
    for name, p in model.named_parameters():
        if "audio_emb" in name:
            p.requires_grad_(flag)


def _extract_rollout_codes(gen_out: Any) -> "torch.Tensor | None":
    """Pull the `codes` (B, 2, K, T_gen) tensor out of a `model.generate` return. (ARCH 5.0.3/5.4)

    Accepts an object with a `.codes` attribute or a dict with a `"codes"` key (the documented
    simulation-mode return, RISKS 7.3). Returns None if neither is present so the caller can skip
    scheduled sampling for this batch without failing the step.
    """
    codes = getattr(gen_out, "codes", None)
    if codes is None and isinstance(gen_out, dict):
        codes = gen_out.get("codes")
    if isinstance(codes, torch.Tensor) and codes.dim() == 4:
        return codes
    return None


@torch.no_grad()
def _apply_scheduled_sampling(
    model: "torch.nn.Module", batch: dict, args: TrainArgs, global_step: int
) -> dict:
    """Scheduled-sampling self-rollout to correct exposure bias (RISKS 7.3). ONLY reached when
    `on_policy_ratio > 0`; the `== 0` path never calls this, so it stays byte-for-byte unchanged.

    INPUT vs TARGET contract (fix4): `batch['audio_codes']` is the ground-truth Mimi code tensor
    that the audio-CE loss supervises against -- it MUST stay untouched, otherwise the model would
    be trained to predict its OWN (possibly wrong) rollout instead of the GT (a corrupted target).
    Scheduled sampling only changes what the model CONDITIONS on, so the self-rollout is written to
    a SEPARATE, documented key `batch['input_audio_codes']`: a clone of `audio_codes` whose trailing
    self-frames (role 0) of a random subset of rows are replaced by the model's own rollout. The
    model reads `input_audio_codes` as its input conditioning while the loss keeps scoring against
    the untouched GT `audio_codes`. Only a TRAILING conditioning window (not the whole T-frame
    stream) is overwritten, so the leading GT context is preserved and the model still receives real
    history before switching to its own predictions.

    fix5 (KD validity): the offline `teacher_topk_*` dump is computed on the GT trajectory, so for any
    (row, trailing-frame) position that we splice with the rollout it is an INVALID teacher -- the
    student output there is now rollout-conditioned. We set `kd_valid=False` for those positions so
    the KD KL never scores a rollout-conditioned student against a GT-trajectory teacher. Both role
    outputs at the spliced frames are conditioned on the modified self stream, so both are
    invalidated. True on-policy KD would need a LIVE teacher re-inferred on the rollout (future work,
    RISKS 7.3); until then those frames simply drop out of the KD term.

    Minimal and guarded: returns the batch unchanged whenever nothing can be spliced (no rows
    selected, no `audio_codes`, or `generate` yields no `codes`). Requires `model.generate` -- a real
    model per the I/O contract exposes it; its absence under on-policy is a contract violation and
    raises.

    fix11 (determinism on resume): the per-row Bernoulli selection is drawn from a LOCAL
    `torch.Generator` seeded on `(args.seed, global_step)` rather than the global RNG, so the exact
    same rows are selected when a run resumes at the same `global_step` -- the on-policy decision is
    reproducible and independent of how many global RNG draws happened before the crash.
    """
    audio_codes = batch.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.dim() != 4:
        return batch  # nothing to splice into

    ratio = min(max(float(args.on_policy_ratio), 0.0), 1.0)
    B, R, K, T = audio_codes.shape
    # Trailing conditioning window: replace only the tail, never the whole stream (fix4). Bounding
    # the rollout to the second half keeps a leading block of real GT history as context.
    cond_len = max(1, T // 2)
    # Reproducible per-(seed, step) selection: a local Generator keyed on the step keeps the choice
    # identical across a resume without perturbing the global RNG stream (fix11).
    gen = torch.Generator(device=audio_codes.device)
    gen.manual_seed((int(args.seed) * 1_000_003 + int(global_step)) & 0x7FFF_FFFF_FFFF_FFFF)
    sel = torch.rand(B, generator=gen, device=audio_codes.device) < ratio  # per-row Bernoulli ~ ratio
    if not bool(sel.any()):
        return batch

    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise NotImplementedError(
            "on_policy_ratio > 0 requires model.generate(batch, mode='simulation', ...) -> codes "
            "(B,2,K,T_gen) (ARCH 5.0.3/5.4, RISKS 7.3); the model exposes no generate()."
        )

    gen_out = generate(batch, mode="simulation", max_new_frames=cond_len)
    codes = _extract_rollout_codes(gen_out)
    if codes is None or codes.shape[1] < 1:
        return batch

    gen_self = codes[:, 0]                       # (B, K_gen, T_gen) -- self stream (role 0)
    L = min(cond_len, gen_self.shape[-1])        # trailing overlap length to splice (<= cond_len < T)
    k = min(K, gen_self.shape[1])                # codebooks common to GT and rollout
    if L <= 0 or k <= 0:
        return batch

    rows = sel.nonzero(as_tuple=False).flatten()
    spliced = dict(batch)

    # fix4: build the INPUT conditioning tensor as a clone of the GT codes and splice the rollout
    # into ONLY the trailing L self-frames (codebooks 0..k-1) of the selected rows. LHS and RHS are
    # both (n_selected, k, L). The GT `audio_codes` (the CE target) is left entirely untouched.
    input_codes = audio_codes.clone()
    input_codes[rows, 0, :k, T - L:] = gen_self[rows, :k, gen_self.shape[-1] - L:].to(input_codes.dtype)
    spliced["input_audio_codes"] = input_codes

    # fix5: invalidate the offline GT-trajectory teacher for the rollout-conditioned frames of the
    # selected rows (both roles) so KD never scores a rollout-conditioned student against it.
    kd_valid = batch.get("kd_valid")
    if isinstance(kd_valid, torch.Tensor) and kd_valid.dim() == 3 and kd_valid.shape[-1] == T:
        new_valid = kd_valid.clone()
        new_valid[rows, :, T - L:] = False
        spliced["kd_valid"] = new_valid

    return spliced


def train_loop(
    model: "torch.nn.Module",
    loader: "DataLoader",
    optimizer: Any,
    scheduler: Any,
    loss_fn: Callable[[Any, dict], "torch.Tensor"],
    args: TrainArgs,
    start_step: int = 0,
    start_epoch: int = 0,
    start_micro: int = 0,
) -> None:
    """Canonical PyTorch loop + LR schedule + combined KD/CE + diagnostics + ckpt/logging (§2).

    Control flow (fully implemented; the pieces that need models/datasets are `model`, `loader`,
    `loss_fn`, which are passed in):
        model.train()
        for each epoch: loader.set_epoch(epoch)           # LoaderBundle: datasets+sampler; hasattr-guarded
          for each batch in loader.loader:                # the bundle's DataLoader (falls back to loader)
            batch -> device
            outputs = model(**batch)                       # pass the collator batch straight through
            total_loss, metrics = loss_fn(outputs, batch)  # KD + CE + anchor (5.1/7.6)
            (total_loss / grad_accum).backward()
            on accumulation boundary: clip grad -> optimizer.step() -> scheduler.step() -> zero_grad()
            periodic: run_diagnostics / log / save_checkpoint

    `global_step` counts optimizer steps and starts at `start_step` (resume, fix7); `micro_step`
    counts micro-batches (backward calls) and resumes at `start_micro`; the epoch loop resumes at
    `start_epoch` (fix3), so `loader.set_epoch(epoch)` re-seats the LoaderBundle's datasets + sampler
    RNG on the epoch the checkpoint captured rather than restarting at 0. The LR schedule advances with
    `scheduler.step()` AFTER `optimizer.step()`, once per optimizer step (fix1); `lr` is logged from
    `scheduler.get_last_lr()`. Notes: if `freeze_audio_emb_steps` > 0 the audio embeddings are
    frozen for the first N steps then unfrozen (RISKS 7.5); the scheduler still advances the (frozen)
    group's LR during that window, which is intentional. `on_policy_ratio` > 0 mixes in self-rollout
    via scheduled sampling (RISKS 7.3); the `== 0` default path is byte-for-byte unchanged.
    """
    # fix2: seed ONLY a fresh run. On resume (start_step > 0) the RNG restored by load_checkpoint
    # must win -- re-seeding here would clobber the restored generator and desynchronize on-the-fly
    # data augmentation / dropout. Seed per-rank-decorrelated (`seed + rank`) so multi-rank runs do
    # not all draw the identical RNG stream.
    if int(start_step) == 0:
        torch.manual_seed(int(args.seed) + _dist_rank())

    device = _resolve_device()
    model.train()

    # Optional early-freeze of audio embeddings, unfrozen once past the warmup window.
    if args.freeze_audio_emb_steps > 0:
        _set_audio_emb_requires_grad(model, False)

    global_step = int(start_step)
    micro_step = int(start_micro)  # fix3: resume the grad-accumulation cursor, not reset to 0
    epoch = int(start_epoch)       # fix3: default in case the epoch range is empty
    optimizer.zero_grad(set_to_none=True)

    # fix3: resume the epoch loop at the restored epoch. `max(start_epoch + 1, args.epochs)` keeps
    # the fresh-run behavior identical (start_epoch=0 -> range(0, max(1, epochs))) while guaranteeing
    # the resumed epoch still runs even if `epochs` was lowered.
    for epoch in range(int(start_epoch), max(int(start_epoch) + 1, int(args.epochs))):
        # Advance every RNG stream for this epoch. The LoaderBundle's set_epoch advances its
        # datasets AND sampler together (call BEFORE iterating). hasattr-guarded so a plain
        # DataLoader / stub without set_epoch is a silent no-op (CPU smoke path).
        if hasattr(loader, "set_epoch"):
            try:
                loader.set_epoch(epoch)
            except Exception:
                pass

        # Iterate the bundle's DataLoader (`loader.loader`); fall back to `loader` itself for a
        # plain-DataLoader / stub loader that has no `.loader` attribute (CPU smoke path).
        for batch in getattr(loader, "loader", loader):
            if global_step >= args.steps:
                break

            batch = _to_device(batch, device)

            # Unfreeze audio embeddings once we cross the freeze window.
            if args.freeze_audio_emb_steps > 0 and global_step == args.freeze_audio_emb_steps:
                _set_audio_emb_requires_grad(model, True)

            # Scheduled-sampling self-rollout (RISKS 7.3 exposure-bias correction). Engaged ONLY
            # when on_policy_ratio > 0; the == 0 default path skips this entirely and reaches the
            # forward below with `batch` byte-for-byte unchanged (no RNG draw, no model.generate).
            if args.on_policy_ratio > 0 and isinstance(batch, dict):
                batch = _apply_scheduled_sampling(model, batch, args, global_step)

            # --- forward ---
            outputs = model(**batch) if isinstance(batch, dict) else model(batch)
            total_loss, metrics = loss_fn(outputs, batch)

            # --- backward (scaled for gradient accumulation) ---
            (total_loss / args.grad_accum).backward()
            micro_step += 1

            # --- optimizer step on accumulation boundary ---
            if micro_step % args.grad_accum == 0:
                # fix6: on a real FSDP2 run `model.parameters()` are DTensors, and modern torch
                # `clip_grad_norm_` computes a DTensor-aware GLOBAL norm across shards (not a
                # per-shard local norm). This is a real-hardware verification point (plan §5): if a
                # given torch build clipped only the local shard, a manual all_reduce of the squared
                # norm would be needed here.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # fix1: advance the LR schedule once per optimizer step, AFTER optimizer.step().
                if scheduler is not None and hasattr(scheduler, "step"):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if isinstance(metrics, dict):
                    metrics = {**metrics, "grad_norm": float(grad_norm)}
                    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
                        try:
                            metrics["lr"] = float(scheduler.get_last_lr()[0])
                        except Exception:
                            pass

                # --- periodic diagnostics / logging / checkpoint (keyed on optimizer steps) ---
                if args.probe_interval > 0 and global_step % args.probe_interval == 0:
                    diag = run_diagnostics(model, outputs, batch, global_step)
                    _log(global_step, {**(metrics or {}), **diag}, args, tag="probe")

                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    _log(global_step, metrics or {}, args, tag="train")

                if args.ckpt_interval > 0 and global_step % args.ckpt_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, loader, global_step, micro_step, epoch, args
                    )

        if global_step >= args.steps:
            break

    # Final checkpoint at the end of training.
    save_checkpoint(model, optimizer, scheduler, loader, global_step, micro_step, epoch, args)


def _resolve_device() -> "torch.device":
    """Return the local CUDA device (respecting LOCAL_RANK) if available, else CPU."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _log(step: int, metrics: dict, args: TrainArgs, tag: str = "train") -> None:
    """Rank-0 logging of scalar metrics. Kept dependency-free (stdout); swap for wandb/TB as needed."""
    if _dist_rank() != 0:
        return
    scalars = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in (metrics or {}).items()}
    print(f"[{args.phase}][{tag}] step={step} {scalars}", flush=True)


# ======================================================================================
# 6. Entry point
# ======================================================================================

def _overlay_cli_over_config(argv: list, dataclasses_: tuple) -> None:
    """fix1: overlay `--<field> <value>` argv tokens onto the JSON-parsed dataclasses, in place.

    `HfArgumentParser.parse_json_file` reads ONLY the config file and ignores every other argv
    token, so runner.py's `--config cfg --resume dir --out_dir od --phase name` would silently drop
    `--resume`/`--out_dir`/`--phase` (and any other TrainArgs/DataArgs field passed alongside a
    `--config`). This manual overlay restores the expected precedence (CLI flag > config file).

    We build a field-name -> owning-dataclass map from the parsed instances and, for each
    `--flag value` (or `--flag=value`) token whose flag names a known field, `setattr` it -- coercing
    the raw string to the field's RUNTIME type inferred from the current value. (Type must be read
    off the live value, not `dataclasses.fields(...).type`: `from __future__ import annotations`
    turns every field annotation into a string.) Only bool/int/float/str fields are overlaid; unknown
    flags (e.g. `--config` itself) and container fields (tuple/dict such as `betas`/`liger_config`)
    are left untouched, and a bad value is ignored rather than crashing the launch.
    """
    field_owner: dict[str, Any] = {}
    for dc in dataclasses_:
        for name in vars(dc):  # dataclass instance __dict__ == its field values
            field_owner[name] = dc

    i, n = 0, len(argv)
    while i < n:
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        inline = None
        if "=" in key:  # support the --flag=value form as well as --flag value
            key, inline = key.split("=", 1)
        if key not in field_owner:
            i += 1  # unknown flag (e.g. --config); its value token is skipped naturally next loop
            continue
        if inline is not None:
            raw = inline
            i += 1
        elif i + 1 < n:
            raw = argv[i + 1]
            i += 2
        else:
            break  # trailing flag with no value
        dc = field_owner[key]
        cur = getattr(dc, key)
        try:
            if isinstance(cur, bool):  # check bool BEFORE int (bool is a subclass of int)
                setattr(dc, key, raw.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(dc, key, int(raw))
            elif isinstance(cur, float):
                setattr(dc, key, float(raw))
            elif isinstance(cur, str) or cur is None:
                setattr(dc, key, raw)
            # else: tuple/dict container -> leave untouched (runner never threads these)
        except (TypeError, ValueError):
            pass  # ignore an unparseable value rather than aborting the launch


def parse_args() -> tuple[ModelArgs, DataArgs, TrainArgs]:
    """Parse (ModelArgs, DataArgs, TrainArgs) with HfArgumentParser. `--config X.json` loads JSON.

    Convention (HF run_clm.py): a single `--config configs/phase2_joint.json` reads the fields from
    JSON; otherwise fields are parsed from command-line flags. Config is JSON only (no yaml,
    filemap.md fixed rule).

    fix1 (config + argv overlay): with `--config`, `HfArgumentParser.parse_json_file` reads only the
    JSON and drops every other flag. runner.py drives phases by threading
    `['--config', cfg, '--resume', dir, '--out_dir', od, '--phase', name]`, so after the JSON parse
    we OVERLAY any `--<field> <value>` tokens still in argv onto the parsed dataclasses. That makes
    runner's `--resume`/`--out_dir`/`--phase` (and any other TrainArgs/DataArgs field) take effect
    on top of the config file, which is the whole mechanism the phase manager relies on.
    """
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))

    argv = sys.argv[1:]
    # `--config X.json` idiom: fill all three dataclasses from a single JSON file.
    if "--config" in argv:
        i = argv.index("--config")
        config_path = argv[i + 1]
        margs, dargs, targs = parser.parse_json_file(json_file=config_path)
        # fix1: honor CLI flags threaded alongside --config (runner's --resume/--out_dir/--phase).
        _overlay_cli_over_config(argv, (margs, dargs, targs))
    else:
        margs, dargs, targs = parser.parse_args_into_dataclasses()
    return margs, dargs, targs


def main() -> None:
    """Entry point (plan §1). Order matters for correctness -- notably wrap -> build_optimizer so the
    32-bit optimizer override lands on the wrapped param identities (fix9), and gradient
    checkpointing enabled BEFORE the FSDP2 wrap (fix3).

    parse -> _init_distributed (fix2) -> build_model -> gradient_checkpointing_enable (fix3) ->
    setup_fsdp2 -> build_optimizer (fix9) -> get_scheduler (fix1) -> build_dataloader ->
    [load_checkpoint if resume (fix7/12)] -> _assert_optim_precision (fix9) -> train_loop, with a
    guaranteed `_shutdown_distributed` in `finally` (fix2). Usually runner.py drives this per phase.
    """
    # `get_scheduler` is imported lazily here (not at module top): transformers is not a dependency
    # of the CPU import/py_compile smoke, and this module must import with torch alone (plan §7).
    from transformers import get_scheduler  # noqa: PLC0415

    margs, dargs, targs = parse_args()

    dist_started = _init_distributed()  # fix2: guarded env(WORLD_SIZE/RANK) init, no-op single-process
    try:
        # build_model raises NotImplementedError until the models/ forward + warm-start land
        # (datasets/ and losses/ are already real); everything wrapped around it -- FSDP2,
        # optimizer, scheduler, resume, the loop -- is real.
        model = build_model(margs, targs)

        # fix3: enable activation (gradient) checkpointing BEFORE the FSDP2 wrap. hasattr/try guarded
        # so a stub model without the hook (or an older signature) does not break import/CPU runs.
        if targs.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()  # older signature without the kwargs arg

        model = setup_fsdp2(model, targs)               # fully_shard + bf16 MixedPrecisionPolicy
        optimizer = build_optimizer(model, targs)       # fix9: wrap -> build -> 32-bit override identity
        scheduler = get_scheduler(                      # fix1: warmup + cosine, unit = optimizer step
            targs.lr_scheduler_type,
            optimizer,
            num_warmup_steps=targs.warmup_steps,
            num_training_steps=targs.steps,
        )
        loader = build_dataloader(dargs, targs)
        loss_fn = build_loss_fn(targs)

        start_step = 0
        start_epoch = 0
        start_micro = 0
        if targs.resume:
            # fix3: load_checkpoint now returns (global_step, epoch, micro_step); thread all three
            # into the loop so a resume continues at the exact epoch / grad-accum cursor.
            start_step, start_epoch, start_micro = load_checkpoint(
                model, optimizer, scheduler, loader, targs.resume, targs
            )
            _log(
                start_step,
                {"resumed_from": targs.resume, "epoch": start_epoch, "micro_step": start_micro},
                targs,
                tag="resume",
            )

        _assert_optim_precision(model, optimizer)       # fix9: sensitive-param exp_avg dtype check
        train_loop(
            model, loader, optimizer, scheduler, loss_fn, targs,
            start_step=start_step, start_epoch=start_epoch, start_micro=start_micro,
        )
    finally:
        _shutdown_distributed(dist_started)             # fix2: destroy the group we started


if __name__ == "__main__":
    main()
