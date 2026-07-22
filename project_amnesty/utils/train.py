"""train.py -- training entry point (with loop).

`runner.py` (the phase manager) drives this entry point per phase.

On top of the canonical PyTorch training loop
(`train_loop`: forward -> loss -> backward -> optimizer.step -> zero_grad) we layer
the Haan training stack:
  - **Distributed**: FSDP2 (`fully_shard`). Default `reshard_after_forward=False`
    (keep parameters replicated, shard grad/optim state only), so the parameters
    themselves are not sharded. Fall back to True (ZeRO-3 class) only when VRAM is
    tight.
  - **Optimizer**: PagedAdamW8bit -- the A100 (Ampere) cannot accelerate FP8 tensor
    cores, so an 8-bit optimizer yields the same memory savings.
  - **Kernels**: Liger Kernel (fused RMSNorm/RoPE/SwiGLU/CE/FusedLinearCE). Compatible
    with FlashAttention and FSDP.
  - **Loss**: semantic-KD KL (Mimi level-0 logit) + Korean TTS CE + voice-cloning CE +
    text anchor CE combined. Acoustic codebooks (1~7) are excluded from KD by default
    (timbre carriers). It consumes the per-token weights the collator ships (stream PAD
    text x0.3, non-semantic audio x0.02) and the semantic-KD internal frame weights
    (speech / turn-transition regions) as-is. Zone A (system prompt) regions and batch
    pad are fully masked out of the loss.
  - **Diagnostic hooks**: grad-norm (per-task gradient dominance watch) - self/user role
    vector cosine separation - Depth batch-2 two-element output-collapse probing.

Run: `python -m project_amnesty.utils.train --config configs/phase2_joint.json`
Config is JSON (no yaml). ModelArgs/DataArgs/TrainArgs are defined at the top of this
file as @dataclass per the HF `run_clm.py` convention and parsed with `HfArgumentParser`.

Note: `project_amnesty/datasets/` is a REAL, fully-implemented data stack and
`project_amnesty/losses/` provides the real semantic-KD loss, so `build_dataloader` and
`build_loss_fn` delegate to them for real (lazy import). `project_amnesty/models/haan/` is a
real transformers Moshi subclass (HaanConfig / HaanModel / HaanForConditionalGeneration), so
`build_model` constructs and warm-starts a working model. Everything else here (FSDP2 wrapping,
optimizer construction, the training loop, checkpoint I/O, diagnostics, argument parsing) is
implemented for real and imports cleanly with no heavy dependencies installed.

`models/haan` is a faithful Moshi subclass and does NOT override `forward`, so the model speaks
Moshi's I/O (`assistant_audio_codes` / `user_audio_codes`, text logits on `.logits`) rather than
the MODEL I/O + BATCH contract this file's loss is written against. `forward_with_contract`
(section 2b) is the translation, and it lives here rather than in the model deliberately -- the
KDCollator batch is therefore NOT passed to the model directly.
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
    """Backbone / audio / Depth / warm-start settings."""

    backbone: str = "Qwen/Qwen3-8B"
    moshi_ckpt: str = "kmhf/hf-moshiko"  # warm-start source (emb.8~15, depformer, linears)
    num_codebooks: int = 8  # K (number of audio codebooks in the self stream = dep_q)
    depth_dim: int = 1024  # Depth Transformer internal dim (independent of backbone dim)
    audio_cardinality: int = 2048  # frozen Mimi shared -> teacher/student identical
    # Audio embedding table is shared across self/user (8 books). Role is distinguished by a
    # learned additive Role Token.
    share_scope: str = "semantic+acoustic"  # ablation: "semantic" | "semantic+acoustic"
    init_source: str = "user"  # codebook init: "user"(emb.8~15) | "self"(emb.0~7) | "random"
    # Depth parallel-prediction mode switch: train q16 (self+user, batch 2) / live inference
    # q8 (self, batch 1).
    depth_mode: str = "q16"  # "q16"(training/simulation) | "q8"(live conversation)


@dataclass
class DataArgs:
    """Data root / frames / collator wiring settings."""

    # SUPERSEDED: the whole data stack (root/dataset/mix/sampler/dataloader) now comes from
    # `configs/data/loader.yaml` via `datasets.loader.load_configs` (see build_dataloader). The
    # fields below are kept only for back-compat / phase-JSON tolerance (a phase config may still
    # carry them) -- removing them would break `parse_args` on those JSONs. They are NOT read by the
    # real data stack anymore.
    root: str = "data/prepared"
    max_frames: int = 750  # context cap at 12.5Hz (60 seconds)
    double_ab: bool = True  # reuse the same conversation once in each A/B direction (role swap)
    config_json: str = "configs/data.json"  # datasets pipeline config (JSON)
    ko_ratio: float = 0.1  # Korean data ratio (ramped up in Phase 2: 0.1->0.3->0.5)
    num_workers: int = 4

    # --- Collator delay (per-phase knob; NOT in loader.yaml by design) ---
    # `delay` changes between curriculum phases, so it is supplied here per phase rather than baked
    # into the static YAML. build_dataloader threads these into the KDCollator's DelayConfig.
    acoustic_delay: int = 1        # Phase 1+ conversation default (runner: acoustic 1, text 0).
    text_delay_frames: int = 0     # Phase 0 pre-training override = acoustic 2 / text +-0.6 via per-phase config.


@dataclass
class TrainArgs:
    """Training hyperparameters / distributed / optimizer / loss weights / diagnostic intervals."""

    phase: str = "phase2_joint"  # phase identifier selected by runner.py

    # --- Distributed (FSDP2) ---
    reshard_after_forward: bool = False  # False=ZeRO-2 class (param replicate), True=ZeRO-3 class

    # --- Optimizer ---
    optim: str = "paged_adamw_8bit"  # A100 -> 8-bit optimizer for memory savings
    lr: float = 2e-5
    audio_param_lr: float = 2e-5  # dedicated lr for audio embedding / Depth heads / Role Token
    #                               (can be lowered early to prevent drift)
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    grad_accum: int = 4
    warmup_steps: int = 500
    steps: int = 45000
    epochs: int = 1  # optional epoch cap; the loop stops at whichever of steps/epochs comes first
    lr_scheduler_type: str = "cosine"  # transformers get_scheduler type; cosine decay w/ warmup

    # --- Kernels ---
    use_liger_kernel: bool = True
    liger_config: dict = field(default_factory=lambda: {
        "rope": True, "swiglu": True, "cross_entropy": True,
        "fused_linear_cross_entropy": True, "rms_norm": True,
    })

    # --- Fine-tuning mode ---
    full_ft: bool = True  # Phase 1~3/5 are Full FT (avoid shortcuts). Only Phase 4 uses LoRA.
    freeze_audio_emb_steps: int = 0  # if >0, freeze audio embeddings for the first N steps then warm up
    # Activation (gradient) checkpointing: trade compute for memory. Enabled BEFORE the FSDP2 wrap
    # in main() so re-materialized activations live under the sharded module.
    gradient_checkpointing: bool = True

    # --- Combined loss weights ---
    kd_weight: float = 1.0  # semantic KD KL term
    ce_weight: float = 1.0  # Korean TTS CE + voice-cloning CE term
    text_anchor_weight: float = 0.1  # multilingual text-ability anchor (small, all regions)
    kd_temperature: float = 1.0  # KL distillation temperature
    pad_text_weight: float = 0.3  # down-weight stream PAD text tokens
    non_sem_audio_weight: float = 0.02  # down-weight non-semantic (acoustic) audio tokens

    # --- KD operating mode ---
    kd_logit_dump: str = ""  # empty = live teacher; if set, path to pre-dumped top-k logits (offline)
    on_policy_ratio: float = 0.0  # if >0, fraction of self-rollout via scheduled sampling

    # --- Diagnostics / checkpoint / logging ---
    log_interval: int = 20
    probe_interval: int = 500  # role-vector cosine / batch-collapse probing interval
    ckpt_interval: int = 2000
    keep_last_n_ckpts: int = 3  # keep-last-N rotation of periodic step_* checkpoints; final always kept
    out_dir: str = "checkpoints"
    resume: str = ""  # checkpoint path to resume from (injected by runner.py on phase transition)
    seed: int = 42
    bf16: bool = True

    # Name PREFIXES identifying the "brand-new" parameters (audio RVQ embeddings, Role Token,
    # shared Depth Transformer) that must always train in full precision. See
    # NEW_PARAM_PREFIXES below for why these are prefixes and not substrings.
    new_param_prefixes: tuple[str, ...] = ()  # empty -> NEW_PARAM_PREFIXES (the model's real names)


# ======================================================================================
# 1b. Parameter-name contract shared with project_amnesty.models.haan
# ======================================================================================

# Haan's brand-new parameters, spelled EXACTLY as
# `HaanForConditionalGeneration.named_parameters()` yields them:
#
#   embed_tokens.{0..K-1}.weight              shared audio input embeddings
#   role_embedding.{role_scale|role_emb}      Temporal-side role signal
#   depth_decoder.*                           the whole shared Depth Transformer,
#                                             i.e. its own role_embedding, the per-index
#                                             input_projections, and the lm_heads
#
# These are matched with `str.startswith`, NOT as substrings, and that distinction is load
# bearing: the backbone's TEXT embedding is `model.embed_tokens.weight`, so a bare
# "embed_tokens" substring test would sweep the 151936-row text table into the new-parameter
# group -- training it at `audio_param_lr` and forcing 32-bit optimizer state for it.
#
# (The pre-rewrite keys "audio_emb" / "linears" / "depformer" named an earlier stub model and
# match NOTHING in models/haan -- they would have silently produced an empty new-param group.)
NEW_PARAM_PREFIXES = ("embed_tokens.", "role_embedding.", "depth_decoder.")

# The audio input embeddings alone (the optional early-freeze warmup) -- a strict
# subset of NEW_PARAM_PREFIXES, kept separate because freezing must NOT touch the Depth decoder.
AUDIO_EMB_PREFIXES = ("embed_tokens.",)

# Wrappers that prefix parameter names. FSDP2 (`fully_shard`) preserves names, but a
# `torch.compile` / DDP layer would not, so names are normalized before matching.
_NAME_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")


def _strip_wrappers(name: str) -> str:
    """Drop DDP / torch.compile / FSDP wrapper prefixes so prefix matching sees the real name."""
    changed = True
    while changed:
        changed = False
        for wrapper in _NAME_WRAPPER_PREFIXES:
            if name.startswith(wrapper):
                name = name[len(wrapper):]
                changed = True
    return name


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    """True when `name` (wrapper-stripped) starts with any of `prefixes`."""
    return _strip_wrappers(name).startswith(prefixes)


def _new_param_prefixes(args: TrainArgs) -> tuple[str, ...]:
    """The configured new-parameter prefixes, defaulting to the model's real names."""
    return tuple(args.new_param_prefixes) or NEW_PARAM_PREFIXES


# ======================================================================================
# 2. Component builders (datasets/losses/models are real; only the Moshi warm-start is bespoke)
# ======================================================================================

def build_model(margs: ModelArgs, targs: TrainArgs) -> "torch.nn.Module":
    """Build the Haan model: Qwen3 backbone + shared audio embeddings (K) + Role Token + shared Depth.

    Delegates the transfer to `utils.warm_start_haan`, which owns the mapping and reports what did
    NOT transfer. In summary: the Temporal backbone, `lm_head` and QK-Norm come from Qwen3; the
    audio input embeddings from Moshi's user-side tables; the Depth body / heads / per-index
    projections from Moshi verbatim, with `input_projections` initialized from Moshi but retrained
    as an adapter. The role embeddings start at identity and the depth text embedding starts cold
    (different tokenizer).

    `backbone=""` selects the baseline arm instead -- everything, backbone included, from Moshi.
    That is the control the Qwen3 substitution is measured against, so it is a supported mode
    rather than a fallback.

    The dimension-bearing ModelArgs (`num_codebooks` / `depth_dim` / `audio_cardinality`) are
    VALIDATED against the checkpoints rather than applied to them: the warm-start reuses Moshi's
    audio cardinality and depth width as-is, so a config asking for different ones is not a variant
    of the warm-start, it is a silent cancellation of it.
    """
    from project_amnesty.utils import warm_start_haan  # noqa: PLC0415

    _reject_unimplemented_model_args(margs, targs)

    dtype = torch.bfloat16 if targs.bf16 else None
    verbose = _dist_rank() == 0

    if margs.backbone:
        model = warm_start_haan.warm_start_qwen3_moshi(
            margs.backbone,
            margs.moshi_ckpt,
            init_source=margs.init_source,
            dtype=dtype,
            verbose=verbose,
        )
    else:
        # Baseline: Moshi backbone, so the Korean-emergence comparison has a control.
        model = warm_start_haan.warm_start_from_moshi(
            margs.moshi_ckpt,
            init_source=margs.init_source,
            dtype=dtype,
            verbose=verbose,
        )

    # The warm-start keeps Moshi's audio cardinality (2048) and depth dim (1024); K is the stream width.
    for field, expected, actual in (
        ("num_codebooks", margs.num_codebooks, model.config.num_codebooks),
        ("depth_dim", margs.depth_dim, model.config.depth_decoder_config.hidden_size),
        ("audio_cardinality", margs.audio_cardinality, model.config.audio_vocab_size),
    ):
        if int(expected) != int(actual):
            raise ValueError(
                f"ModelArgs.{field}={expected} but the warm-start source "
                f"{margs.moshi_ckpt!r} has {actual}. ARCH 5.4.1 reuses Moshi's audio cardinality "
                f"and depth width as-is, so these must agree -- change the config, not the "
                f"checkpoint."
            )

    # q16 rolls out self+user (training / simulation), q8 self only (live conversation).
    # Training goes through `forward` and never consults this switch; set it anyway so a model
    # handed straight to `generate` behaves as the phase config says.
    model.set_depth_mode({"q16": "simulation", "q8": "live"}[margs.depth_mode])
    return model


def _reject_unimplemented_model_args(margs: ModelArgs, targs: TrainArgs) -> None:
    """Fail on ModelArgs/TrainArgs that models/haan does not currently honor.

    Every knob here is one the config can set and the model would silently ignore -- the failure
    mode is a run that looks configured and is not, discovered only at eval time.
    """
    if margs.depth_mode not in ("q16", "q8"):
        raise ValueError(f"ModelArgs.depth_mode={margs.depth_mode!r} must be 'q16' or 'q8' (ARCH 5.4).")

    if margs.share_scope != "semantic+acoustic":
        # models/haan shares all K audio tables unconditionally (modeling_haan.HaanForConditionalGeneration),
        # so the "semantic"-only ablation has no implementation to select.
        raise NotImplementedError(
            f"ModelArgs.share_scope={margs.share_scope!r} is not implemented: models/haan shares all "
            "K audio tables unconditionally, so only 'semantic+acoustic' is realizable. The ARCH 3.6 "
            "semantic-only ablation needs a modeling-side change first."
        )

    # `backbone` now selects the warm-start arm rather than being rejected: a non-empty value is the
    # Qwen3 substitution, `""` is the Moshi-backbone control. Nothing to validate here beyond what
    # warm_start_haan itself checks against the two checkpoints.

    if targs.use_liger_kernel:
        # `use_liger_kernel` / `liger_config` are read NOWHERE else in this file, and liger_kernel
        # is not even installed -- so leaving this silent would size a phase for fused-kernel memory
        # and throughput it does not get.
        detail = (
            "the package is not installed" if not _liger_available()
            else "the package is installed but train.py never applies it"
        )
        raise NotImplementedError(
            f"TrainArgs.use_liger_kernel=True but Liger Kernel is not wired up ({detail}), so the "
            "fused RMSNorm/RoPE/SwiGLU/CE path of ARCH 5.1 is inactive. Set use_liger_kernel=false "
            "to train without it deliberately, or apply it to the model in build_model first "
            "(note liger_kernel ships no patcher for the `moshi` model type, which is what "
            "models/haan's backbone is)."
        )


def _liger_available() -> bool:
    """Whether the Liger Kernel package can be imported."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("liger_kernel") is not None


# ======================================================================================
# 2b. Batch/output adapter -- KDCollator contract <-> the model's Moshi-inherited signature
# ======================================================================================
#
# `models/haan` does not override `forward`, so the model speaks Moshi's I/O while this file's
# loss is written against the MODEL I/O + BATCH contract documented in section 3. The two differ
# in three ways, and the translation lives HERE rather than in the model on purpose: the model is
# a faithful Moshi subclass and stays that way.
#
#   1. Audio input. Moshi takes `assistant_audio_codes` / `user_audio_codes` as two `(B, K, T)`
#      tensors; the collator ships one `(B, 2, K, T)` with the streams on axis 1. Splitting is
#      just indexing -- and the order is load bearing: `MoshiForConditionalGeneration.forward`
#      concatenates `[assistant, user]`, and `generation_haan._embed_audio_codes` recovers the
#      role as `codebook // K`, so axis-1 index 0 must be the assistant (self) stream.
#
#   2. Audio output. Moshi only populates `audio_logits` when BOTH `text_labels` and
#      `audio_labels` are passed, and then shaped `(B * T, 2K, C)`. The loss needs
#      `(B, 2, K, T, C)`.
#
#   3. **The delay pattern, which is why the labelled path is not used at all.** Moshi's
#      labelled branch runs `build_delay_pattern_mask` over `audio_labels` internally. The
#      KDCollator has ALREADY applied the delay (`KDCollator.set_delay`, `DelayConfig`), so
#      going through that branch would delay the codes a second time -- a silent, phase-dependent
#      misalignment between the KD teacher and the student. Calling the decoder without labels
#      skips it entirely, and the depth decoder is then driven directly with the collator's
#      already-delayed codes.
#
# The depth decoder is teacher-forced the way Moshi teacher-forces it: every frame becomes one
# row, and position `p` of that row is fed codebook `p - 1` (position 0 takes the frame's text
# token), so position `p` predicts codebook `p`.


def _split_streams(audio_codes: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
    """`(B, 2, K, T)` -> `(assistant, user)`, each `(B, K, T)` (stream order [self, user])."""
    if audio_codes.dim() != 4 or audio_codes.shape[1] != 2:
        raise ValueError(
            f"audio codes must be (B, 2, K, T) with axis 1 = [self, user]; got {tuple(audio_codes.shape)}."
        )
    return audio_codes[:, 0], audio_codes[:, 1]


def forward_with_contract(model: "torch.nn.Module", batch: dict) -> dict:
    """Run the model on a KDCollator batch and return the loss's documented output contract.

    Returns a dict with `text_logits` `(B, T, V)` and `audio_logits` `(B, 2, K, T, C)` -- the two
    keys `build_loss_fn` resolves. A dict (rather than the model's output object) because the
    audio logits are assembled here and there is no Moshi output class that describes them in
    this layout; `loss_fn` already accepts either.

    Memory note: `audio_logits` is `B * 2 * K * T * C` floats. That cost is inherent to
    supervising every codebook per frame, not to this adapter, but it is the tensor to look at
    first when sizing a phase.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"expected the KDCollator batch dict; got {type(batch).__name__}.")

    # A text-anchor micro-batch carries no audio at all -- `TextAnchorCollator` refuses to fabricate
    # silence to match shapes. `RoutingCollator` labels the two kinds because branching on a shared
    # key would silently do the wrong thing on one path (loader.py), so branch on the label.
    if batch.get("batch_kind") == "text_anchor":
        if "input_ids" not in batch:
            raise KeyError("a text_anchor batch must carry `input_ids` (B, L) (RISKS 7.6).")
        # Text only: no audio in, so no audio or depth logits out. The loss reads `audio_logits`
        # only when it is present, so the anchor term is all this contributes.
        return {"text_logits": model(input_ids=batch["input_ids"]).logits}

    if "input_ids" not in batch or "audio_codes" not in batch:
        raise KeyError("batch must carry `input_ids` (B, T) and `audio_codes` (B, 2, K, T) (ARCH 7.6).")

    input_ids = batch["input_ids"]
    targets = batch["audio_codes"]
    # `input_audio_codes` is the scheduled-sampling conditioning copy; when present the model
    # conditions on it while the loss still supervises the ground-truth `audio_codes`.
    conditioning = batch.get("input_audio_codes")
    conditioning = targets if conditioning is None else conditioning

    assistant, user = _split_streams(conditioning)
    batch_size, _, codebooks, frames = conditioning.shape

    outputs = model(
        input_ids=input_ids,
        assistant_audio_codes=assistant,
        user_audio_codes=user,
        use_cache=False,
    )

    hidden = outputs.last_hidden_state
    text_logits = outputs.logits

    # One row per frame: (B, T, H) -> (B * T, 1, H). The depth decoder's "sequence" axis is the
    # codebook axis, so each frame is an independent row.
    hidden = hidden.reshape(-1, 1, hidden.shape[-1])

    # (B, 2, K, T) -> (B, 2K, T) -> (B, T, 2K) -> (B * T, 2K), assistant codebooks then user's.
    per_frame = conditioning.reshape(batch_size, 2 * codebooks, frames).transpose(1, 2).reshape(-1, 2 * codebooks)
    # Position 0 is the frame's text token; position p > 0 is fed codebook p - 1. Dropping the last
    # codebook keeps the row `2K` long -- it is a target at position 2K-1, never an input.
    depth_input_ids = torch.cat([input_ids.reshape(-1, 1), per_frame[:, :-1]], dim=1)

    depth_logits = model.depth_decoder(last_hidden_state=hidden, input_ids=depth_input_ids).logits

    # (B * T, 2K, C) -> (B, T, 2, K, C) -> (B, 2, K, T, C). The row index is `b * T + t`, so the
    # leading split is (B, T) in that order; axis 1 then carries [assistant, user] as concatenated.
    audio_logits = depth_logits.view(batch_size, frames, 2, codebooks, -1).permute(0, 2, 3, 1, 4)

    return {"text_logits": text_logits, "audio_logits": audio_logits}


def rollout_with_contract(model, batch: dict, n_frames: int, prompt_frames: int):
    """Self-play rollout: generate `n_frames` of BOTH streams, frame-aligned.

    Conditions on the FIRST `prompt_frames` frames of `batch` and lets the model invent the rest.
    Returns `(codes, gen)`:

      * `codes` -- `(B, 2, K, n_gen)` long, axis 1 = `[self, user]`: the layout both the on-policy
        scheduled sampler and the offline simulation consume.
      * `gen`   -- the raw `generate` output, so a caller that also needs the text stream reads
        `gen.sequences` off the SAME rollout instead of generating a second time.

    This is the ONE place the assistant stream is end-anchored against the predicted user tail.
    `generate` trims the leading all-BOS run off the assistant codes, so a prompt-anchored slice
    (`[..., prompt_frames:]`) is off by that trim on every first turn -- and if the two consumers
    each re-derived the alignment, a one-frame disagreement between them would be invisible in both.

    Both roles must be rolled out, so the depth decoder is switched to `"simulation"` for the call
    and restored afterwards: in `"live"` mode `generate` returns no predicted user stream at all.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"expected the collator batch dict; got {type(batch).__name__}.")
    if "audio_codes" not in batch or "input_ids" not in batch:
        raise KeyError("rollout needs `input_ids` (B, T) and `audio_codes` (B, 2, K, T) (ARCH 7.6).")

    audio = batch["audio_codes"]
    prompt_frames = max(int(prompt_frames), 1)

    set_mode = getattr(model, "set_depth_mode", None)
    was_simulation = getattr(model, "predicts_user_stream", None)
    if callable(set_mode) and was_simulation is not True:
        set_mode("simulation")
    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=batch["input_ids"][:, :prompt_frames],
                assistant_audio_codes=audio[:, 0, :, :prompt_frames],
                user_audio_codes=audio[:, 1, :, :prompt_frames],
                # One extra step: the predicted user tail is one frame shorter than the self span.
                max_new_tokens=int(n_frames) + 1,
                do_sample=True,             # mandatory for Moshi generation
                return_audio_codes=True,    # without this the codes come back None
                concat_unconditional_inputs=False,
            )
    finally:
        if callable(set_mode) and was_simulation is False:
            set_mode("live")

    user_c = getattr(gen, "user_audio_codes", None)
    if user_c is None:
        raise ValueError(
            "generate returned no predicted user stream, so there is no simulated dialogue to "
            "measure (ARCH §5.0.3). Either the supplied user_audio_codes already covered the "
            "generation horizon, or generation stopped early on EOS."
        )
    n_gen = user_c.shape[-1]
    self_c = gen.audio_codes[:, :, -(n_gen + 1):-1]
    # A negative-index slice does NOT raise when the source is too short -- it quietly returns a
    # shorter tensor, and a downstream min() would absorb the difference, leaving every metric
    # computed on silently fewer frames than asked for.
    if tuple(self_c.shape) != tuple(user_c.shape):
        raise ValueError(
            f"self {tuple(self_c.shape)} and user {tuple(user_c.shape)} streams disagree: the "
            f"end-anchored slice needs audio_codes to be at least one frame longer than the "
            f"predicted user tail (got {gen.audio_codes.shape[-1]} vs {n_gen}). Frame alignment "
            f"is not recoverable."
        )
    codes = torch.stack([torch.as_tensor(self_c), torch.as_tensor(user_c)], dim=1)
    return codes, gen


def _assert_stream_tokens_are_reserved(tokens) -> None:
    """Fail before step 0 if a stream token id is an id the backbone already means something by.

    `TokenConfig` checks that the three ids differ from each other; it cannot check what they ARE,
    because it holds `tokenizer_name` as a string and never loads a tokenizer. That gap is where the
    damaging mistake lives: an id below `len(tokenizer)` is a REAL token, and the stream PAD alone
    fills most of the text channel, so training would overwrite whatever that token meant.
    `<|im_end|>` is the loud version of this -- it would teach "generation over" as the resting state
    of the channel and take the Qwen3 backbone's instruction-following with it.

    Qwen3-8B has 151936 embedding rows for 151669 real tokens; `configs/tokens.yaml` deliberately
    assigns all three ids into that gap. No UPPER bound is checked here -- only the model knows how
    many rows it has, and reaching for it would drag the model config into the data path.

    Checked here rather than in `HaanProcessor._pad_token_id`, which enforces the same contract for
    anyone loading a published checkpoint: this project's pipeline never constructs that processor,
    so the collator (collator.py) would have consumed the bad id with nothing in the way.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    named = {
        "text_pad_id": tokens.text_pad_id,
        "text_epad_id": tokens.text_epad_id,
        "batch_pad_id": tokens.batch_pad_id,
    }
    if all(v is None for v in named.values()):
        return  # `TokenConfig.require` owns "unset"; nothing to compare here.

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokens.tokenizer_name)
    except Exception as error:  # noqa: BLE001 -- offline/no-cache must not block a run
        print(
            f"[haan] WARNING: could not load {tokens.tokenizer_name!r} to verify the stream token "
            f"assignment ({type(error).__name__}). configs/tokens.yaml is UNVERIFIED this run.",
            flush=True,
        )
        return

    real_tokens = len(tokenizer)
    for name, token_id in named.items():
        if token_id is None or token_id >= real_tokens:
            continue
        raise ValueError(
            f"configs/tokens.yaml: {name}={token_id} is the existing token "
            f"{tokenizer.convert_ids_to_tokens(token_id)!r}, not a reserved slot. These ids are written "
            f"into the text channel every frame, so a trained token's meaning would be overwritten "
            f"(ARCHITECTURE 7.6). Use an id at or above {real_tokens} -- an embedding row that exists "
            f"but carries no token."
        )


def build_dataloader(dargs: DataArgs, targs: TrainArgs, split: str = "train") -> "LoaderBundle":
    """Assemble the real data stack via datasets.load_configs + build_dataloader -> LoaderBundle.
    Config (root/dataset/mix/sampler/dataloader) comes from configs/data/loader.yaml; only `delay`
    is a per-phase knob supplied here (not in the static YAML).

    The returned `LoaderBundle` exposes `.loader` (the iterable DataLoader), `.set_epoch(epoch)`
    (advances datasets + sampler RNG together), and `.state_dict()`/`.load_state_dict()` for resume;
    train_loop / save_checkpoint / load_checkpoint drive those. The collator batch is passed to the
    model as-is; it carries `input_ids`, `audio_codes`, the teacher top-k dump, `kd_frame_weight`,
    and the per-token loss weights the loss consumes.
    """
    # Heavy datasets imports stay LAZY here so the module py_compiles / imports with torch alone.
    from project_amnesty.datasets.loader import build_dataloader as _build, load_configs
    from project_amnesty.datasets.collator import KDCollatorConfig, DelayConfig

    # All static config (root/dataset/mix/sampler/dataloader) is read from configs/data/loader.yaml.
    data_cfg, loader_cfg, mix_cfg = load_configs()
    _assert_stream_tokens_are_reserved(data_cfg.tokens)
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
# `datasets/` emits this batch for real, and `forward_with_contract` (section 2b) produces this
# output shape from the model's Moshi-inherited signature. The model itself does NOT speak this
# contract -- it is a faithful Moshi subclass, and the translation is deliberately on this side.
# This is the documented interface both this file and the sibling `evaluate.py` are written
# against, so they stay aligned.
# Every access below is guarded (getattr / `in` / `.get`) with a clear error, so a conforming
# model/collator "just works" and a non-conforming one fails loud.
#
# MODEL OUTPUT (`outputs = forward_with_contract(model, batch)`):
#   - `outputs.text_logits`  Float (B, T, V_text)   Inner-Monologue text logits.
#                            Falls back to `outputs.logits` when `text_logits` is absent.
#   - `outputs.audio_logits` Float (B, 2, K, T, C)  role r in {0=self, 1=user}, codebook
#                            k in {0..K-1}, C=2048 (frozen Mimi). Semantic level-0 == k==0.
#   - `model.generate(batch, mode="simulation", max_new_frames=..., ...)` -> obj/dict with
#                            `codes` (B, 2, K, T_gen) -- used only by the on-policy hook.
#
# BATCH (KDCollator output; keys guarded, teacher keys optional -> KD contributes 0):
#   - `input_ids`         (B, T)          long   text-stream token ids.
#   - `audio_codes`       (B, 2, K, T)    long   ground-truth Mimi codes (self/user, all K books).
#                                                This is the audio-CE supervision TARGET; it is never
#                                                overwritten by the on-policy hook.
#   - `input_audio_codes` (B, 2, K, T)    long   OPTIONAL scheduled-sampling INPUT conditioning: a
#                                                copy of `audio_codes` whose trailing self-frames may
#                                                be the model's own rollout. Present only when
#                                                on_policy_ratio>0; the model conditions on it while
#                                                the loss still supervises the GT `audio_codes`.
#   - `role_ids`          (B, 2)          long   {self, user} row order (documentation; not read here).
#   - `text_loss_weight`  (B, T)          float  stream-PAD x0.3, EPAD x1, Zone A / batch-pad = 0.
#   - `audio_loss_weight` (B, 2, K, T)    float  semantic=1, non-sem x0.02, synthetic user ch=0,
#                                                Zone A / batch-pad = 0.
#   - `teacher_topk_val`  (B, 2, T, topk) float  teacher top-k logits (semantic k=0), real collator key.
#   - `teacher_topk_idx`  (B, 2, T, topk) long   teacher top-k support indices.
#   - `kd_valid`          (B, 2, T)       bool   frames with a valid teacher dump.
#   - `kd_frame_weight`   (B, 2, T)       float  silence/speech imbalance weight.
#   - `target_aligned`    True                   collator tripwire (semantic_kd_loss_from_batch asserts it).
#   - `is_text_only`      (B,)            bool   pure-text rows for the anchor term.
# --------------------------------------------------------------------------------------

_LOSS_EPS = 1e-8  # denominator floor -> a term with zero valid tokens reduces to 0 (never NaN)


def _weighted_token_ce(
    logits: "torch.Tensor", targets: "torch.Tensor", weights: "torch.Tensor"
) -> "torch.Tensor":
    """Per-token weighted cross-entropy, reduced to a scalar weighted mean.

    `logits` (..., C) float, `targets` (...) long, `weights` (...) float share the same leading
    dims. Returns sum(nll * weights) / sum(weights); when the weights sum to zero (every token
    masked -- Zone A / batch pad) it returns 0 with no NaN (the floor `_LOSS_EPS`
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

      - **KD**: delegated to `losses.semantic_kd_loss_from_batch` -- KL over Mimi semantic
        (level-0, k==0) logits. teacher/student share the frozen Mimi output space (2048)
        so no projection is needed. Teacher = softmax of the collator's top-k dump
        (`teacher_topk_val`/`teacher_topk_idx`) at temperature `kd_temperature`; student = log-softmax
        of `audio_logits[:, 0, 0]` (role 0, cb 0). Per-frame KL is masked by `kd_valid` and weighted by
        `kd_frame_weight` (silence/speech imbalance -- no separate auxiliary term). Scaled by
        `kd_weight`. Contributes exactly 0 when the batch carries no teacher dump. The SAME
        `losses.semantic_kd_loss` serves the on-policy path so the objective cannot drift.
      - **Audio CE**: cross-entropy of `audio_logits` vs `audio_codes` over all K
        codebooks, per-token weighted by `audio_loss_weight` (semantic=1, non-sem x0.02, synthetic
        user channel=0, Zone A / batch pad=0). Scaled by `ce_weight`.
      - **Text CE**: cross-entropy of `text_logits` vs frame-aligned `input_ids`
        (Inner Monologue -- no shift; each frame already carries its text token), per-token weighted
        by `text_loss_weight` (stream PAD x0.3, Zone A / batch pad=0). Scaled by `ce_weight`.
      - **Anchor**: a light pure-text CE restricted to `is_text_only` rows, guarding the
        backbone's multilingual text ability against catastrophic forgetting. Scaled by
        `text_anchor_weight`.

    total = kd + ce_audio + ce_text + anchor. Returned `metrics` are the post-weight contribution of
    each term (they sum to `total`) plus `total` itself -- floats for logging; per-task grad-norm
    dominance is watched separately by train_loop diagnostics (2). torch-only; numerically safe (no
    NaN when any term has zero valid tokens, no -inf materialization).
    """
    # The output-head / vocab layout this loss reads is defined by
    # `project_amnesty.models.haan.HaanForConditionalGeneration`. Not imported: the loss resolves
    # `outputs` duck-typed (getattr chain below), so an import here would only drag transformers
    # onto this path for a name that is never used.

    # Semantic KD is delegated to the ONE shared objective (losses/kd.py) so the offline path here
    # and the on-policy path use an identical KL -- no diverging in-file copy.
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
            text_logits = getattr(outputs, "logits", None)  # fallback
        if text_logits is None and isinstance(outputs, dict):
            text_logits = outputs.get("text_logits", outputs.get("logits"))
        audio_logits = getattr(outputs, "audio_logits", None)
        if audio_logits is None and isinstance(outputs, dict):
            audio_logits = outputs.get("audio_logits")
        assert text_logits is not None, (
            "model outputs.text_logits (B,T,V) [or .logits] is required for the text CE / anchor "
            "terms (ARCH 5.0.1/7.6)."
        )
        # A text-anchor micro-batch has no audio, so no audio logits either -- for that kind alone,
        # their absence is the contract rather than a violation of it.
        text_anchor_batch = batch.get("batch_kind") == "text_anchor"
        assert audio_logits is not None or text_anchor_batch, (
            "model outputs.audio_logits (B,2,K,T,C) is required for the audio CE / semantic KD "
            "terms (ARCH 5/5.1/7.6); model does not follow the I/O contract."
        )

        # --- (1) audio CE over all K codebooks, per-token weighted ---
        if text_anchor_batch:
            ce_audio_raw = None
        else:
            assert "audio_codes" in batch and "audio_loss_weight" in batch, (
                "batch must carry audio_codes (B,2,K,T) and audio_loss_weight (B,2,K,T) (ARCH 7.6)."
            )
            ce_audio_raw = _weighted_token_ce(
                audio_logits, batch["audio_codes"], batch["audio_loss_weight"]
            )

        # --- (2) text CE (Inner Monologue), per-token weighted ---
        assert "input_ids" in batch and "text_loss_weight" in batch, (
            "batch must carry input_ids (B,T) and text_loss_weight (B,T) (ARCH 5.0.1/7.6)."
        )
        input_ids = batch["input_ids"]
        text_w = batch["text_loss_weight"]
        # The model returns UNSHIFTED text logits: logits[t] predicts frame t+1, and the 1-step
        # autoregressive shift lives in the model's own loss_function -- which forward_with_contract
        # bypasses by calling forward WITHOUT text_labels. So every text term
        # must reproduce that shift exactly as ForCausalLMLoss does -- logits[:-1] against
        # input_ids[1:], the weight following the TARGET frame -- or it trains a frame-late objective
        # whose loss still falls silently. Audio is deliberately NOT shifted here: the depth decoder
        # predicts the current delay-patterned frame from hidden[t], matching Moshi's native depth
        # loss (modeling_moshi: audio_labels get the delay pattern, not a 1-step shift).
        text_logits_shifted = text_logits[:, :-1]
        text_targets = input_ids[:, 1:]
        text_w_shifted = text_w[:, 1:]
        # On a text-anchor batch this exact quantity IS the anchor term below -- computing it here as
        # well would count one loss twice, under two different weights.
        ce_text_raw = None if text_anchor_batch else _weighted_token_ce(
            text_logits_shifted, text_targets, text_w_shifted
        )

        # --- (3) semantic KD KL on k=0, delegated to losses.semantic_kd_loss.
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

        # --- (4) text anchor: light pure-text CE guarding Qwen3's multilingual ability ---
        #
        # Gated on `batch_kind`, which is what `RoutingCollator` actually emits. `is_text_only` is a
        # ROW-level field consumed inside the collators and never present on a batch, so gating the
        # anchor on it would leave `anchor_raw` always None and the term an exact zero on every step
        # -- the single guard against catastrophic forgetting silently switched off.
        is_text_only = batch.get("is_text_only")
        if text_anchor_batch:
            # The whole batch is the anchor: every row is pure text, so no row mask is needed.
            # Same 1-step shift as the text CE above (the model owns no shift here either).
            anchor_raw = _weighted_token_ce(text_logits_shifted, text_targets, text_w_shifted)
        elif is_text_only is not None:
            # Still honoured: a mixed batch tagging individual rows. `RoutingCollator` asserts batches
            # are homogeneous, so this is the fixture/experimental path rather than the live one.
            row = is_text_only.to(text_w.dtype).view(-1, *([1] * (text_w.dim() - 1)))
            anchor_raw = _weighted_token_ce(text_logits_shifted, text_targets, (text_w * row)[:, 1:])
        else:
            anchor_raw = None

        # --- combine. Skipped terms are exact scalar zeros on the graph dtype. ---
        # Anchored on `text_logits`: it is the one output present on both batch kinds.
        zero = text_logits.new_zeros(())
        ce_audio = ce_weight * ce_audio_raw if ce_audio_raw is not None else zero
        ce_text = ce_weight * ce_text_raw if ce_text_raw is not None else zero
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
    """Wrap the backbone with `fully_shard`. The `reshard_after_forward` flag switches ZeRO-2/3 class.

    Default False: keep parameters replicated, shard only grad/optim state. If VRAM is tight, fall
    back to True to also shard parameters (ZeRO-3 class). bf16 mixed precision. Each transformer
    block is sharded individually first, then the top-level module is wrapped -- compatible with
    Liger Kernel / FlashAttention.
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
    train in a full-precision group.

    LoRA targets existing weight matrices, so it cannot apply to fully-new parameters like the audio
    RVQ embeddings, Depth output heads, and Role Token (two additive vectors) -> in every Phase
    these live in a separate param group and train in full. They use a separate lr
    (`audio_param_lr`) to curb early drift. When `full_ft=False` (Phase 4) the backbone
    contributes only its LoRA adapters, while this new-parameter group still trains in full.

    "Full precision" here means the optimizer keeps 32-bit optimizer state for these parameters
    (registered via bitsandbytes GlobalOptimManager override) instead of the default 8-bit state,
    which is important for the sensitive new embedding/head parameters.
    """
    # Lazy heavy imports.
    import bitsandbytes as bnb  # noqa: PLC0415  (bnb.optim.PagedAdamW8bit)

    prefixes = _new_param_prefixes(args)

    def _is_new_param(name: str) -> bool:
        return _matches(name, prefixes)

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
            # `named_modules` yields "role_embedding", while the prefixes are parameter-name
            # prefixes ("role_embedding."). Append the separator so a module that IS the match
            # target -- and holds the parameter directly -- is not skipped.
            if _matches(f"{module_name}.", prefixes):
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
# 3. Diagnostic hooks
# ======================================================================================

def run_diagnostics(model: "torch.nn.Module", outputs: Any, batch: dict, step: int) -> dict:
    """Early-warning probing during training. Instrumentation to leave failure mechanisms in an
    explainable state.

    - **grad-norm dominance**: compare per-group gradient norms (new-param vs backbone). If one
      dominates, loss weights / PCGrad may need rebalancing.
    - **role-vector separation**: cosine similarity of the self/user Role Token (and Depth role
      embedding). Watch that role distinction is not diluted under other loss pressure; role
      differentiation concentrates at the semantic level.
    - **batch-element collapse**: probe that the Depth batch-2 (self/user) elements do not collapse
      to identical output ignoring role; measure divergence of the two batch elements. On
      collapse, signal to promote the projection to two role-specific ones (split MLP).

    Returns a metrics dict; never raises on missing tensors (best-effort probing).
    """
    metrics: dict[str, float] = {"diag_step": float(step)}

    # --- grad-norm dominance (new-param group vs backbone group) ---
    # Under FSDP2 each rank owns only a shard of every parameter's grad, so a purely local sum is a
    # per-shard partial. Accumulate the per-group squared grad sums locally, then all_reduce(SUM)
    # across ranks BEFORE taking the sqrt/ratio so the reported dominance ratio is GLOBAL, not a
    # shard artifact. Best-effort: guarded on is_initialized() and never raises.
    new_sq = 0.0
    back_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        # Under FSDP2 `p.grad` is a sharded DTensor. Calling `.pow(2).sum().item()` on it
        # would trigger an implicit redistribution/all-reduce to materialize the global scalar --
        # and the explicit `dist.all_reduce(SUM)` below would then reduce it a SECOND time
        # (double-counting, ~sqrt(world_size)x inflation in the reported norm). Take THIS rank's
        # local shard via `.to_local()` (no implicit comm) so `val` is a pure per-shard partial;
        # the single all_reduce below combines the partials exactly once. Plain (non-DTensor) grads
        # -- CPU/stub, or a non-sharded param -- have no `to_local`, so they pass through unchanged.
        local = g.to_local() if hasattr(g, "to_local") else g
        val = float(local.pow(2).sum().item())
        if _matches(name, NEW_PARAM_PREFIXES):
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
    # Haan carries TWO role signals -- the Temporal one and the Depth one -- and they are diagnosed
    # separately: the role-collapse alarm for the Temporal signal is distinct from the one for the
    # Depth signal. Reporting whichever the parameter iteration happened to reach first would
    # silently label one as the other.
    #
    # Reported as a RELATIVE DISTANCE, `||r0 - r1|| / mean(||r0||, ||r1||)`, not a cosine
    # similarity. Cosine reads the angle between two vectors, and neither mode is asking that question:
    #
    #   scale (the default)  the rows are elementwise GAINS initialised to all-ones, so cosine starts
    #                        at exactly 1.0 and barely moves -- `[1,1]` and `[4,4]` are parallel and
    #                        behave completely differently. Magnitude is the whole signal here.
    #   additive             the rows initialise to ZEROS, so cosine is 0/0 -> 0.0 at step 0 and stays
    #                        pinned while either row is near zero -- perturbing one role would leave
    #                        the reported cosine at exactly 0.0.
    #
    # One metric for both, so the series stays comparable across an ablation that flips the mode:
    # 0.0 means the roles are identical (the degenerate reading in BOTH modes, and the step-0 value
    # by construction), and it grows as they separate. Scale-invariant, so it does not drift with
    # overall weight magnitude.
    for key, role_emb in _find_role_embeddings(model).items():
        if role_emb.shape[0] >= 2:
            with torch.no_grad():
                self_vec = role_emb[0].flatten().float()
                user_vec = role_emb[1].flatten().float()
                mean_norm = (self_vec.norm() + user_vec.norm()) / 2
                metrics[key] = float(((self_vec - user_vec).norm() / (mean_norm + 1e-8)).item())

    # --- self/user output collapse: compare the ROLE axis of audio_logits (B,2,K,T,C) ---
    # The Depth batch-2 probe must detect self/user collapsing to identical audio
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


def _find_role_embeddings(model: "torch.nn.Module") -> "dict[str, torch.Tensor]":
    """Locate Haan's `(num_roles, dim)` role parameters, keyed by the metric they feed.

    `models.haan.RoleEmbedding` names its parameter `role_scale` (role_mode="scale", the default)
    or `role_emb` (role_mode="additive"), and the model holds two of them:

        role_embedding.<param>                     Temporal side  -> "role_sep"
        depth_decoder.model.role_embedding.<param> Depth side     -> "depth_role_sep"

    Matched by the OWNING MODULE path rather than by scanning for a "role"-ish substring, because
    the two are indistinguishable by shape and a substring scan returns whichever comes first in
    `named_parameters()` order -- which for Haan is the Depth one, reported under the Temporal
    metric's name. Missing entries are simply omitted (best-effort probing, never raises).

    Note both modes initialise to the identity, so the step-0 separation is 0.0 by construction --
    that is expected, not collapse. The diagnostic is the TREND.
    """
    found: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        parent = _strip_wrappers(name).rsplit(".", 1)[0]  # drop the parameter, keep the module path
        if not parent.endswith("role_embedding"):
            continue
        if p.dim() < 2 or p.shape[0] < 2:
            continue
        # The two are told apart by the owning subtree, so an FSDP/compile wrapper inserted
        # mid-path cannot swap them the way an exact-path match could.
        key = "depth_role_sep" if "depth_decoder" in parent else "role_sep"
        found.setdefault(key, p.detach())
    return found


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
    """Save the full resume manifest to `out_dir/{phase}/step_{global_step}`.

    Manifest: global_step/micro_step/epoch, model (DCP full state_dict), optimizer (DCP
    `get_optimizer_state_dict(full_state_dict, cpu_offload)`), scheduler state, sampler state
    (hasattr-guarded), RNG states (torch/cuda/numpy/python -- saved PER RANK as `rng_rank{r}.pt`
    under distributed so each rank restores its own stream), and a provenance JSON
    (world_size/grad_accum/seed/phase/schema_version) for resume-compatibility checks.

    Stability: the payload is written into a sibling `<dir>.tmp` and then `os.rename`d onto
    the final path, so a kill mid-write cannot leave a torn checkpoint. A keep-last-N rotation
    prunes older periodic `step_*` dirs (the just-written one is always retained; gate exports made
    by runner.py under other names are untouched). All ranks participate in the (collective) state
    gathers; rank 0 writes the shared payload (state.pt / provenance / train_args) while each rank
    writes its own `rng_rank{r}.pt`; every rank meets the closing barrier.
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
    # Every rank snapshots its OWN RNG (torch/cuda/numpy/python). Saving only rank 0's RNG and
    # restoring it onto every rank collapses all ranks to an identical generator after a resume, so
    # per-rank data augmentation / dropout stops diverging. Under real distributed each rank commits
    # its own `rng_rank{r}.pt`; `state.pt` still carries rank 0's RNG for the single-process /
    # backward-compatible path.
    rng_state = _collect_rng_state()

    if rank == 0:
        _rmtree_quiet(tmp_dir)  # clear any stale tmp from a previous crash
        os.makedirs(tmp_dir, exist_ok=True)

    # Barrier so the tmp dir rank 0 just created is visible before other ranks write into it.
    _dist_barrier()

    # Under genuine distributed (world_size > 1) each rank writes its own RNG file INTO the
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
    RNG states. Runs a provenance check first: world_size / grad_accum / schema mismatches are
    warned about (they break the resume's determinism keys) but do not block the resume.

    `epoch` and `micro_step` are saved in the manifest and restored and RETURNED (as a 3-tuple with
    `global_step`) so `train_loop` can resume the epoch loop and grad-accumulation cursor at the
    exact position the checkpoint captured -- without them a resume silently restarts the epoch loop
    at 0 and resets the micro-batch counter.
    RNG is restored PER RANK from `rng_rank{r}.pt` when present (each rank its own stream), falling
    back to the shared rng embedded in state.pt for single-process / legacy checkpoints.
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

    # Provenance check -- warn on determinism-key changes, then proceed tolerantly.
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
    # restore the epoch / micro-batch cursor (already in the manifest) for a faithful resume.
    epoch = int(payload.get("epoch", 0))
    micro_step = int(payload.get("micro_step", 0))
    _restore_sampler(loader, payload.get("sampler"), global_step)  # sampler cursor + set_step
    _restore_rng_for_rank(os.path.dirname(state_path), payload.get("rng"))  # per-rank rng
    return global_step, epoch, micro_step


# --------------------------------------------------------------------------------------
# Checkpoint state helpers (all guarded for the CPU/stub environment: DCP falls back to plain
# state_dict I/O, and missing sampler / gradient-checkpointing hooks degrade to no-ops).
# --------------------------------------------------------------------------------------

def _warn_if_distributed_partial(what: str, err: Exception) -> None:
    """Warn LOUDLY when a DCP full-gather fails under real distributed (world_size > 1).

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
        # never SILENTLY save a shard-only model under distributed.
        _warn_if_distributed_partial("_gather_full_state_dict", e)
        try:
            return model.state_dict()
        except Exception:
            return {}


def _gather_full_optim_state_dict(model: "torch.nn.Module", optimizer: Any) -> "dict | None":
    """Return a full (unsharded) optimizer state_dict via DCP, CPU-offloaded on rank 0.

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
        # never SILENTLY save a shard-only optimizer state under distributed.
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

    Tolerant: after a param-group change (Full-FT <-> LoRA/freeze phase transition) the
    optimizer state legitimately mismatches; any failure resumes model weights only.
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
    without these methods) is a silent no-op.
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
    """Restore THIS rank's own `rng_rank{r}.pt` if present, else the shared rng from state.pt.

    Under distributed, `save_checkpoint` writes one RNG file per rank, so each rank restores exactly
    the generator state IT held at checkpoint time -- combined with the per-rank seeding this
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
    """Dataset SCHEMA_VERSION from `project_amnesty.datasets.schema`, or 0 when absent.

    A failed import silently hits the `except` and pins every provenance record to
    schema_version=0, which disables the dataset-layout half of `_check_provenance`.
    """
    try:
        from project_amnesty.datasets.schema import SCHEMA_VERSION  # noqa: PLC0415

        return int(SCHEMA_VERSION)
    except Exception:
        return 0


def _check_provenance(ckpt_dir: str, args: TrainArgs) -> None:
    """Warn (never block) when a resume changes a determinism key vs the saved provenance.

    `world_size` and `grad_accum` feed the determinism keys (group draw `rng(seed, step)`, step
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
    """Keep the newest `keep` periodic `step_*` dirs; prune older ones and any stale `*.tmp`.

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
    """Guarded process-group init at `main` entry. Returns True iff this call started a group.

    Only initializes when the launcher exported WORLD_SIZE/RANK (i.e. torchrun) and no group is up
    yet: on CUDA it pins the local device with `set_device(LOCAL_RANK)` then `init_process_group`s
    NCCL; on CPU it uses Gloo. A plain single-process CPU smoke (no env vars) is a no-op, so import
    / py_compile / CPU runs are unaffected. Never raises.
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
    """Tear down the process group iff `_init_distributed` started it. No-op otherwise."""
    if not started:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _assert_optim_precision(model: "torch.nn.Module", optimizer: Any) -> dict:
    """Best-effort check that sensitive new params keep float32 optimizer state.

    The 8-bit optimizer must NOT quantize the brand-new audio-embedding / Depth-head / Role-Token
    params -- `build_optimizer` registers a 32-bit bitsandbytes override for them. If that override
    evaporated (e.g. the param identity changed across the FSDP2 wrap) their `exp_avg` would come
    back as int8/uint8, a silent precision loss. This confirms `exp_avg` is float32 for those params.

    Called once before the loop, so `optimizer.state` is usually still empty (bnb populates it on
    the first step); this stays best-effort -- it warns and returns a summary rather than raising on
    CPU/stub or when bnb state is absent. Authoritative verification requires a real multi-GPU run
    right after the first step.
    """
    summary = {"checked": 0, "float32": 0, "nonfloat32": 0, "missing_state": 0}
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        return summary
    try:
        for name, p in model.named_parameters():
            if not _matches(name, NEW_PARAM_PREFIXES):
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


def _set_audio_emb_requires_grad(model: "torch.nn.Module", flag: bool) -> int:
    """Freeze/unfreeze the audio input embeddings (the optional early-freeze warmup).

    Matches `embed_tokens.*` at the TOP level only (AUDIO_EMB_PREFIXES) -- the model's K shared
    audio tables. Deliberately not the backbone's text table (`model.embed_tokens`)
    nor the Depth decoder's own input tables (`depth_decoder.model.embed_tokens`), neither of
    which this warmup is about. Returns how many parameters were touched so the caller can tell
    a real freeze from a silent no-op.
    """
    touched = 0
    for name, p in model.named_parameters():
        if _matches(name, AUDIO_EMB_PREFIXES):
            p.requires_grad_(flag)
            touched += 1
    return touched


@torch.no_grad()
def _apply_scheduled_sampling(
    model: "torch.nn.Module", batch: dict, args: TrainArgs, global_step: int
) -> dict:
    """Scheduled-sampling self-rollout to correct exposure bias. ONLY reached when
    `on_policy_ratio > 0`; the `== 0` path never calls this, so it stays byte-for-byte unchanged.

    INPUT vs TARGET contract: `batch['audio_codes']` is the ground-truth Mimi code tensor
    that the audio-CE loss supervises against -- it MUST stay untouched, otherwise the model would
    be trained to predict its OWN (possibly wrong) rollout instead of the GT (a corrupted target).
    Scheduled sampling only changes what the model CONDITIONS on, so the self-rollout is written to
    a SEPARATE, documented key `batch['input_audio_codes']`: a clone of `audio_codes` whose trailing
    self-frames (role 0) of a random subset of rows are replaced by the model's own rollout. The
    model reads `input_audio_codes` as its input conditioning while the loss keeps scoring against
    the untouched GT `audio_codes`. Only a TRAILING conditioning window (not the whole T-frame
    stream) is overwritten, so the leading GT context is preserved and the model still receives real
    history before switching to its own predictions.

    KD validity: the offline `teacher_topk_*` dump is computed on the GT trajectory, so for any
    (row, trailing-frame) position that we splice with the rollout it is an INVALID teacher -- the
    student output there is now rollout-conditioned. We set `kd_valid=False` for those positions so
    the KD KL never scores a rollout-conditioned student against a GT-trajectory teacher. Both role
    outputs at the spliced frames are conditioned on the modified self stream, so both are
    invalidated. True on-policy KD would need a LIVE teacher re-inferred on the rollout (future
    work); until then those frames simply drop out of the KD term.

    Minimal and guarded: returns the batch unchanged whenever nothing can be spliced (no rows
    selected, no `audio_codes`, or `generate` yields no `codes`). Requires `model.generate` -- a real
    model per the I/O contract exposes it; its absence under on-policy is a contract violation and
    raises.

    Determinism on resume: the per-row Bernoulli selection is drawn from a LOCAL
    `torch.Generator` seeded on `(args.seed, global_step)` rather than the global RNG, so the exact
    same rows are selected when a run resumes at the same `global_step` -- the on-policy decision is
    reproducible and independent of how many global RNG draws happened before the crash.
    """
    audio_codes = batch.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.dim() != 4:
        return batch  # nothing to splice into

    ratio = min(max(float(args.on_policy_ratio), 0.0), 1.0)
    B, R, K, T = audio_codes.shape
    # Trailing conditioning window: replace only the tail, never the whole stream. Bounding
    # the rollout to the second half keeps a leading block of real GT history as context.
    cond_len = max(1, T // 2)
    # Reproducible per-(seed, step) selection: a local Generator keyed on the step keeps the choice
    # identical across a resume without perturbing the global RNG stream.
    gen = torch.Generator(device=audio_codes.device)
    gen.manual_seed((int(args.seed) * 1_000_003 + int(global_step)) & 0x7FFF_FFFF_FFFF_FFFF)
    sel = torch.rand(B, generator=gen, device=audio_codes.device) < ratio  # per-row Bernoulli ~ ratio
    if not bool(sel.any()):
        return batch

    # Self-play rollout through the SHARED contract helper (section 2b): condition on the leading GT
    # frames and invent exactly the trailing window about to be spliced. Both streams are rolled out
    # there and the assistant stream is end-anchored there, so this path and the offline simulation
    # (`evaluate._simulate_dialogue`) cannot drift apart by a frame -- they call the same code.
    codes, _ = rollout_with_contract(model, batch, n_frames=cond_len, prompt_frames=T - cond_len)
    if codes.shape[1] < 1:
        return batch

    gen_self = codes[:, 0]                       # (B, K_gen, T_gen) -- self stream (role 0)
    L = min(cond_len, gen_self.shape[-1])        # trailing overlap length to splice (<= cond_len < T)
    k = min(K, gen_self.shape[1])                # codebooks common to GT and rollout
    if L <= 0 or k <= 0:
        return batch

    rows = sel.nonzero(as_tuple=False).flatten()
    spliced = dict(batch)

    # build the INPUT conditioning tensor as a clone of the GT codes and splice the rollout
    # into ONLY the trailing L self-frames (codebooks 0..k-1) of the selected rows. LHS and RHS are
    # both (n_selected, k, L). The GT `audio_codes` (the CE target) is left entirely untouched.
    input_codes = audio_codes.clone()
    input_codes[rows, 0, :k, T - L:] = gen_self[rows, :k, gen_self.shape[-1] - L:].to(input_codes.dtype)
    spliced["input_audio_codes"] = input_codes

    # invalidate the offline GT-trajectory teacher for the rollout-conditioned frames of the
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
    """Canonical PyTorch loop + LR schedule + combined KD/CE + diagnostics + ckpt/logging.

    Control flow (fully implemented; the pieces that need models/datasets are `model`, `loader`,
    `loss_fn`, which are passed in):
        model.train()
        for each epoch: loader.set_epoch(epoch)           # LoaderBundle: datasets+sampler; hasattr-guarded
          for each batch in loader.loader:                # the bundle's DataLoader (falls back to loader)
            batch -> device
            outputs = forward_with_contract(model, batch)   # adapter: collator batch -> Moshi signature
            total_loss, metrics = loss_fn(outputs, batch)  # KD + CE + anchor
            (total_loss / grad_accum).backward()
            on accumulation boundary: clip grad -> optimizer.step() -> scheduler.step() -> zero_grad()
            periodic: run_diagnostics / log / save_checkpoint

    `global_step` counts optimizer steps and starts at `start_step` (resume); `micro_step`
    counts micro-batches (backward calls) and resumes at `start_micro`; the epoch loop resumes at
    `start_epoch`, so `loader.set_epoch(epoch)` re-seats the LoaderBundle's datasets + sampler
    RNG on the epoch the checkpoint captured rather than restarting at 0. The LR schedule advances with
    `scheduler.step()` AFTER `optimizer.step()`, once per optimizer step; `lr` is logged from
    `scheduler.get_last_lr()`. Notes: if `freeze_audio_emb_steps` > 0 the audio embeddings are
    frozen for the first N steps then unfrozen; the scheduler still advances the (frozen)
    group's LR during that window, which is intentional. `on_policy_ratio` > 0 mixes in self-rollout
    via scheduled sampling; the `== 0` default path is byte-for-byte unchanged.
    """
    # Seed ONLY a fresh run. On resume (start_step > 0) the RNG restored by load_checkpoint
    # must win -- re-seeding here would clobber the restored generator and desynchronize on-the-fly
    # data augmentation / dropout. Seed per-rank-decorrelated (`seed + rank`) so multi-rank runs do
    # not all draw the identical RNG stream.
    if int(start_step) == 0:
        torch.manual_seed(int(args.seed) + _dist_rank())

    device = _resolve_device()
    model.train()

    # Optional early-freeze of audio embeddings, unfrozen once past the warmup window.
    if args.freeze_audio_emb_steps > 0:
        frozen = _set_audio_emb_requires_grad(model, False)
        if frozen == 0 and _dist_rank() == 0:
            # A silent no-op here reads as "the warmup ran" while the embeddings trained from
            # step 0 -- exactly the drift the warmup is meant to prevent.
            print(
                f"[warn] freeze_audio_emb_steps={args.freeze_audio_emb_steps} but no parameter "
                f"matched {AUDIO_EMB_PREFIXES}; the audio embeddings were NOT frozen. The model "
                "does not use the expected parameter names (project_amnesty.models.haan).",
                flush=True,
            )

    global_step = int(start_step)
    micro_step = int(start_micro)  # resume the grad-accumulation cursor, not reset to 0
    epoch = int(start_epoch)       # default in case the epoch range is empty
    optimizer.zero_grad(set_to_none=True)

    # Resume the epoch loop at the restored epoch. `max(start_epoch + 1, args.epochs)` keeps
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

            # Scheduled-sampling self-rollout (exposure-bias correction). Engaged ONLY
            # when on_policy_ratio > 0; the == 0 default path skips this entirely and reaches the
            # forward below with `batch` byte-for-byte unchanged (no RNG draw, no model.generate).
            if args.on_policy_ratio > 0 and isinstance(batch, dict):
                batch = _apply_scheduled_sampling(model, batch, args, global_step)

            # --- forward ---
            # Routed through the adapter (section 2b) rather than `model(**batch)`: the model
            # speaks Moshi's signature, and passing the collator batch straight through would
            # both miss `assistant_audio_codes`/`user_audio_codes` and double-apply the delay.
            outputs = forward_with_contract(model, batch) if isinstance(batch, dict) else model(batch)
            total_loss, metrics = loss_fn(outputs, batch)

            # --- backward (scaled for gradient accumulation) ---
            (total_loss / args.grad_accum).backward()
            micro_step += 1

            # --- optimizer step on accumulation boundary ---
            if micro_step % args.grad_accum == 0:
                # On a real FSDP2 run `model.parameters()` are DTensors, and modern torch
                # `clip_grad_norm_` computes a DTensor-aware GLOBAL norm across shards (not a
                # per-shard local norm). If a given torch build clipped only the local shard, a
                # manual all_reduce of the squared norm would be needed here.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # advance the LR schedule once per optimizer step, AFTER optimizer.step().
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
    """Overlay `--<field> <value>` argv tokens onto the JSON-parsed dataclasses, in place.

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
    JSON; otherwise fields are parsed from command-line flags. Config is JSON only (no yaml).

    Config + argv overlay: with `--config`, `HfArgumentParser.parse_json_file` reads only the
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
        # honor CLI flags threaded alongside --config (runner's --resume/--out_dir/--phase).
        _overlay_cli_over_config(argv, (margs, dargs, targs))
    else:
        margs, dargs, targs = parser.parse_args_into_dataclasses()
    return margs, dargs, targs


def main() -> None:
    """Entry point. Order matters for correctness -- notably wrap -> build_optimizer so the
    32-bit optimizer override lands on the wrapped param identities, and gradient
    checkpointing enabled BEFORE the FSDP2 wrap.

    parse -> _init_distributed -> build_model -> gradient_checkpointing_enable ->
    setup_fsdp2 -> build_optimizer -> get_scheduler -> build_dataloader ->
    [load_checkpoint if resume] -> _assert_optim_precision -> train_loop, with a
    guaranteed `_shutdown_distributed` in `finally`. Usually runner.py drives this per phase.
    """
    # `get_scheduler` is imported lazily here (not at module top): transformers is not a dependency
    # of the CPU import/py_compile smoke, and this module must import with torch alone.
    from transformers import get_scheduler  # noqa: PLC0415

    margs, dargs, targs = parse_args()

    dist_started = _init_distributed()  # guarded env(WORLD_SIZE/RANK) init, no-op single-process
    try:
        # build_model raises NotImplementedError until the models/ forward + warm-start land
        # (datasets/ and losses/ are already real); everything wrapped around it -- FSDP2,
        # optimizer, scheduler, resume, the loop -- is real.
        model = build_model(margs, targs)

        # Enable activation (gradient) checkpointing BEFORE the FSDP2 wrap. hasattr/try guarded
        # so a stub model without the hook (or an older signature) does not break import/CPU runs.
        if targs.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()  # older signature without the kwargs arg

        model = setup_fsdp2(model, targs)               # fully_shard + bf16 MixedPrecisionPolicy
        optimizer = build_optimizer(model, targs)       # wrap -> build -> 32-bit override identity
        scheduler = get_scheduler(                      # warmup + cosine, unit = optimizer step
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
            # load_checkpoint returns (global_step, epoch, micro_step); thread all three
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

        _assert_optim_precision(model, optimizer)       # sensitive-param exp_avg dtype check
        train_loop(
            model, loader, optimizer, scheduler, loss_fn, targs,
            start_step=start_step, start_epoch=start_epoch, start_micro=start_micro,
        )
    finally:
        _shutdown_distributed(dist_started)             # destroy the group we started


if __name__ == "__main__":
    main()
