"""Dual-source warm-start for Haan (ARCHITECTURE 1 / 5.4.1 / 5.4.2).

Haan is assembled from two pretrained models, because its two halves come from different places:

    model.* / lm_head / self_attn.{q,k}_norm     <- Qwen3      (ARCHITECTURE 1)
    embed_tokens.{0..K-1}                        <- Moshi, user half   (ARCHITECTURE 5.4.2)
    depth_decoder.*                              <- Moshi      (ARCHITECTURE 5.4.1)

**Why the Qwen3 side needs surgery and the Moshi side does not.** Haan's backbone is
`HaanModel(MoshiModel)`, so it is laid out like Moshi even while holding Qwen3's weights. The two
describe the same transformer with different module shapes:

  - Qwen3 keeps `gate_proj` and `up_proj` separate; `MoshiGatingMLP` fuses them into one `fc1` and
    splits the result in half at run time. So `fc1 = cat([gate, up], dim=0)`, **gate first** --
    `MoshiGatingMLP.forward` does `view(..., 2, -1)` and activates `h[..., 0, :]`, which is the
    contiguous first half. Swapping the halves is silent and trains a model whose gate and value
    are exchanged. `ffn_dim` must be `2 * intermediate_size` for the fused shape to exist at all.
  - `MoshiLinear` wraps `nn.Linear` as `.linear`, so `q_proj.weight` -> `q_proj.linear.weight`.
  - `MoshiModel` allocates `vocab_size + 1` embedding rows -- one reserved id used as the text
    stream's audio-only/BOS position. Qwen3 has no such row, so it is appended, not loaded.
  - Qwen3's QK-Norm has a home only because `HaanAttention` adds `q_norm`/`k_norm`
    (`config.use_qk_norm`); a stock `MoshiAttention` would drop 2 tensors per layer.

Everything else is a rename or a straight copy. The mapping was verified end to end: a tiny
`Qwen3Model` and a `HaanModel` built through it produce bit-identical fp32 output.

**What starts cold.** `depth_decoder.model.text_embed_tokens` (~155M parameters) has no source at
all -- Moshi's is sized for its own 32k tokenizer and the rows do not correspond to Qwen3's. That,
plus the reserved embedding row and the two role embeddings, is ~1.7% of the model (the depth text
table dominates the figure). Everything unsourced is printed rather than allowed to pass as
transferred.

Location note: this module lives under `utils/` rather than `models/haan/` so that `models/` never
imports from `utils/`. It is model *assembly*, not model *definition* -- the classes in
`models/haan/` stay independent of where their initial weights come from.
"""

from __future__ import annotations

import re

import torch

__all__ = [
    "INIT_SOURCES",
    "UNSOURCED_BY_DESIGN",
    "haan_config_from_moshi",
    "haan_config_from_qwen3_and_moshi",
    "moshi_audio_tables",
    "warm_start_from_moshi",
    "warm_start_qwen3_moshi",
]

# ARCHITECTURE 5.4.2. Which half of Moshi's `2 * K` audio tables seeds Haan's `K` shared ones.
#
#   "user"   emb[K:2K] -- the DEFAULT and the documented choice. Moshi's user stream was trained
#            with the second speaker's voice resampled per example, so this half never collapsed
#            onto a single actor. Removing fixed-speaker bias is this project's premise, which
#            makes it the better prior for voice-prompt cloning (5.2).
#   "self"   emb[0:K]  -- Moshi's own stream, narrowed to one actor's voice by instruct tuning.
#            Kept for the 3.6 ablation.
#   "random" leave Haan's own init; measures what the warm-start is actually worth.
#
# The choice is NOT cosmetic and cannot be left to name matching: Moshi's tables 0..7 carry the
# same NAMES as Haan's 0..7, so a naive load_state_dict silently takes the "self" half.
INIT_SOURCES = ("user", "self", "random")

# Moshi audio tables are `embed_tokens.<i>.weight` in HF format. Anchored at the start so it
# matches neither `model.embed_tokens.weight` (the backbone's text table) nor
# `depth_decoder.model.embed_tokens.N.weight`.
_AUDIO_TABLE_RE = re.compile(r"^embed_tokens\.(\d+)\.weight$")

# Backbone projections whose Moshi name carries the `MoshiLinear` wrapper. Restricted to
# `model.layers.` on purpose: the depth decoder has leaves with the identical final name whose
# weights are rank-3 `MoshiFlexibleLinear` tensors and must never receive a 2-D Qwen3 tensor.
_PROJ_RENAME_RE = re.compile(r"^(model\.layers\.\d+\.self_attn\.[qkvo]_proj)\.weight$")
_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.(gate|up)_proj\.weight$")
_DOWN_RE = re.compile(r"^(model\.layers\.\d+)\.mlp\.down_proj\.weight$")

# Haan parameters that no source was ever going to supply, under the Qwen3+Moshi assembly.
# Enumerated rather than inferred so that a genuine shape mismatch still fails loudly instead of
# being absorbed as "expected" -- in particular `text_embed_tokens`, whose Moshi counterpart has
# the right NAME and the wrong SHAPE and would otherwise abort the whole warm-start.
UNSOURCED_BY_DESIGN = (
    # Moshi's is (32001, depth_hidden), sized for its own tokenizer; the rows do not correspond to
    # Qwen3's vocabulary. Not sliceable, not remappable -- this is the one large cold block.
    "depth_decoder.model.text_embed_tokens.weight",
)


# ======================================================================================
# Config construction
# ======================================================================================

def haan_config_from_moshi(
    moshi_config,
    *,
    num_roles: int = 2,
    role_mode: str = "scale",
    predict_user_stream: bool = True,
    use_qk_norm: bool = False,
    **overrides,
):
    """Derive a `HaanConfig` describing a MOSHI-backbone Haan.

    Used by the Moshi-only warm-start (the ARCHITECTURE 3.6 baseline arm). Everything dimensional
    is Moshi's, unchanged -- 5.4.1 reuses audio cardinality 2048 and depth dim 1024 as-is.

    `use_qk_norm` defaults to **False** here, and that default is load bearing: `HaanConfig`'s own
    default is True (Haan's normal shape is a Qwen3 backbone) and `MoshiConfig.to_dict()` carries
    no `use_qk_norm` key, so without passing it explicitly every Moshi warm-start would silently
    build QK-Norm the Moshi weights never saw. An all-ones RMSNorm still rescales its input, so
    that is not a harmless extra module -- it breaks the transfer at every layer's q and k.
    """
    from project_amnesty.models.haan.configuration_haan import HaanConfig  # noqa: PLC0415

    config_dict = moshi_config.to_dict()
    config_dict.pop("model_type", None)
    config_dict.pop("architectures", None)

    depth_dict = dict(config_dict.get("depth_decoder_config") or {})
    depth_dict.pop("model_type", None)
    codebooks_per_role = int(config_dict["num_codebooks"])
    depth_dict["num_codebooks"] = num_roles * codebooks_per_role if predict_user_stream else codebooks_per_role

    config_dict["depth_decoder_config"] = depth_dict
    config_dict["num_roles"] = num_roles
    config_dict["role_mode"] = role_mode
    config_dict["use_qk_norm"] = use_qk_norm
    # `moshi_config.to_dict()` carries Moshi's own `sliding_window=3000`, which would override
    # `HaanConfig`'s deliberate None. Cleared so this arm differs from the Qwen3 arm ONLY in the
    # backbone -- it exists to isolate the Qwen3 substitution (ARCH 3.6), and a context limit
    # present in one arm and absent in the other is a confound. Harmless in practice: a 3000-frame
    # window is bit-identical to full causal below 3000, and this arm is capped at Moshi's
    # `max_position_embeddings` anyway.
    config_dict["sliding_window"] = None
    config_dict.update(overrides)

    return HaanConfig(**config_dict)


def haan_config_from_qwen3_and_moshi(
    qwen3_config,
    moshi_config,
    *,
    num_roles: int = 2,
    role_mode: str = "scale",
    predict_user_stream: bool = True,
    **overrides,
):
    """Derive a `HaanConfig` describing a QWEN3-backbone Haan with Moshi's audio stack.

    Backbone dimensions come from Qwen3; the audio, Mimi and depth sub-configs from Moshi. Four
    fields are handled specially because getting them wrong is silent rather than fatal:

      - **`rope_parameters`** is set as a whole dict, never as a bare `rope_theta=`. Layering
        `rope_theta=1_000_000` onto a Moshi-derived config does nothing: the dict already carries
        `rope_parameters={"rope_theta": 10000.0}` and transformers resolves it with `setdefault`,
        so the existing value wins and the override is dropped with no warning at all.
      - **`pad_token_id`** is forced to None. `MoshiModel` passes it straight into
        `nn.Embedding(..., padding_idx=)`, so a real id there zeroes that row AND pins its gradient
        to zero permanently. Moshi keeps its BOS/pad ids in the GenerationConfig instead.
      - **`ffn_dim`** is `2 * intermediate_size`, because `MoshiGatingMLP` fuses gate and up into
        one `fc1`. Leaving Moshi's own value makes both MLP projections fail to load.
      - **`sliding_window`** is cleared to None. Qwen3 is full-attention, and Moshi's inherited
        3000 is not inert: `MoshiModel.forward` selects `create_sliding_window_causal_mask`
        whenever it is non-None, and the cache builds window layers that physically evict KV. It
        is bit-identical to full causal up to 3000 positions, so it passes every short test and
        diverges only on long context -- which is exactly why it has to be set here rather than
        noticed later. It must also be set BEFORE any cache is constructed.
    """
    from project_amnesty.models.haan.configuration_haan import HaanConfig  # noqa: PLC0415

    qwen3 = qwen3_config.to_dict()
    moshi = moshi_config.to_dict()

    head_dim = qwen3.get("head_dim") or qwen3["hidden_size"] // qwen3["num_attention_heads"]
    rope_theta = (qwen3.get("rope_parameters") or {}).get("rope_theta") or qwen3.get("rope_theta")

    config_dict = {
        # --- backbone: Qwen3 (ARCHITECTURE 1) ---
        "vocab_size": qwen3["vocab_size"],
        "hidden_size": qwen3["hidden_size"],
        "num_hidden_layers": qwen3["num_hidden_layers"],
        "num_attention_heads": qwen3["num_attention_heads"],
        "num_key_value_heads": qwen3["num_key_value_heads"],
        # Explicit: MoshiConfig derives head_dim as hidden_size // num_attention_heads when unset,
        # which happens to be right for Qwen3-8B (4096/32) and is not right in general.
        "head_dim": head_dim,
        "hidden_act": qwen3.get("hidden_act", "silu"),
        "max_position_embeddings": qwen3["max_position_embeddings"],
        "rms_norm_eps": qwen3["rms_norm_eps"],
        "ffn_dim": 2 * qwen3["intermediate_size"],
        "rope_parameters": {"rope_type": "default", "rope_theta": rope_theta},
        "initializer_range": qwen3.get("initializer_range", 0.02),
        "attention_dropout": qwen3.get("attention_dropout", 0.0),
        # Impossible to tie here regardless: embed_tokens has one row lm_head does not.
        "tie_word_embeddings": False,
        "use_cache": qwen3.get("use_cache", True),
        "pad_token_id": None,
        "sliding_window": None,
        "use_qk_norm": True,
        # --- audio stack: Moshi (ARCHITECTURE 4 / 5.4) ---
        "audio_vocab_size": moshi["audio_vocab_size"],
        "num_codebooks": moshi["num_codebooks"],
        "audio_encoder_config": moshi.get("audio_encoder_config"),
        "num_roles": num_roles,
        "role_mode": role_mode,
    }

    # The depth decoder is Moshi's end to end, so it keeps Moshi's dimensions AND Moshi's
    # `rms_norm_eps` -- only the backbone becomes Qwen3.
    depth_dict = dict(moshi.get("depth_decoder_config") or {})
    depth_dict.pop("model_type", None)
    codebooks_per_role = int(moshi["num_codebooks"])
    depth_dict["num_codebooks"] = num_roles * codebooks_per_role if predict_user_stream else codebooks_per_role
    config_dict["depth_decoder_config"] = depth_dict

    config_dict.update(overrides)
    return HaanConfig(**config_dict)


# ======================================================================================
# Checkpoint reading
# ======================================================================================

def moshi_audio_tables(
    moshi_ckpt: str,
    *,
    num_codebooks: int = 8,
    side: str = "user",
    revision: str | None = None,
) -> list["torch.Tensor"]:
    """Read Moshi's audio input embedding tables straight out of the checkpoint shards.

    Returns `num_codebooks` tensors of shape `(audio_vocab_size + 1, hidden_size)` -- the half of
    Moshi's `2 * K` tables named by `side` ("user" -> `[K, 2K)`, "self" -> `[0, K)`).

    Reads only the `embed_tokens.*` tensors via safetensors' lazy `safe_open`, so the backbone is
    never materialized. This is the same quantity the warm-start copies in, which is why
    `utils/evaluate.py` measures the ARCHITECTURE 5.4.2 / RISKS 3 embedding drift against THIS
    function rather than re-deriving the initialization independently.

    HF-format checkpoints only (`kmhf/hf-moshiko` and friends), where the tables are
    `embed_tokens.<i>.weight`. The original Kyutai release names them `emb.<i>.weight`;
    `project_amnesty/tools/inspect_moshi_weights.py` reads that layout instead.
    """
    if side not in ("user", "self"):
        raise ValueError(f"`side={side!r}` must be 'user' or 'self'.")

    from safetensors import safe_open  # noqa: PLC0415

    wanted_start = num_codebooks if side == "user" else 0
    wanted = {wanted_start + k: k for k in range(num_codebooks)}  # checkpoint index -> Haan slot

    tables: dict[int, torch.Tensor] = {}
    for shard in _resolve_shards(moshi_ckpt, _AUDIO_TABLE_RE, revision=revision):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                match = _AUDIO_TABLE_RE.match(key)
                if match is not None and int(match.group(1)) in wanted:
                    tables[wanted[int(match.group(1))]] = f.get_tensor(key)

    missing = sorted(set(range(num_codebooks)) - set(tables))
    if missing:
        raise KeyError(
            f"{moshi_ckpt} is missing the {side}-side audio tables for codebooks {missing} "
            f"(looked for embed_tokens.{{{wanted_start}..{wanted_start + num_codebooks - 1}}}.weight). "
            "Is this an HF-format Moshi checkpoint?"
        )
    return [tables[k] for k in range(num_codebooks)]


def _resolve_shards(ckpt: str, key_filter: "re.Pattern | None" = None, *, revision: str | None = None) -> list[str]:
    """Local paths of a checkpoint's safetensors shards.

    `key_filter` restricts the result to shards that actually hold a matching tensor, which for a
    cold cache also skips downloading the rest; pass None to take every shard.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    def _wanted(weight_map: dict) -> list[str]:
        return sorted({s for k, s in weight_map.items() if key_filter is None or key_filter.match(k)})

    if os.path.isdir(ckpt):
        index_path = os.path.join(ckpt, "model.safetensors.index.json")
        single = os.path.join(ckpt, "model.safetensors")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                return [os.path.join(ckpt, name) for name in _wanted(json.load(f)["weight_map"])]
        if os.path.isfile(single):
            return [single]
        raise FileNotFoundError(f"no safetensors weights under {ckpt}")

    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import EntryNotFoundError  # noqa: PLC0415

    try:
        index_path = hf_hub_download(repo_id=ckpt, filename="model.safetensors.index.json", revision=revision)
    except (EntryNotFoundError, OSError):
        # Unsharded checkpoint: a single model.safetensors and no index.
        return [hf_hub_download(repo_id=ckpt, filename="model.safetensors", revision=revision)]

    with open(index_path, encoding="utf-8") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    return [hf_hub_download(repo_id=ckpt, filename=name, revision=revision) for name in _wanted(weight_map)]


def _stream_tensors(ckpt: str, *, revision: str | None = None):
    """Yield `(name, tensor)` for every tensor in a checkpoint, one shard at a time.

    Streamed rather than loaded through `Qwen3ForCausalLM.from_pretrained` because holding Qwen3,
    Moshi and Haan resident at once is ~50 GB in bf16. Safe to read raw for Qwen3: unlike Moshi --
    whose released checkpoints need transformers' `_checkpoint_conversion_mapping` to reach the
    current module layout -- Qwen3's on-disk keys already match its modules.
    """
    from safetensors import safe_open  # noqa: PLC0415

    for shard in _resolve_shards(ckpt, revision=revision):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


# ======================================================================================
# Warm-start entry points
# ======================================================================================

def warm_start_qwen3_moshi(
    qwen3_ckpt: str = "Qwen/Qwen3-8B",
    moshi_ckpt: str = "kmhf/hf-moshiko",
    *,
    init_source: str = "user",
    num_roles: int = 2,
    role_mode: str = "scale",
    predict_user_stream: bool = True,
    dtype: "torch.dtype | None" = None,
    revision: str | None = None,
    verbose: bool = True,
    **config_overrides,
):
    """Assemble a Haan model: Qwen3 backbone + Moshi audio embeddings + Moshi depth decoder.

    ARCHITECTURE 1 + 5.4.1 + 5.4.2 as one operation. See the module docstring for the full mapping
    and for what starts cold.

    Args:
        qwen3_ckpt: Qwen3 checkpoint for the Temporal backbone, `lm_head` and QK-Norm.
        moshi_ckpt: Moshi checkpoint for the audio tables and the whole depth decoder.
        init_source: which half of Moshi's audio tables seeds the shared ones (`INIT_SOURCES`).
        num_roles / role_mode / predict_user_stream: forwarded to the config builder.
        dtype: dtype to build Haan as. Defaults to Qwen3's checkpoint dtype.
        verbose: print the transfer summary, including everything that did NOT transfer.
        **config_overrides: forwarded onto the `HaanConfig`.

    Returns:
        `HaanForConditionalGeneration`, warm-started.
    """
    from transformers import AutoConfig  # noqa: PLC0415
    from transformers.models.moshi.configuration_moshi import MoshiConfig  # noqa: PLC0415

    from project_amnesty.models.haan.modeling_haan import HaanForConditionalGeneration  # noqa: PLC0415

    if init_source not in INIT_SOURCES:
        raise ValueError(f"`init_source={init_source!r}` must be one of {INIT_SOURCES}.")

    qwen3_config = AutoConfig.from_pretrained(qwen3_ckpt, revision=revision)
    moshi_config = MoshiConfig.from_pretrained(moshi_ckpt)
    config = haan_config_from_qwen3_and_moshi(
        qwen3_config,
        moshi_config,
        num_roles=num_roles,
        role_mode=role_mode,
        predict_user_stream=predict_user_stream,
        **config_overrides,
    )

    dtype = dtype or getattr(qwen3_config, "dtype", None) or getattr(qwen3_config, "torch_dtype", None)
    model = HaanForConditionalGeneration._from_config(config, dtype=dtype)

    summary = _new_summary()
    _copy_qwen3_backbone(qwen3_ckpt, model, summary, revision=revision)
    _copy_moshi_audio_and_depth(moshi_ckpt, model, summary, init_source=init_source, dtype=dtype)

    _finalize(model, summary)
    if verbose:
        _report(summary, sources=f"{qwen3_ckpt} (backbone) + {moshi_ckpt} (audio/depth)", init_source=init_source)
    return model


def warm_start_from_moshi(
    moshi_ckpt: str = "kmhf/hf-moshiko",
    *,
    init_source: str = "user",
    num_roles: int = 2,
    role_mode: str = "scale",
    predict_user_stream: bool = True,
    dtype: "torch.dtype | None" = None,
    revision: str | None = None,
    verbose: bool = True,
    **config_overrides,
):
    """Assemble a Haan model warm-started ENTIRELY from Moshi -- backbone included.

    The ARCHITECTURE 3.6 baseline arm: the same audio/Depth deltas, but Helium rather than Qwen3
    behind them, so the Korean-emergence comparison has a control. Built with `use_qk_norm=False`,
    since a Moshi backbone has no QK-Norm to load.

    Memory: both models are resident while the tensors are copied (~2x the checkpoint). The Moshi
    side is released before returning. Load in bf16 (`dtype=torch.bfloat16`) to halve it.
    """
    from transformers.models.moshi.modeling_moshi import MoshiForConditionalGeneration  # noqa: PLC0415

    from project_amnesty.models.haan.modeling_haan import HaanForConditionalGeneration  # noqa: PLC0415

    if init_source not in INIT_SOURCES:
        raise ValueError(f"`init_source={init_source!r}` must be one of {INIT_SOURCES}.")

    # Loaded through the class (rather than raw shards) on purpose: transformers'
    # `_checkpoint_conversion_mapping` for Moshi rewrites the released checkpoints' older key
    # layout (`decoder.model.*`, `depth_decoder.layers.*`) onto the current module paths. Reading
    # raw tensors would reimplement that mapping, and drift from it on the next upstream change.
    moshi = MoshiForConditionalGeneration.from_pretrained(moshi_ckpt, dtype=dtype, revision=revision)
    try:
        config = haan_config_from_moshi(
            moshi.config,
            num_roles=num_roles,
            role_mode=role_mode,
            predict_user_stream=predict_user_stream,
            use_qk_norm=False,
            **config_overrides,
        )
        model = HaanForConditionalGeneration._from_config(config, dtype=dtype or moshi.dtype)
        source = dict(moshi.named_parameters())
        source.update(moshi.named_buffers())
        summary = _new_summary()
        _copy_named(source, model, summary, init_source=init_source)
    finally:
        del moshi

    _finalize(model, summary)
    if verbose:
        _report(summary, sources=moshi_ckpt, init_source=init_source)
    return model


# ======================================================================================
# Copy machinery
# ======================================================================================

def _new_summary() -> dict:
    return {"copied": [], "kept": [], "mismatched": [], "unused": [], "notes": []}


@torch.no_grad()
def _copy_qwen3_backbone(qwen3_ckpt: str, haan, summary: dict, *, revision: str | None = None) -> None:
    """Stream Qwen3's parameters into Haan's Moshi-shaped backbone."""
    targets = _targets(haan)

    # gate/up fuse into one `fc1`, so they are buffered until both halves are in hand.
    mlp_halves: dict[str, dict[str, torch.Tensor]] = {}
    embed_source: torch.Tensor | None = None
    seen: set[str] = set()

    for name, tensor in _stream_tensors(qwen3_ckpt, revision=revision):
        seen.add(name)

        if name == "model.embed_tokens.weight":
            embed_source = tensor
            continue

        gate_up = _GATE_UP_RE.match(name)
        if gate_up is not None:
            mlp_halves.setdefault(gate_up.group(1), {})[gate_up.group(2)] = tensor
            continue

        target_name = _DOWN_RE.sub(r"\1.mlp.fc2.weight", name)
        target_name = _PROJ_RENAME_RE.sub(r"\1.linear.weight", target_name)
        _assign(targets, target_name, tensor, summary, source_name=name)

    # --- fused MLP: gate FIRST (see the module docstring; the order is not recoverable later) ---
    for layer, halves in sorted(mlp_halves.items()):
        if set(halves) != {"gate", "up"}:
            summary["mismatched"].append(f"{layer}.mlp: expected gate_proj and up_proj, got {sorted(halves)}")
            continue
        _assign(
            targets,
            f"{layer}.mlp.fc1.weight",
            torch.cat([halves["gate"], halves["up"]], dim=0),
            summary,
            source_name=f"{layer}.mlp.{{gate,up}}_proj.weight",
        )

    # --- text embedding: Qwen3's rows, plus Moshi's one reserved row ---
    # `MoshiModel` allocates `vocab_size + 1` rows; the extra id is the text stream's audio-only /
    # BOS position and has no Qwen3 counterpart. Seeded from the mean of the real rows so it starts
    # in-distribution rather than at whatever the random init produced.
    target = targets.get("model.embed_tokens.weight")
    if embed_source is None:
        summary["kept"].append("model.embed_tokens.weight")
    elif target is None:
        summary["unused"].append("model.embed_tokens.weight")
    elif tuple(target.shape) != (embed_source.shape[0] + 1, embed_source.shape[1]):
        summary["mismatched"].append(
            f"model.embed_tokens.weight: haan {tuple(target.shape)} != qwen3 {tuple(embed_source.shape)} + 1 row"
        )
    else:
        source = embed_source.to(device=target.device, dtype=target.dtype)
        target[: source.shape[0]].copy_(source)
        target[source.shape[0]].copy_(source.mean(dim=0))
        summary["copied"].append("model.embed_tokens.weight")
        summary["notes"].append(
            f"model.embed_tokens.weight: rows [0:{source.shape[0]}) from Qwen3; reserved row "
            f"{source.shape[0]} (Moshi's audio-only/BOS id) seeded with the row mean -- no Qwen3 source."
        )


@torch.no_grad()
def _copy_moshi_audio_and_depth(
    moshi_ckpt: str, haan, summary: dict, *, init_source: str, dtype: "torch.dtype | None" = None
) -> None:
    """Copy Moshi's audio tables and its whole depth decoder into Haan, leaving the backbone alone."""
    from transformers.models.moshi.modeling_moshi import MoshiForConditionalGeneration  # noqa: PLC0415

    moshi = MoshiForConditionalGeneration.from_pretrained(moshi_ckpt, dtype=dtype)
    try:
        source = {
            name: tensor
            for name, tensor in moshi.named_parameters()
            if _AUDIO_TABLE_RE.match(name) or name.startswith("depth_decoder.")
        }
        _copy_named(source, haan, summary, init_source=init_source, audio_and_depth_only=True)
    finally:
        del moshi


@torch.no_grad()
def _copy_named(
    source: dict, haan, summary: dict, *, init_source: str, audio_and_depth_only: bool = False
) -> None:
    """Name-matched copy with the one audio remap applied.

    `audio_and_depth_only=True` restricts the walk to the parameters a Moshi source is responsible
    for in the dual-source assembly, leaving the Qwen3-filled backbone untouched.
    """
    codebooks_per_role = int(haan.config.num_codebooks)
    audio_offset = codebooks_per_role if init_source == "user" else 0

    targets = _targets(haan)
    consumed: set[str] = set()

    for name, target in targets.items():
        audio_table = _AUDIO_TABLE_RE.match(name)
        if audio_and_depth_only and audio_table is None and not name.startswith("depth_decoder."):
            continue

        if audio_table is not None:
            # The one remap: Haan's shared table `k` takes Moshi's `k + offset`. Same NAME space as
            # Moshi's assistant half, so this must not fall through to the identity branch.
            if init_source == "random":
                summary["kept"].append(name)
                continue
            source_name = f"embed_tokens.{int(audio_table.group(1)) + audio_offset}.weight"
        else:
            source_name = name

        tensor = source.get(source_name)
        if tensor is None:
            summary["kept"].append(name)
            continue
        consumed.add(source_name)
        _assign(targets, name, tensor, summary, source_name=source_name)

    summary["unused"].extend(sorted(set(source) - consumed))


def _targets(haan) -> dict:
    targets = dict(haan.named_parameters())
    targets.update(haan.named_buffers())
    return targets


def _assign(targets: dict, name: str, tensor: torch.Tensor, summary: dict, *, source_name: str) -> None:
    """Copy one tensor into `targets[name]`, recording the outcome."""
    target = targets.get(name)
    if target is None:
        summary["unused"].append(source_name)
        return
    if target.shape != tensor.shape:
        # Not raised here: the walk finishes first so `_finalize` can report every mismatch at
        # once, and so an expected-no-source name (UNSOURCED_BY_DESIGN) can be reclassified.
        summary["mismatched"].append(
            f"{name}: haan {tuple(target.shape)} != source {source_name} {tuple(tensor.shape)}"
        )
        return
    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    summary["copied"].append(name)


def _finalize(model, summary: dict) -> None:
    """Reclassify the by-design gaps, complete the `kept` set, then fail on real mismatches."""
    present = _targets(model)
    copied = set(summary["copied"])

    for name in UNSOURCED_BY_DESIGN:
        if name not in present or name in copied:
            continue
        # A same-name/different-shape source landed in `mismatched`; that is the expected outcome
        # for these, not a broken checkpoint. Drop those entries and record the gap instead.
        summary["mismatched"] = [m for m in summary["mismatched"] if not m.startswith(f"{name}:")]

    # `kept` is derived here rather than accumulated during the walk, because no single walk sees
    # every parameter: the Qwen3 pass iterates its own SOURCE tensors, and the Moshi pass is scoped
    # to the audio tables and the depth decoder. A parameter belonging to neither -- notably the
    # Temporal `role_embedding` -- was therefore correctly left at its init and silently omitted
    # from the report. Since the report exists precisely to name what did not transfer, it is
    # computed as "everything not copied" instead.
    mismatched_names = {m.split(":", 1)[0] for m in summary["mismatched"]}
    summary["kept"] = sorted(
        name
        for name in present
        if name not in copied and name not in mismatched_names and not _is_derived(name)
    )

    if summary["mismatched"]:
        # A surviving mismatch means the checkpoint does not describe this architecture -- the
        # result would be neither warm-started nor cleanly random, so it is an error.
        raise ValueError(
            "warm-start shape mismatch (a source checkpoint does not match this Haan config):\n  "
            + "\n  ".join(sorted(set(summary["mismatched"])))
        )


def _is_derived(name: str) -> bool:
    """Buffers reconstructed from the config at build time, so never 'missing' from a checkpoint.

    `rotary_emb.inv_freq` / `original_inv_freq` are a pure function of `rope_parameters` and
    `head_dim`, and Qwen3 does not ship them (they are non-persistent). Listing them alongside the
    genuinely cold parameters would bury the ones that matter.
    """
    return ".rotary_emb." in f".{name}"


def _report(summary: dict, *, sources: str, init_source: str) -> None:
    """Print what did and did not transfer. The 'kept' list is the point of this function."""
    kept = sorted(set(summary["kept"]))
    print(
        f"[haan] warm-start from {sources}: {len(set(summary['copied']))} tensors copied "
        f"(audio init_source={init_source!r}).",
        flush=True,
    )
    for note in summary["notes"]:
        print(f"[haan]   {note}", flush=True)
    if kept:
        print(f"[haan] {len(kept)} parameter(s) had NO source and keep their own init:", flush=True)
        for name in kept:
            print(f"[haan]   {name}", flush=True)
    if summary["unused"]:
        print(
            f"[haan] {len(set(summary['unused']))} source tensor(s) unused "
            f"(expected: the opposite-side audio tables).",
            flush=True,
        )
