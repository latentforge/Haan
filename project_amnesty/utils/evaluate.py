"""evaluate.py -- standalone evaluation entry point (offline, run against a checkpoint).

Core separation: **"turn-taking mechanism transfer" and
"conversational content consistency transfer" MUST be measured separately** -- if the
two are not split apart, we cannot even decide success vs failure. This file enforces
that separation in the code structure itself: `eval_content` (content) and
`eval_mechanism` (mechanism) return different metrics and are NEVER summed into a
single scalar.

  Checkpoint A (after Phase 1): causal probing of whether any turn-taking
                                representation activates on a Korean audio prefix
                                (an early signal).
  Checkpoint B (after Phase 2): whether English held-out multi-turn performance
                                collapses after Korean is injected (interference),
                                plus re-measuring the Korean-prefix probe
                                (has it risen vs Phase1?).
  Checkpoint C (after Phase 3): first emergence judgment for Korean multi-turn --
                                mechanism vs content, kept separate.

Metrics:
  - **Content accuracy**: WER/CER from ASR re-transcription (guards against
    word-salad -- ARCHITECTURE §4.4). MOS/naturalness alone cannot catch "natural
    sounding but wrong" speech. Results are attributed against the Mimi round-trip
    ceiling (ARCHITECTURE §4.3, computed inline here) to separate the
    "codec bottleneck" from the "LM transfer bottleneck" (RISKS §6).
  - **Mechanism**: timing / overlap / barge-in pattern analysis (Full-Duplex-Bench
    family). Kept separate from content.
  - **Probing**: role-vector separability, user-channel
    embedding shift from its initialization (L2 / cosine), and the
    turn-taking causal probe.

Only the probe split (held out by prepare) is used; it is NEVER exposed to
training.

Run: `python -m project_amnesty.utils.evaluate --ckpt <path> --checkpoint-tag C`

Note: this is a standalone entry point. `project_amnesty/datasets/` is real
and already used here (eval_mechanism's probe loader imports it), and
`project_amnesty/models/haan/` is a real transformers Moshi subclass -- forward and generate
are inherited and work. Heavy imports (torch, transformers, ASR libraries) are deferred: they
live INSIDE each function body, not at module top, to keep entry-point load cost minimal.
"""

from __future__ import annotations

import argparse

# References used here (imported lazily INSIDE each eval body, off the entry-point load path):
#   from project_amnesty.models.haan import HaanForConditionalGeneration          (real)
#   from project_amnesty.datasets.loader import build_dataloader   (real; split="probe" holdout)
# datasets/ is fully implemented, so the probe loader runs for real; the model's forward is the
# only piece still TODO. What lives in eval_content / eval_mechanism / probe_representations is
# genuine *implementation* (generation loop, external ASR, timing/overlap, probing) that belongs
# in evaluate.py -- not something models/ or datasets/ can supply.
# NOTE: the standalone Mimi round-trip tool was removed; the §4.3 codec ceiling used for WER
# attribution (RISKS §6) is now computed inline in eval_content (Mimi encode->decode->ASR on the
# same probe audio), not via a separate tool.


# Checkpoint tag -> the bundle of judgments that must be reached at that stage
# (a human-readable documentation constant). The real gating logic does not exist
# yet (NotImplementedError).
CHECKPOINT_JUDGMENTS: dict[str, tuple[str, ...]] = {
    # After Phase 1 -- we only look at "early signals" (not English dialogue quality;
    # modality stability + probing).
    "A": (
        "Does the Korean-prefix turn-taking causal probe show an activation signal "
        "above baseline?",
        "Do the audio tokens avoid collapse and generate a plausible sequence "
        "(modality stability)?",
    ),
    # After Phase 2 -- we look at "interference" and "rise" at the same time.
    "B": (
        "Does English held-out multi-turn performance avoid a sharp drop after Korean "
        "injection (interference, RISKS §2)?",
        "Has the Korean-prefix probe risen vs Phase1 (A) (transfer-in-progress signal)?",
        "Do role-vector separability and user-embedding shift show no signs of collapse "
        "(RISKS §3, ARCH §3.5)?",
    ),
    # After Phase 3 -- "first emergence judgment". MUST keep mechanism vs content
    # separate.
    "C": (
        "Mechanism transfer: are timing / overlap / barge-in patterns reproduced in Korean?",
        "Content transfer: is ASR WER/CER within an acceptable range vs the Mimi "
        "round-trip ceiling (not word-salad)?",
        "For a partial failure, can we attribute where it broke, separately "
        "(RISKS §4 · §8 negative result)?",
    ),
}


# ---------------------------------------------------------------------------
# Real, model-free helper functions (used by eval_content).
# ---------------------------------------------------------------------------

def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein (edit) distance between two token sequences.

    Standard dynamic-programming edit distance with unit cost for insertion,
    deletion, and substitution. Operates on any list of hashable tokens, so it
    serves both WER (word tokens) and CER (character tokens). Uses two rolling
    rows for O(min(len)) memory.

    Args:
        ref: reference token sequence.
        hyp: hypothesis token sequence.

    Returns:
        Minimum number of single-token edits to turn ``ref`` into ``hyp``.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    # Keep the shorter sequence on the inner axis to minimize memory.
    if m < n:
        ref, hyp = hyp, ref
        n, m = m, n
    previous = list(range(m + 1))
    current = [0] * (m + 1)
    for i in range(1, n + 1):
        current[0] = i
        ref_tok = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_tok == hyp[j - 1] else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost,  # substitution / match
            )
        previous, current = current, previous
    return previous[m]


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate = edit distance over whitespace-split words / #reference words.

    Content-accuracy metric backing `eval_content` (ARCHITECTURE §4.4). A word is a
    maximal whitespace-delimited token.

    Korean spacing note: Korean word spacing (word segmentation) is not deterministic --
    the same utterance can be spaced differently while meaning the same thing, so WER
    is inflated by spacing disagreements and is a noisy content metric for Korean.
    Prefer `cer` for Korean content accuracy; WER is still reported for continuity
    with the English held-out evaluation (RISKS §2) where word boundaries are stable.

    Args:
        ref: reference (ground-truth) transcript.
        hyp: hypothesis (ASR-of-generation) transcript.

    Returns:
        WER as a float. Returns 0.0 when both are empty; returns 1.0 when the
        reference is empty but the hypothesis is not (every hypothesis word is an
        insertion error, normalized to the hypothesis length).
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = _levenshtein(ref_words, hyp_words)
    return distance / len(ref_words)


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate = edit distance over characters / #reference characters.

    Content-accuracy metric backing `eval_content` (ARCHITECTURE §4.4). Whitespace is
    stripped from both sides before comparison so that spacing disagreements do not
    count as errors.

    Korean spacing note: because Korean spacing is inconsistent (see `wer`), CER on
    despaced strings is the more robust content-accuracy metric for Korean -- it
    compares the actual syllable/character stream and is insensitive to how words are
    split. This is why CER is the primary content metric for the Korean emergence
    judgment (Checkpoint C).

    Args:
        ref: reference (ground-truth) transcript.
        hyp: hypothesis (ASR-of-generation) transcript.

    Returns:
        CER as a float. Returns 0.0 when both (despaced) are empty; returns 1.0 when
        the reference is empty but the hypothesis is not.
    """
    ref_chars = list("".join(ref.split()))
    hyp_chars = list("".join(hyp.split()))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    distance = _levenshtein(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def _corpus_error_rate(
    pairs: list[tuple[str, str]], level: str = "char"
) -> float:
    """Micro-averaged corpus error rate over (reference, hypothesis) pairs.

    Aggregates edit distance and reference length across all utterances before
    dividing (micro-average), which is the standard way to report corpus WER/CER --
    it weights each utterance by its length rather than averaging per-utterance rates.

    Args:
        pairs: list of (reference, hypothesis) transcript pairs.
        level: "word" for WER-style tokenization, "char" for CER-style.

    Returns:
        Micro-averaged error rate. Returns 0.0 for an empty corpus (no pairs at all), and `nan`
        when there are pairs but every reference is empty -- that is a broken corpus, not a
        perfect transcription. `total_length` counts reference tokens only, so without this the
        function returns 0.0 (a flawless score) for a corpus whose transcripts all failed to load,
        while `wer("", "hallucinated words")` on the same pair returns 1.0.
    """
    total_distance = 0
    total_length = 0
    for ref, hyp in pairs:
        if level == "word":
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
        else:
            ref_tokens = list("".join(ref.split()))
            hyp_tokens = list("".join(hyp.split()))
        total_distance += _levenshtein(ref_tokens, hyp_tokens)
        total_length += len(ref_tokens)
    if total_length == 0:
        # No pairs at all -> nothing was measured, 0.0 is honest. Pairs present but no reference
        # tokens -> the references are missing, and any number here would be a lie.
        return 0.0 if not pairs else float("nan")
    return total_distance / total_length


# ---------------------------------------------------------------------------
# Model-dependent evaluations (datasets/ is real; these raise NotImplementedError until the models/ forward lands).
# ---------------------------------------------------------------------------

def eval_content(ckpt: str, split: str = "probe") -> dict:
    """Generation -> ASR re-transcription -> WER/CER (content accuracy). -- ARCH §4.4, RISKS §4

    Measures "conversational content consistency transfer". Acoustic RVQ (§4.1) can
    produce natural-sounding speech texture even when the semantics are wrong (word
    salad), so we do NOT rely on MOS/naturalness; we measure real utterance-content
    accuracy with **ASR-based WER/CER** (§4.4). The pure WER/CER math is implemented
    in `wer`/`cer`/`_corpus_error_rate` above; the model-dependent part is generating
    the audio and running ASR.

    Attribution: the observed WER/CER is compared against the Mimi round-trip WER
    ceiling (§4.3, computed inline: encode->decode->ASR) to separate "the portion
    the codec cannot represent at all" from "the portion of LM transfer failure"
    (RISKS §6). Pure LM transfer performance is reported around
    (observed WER - codec ceiling).

    Args:
        ckpt: path to the checkpoint under evaluation.
        split: data split. **Only "probe" is allowed** -- the probe split is a holdout
            set aside by prepare and never exposed to training (RISKS §1, CURRICULUM §0).

    Returns:
        dict -- minimum keys (proposed):
            {"wer": float, "cer": float,
             "mimi_ceiling_wer": float,       # §4.3 codec ceiling
             "lm_attributed_wer": float,      # observed - ceiling (LM transfer share, RISKS §6)
             "n_utterances": int, "split": str}
        Mechanism metrics are never mixed in (eval_mechanism owns those, RISKS §4).
    """
    # TODO(generation): once models/ exists, load HaanForConditionalGeneration and
    #   generate self audio from probe prefixes in simulation mode (q16, batch2,
    #   ARCH §5.0.3·§5.4).
    # TODO(ASR): generated audio -> Mimi decode -> external (Korean) ASR re-transcription
    #   -> WER/CER against the reference text, via wer()/cer()/_corpus_error_rate().
    # TODO(attribution): compute the §4.3 codec ceiling inline (Mimi encode->decode->ASR on the
    #   same probe audio), then derive lm_attributed_wer = observed_wer - ceiling.
    raise NotImplementedError(
        "TODO: generation -> Mimi decode -> ASR re-transcription -> WER/CER, "
        "attributed against the Mimi round-trip ceiling (§4.3·§4.4·§6)"
    )


# ---------------------------------------------------------------------------
# Model / checkpoint access helpers -- the DOCUMENTED model I/O contract that
# eval_mechanism and probe_representations are implemented against.
#
# `project_amnesty/datasets/` and `project_amnesty/models/haan/` are both real; the model's
# forward / generate are inherited from transformers' Moshi and run. All heavy imports
# (torch / numpy / model / datasets) are kept LOCAL to each body so this entry-point module
# still imports with no heavy dependencies installed.
#
# CAVEAT -- the forward signature below is Haan's INTENDED contract, not Moshi's inherited one.
# `MoshiForConditionalGeneration.forward` takes `user_audio_codes` / `assistant_audio_codes`
# (not a single `audio_codes`) and only populates `audio_logits` when `text_labels` AND
# `audio_labels` are both given, shaped `(batch * sequence_length, 2K, C)` rather than
# `(B, 2, K, T, C)`. models/haan does not override `forward`, so the eval bodies below that
# call the model directly need that adapter to land first. Flagged rather than silently
# reshaped here: guessing the axis order would corrupt every metric downstream.
#
# Attribute names are kept IDENTICAL to what utils/train.py's diagnostics read, so
# training and evaluation see the same model. They are the real names on
# `project_amnesty.models.haan.HaanForConditionalGeneration`:
#   model.embed_tokens                       nn.ModuleList of K shared audio input embedding
#                                            tables; the init is copied from Moshi's user-side
#                                            tables
#   model.role_embedding                     RoleEmbedding -- Temporal self/user role signal,
#                                            parameter `role_scale` (role_mode="scale", default)
#                                            or `role_emb` (role_mode="additive"), (num_roles, D)
#   model.depth_decoder.model.role_embedding Depth-side RoleEmbedding, (num_roles, D_depth)
#   model(input_ids=, audio_codes=[, role_ids=]) -> outputs with
#       outputs.audio_logits (B, 2, K, T, C)  # axis 1 = [self, user] streams
#       outputs.text_logits  (B, T, V)
#   model.generate(prefix, *, mode="q16", max_new_frames=...) -> obj/dict with
#       .codes (B, 2, K, T_gen)               # ALSO generates the user stream
# ---------------------------------------------------------------------------

# Mimi frame rate: 1 text token per frame, 12.5 frames per second.
FRAME_RATE_HZ = 12.5

# Baseline self/user row cosine of the original Moshi tables (tools/inspect_moshi_weights.py):
# semantic (codebook 0) ~= 0.501, acoustic (codebook 1) ~= 0.751. The PRIMARY check is the
# semantic level -- role differentiation concentrates there.
ROLE_COS_BASELINE_SEMANTIC = 0.501
ROLE_COS_BASELINE_ACOUSTIC = 0.751

# Not a documented value. The docs' only 0.6 s figure is a TEXT delay (text delay = +-0.6 s
# pre-training -> 0 after), not a backchannel-duration cutoff. This threshold is
# round(0.6 * FRAME_RATE_HZ) = round(7.5) = 8 -- a rounding of a 0.6 s estimate that needs
# empirical confirmation.
# _bargein_backchannel classifies with a STRICT `<` (`length < BACKCHANNEL_MAX_FRAMES`),
# so the effective boundary is "a self run of <= 7 frames (~0.56 s) fully inside a user
# turn = backchannel; 8+ frames = barge-in".
BACKCHANNEL_MAX_FRAMES = 8


def _load_model(ckpt: str):
    """Construct the Haan model and load `ckpt` weights (model I/O contract above).

    Delegates to `project_amnesty.models.haan.HaanForConditionalGeneration`, whose
    `from_pretrained` is transformers' own -- `ckpt` is a trained Haan checkpoint directory, NOT
    a Moshi warm-start source (that is `warm_start_from_moshi`, called by train.build_model).
    Heavy imports are local so the entry point stays importable without torch/transformers.
    """
    from project_amnesty.models.haan import HaanForConditionalGeneration  # noqa: PLC0415 (lazy)

    model = HaanForConditionalGeneration.from_pretrained(ckpt)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _require_model_attr(model, name: str, doc: str):
    """getattr with a clear, contract-pointing error (so a real models/ 'just works')."""
    attr = getattr(model, name, None)
    if attr is None:
        raise AttributeError(
            f"model loaded from checkpoint is missing required attribute '{name}' ({doc}); the "
            f"models/ implementation must expose it under this exact name (kept identical to the "
            f"names utils/train.py's diagnostics read)."
        )
    return attr


def _as_ndarray(tensor):
    """Return a detached float64 numpy view of a torch tensor / parameter / ndarray (on CPU)."""
    import numpy as np  # noqa: PLC0415

    if hasattr(tensor, "detach"):  # torch.Tensor / nn.Parameter
        tensor = tensor.detach()
        if hasattr(tensor, "to"):
            tensor = tensor.to("cpu")
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        return np.asarray(tensor.numpy(), dtype=np.float64)
    return np.asarray(tensor, dtype=np.float64)


def _weight_2d(table):
    """Extract a 2D (C, D) weight matrix from an nn.Embedding / Parameter / tensor / ndarray."""
    weight = getattr(table, "weight", table)  # nn.Embedding -> .weight
    return _as_ndarray(weight)


def _audio_emb_tables(audio_emb) -> "list":
    """Normalize `model.embed_tokens` into a list of per-codebook (C, D) numpy tables.

    Haan builds this as an `nn.ModuleList` of K `nn.Embedding`s (modeling_haan.py); the
    (K, C, D) tensor and bare (C, D) forms are also accepted so a checkpoint that stacked them
    still reads.
    """
    # A container (list / tuple / nn.ModuleList) exposes __len__ but -- unlike a tensor/ndarray --
    # has no .dim / .ndim; a single nn.Embedding has neither __len__ nor .dim.
    if hasattr(audio_emb, "__len__") and not hasattr(audio_emb, "dim") and not hasattr(audio_emb, "ndim"):
        return [_weight_2d(t) for t in audio_emb]
    arr = _weight_2d(audio_emb)
    if arr.ndim == 3:      # (K, C, D)
        return [arr[k] for k in range(arr.shape[0])]
    if arr.ndim == 2:      # single shared (C, D) table
        return [arr]
    raise ValueError(f"model.embed_tokens has unsupported shape {getattr(arr, 'shape', None)}")


def _audio_embeddings(model):
    """`model.embed_tokens` -- Haan's K shared audio input tables.

    Deliberately NOT `model.model.embed_tokens`, which is the backbone's text embedding.
    """
    return _require_model_attr(model, "embed_tokens", "K shared audio input embeddings, ARCH §5.4.2")


def _role_matrix(role_embedding, where: str):
    """The `(num_roles, D)` parameter out of a `models.haan.RoleEmbedding`.

    Which attribute holds it depends on `role_mode`: `role_scale` for "scale" (the default) and
    `role_emb` for "additive". Returns `(parameter, role_mode)` so callers apply the matching
    formula rather than assuming one (see `_role_row_cosine`).
    """
    role_mode = getattr(role_embedding, "role_mode", None)
    for attr, mode in (("role_scale", "scale"), ("role_emb", "additive")):
        parameter = getattr(role_embedding, attr, None)
        if parameter is not None:
            return parameter, (role_mode or mode)
    raise AttributeError(
        f"{where} exposes neither `role_scale` nor `role_emb`; models/haan's RoleEmbedding must "
        f"hold its (num_roles, D) parameter under one of those names (role_mode={role_mode!r})."
    )


def _temporal_role(model):
    """Temporal-side role signal -> (parameter, role_mode)."""
    role_embedding = _require_model_attr(model, "role_embedding", "Temporal self/user Role Token, ARCH §3.3")
    return _role_matrix(role_embedding, "model.role_embedding")


def _depth_role(model):
    """Depth-side role signal -> (parameter, role_mode), or (None, None) if absent."""
    depth = getattr(model, "depth_decoder", None)
    role_embedding = getattr(getattr(depth, "model", None), "role_embedding", None)
    if role_embedding is None:
        return None, None
    return _role_matrix(role_embedding, "model.depth_decoder.model.role_embedding")


# -- numeric primitives (numpy; inspect_moshi_weights.py style) ---------------

def _row_cosine(a, b, eps: float = 1e-8) -> float:
    """Mean over rows of cosine(a_i, b_i) for two (N, D) matrices."""
    import numpy as np  # noqa: PLC0415

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return float((an * bn).sum(axis=1).mean())


def _row_l2(a, b) -> float:
    """Mean over rows of ||a_i - b_i||_2 for two (N, D) matrices."""
    import numpy as np  # noqa: PLC0415

    return float(np.linalg.norm(a - b, axis=1).mean())


def _vec_cosine(a, b, eps: float = 1e-8) -> float:
    """Cosine similarity of two flattened vectors."""
    import numpy as np  # noqa: PLC0415

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a.dot(b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def _role_row_cosine(table, role0, role1, role_mode: str = "scale") -> float:
    """Self/user row cosine on the SHARED audio table under the Role Token.

    In Haan self and user index the SAME table and differ only by the role signal, so the
    self-vs-user row cosine becomes cos(f(table_i, role0), f(table_i, role1)) where `f` is however
    `models.haan.RoleEmbedding` applies the role. That is NOT one fixed formula -- the two modes
    apply the role differently and mixing them up silently reports a number for the wrong model:

      role_mode="scale"     (default)  f = table_i * role      elementwise per-role gain
      role_mode="additive"  (ablation) f = table_i + role      additive role, applied as written

    The result is directly comparable to the Moshi baseline (semantic ~0.50, acoustic ~0.75).

    Note the additive mode is degenerate at the Temporal input (RoleEmbedding's docstring derives
    why: adding a constant to a symmetric sum leaves it invariant under a stream swap); it is kept
    only so the ablation can measure the gap.
    """
    import numpy as np  # noqa: PLC0415

    r0 = np.asarray(role0, dtype=np.float64).ravel()
    r1 = np.asarray(role1, dtype=np.float64).ravel()
    if role_mode == "scale":
        return _row_cosine(table * r0[None, :], table * r1[None, :])
    if role_mode == "additive":
        return _row_cosine(table + r0[None, :], table + r1[None, :])
    raise ValueError(f"`role_mode={role_mode!r}` must be 'scale' or 'additive' (models.haan.ROLE_MODES).")


# -- frame/code-level turn-taking primitives (eval_mechanism) -----------------

def _silence_codes(silence_bank) -> set:
    """Semantic (level-0) silence codes, read from the measured silence bank.

    0 is a valid Mimi code, so silence must be the measured bank (tools/derive_silence_codes.py),
    never zeros. Takes `DataConfig.tokens.silence_bank` -- the `(K, P)` tuple datasets/config.py
    loads with a repo-root fallback and a `mimi_ckpt_id` cross-check -- and returns codebook 0's
    row (semantic level-0) as a set of codes.

    The silence bank is required; there is no modal fallback. Inferring silence as "the single most
    frequent code across the generated streams" is actively wrong: if the model collapses to one
    non-silence code, that code becomes the mode, every frame is marked silence, the speech mask
    goes all-False, and every timing metric reads zero activity -- laundering the exact
    silence/speech collapse into a clean-looking result.
    """
    if not silence_bank:
        raise ValueError(
            "silence_bank is required for the semantic silence mask (ARCH §4.1 / configs/tokens.yaml "
            "silence_bank_path). It comes from DataConfig.tokens.silence_bank -- there is no modal "
            "fallback, because a collapsed generation would poison it silently (RISKS §7.4)."
        )
    semantic_row = silence_bank[0]  # codebook 0 = semantic level-0
    return {int(c) for c in semantic_row}


def _speech_mask_from_semantic(semantic_codes, silence_codes) -> "np.ndarray":
    """Per-frame speech mask from Mimi semantic (level-0) codes: True where NOT a silence code."""
    import numpy as np  # noqa: PLC0415

    codes = np.asarray(semantic_codes).astype(int).ravel()
    silent = np.isin(codes, np.asarray(list(silence_codes), dtype=int))
    return ~silent


def _pad_token_ids(tokens) -> set:
    """Plain stream-PAD id(s) from the loaded token config: between-word silence
    on the text stream (~65% of tokens). A non-PAD frame is a word or the EPAD onset trigger (both
    = the agent turn is active).

    Takes the already-loaded `DataConfig.tokens` (`text_pad_id` / `text_epad_id`) rather than
    re-reading a file. A lookup miss would return an empty set silently, so `_self_speech_mask`
    would drop its text-stream term and every timing metric would lose the EPAD onset signal (EPAD
    is the "a word is about to start" trigger) while still reporting a number.

    Only the plain PAD is returned. EPAD is deliberately NOT included: it is the speech-onset
    signal, so a frame carrying EPAD counts as the agent being active.
    """
    pad = getattr(tokens, "text_pad_id", None)
    return set() if pad is None else {int(pad)}


def _self_speech_mask(self_text, self_sem, silence_codes, pad_ids) -> "np.ndarray":
    """Self-stream per-frame speech presence.

    Continuous presence comes from the semantic-code silence bank (audio). When the generator
    also returns the self text stream and PAD ids are known, UNION the text utterance signal --
    a non-PAD frame is a word or the EPAD onset trigger -- which pins onsets to the
    EPAD trigger; the audio term fills the within-utterance PAD gaps (text is ~65% PAD even mid-
    speech). Offsets fall on PAD/silence transitions.
    """
    import numpy as np  # noqa: PLC0415

    audio_speech = _speech_mask_from_semantic(self_sem, silence_codes)
    if self_text is None or not pad_ids:
        return audio_speech
    text_speaking = ~np.isin(np.asarray(self_text).astype(int), np.asarray(list(pad_ids), dtype=int))
    n = min(audio_speech.shape[0], text_speaking.shape[0])
    return audio_speech[:n] | text_speaking[:n]


def _runs(mask):
    """List of (start, end) speech runs (contiguous True regions) in a boolean frame mask.

    `end` is EXCLUSIVE: it is the first frame *after* the run stops. So when the next turn's onset
    equals a run's `end`, the between-turn gap is exactly 0 -- a zero-silence adjacent turn switch
    (the fastest possible clean response), NOT an overlap. The file-wide sign convention is
    therefore: gap >= 0 == clean turn switch (0 == zero-silence switch); gap < 0 == overlap
    (responder onset fell strictly inside the partner's run). See _response_gaps / _timing_stats.
    """
    import numpy as np  # noqa: PLC0415

    m = np.asarray(mask).astype(bool).astype(int)
    if m.size == 0:
        return []
    diff = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _response_gaps(partner_runs, responder_runs):
    """Signed frame gaps from each partner-utterance to the responder's next onset.

    For each partner turn, take the first responder onset in [partner_start, next_partner_start):
    gap = responder_onset - partner_offset. `partner_offset` is `_runs`' EXCLUSIVE end (first frame
    after the partner stops), so a non-negative gap (>= 0) = a clean turn switch -- with gap == 0
    the zero-silence adjacent switch (fastest clean response); a negative gap (< 0) = the responder
    started strictly before the partner finished (overlap / barge-in). Callers split the two. One
    response is attributed per partner turn.
    """
    import numpy as np  # noqa: PLC0415

    responder_starts = np.asarray([r[0] for r in responder_runs], dtype=int)
    partner_starts = [p[0] for p in partner_runs]
    gaps = []
    for i, (ps, pe) in enumerate(partner_runs):
        next_ps = partner_starts[i + 1] if i + 1 < len(partner_starts) else np.inf
        cand = responder_starts[(responder_starts >= ps) & (responder_starts < next_ps)]
        if cand.size == 0:
            continue
        gaps.append(int(cand.min()) - pe)  # >=0 clean gap (0 == zero-silence switch), <0 overlap
    return gaps


def _bargein_backchannel(self_runs, user_speech, backchannel_max):
    """Classify self runs that begin during user speech into barge-ins vs backchannels.

    - barge-in: self takes the floor while the user is speaking (onset during user speech and the
      self run is not merely a short burst) -> interruption (full-duplex behavior).
    - backchannel: a short self burst fully inside a user turn (user keeps the floor before and
      after) -> "uh-huh" (Full-Duplex-Bench family).
    Returns (n_bargein, n_backchannel, bargein_frame_mask). The mask marks the frames where a
    barge-in self run overlaps the ongoing user speech -- the local speaker-similarity target.
    """
    import numpy as np  # noqa: PLC0415

    user_speech = np.asarray(user_speech).astype(bool)
    total = int(user_speech.shape[0])
    bargein_mask = np.zeros(total, dtype=bool)
    n_barge = 0
    n_back = 0
    for (ss, se) in self_runs:
        if ss >= total or not user_speech[ss]:
            continue  # self did not start while the user was speaking
        length = se - ss
        user_after = se < total and bool(user_speech[min(se, total - 1)])
        if length < backchannel_max and user_after:
            n_back += 1
        else:
            n_barge += 1
            overlap = np.zeros(total, dtype=bool)
            overlap[ss:min(se, total)] = True
            bargein_mask |= overlap & user_speech
    return n_barge, n_back, bargein_mask


def _timing_stats(gaps_s) -> dict:
    """Turn-transition timing statistics (seconds) for `turn_switch_timing`."""
    import numpy as np  # noqa: PLC0415

    if not gaps_s:
        return {
            "n_transitions": 0, "mean_gap_s": float("nan"), "median_gap_s": float("nan"),
            "std_gap_s": float("nan"), "min_gap_s": float("nan"), "max_gap_s": float("nan"),
            "n_gap_transitions": 0, "n_overlap_transitions": 0,
        }
    g = np.asarray(gaps_s, dtype=np.float64)
    return {
        "n_transitions": int(g.size),
        "mean_gap_s": float(g.mean()),
        "median_gap_s": float(np.median(g)),
        "std_gap_s": float(g.std()),
        "min_gap_s": float(g.min()),
        "max_gap_s": float(g.max()),
        # g==0 reclassified from overlap to clean switch; makes the whole file consistent
        # (overlap_rate/bargein already exclude it). `_runs` end is exclusive, so gap==0 is a
        # zero-silence adjacent turn switch, not an overlap.
        "n_gap_transitions": int((g >= 0).sum()),     # clean turn switches (>=0: silence gap or 0-gap switch)
        "n_overlap_transitions": int((g < 0).sum()),  # responder started strictly before partner ended
    }


def _simulate_dialogue(model, prefix, max_new_frames):
    """q16 simulation generate: produce BOTH self and user streams for offline eval.

    **Self-play, not replay.** The Moshi paper explains why the user stream is modeled
    as an output at all: *"prediction for the audio coming from the user is actually ignored, as
    the actual user audio is used instead. However, modeling the user stream as output allows
    generating simulated dialogues, which is necessary for offline evaluation."* Generating the
    simulated dialogue IS the purpose of the user head, so the recorded user stream is supplied
    only as far as the prompt and the model invents the rest. Feeding the recorded user across the
    horizon would not merely change the measurement -- `generate` returns the predicted user tail
    only for frames the caller left open, so a horizon-covering user stream yields no user stream
    at all.

    The prompt is cut at the Zone A + Zone B boundary (static ChatML prefix -> voice
    prompt -> Zone C dialogue), so exactly the dialogue is simulated and the conditioning prefix is
    not.

    Returns three per-batch-element lists (self_semantic, user_semantic, self_text), each entry a
    frame-aligned 1D int array. Frame / code level only -- NO waveform, NO Mimi decode (keeps this
    off the content path).
    """
    if not isinstance(prefix, dict):
        raise TypeError(f"expected the collator batch dict; got {type(prefix).__name__}.")

    # Zone A + Zone B is the conditioning prefix; Zone C is what gets simulated. The zone lengths
    # are per row, and `generate` needs one rectangular prompt, so the batch-wide minimum is used:
    # cutting shorter than a row's Zone B only means a little of its voice prompt is regenerated,
    # whereas cutting longer would consume Zone C frames that are supposed to be simulated.
    zone_a = prefix["zone_a_frames"]
    zone_b = prefix.get("zone_b_frames")
    prompt = zone_a if zone_b is None else zone_a + zone_b
    prompt_frames = max(int(prompt.min()), 1)

    # The rollout -- `generate` plus the END-ANCHORED self/user alignment -- is SHARED with the
    # on-policy scheduled sampler: `train.rollout_with_contract` owns it, so the two paths cannot
    # drift apart by a frame. Lazy import: evaluate is the higher layer.
    from project_amnesty.utils.train import rollout_with_contract  # noqa: PLC0415

    codes_t, gen = rollout_with_contract(model, prefix, max_new_frames, prompt_frames)
    codes = _as_ndarray(codes_t).astype(int)                # (B, 2, K, Tu), axis 1 = [self, user]
    batch = codes.shape[0]
    n_gen = codes.shape[-1]
    self_list = [codes[b, 0, 0, :] for b in range(batch)]   # semantic level-0, self stream
    user_list = [codes[b, 1, 0, :] for b in range(batch)]   # semantic level-0, user stream

    # The generated text stream lands on `sequences` (there is no `text` field); reading a missing
    # field would return None, leaving `self_text=None` and silently dropping the text term from
    # `_self_speech_mask`.
    #
    # Slice with the SAME end-anchored window as `self_c` (`-(n_gen + 1):-1`), NOT `-n_gen:`. Text
    # delay is 0 post-training, so text frame t and self-audio frame t are the same absolute frame,
    # and `_self_speech_mask` ORs the two elementwise (see its `[:n]` union) -- index i of both
    # terms MUST name the same frame. `-n_gen:` puts the text one frame LATER than `self_c` at every
    # index, so the text onset (which PINS the self-speech onset) lands one frame early: a zero-gap
    # clean switch (agent onset == user offset) reads as gap -1 and gets miscounted as an overlap.
    # Still end-anchored, so an early EOS cannot bleed prompt tokens into the dialogue window.
    text_arr = _as_ndarray(gen.sequences).astype(int).reshape(batch, -1)[:, -(n_gen + 1):-1]
    text_list = [text_arr[b] for b in range(batch)]
    return self_list, user_list, text_list


# -- representation-level primitives (probe_representations) -------------------

def _depth_role_divergence(model) -> dict:
    """Batch-2 self/user forward divergence probe for the shared Depth.

    A single forward already emits both streams: outputs.audio_logits is (B, 2, K, T, C) with
    axis 1 = [self, user]. We forward a minimal synthetic frame-level prefix
    (identical content for both streams, so only the additive role embedding differs) and measure
    how far the self-role output diverges from the user-role output. If the additive Depth role
    embedding is diluted under other loss pressure the two collapse to the same output (role
    ignored) -> the fallback is to promote the shared projection to two role-specific
    ones. Also reports the static cosine / L2 of the Depth role embedding when present.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # The model speaks Moshi's signature, not this contract, so the forward goes through the
    # shared adapter. Calling `model(input_ids=, audio_codes=)` directly
    # does not merely fail -- passing labels to make `audio_logits` appear returns it shaped
    # `(B * T, 2K, C)`, whose axis 1 is 2K, so a `shape[1] >= 2` check PASSES and the probe then
    # compares assistant codebook 0 against assistant codebook 1 while reporting the number as
    # self-vs-user role divergence.
    from project_amnesty.utils.train import forward_with_contract  # noqa: PLC0415 (lazy)

    n_codebooks = len(_audio_emb_tables(_audio_embeddings(model)))
    n_frames = 8
    # (1, 2, K, T) -- the ROLE AXIS is required. Identical content in both streams, so the only
    # thing that can separate the two outputs is the role signal itself, which is the point.
    audio_codes = torch.zeros((1, 2, n_codebooks, n_frames), dtype=torch.long)
    input_ids = torch.zeros((1, n_frames), dtype=torch.long)
    with torch.no_grad():
        outputs = forward_with_contract(model, {"input_ids": input_ids, "audio_codes": audio_codes})
    logits = _as_ndarray(outputs["audio_logits"])
    # Exact, not `>= 2`: the loose check is what let the `(B * T, 2K, C)` layout through.
    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError(
            f"audio_logits must be (B, 2, K, T, C) with the self/user role axis at dim 1; "
            f"got {logits.shape}"
        )
    self_l = logits[:, 0, ...].ravel()
    user_l = logits[:, 1, ...].ravel()
    diff = float(np.linalg.norm(self_l - user_l))
    denom = float(np.linalg.norm(self_l) + np.linalg.norm(user_l) + 1e-8)
    batch_output_divergence = diff / denom  # matches train.py run_diagnostics' collapse metric

    depth_role_emb, _ = _depth_role(model)
    if depth_role_emb is not None:
        dre = _as_ndarray(depth_role_emb)
        depth_role_cos = _vec_cosine(dre[0], dre[1]) if dre.shape[0] >= 2 else float("nan")
        depth_role_l2 = float(np.linalg.norm(dre[0].ravel() - dre[1].ravel())) if dre.shape[0] >= 2 else float("nan")
    else:
        depth_role_cos = None
        depth_role_l2 = None

    # The Depth role starts at the identity BY DESIGN (`RoleEmbedding`: additive mode initialises
    # to zeros so a Moshi warm-start reproduces the teacher exactly at step 0). So a freshly
    # warm-started checkpoint has divergence 0 and would trip the collapse alarm at Checkpoint A
    # -- a false positive on the very checkpoint the alarm is first read. `depth_role_l2 == 0`
    # separates "never trained" from "trained and collapsed"; `collapsed` is meaningless while it
    # holds, and the caller must read this flag before the alarm.
    role_is_at_init = depth_role_l2 is not None and depth_role_l2 == 0.0

    return {
        "batch_output_divergence": batch_output_divergence,   # normalized L2 self vs user
        "self_user_output_cos": _vec_cosine(self_l, user_l),  # cosine of the two stream outputs
        "collapsed": bool(batch_output_divergence < 1e-3),    # role-collapse alarm -> split projection
        "role_emb_is_init": bool(role_is_at_init),            # if True, `collapsed` says nothing
        "depth_role_cos": depth_role_cos,                     # static cos of the two Depth role rows
        "depth_role_l2": depth_role_l2,
        "n_frames": n_frames,
    }


def _load_probe_set(probe_path: str):
    """Load the turn-taking causal-probe holdout (never trained on).

    Documented schema (whichever the prepare step wrote):
      * .jsonl -- one object per line: {"prefix": <frame-level prefix payload for model forward>,
                    "label": 0/1 (turn-taking event present), "lang": "ko" | "en"}
      * .npz   -- object array `prefixes` plus arrays `labels` (N,) and `langs` (N,)
    Returns (prefixes: list, labels: np.ndarray[int], langs: np.ndarray[str]).
    """
    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    if probe_path.endswith(".npz"):
        data = np.load(probe_path, allow_pickle=True)
        return list(data["prefixes"]), np.asarray(data["labels"]).astype(int), np.asarray(data["langs"]).astype(str)

    prefixes, labels, langs = [], [], []
    with open(probe_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefixes.append(obj["prefix"])
            labels.append(int(obj["label"]))
            langs.append(str(obj["lang"]))
    return prefixes, np.asarray(labels, dtype=int), np.asarray(langs, dtype=str)


def _forward_prefix(model, prefix):
    """Run one holdout prefix through the model forward (documented I/O contract).

    `output_hidden_states` is forced on: `outputs.hidden_states` is `None` unless it is requested,
    and the probe reads z_s off it. Without this every lookup in
    `_turntaking_features` resolves to `None` and the probe raises on the first prefix. A prefix
    payload that sets it itself wins, so a caller can still turn it off deliberately.
    """
    if isinstance(prefix, dict):
        return model(**{"output_hidden_states": True, **prefix})
    return model(prefix, output_hidden_states=True)


def _turntaking_features(model, prefixes):
    """Forward each holdout prefix and pull the turn-taking probe representation (one vector each).

    Uses the per-frame Temporal context vector z_s -- the representation the Depth turn-taking
    prediction reads -- exposed as `outputs.hidden_states` (last layer) or
    `outputs.z`, mean-pooled over the frame axis.

    `hidden_states[-1]` is preferred over `last_hidden_state` deliberately, even though they are
    equal on the label-free path: when a prefix carries both `text_labels` and `audio_labels`,
    Moshi reshapes `last_hidden_state` to `(B * T, 1, H)` while `hidden_states[-1]` stays
    `(B, T, H)`. The pooling below is shape-invariant, so that difference would not raise -- it
    would silently pool a different thing.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    feats = []
    with torch.no_grad():
        for prefix in prefixes:
            outputs = _forward_prefix(model, prefix)
            hidden = getattr(outputs, "hidden_states", None)
            if hidden is None:
                hidden = outputs.get("hidden_states") if isinstance(outputs, dict) else None
            if hidden is None:
                hidden = getattr(outputs, "z", None)
            if hidden is None:
                # Always populated, and equal to hidden_states[-1] on this path -- so a prefix
                # that declined the flag still probes rather than aborting the whole gate.
                hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise AttributeError(
                    "forward outputs expose neither `hidden_states` nor `z`; the turn-taking causal "
                    "probe (RISKS §1) needs the per-frame Temporal context vector z_s."
                )
            h = hidden[-1] if isinstance(hidden, (list, tuple)) else hidden
            hn = _as_ndarray(h)
            feats.append(hn.reshape(-1, hn.shape[-1]).mean(axis=0))
    return np.asarray(feats, dtype=np.float64)


def _fit_logreg(x, y, iters: int = 300, lr: float = 0.1, l2: float = 1e-3):
    """Tiny full-batch logistic-regression probe (numpy; no sklearn dependency).

    Features are standardized for stable optimization, then the standardization is folded back
    into (w, b) so the returned weights apply directly to raw features.
    """
    import numpy as np  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-8
    xs = (x - mu) / sd
    n, d = xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xs.dot(w) + b)))
        w -= lr * (xs.T.dot(p - y) / n + l2 * w)
        b -= lr * float((p - y).mean())
    w_raw = w / sd
    b_raw = b - float((w * mu / sd).sum())
    return w_raw, b_raw


def _accuracy(x, y, w, b) -> float:
    import numpy as np  # noqa: PLC0415

    pred = (np.asarray(x, dtype=np.float64).dot(w) + b >= 0.0).astype(int)
    return float((pred == np.asarray(y, dtype=int)).mean())


def _kfold_heldout_accuracy(x, y, n_splits: int = 5) -> float:
    """Out-of-sample accuracy of the tiny logreg probe via k-fold cross-validation.

    In-language TRAINING accuracy is vacuous for this probe: the feature is a mean-pooled
    hidden-dim vector (~1024 dims) and the probe holdout is small, so feature-dim >= sample-count
    and a linear probe fits the training split to ~1.0 whether or not any turn-taking signal
    exists. We therefore score the in-language probe strictly OUT-OF-SAMPLE. Returns NaN
    ("unreliable" -- never a passing 1.0) when there are too few samples to hold a fold out, when
    only one class is present, or when no fold could be scored; the caller reads NaN as "do not
    decide", not as a high score.
    """
    import numpy as np  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n = x.shape[0]
    if n < 4 or np.unique(y).size < 2:
        return float("nan")            # too small to hold out, or single-class -> unreliable
    k = int(min(n_splits, n))
    # Class-sorted round-robin fold assignment: each fold sees both classes where possible
    # (approximate stratification) and the split does not depend on caller ordering.
    order = np.argsort(y, kind="stable")
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % k
    correct = 0
    counted = 0
    for f in range(k):
        test_mask = folds == f
        train_mask = ~test_mask
        if int(train_mask.sum()) < 2 or np.unique(y[train_mask]).size < 2:
            continue                   # the fit fold must contain both classes
        w, b = _fit_logreg(x[train_mask], y[train_mask])
        pred = (x[test_mask].dot(w) + b >= 0.0).astype(int)
        correct += int((pred == y[test_mask]).sum())
        counted += int(test_mask.sum())
    if counted == 0:
        return float("nan")
    return correct / counted


def _causal_probe_cross(feats, labels, langs):
    """Korean/English turn-taking causal-probe cross.

    Fit a linear probe on one language's representations and test it on the OTHER language (both
    directions). A genuinely language-independent turn-taking circuit transfers across the cross;
    the "if Korean, one turn then stop" language shortcut does NOT. Returns
    (mean cross-lingual probe accuracy, is_shortcut). When no direction can be evaluated (too few
    items) returns (NaN, None).

    `is_shortcut` fires ONLY for a genuine language shortcut = "the in-language probe genuinely
    works AND it does NOT transfer across languages", judged by two guarded conditions:
      (i)  the in-language probe genuinely works, measured OUT-OF-SAMPLE (k-fold CV), because
           in-language TRAINING accuracy is vacuous here (mean-pooled hidden-dim feature on a
           small holdout -> feature-dim >= sample-count -> trivially 1.0). If the sample count is
           too small to hold a fold out, the in-language score is reported unreliable (NaN) and no
           shortcut is declared (is_shortcut = None -- never a false 1.0); AND
      (ii) cross-lingual transfer collapses to chance, measured by DISTANCE FROM CHANCE
           (chance = 0.5): transfer_strength = |cross - 0.5|. A sign-inverted probe (cross ~= 0.0)
           has transferred just as strongly as cross ~= 1.0, so ONLY cross near 0.5 is a transfer
           failure.
    """
    import numpy as np  # noqa: PLC0415

    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    langs = np.asarray(langs)
    en, ko = langs == "en", langs == "ko"

    # "Genuinely works" bar for the OUT-OF-SAMPLE in-language probe -- well above the 0.5 chance.
    in_lang_works = 0.70
    # cross "near chance" band: transfer_strength = |cross - 0.5| <= this counts as chance-level
    # (= transfer failure). 0.10 is the symmetric near-chance band.
    chance_band = 0.10

    cross_scores, in_scores = [], []
    for train_mask, test_mask in ((en, ko), (ko, en)):
        if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 1:
            continue
        w, b = _fit_logreg(feats[train_mask], labels[train_mask])
        cross_scores.append(_accuracy(feats[test_mask], labels[test_mask], w, b))
        # In-language score MUST be out-of-sample (training accuracy overfits to 1.0 at d>=n).
        in_scores.append(_kfold_heldout_accuracy(feats[train_mask], labels[train_mask]))
    if not cross_scores:
        return float("nan"), None

    cross = float(np.mean(cross_scores))
    transfer_strength = abs(cross - 0.5)   # distance from chance, EITHER direction

    reliable_in = [s for s in in_scores if not np.isnan(s)]
    if not reliable_in:
        # In-language reliability unknown (holdout too small to hold out): cannot decide shortcut.
        return cross, None
    in_lang_heldout = float(np.mean(reliable_in))

    # Shortcut = in-language probe reliably works out-of-sample AND cross transfer is at chance.
    is_shortcut = bool(in_lang_heldout >= in_lang_works and transfer_strength <= chance_band)
    return cross, is_shortcut


def eval_mechanism(ckpt: str, split: str = "probe") -> dict:
    """Timing / overlap / barge-in patterns (mechanism transfer). Reported separately
    from content.

    Measures transfer of the turn-taking mechanism (the "form") -- temporal /
    structural behaviors such as the timing of silence, backchannels, and
    interruptions. Uses Full-Duplex-Bench-family metrics and is **never summed with
    content accuracy (eval_content)** -- to catch the "partial failure" where the
    mechanism transfers but the content does not, separate reporting is
    mandatory.

    Barge-in local-measurement caveat: when entering the acoustic prosody graft
    (Phase 3.5), drift monitoring must locally measure the **speaker similarity of
    barge-in / interruption frames**, not of ordinary speech. This
    function may also return that local frame mask.

    Args:
        ckpt: path to the checkpoint under evaluation.
        split: "probe" only (training-unexposed holdout, RISKS §1).

    Returns:
        dict -- minimum keys (proposed):
            {"response_latency": float,           # partner utterance end -> response onset delay
             "overlap_rate": float,               # simultaneous-speech (overlap) ratio
             "backchannel_rate": float,           # backchannel frequency
             "bargein_rate": float,               # interruption occurrence rate
             "turn_switch_timing": dict,          # transition-timing stats (EPAD-onset alignment)
             "bargein_frame_mask_available": bool,  # for local speaker-similarity measurement
             "split": str}
        Does not include content (WER/CER) keys (kept separate from content).
    """
    # Enforce the holdout-only rule: mechanism is measured only on the probe holdout.
    if split != "probe":
        raise ValueError(
            f"eval_mechanism only runs on the 'probe' holdout (training-unexposed, RISKS §1); "
            f"got split={split!r}."
        )

    import numpy as np  # noqa: PLC0415

    model = _load_model(ckpt)

    # Probe-holdout prefixes seed the simulated dialogues (never seen in training).
    # Wire the REAL datasets stack. Imports stay LOCAL to this body so the entry-point
    # module still imports with no heavy deps (torch / datasets) at module top.
    # load_configs() reads configs/data/loader.yaml + tokens.yaml -> (DataConfig, LoaderConfig,
    # mix_cfg). For split="probe" build_dataloader uses a deterministic per-group sampler and
    # does NOT need mix_cfg (only training does); it returns a LoaderBundle. Pass a
    # KDCollatorConfig seeded with the config's token ids so the collator has them.
    #
    # The delay is set EXPLICITLY rather than left to `DelayConfig`'s default. That default is
    # `acoustic_delay=2`, the PRE-training stagger; post-training -- every checkpoint a gate ever
    # sees, since A/B/C run after phases 1/2/3 -- is acoustic 1 / text 0, and that is what
    # `train.py` DataArgs uses. Inheriting the default meant every evaluation conditioned the model
    # on a codebook stagger it was not trained under, which changes the generated distribution and
    # therefore every timing metric, silently.
    from project_amnesty.datasets.collator import DelayConfig, KDCollatorConfig  # noqa: PLC0415 (lazy)
    from project_amnesty.datasets.loader import build_dataloader, load_configs  # noqa: PLC0415 (lazy)

    data_cfg, loader_cfg, _mix = load_configs()
    bundle = build_dataloader(
        data_cfg=data_cfg,
        loader_cfg=loader_cfg,
        split="probe",
        collator_cfg=KDCollatorConfig(
            tokens=data_cfg.tokens,
            delay=DelayConfig(acoustic_delay=1, text_delay_frames=0),  # post-training stagger
        ),
    )

    max_new_frames = 750  # 60 s at 12.5 Hz (matches the training context cap, train.py DataArgs)
    backchannel_max = BACKCHANNEL_MAX_FRAMES  # surfaced module constant; NOT doc-specified (see its def)
    pad_ids = _pad_token_ids(data_cfg.tokens)
    # Semantic silence is a config constant, not a per-dialogue quantity -- resolve it once from the
    # measured bank (no cwd-relative file read, no modal fallback).
    silence = _silence_codes(data_cfg.tokens.silence_bank)

    # Accumulators over every simulated dialogue.
    total_frames = 0
    overlap_frames = 0
    union_frames = 0
    n_dialogues = 0              # simulated dialogues actually measured
    n_skipped_batches = 0        # text_anchor batches skipped (no audio_codes)
    resp_gaps_s: list = []       # non-negative-gap turn latencies (seconds), both directions
    all_gaps_s: list = []        # signed transition gaps (seconds) for turn_switch_timing
    n_user_turns = 0
    n_bargein = 0
    n_backchannel = 0
    bargein_frames: list = []    # per-dialogue barge-in frame indices, for the §7.9 local measure

    # Iterate the bundle's DataLoader (a bare `for batch in bundle` would be wrong).
    for batch in bundle.loader:
        # The probe split carries text_anchor micro-batches alongside the audio ones (they go
        # through TextAnchorCollator). A text_anchor batch has T=0 and no codebook axis
        # -- no `audio_codes` key at all -- so it cannot seed a simulated dialogue and must be
        # skipped rather than reaching `generate`.
        if "audio_codes" not in batch:
            n_skipped_batches += 1   # count the skip so "all skipped" != "measured and got 0"
            continue
        self_streams, user_streams, text_streams = _simulate_dialogue(model, batch, max_new_frames)
        for self_sem, user_sem, self_text in zip(self_streams, user_streams, text_streams):
            n_dialogues += 1
            self_speech = _self_speech_mask(self_text, self_sem, silence, pad_ids)
            user_speech = _speech_mask_from_semantic(user_sem, silence)
            frames = int(min(self_speech.shape[0], user_speech.shape[0]))
            self_speech = self_speech[:frames]
            user_speech = user_speech[:frames]

            total_frames += frames
            overlap_frames += int((self_speech & user_speech).sum())
            union_frames += int((self_speech | user_speech).sum())

            self_runs = _runs(self_speech)
            user_runs = _runs(user_speech)
            n_user_turns += len(user_runs)

            # response latency + signed transition gaps (both directions).
            gaps = _response_gaps(user_runs, self_runs) + _response_gaps(self_runs, user_runs)
            for g in gaps:
                all_gaps_s.append(g / FRAME_RATE_HZ)
                # g==0 reclassified from overlap to clean switch (zero-silence adjacent turn;
                # `_runs` end is exclusive). Makes the whole file consistent -- overlap_rate/
                # bargein already exclude g==0. Include g>=0 (not >0) as a response latency.
                if g >= 0:
                    resp_gaps_s.append(g / FRAME_RATE_HZ)

            # barge-in / backchannel classification on self runs vs the user speech mask.
            nb, nk, bmask = _bargein_backchannel(self_runs, user_speech, backchannel_max)
            n_bargein += nb
            n_backchannel += nk
            bargein_frames.append(np.where(bmask[:frames])[0].tolist())

    response_latency = float(np.mean(resp_gaps_s)) if resp_gaps_s else float("nan")
    # Zero denominator = nothing to measure against, NOT a perfect score. A silence/speech
    # collapse (pure per-frame KD can converge to "always predict silence") yields no
    # union speech / no user turns; reporting 0.0 there reads as a FLAWLESS run (no overlap, no
    # barge-in) and hides the collapse this guards against. Return nan instead -- same precedent as
    # `_kfold_heldout_accuracy` ("nan, never a passing 1.0, when it cannot measure"). A real
    # measured 0.0 (nonzero denominator, no events) stays 0.0: the non-zero-denominator math below
    # is unchanged.
    overlap_rate = float(overlap_frames / union_frames) if union_frames else float("nan")
    backchannel_rate = float(n_backchannel / n_user_turns) if n_user_turns else float("nan")
    bargein_rate = float(n_bargein / n_user_turns) if n_user_turns else float("nan")

    return {
        "response_latency": response_latency,      # seconds: partner utterance end -> response onset
        "overlap_rate": overlap_rate,              # simultaneous-speech frames / union speech frames
        "backchannel_rate": backchannel_rate,      # short backchannels per user turn
        "bargein_rate": bargein_rate,              # interruptions per user turn
        "backchannel_max_frames": BACKCHANNEL_MAX_FRAMES,  # classification cutoff (strict <; NOT doc-specified) -- travels with the rates it governs
        "turn_switch_timing": _timing_stats(all_gaps_s),  # transition-timing stats (EPAD-aligned)
        "bargein_frame_mask_available": any(len(x) > 0 for x in bargein_frames),  # for the local measure
        "bargein_frame_mask": bargein_frames,      # per-dialogue barge-in frame indices
        # Raw counts so a nan rate is legible as "nothing was measured" vs a real zero, and a
        # fully-skipped run is distinguishable from a measured-and-got-0 run. These
        # SURFACE the denominators the rates divide by -- not a threshold, not a doc-specified value.
        "n_dialogues": n_dialogues,                # simulated dialogues measured
        "n_user_turns": n_user_turns,              # denominator of backchannel_rate / bargein_rate
        "total_frames": total_frames,              # frames measured across all dialogues
        "union_frames": union_frames,              # denominator of overlap_rate
        "n_skipped_batches": n_skipped_batches,    # text_anchor batches skipped (no audio_codes)
        "split": split,
    }


def probe_representations(
    ckpt: str,
    moshi_ckpt: str = "kmhf/hf-moshiko",
    probe_path: "str | None" = None,
) -> dict:
    """Role-vector cosine separability · user-channel embedding shift · turn-taking
    causal probe.

    Three representation-level diagnostics (independent of generation quality, to
    confirm the representations were actually rewired):

    1) **Role-vector separability**: a learnable Role Token can
       have its role-distinction signal weakened under pressure from other losses. We
       measure the cosine similarity of the self/user role vectors and use the original
       Moshi measurements as the baseline (semantic cos ~= 0.50, acoustic
       cos ~= 0.75 -- differentiation concentrates at the semantic level). The
       **primary check is the semantic level**. The acoustic figure is
       reported at **codebook 1 only** (`role_cos_acoustic`) to match the
       baseline, which is a codebook-1 value; the per-level `role_cos_acoustic_by_level`
       exposes levels 1..K-1 so the monotonic-1->7 confound stays visible. The
       Depth input-side role embedding is checked the same way -- whether two
       batch elements produce different outputs.

    2) **User-channel embedding shift**: how far the shared embedding (or,
       in the original split structure, emb.8~15) has moved from its initialization
       (copied from Moshi's user-side emb.8~15), measured by L2 distance and
       cosine similarity. A quantitative diagnostic of "was the simultaneous
       listen+speak channel trained under Korean conditions?".

    3) **Turn-taking causal probe**: whether the turn-taking-related internal
       representation activates on a Korean audio prefix **regardless of language**,
       measured directly by causal probing -- distinguishing a genuine mechanism from a
       language-task shortcut of the form "if Korean, speak one turn then stop".
       The shortcut alarm (`probe_is_shortcut`) fires ONLY when the in-language probe
       genuinely works OUT-OF-SAMPLE (k-fold CV, so d>=n overfit accuracy cannot trip it)
       AND cross-lingual transfer sits at chance (measured by distance from 0.5, so a
       sign-inverted but transferred probe does not trip it); see `_causal_probe_cross`.
       Checkpoint A looks for the existence of this signal; B looks for a rise vs A.

    Args:
        ckpt: path to the checkpoint under evaluation. (The probe always uses the
            holdout prefixes.)
        moshi_ckpt: Moshi warm-start source whose user-side tables are the
            initialization the user-channel embedding shift is measured against.
            Read via `utils.warm_start_haan.moshi_audio_tables`, the same function the warm-start uses,
            so the baseline is the model's actual init. Default matches utils/train.py's
            ModelArgs.moshi_ckpt ("kmhf/hf-moshiko").
        probe_path: path to the turn-taking causal-probe holdout set (never trained on).
            When None the probe-score keys are returned as
            NaN / None (the probe set was not provided).

    Returns:
        dict -- minimum keys (proposed):
            {"role_cos_semantic": float,                               # codebook 0 (primary)
             "role_cos_acoustic": float,                               # codebook 1 (baseline-comparable)
             "role_cos_acoustic_by_level": list[float],                # acoustic levels 1..K-1
             "depth_role_divergence": dict,                             # batch-2 divergence
             "user_emb_l2_shift": float, "user_emb_cos_to_init": float,
             "turntaking_probe_score": float,
             "probe_is_shortcut": bool}                                 # language-task shortcut alarm
    """
    import numpy as np  # noqa: PLC0415

    model = _load_model(ckpt)

    # --- (1) role-vector separability -------------------------------------------------
    role_emb, role_mode = _temporal_role(model)
    audio_emb = _audio_embeddings(model)
    role_arr = _as_ndarray(role_emb)
    if role_arr.shape[0] < 2:
        raise ValueError(
            f"model.role_embedding must have 2 rows (self/user); got shape {role_arr.shape}"
        )
    role0, role1 = role_arr[0].ravel(), role_arr[1].ravel()

    cur_tables = _audio_emb_tables(audio_emb)  # list of (C, D), one per codebook
    n_codebooks = len(cur_tables)
    # Self/user row cosine on Haan's SHARED table under the additive Role Token.
    # semantic = codebook 0 (PRIMARY check); acoustic = codebook 1 ONLY, to stay apples-to-apples
    # with ROLE_COS_BASELINE_ACOUSTIC (0.751), which is a codebook-1 value. A mean
    # over codebooks 1..K-1 is NOT comparable to that baseline: the acoustic cosine
    # trends UP monotonically 1->7 (a down-weighting confound), so the levels are not interchangeable.
    # role_cos_acoustic_by_level exposes every acoustic level 1..K-1 so that monotonic trend (the
    # confound check) stays visible in the report.
    if role0.shape[-1] == cur_tables[0].shape[-1]:
        role_cos_semantic = _role_row_cosine(cur_tables[0], role0, role1, role_mode)
        role_cos_acoustic_by_level = [
            _role_row_cosine(cur_tables[k], role0, role1, role_mode) for k in range(1, n_codebooks)
        ]
        # PRIMARY acoustic metric = codebook 1 (the baseline level), NOT the K-1 mean.
        role_cos_acoustic = role_cos_acoustic_by_level[0] if role_cos_acoustic_by_level else float("nan")
    else:
        # Role-token dim does not match the audio-embedding dim: fall back to the raw self/user
        # role-token cosine for both levels (still the ARCH §3.3 role quantity).
        raw = _vec_cosine(role0, role1)
        role_cos_semantic = raw
        role_cos_acoustic = raw
        role_cos_acoustic_by_level = [raw]

    # --- (2) user-channel embedding shift vs the Moshi init ----
    # This must be the SAME tensors the warm-start actually copied in, or the "how far has it
    # moved" number is measured against an initialization the model never had. So it goes through
    # `utils.warm_start_haan`'s own `moshi_audio_tables` -- what `warm_start_from_moshi` uses.
    from project_amnesty.utils.warm_start_haan import moshi_audio_tables  # noqa: PLC0415 (lazy)

    init_tables = [
        _as_ndarray(t) for t in moshi_audio_tables(moshi_ckpt, num_codebooks=n_codebooks, side="user")
    ]
    l2s, coss = [], []
    for cur, init in zip(cur_tables, init_tables):
        if cur.shape[-1] != init.shape[-1]:
            raise ValueError(
                f"embed_tokens dim {cur.shape[-1]} != Moshi init dim {init.shape[-1]} "
                f"(cannot measure the user-channel shift, RISKS §3)."
            )
        n = min(cur.shape[0], init.shape[0])  # compare the real audio-code rows shared by both
        l2s.append(_row_l2(cur[:n], init[:n]))
        coss.append(_row_cosine(cur[:n], init[:n]))
    user_emb_l2_shift = float(np.mean(l2s))
    user_emb_cos_to_init = float(np.mean(coss))

    # --- (3) Depth role divergence: batch-2 self/user forward collapse probe --
    depth_role_divergence = _depth_role_divergence(model)

    # --- (4) turn-taking causal probe, Korean/English cross -----------------
    if probe_path is None:
        # Probe set not provided (holdout absent): report the probe keys as unset.
        turntaking_probe_score = float("nan")
        probe_is_shortcut = None
    else:
        prefixes, labels, langs = _load_probe_set(probe_path)
        feats = _turntaking_features(model, prefixes)
        turntaking_probe_score, probe_is_shortcut = _causal_probe_cross(feats, labels, langs)

    return {
        "role_cos_semantic": role_cos_semantic,       # primary check
        "role_cos_acoustic": role_cos_acoustic,       # codebook 1 (matches ROLE_COS_BASELINE_ACOUSTIC)
        "role_cos_acoustic_by_level": role_cos_acoustic_by_level,  # per-level 1..K-1 (monotonic-trend confound)
        "depth_role_divergence": depth_role_divergence,  # batch-2 divergence
        "user_emb_l2_shift": user_emb_l2_shift,
        "user_emb_cos_to_init": user_emb_cos_to_init,
        "turntaking_probe_score": turntaking_probe_score,  # mean cross-lingual accuracy
        "probe_is_shortcut": probe_is_shortcut,       # shortcut alarm: in-lang works out-of-sample
                                                      # + cross at chance; None if undecidable
    }


# ---------------------------------------------------------------------------
# Real dispatcher (model-free control flow; the model-dependent calls it makes
# are the TODO parts inside the sub-evaluations above).
# ---------------------------------------------------------------------------

def run_checkpoint(ckpt: str, tag: str, probe_path: "str | None" = None) -> dict:
    """Run the judgment bundle for checkpoint tag A/B/C and assemble a report dict.

    Which sub-evaluations each tag calls:
      - A: probe_representations (early signal -- whether the turn-taking probe activates).
      - B: probe_representations + eval_content/eval_mechanism tracking English
           held-out interference + re-measuring the Korean-prefix probe
           (rise vs A).
      - C: eval_content + eval_mechanism (mechanism vs content emergence judgment) +
           probe_representations as support.

    The content section and the mechanism section are kept under **different keys**
    in the report and are NEVER summed into one scalar. This function is
    a real dispatcher; the model-dependent work happens inside the sub-evaluations,
    which currently raise NotImplementedError.

    Args:
        ckpt: checkpoint path.
        tag: "A" | "B" | "C".
        probe_path: optional path to the turn-taking causal-probe holdout, threaded straight
            into probe_representations. This set is a Korean audio prefix held out from training
            and used only for probing, measured by causal probing. Passing it makes Checkpoint A's
            turn-taking probe decidable. When None the probe-score keys stay NaN/None (probe set
            not provided) -- behaviour is UNCHANGED, never a fabricated number.

    Returns:
        dict -- the report, with keys:
            {"tag": str, "ckpt": str,
             "judgments": tuple[str, ...],   # CHECKPOINT_JUDGMENTS[tag]
             "probe": dict,                  # present for A, B, C
             "content": dict,                # present for B, C (separate key)
             "mechanism": dict}              # present for B, C (separate key)
        On a partial failure, the separate keys let us attribute where it broke.
    """
    if tag not in CHECKPOINT_JUDGMENTS:
        raise ValueError(
            f"Unknown checkpoint tag {tag!r}; expected one of "
            f"{tuple(CHECKPOINT_JUDGMENTS)}."
        )

    report: dict = {
        "tag": tag,
        "ckpt": ckpt,
        "judgments": CHECKPOINT_JUDGMENTS[tag],
    }

    # A/B/C all run the representation probe (early signal at A; rise/collapse checks at B/C).
    # probe_path threads the causal-probe holdout through; when None the turn-taking
    # probe score stays NaN (probe set absent), so this only makes the path REACHABLE -- it does
    # not fabricate a result.
    report["probe"] = probe_representations(ckpt, probe_path=probe_path)

    # B and C add content and mechanism -- stored under separate keys and NEVER summed
    # into a single scalar. At B these back the English held-out interference
    # tracking; at C they back the Korean emergence judgment.
    if tag in ("B", "C"):
        # Each half is attempted INDEPENDENTLY so a partial failure can be attributed: mechanism
        # and content must stay separable. `eval_content` raises unconditionally (no Korean ASR
        # yet), so attempting them independently keeps `eval_mechanism` reachable. A failure is
        # recorded under its own key and never merged with the other half.
        for key, run in (("content", eval_content), ("mechanism", eval_mechanism)):
            try:
                report[key] = run(ckpt, split="probe")
            except Exception as exc:  # noqa: BLE001 -- the failure IS the result for this key
                report[key] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


def main() -> None:
    """CLI entry point. Parse --ckpt / --checkpoint-tag (A|B|C) -> run run_checkpoint.

    Only the probe split is used (no training exposure).
    """
    parser = argparse.ArgumentParser(
        description="Haan standalone checkpoint evaluation (mechanism vs content, RISKS §4)."
    )
    parser.add_argument("--ckpt", required=True, help="path to the checkpoint under evaluation")
    parser.add_argument(
        "--checkpoint-tag",
        required=True,
        choices=("A", "B", "C"),
        help="A=early signal / B=interference & rise / C=first emergence judgment (CURRICULUM §2)",
    )
    parser.add_argument(
        "--split",
        default="probe",
        help="only the probe holdout is allowed -- no training exposure (RISKS §1, CURRICULUM §0)",
    )
    parser.add_argument(
        "--probe-path",
        default=None,
        help="path to the turn-taking causal-probe holdout set (CURRICULUM §0 prepares the Korean "
        "audio prefix, training-unused / probing-only; RISKS §1). Threaded into "
        "probe_representations so Checkpoint A's turn-taking probe is decidable. Omit = probe "
        "score stays NaN (behaviour unchanged, not a fabricated number).",
    )
    args = parser.parse_args()

    # Enforce the holdout-only rule explicitly rather than relying on choices,
    # so a misuse produces a clear message.
    assert args.split == "probe", (
        f"Only the 'probe' split is allowed (training-unexposed holdout, RISKS §1); "
        f"got {args.split!r}."
    )

    # run_checkpoint is a real dispatcher; it raises NotImplementedError from within the
    # model-dependent sub-evaluations until the models/ forward lands (datasets/ is already real). The report
    # keeps content / mechanism / probing under separate keys -- never summed into one
    # scalar.
    report = run_checkpoint(args.ckpt, args.checkpoint_tag, probe_path=args.probe_path)
    print(report)


if __name__ == "__main__":
    main()
