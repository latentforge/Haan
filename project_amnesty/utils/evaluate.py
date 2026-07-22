"""evaluate.py -- standalone evaluation entry point (offline, run against a checkpoint).

Core separation required by RISKS §4: **"turn-taking mechanism transfer" and
"conversational content consistency transfer" MUST be measured separately** -- if the
two are not split apart, we cannot even decide success vs failure. This file enforces
that separation in the code structure itself: `eval_content` (content) and
`eval_mechanism` (mechanism) return different metrics and are NEVER summed into a
single scalar (see RISKS §4).

  Checkpoint A (after Phase 1): causal probing of whether any turn-taking
                                representation activates on a Korean audio prefix
                                (an early signal). -- CURRICULUM §2 Phase1
  Checkpoint B (after Phase 2): whether English held-out multi-turn performance
                                collapses after Korean is injected (interference,
                                RISKS §2), plus re-measuring the Korean-prefix probe
                                (has it risen vs Phase1?). -- CURRICULUM §2 Phase2
  Checkpoint C (after Phase 3): first emergence judgment for Korean multi-turn --
                                mechanism vs content, kept separate.
                                -- CURRICULUM §2 Phase3, RISKS §4

Metrics:
  - **Content accuracy**: WER/CER from ASR re-transcription (guards against
    word-salad -- ARCHITECTURE §4.4). MOS/naturalness alone cannot catch "natural
    sounding but wrong" speech. Results are attributed against the Mimi round-trip
    ceiling (ARCHITECTURE §4.3, computed inline here) to separate the
    "codec bottleneck" from the "LM transfer bottleneck" (RISKS §6).
  - **Mechanism**: timing / overlap / barge-in pattern analysis (Full-Duplex-Bench
    family). Kept separate from content.
  - **Probing**: role-vector separability (ARCHITECTURE §3.5·§5.4), user-channel
    embedding shift from its initialization (L2 / cosine, RISKS §3), and the
    turn-taking causal probe (RISKS §1).

Only the probe split (held out by prepare) is used; it is NEVER exposed to
training (RISKS §1 · CURRICULUM §0).

Run: `python -m project_amnesty.utils.evaluate --ckpt <path> --checkpoint-tag C`

Note (file-map rule): this is a standalone entry point. `project_amnesty/datasets/` is
real and already used here (eval_mechanism's probe loader imports it), while
`project_amnesty/models/` exposes the real classes but its forward / warm-start bodies
are still TODO -- so the model-dependent eval bodies raise NotImplementedError at runtime
until that lands. Heavy imports (torch, transformers, ASR libraries) are deferred: they
live INSIDE each function body, not at module top, to keep entry-point load cost minimal.
"""

from __future__ import annotations

import argparse

# References used here (imported lazily INSIDE each eval body, off the entry-point load path):
#   from project_amnesty.models.modeling_haan import HaanForConditionalGeneration  (forward TODO)
#   from project_amnesty.datasets.runtime.loader import build_dataloader   (real; split="probe" holdout)
# datasets/ is fully implemented, so the probe loader runs for real; the model's forward is the
# only piece still TODO. What lives in eval_content / eval_mechanism / probe_representations is
# genuine *implementation* (generation loop, external ASR, timing/overlap, probing) that belongs
# in evaluate.py -- not something models/ or datasets/ can supply.
# NOTE: the standalone Mimi round-trip tool was removed; the §4.3 codec ceiling used for WER
# attribution (RISKS §6) is now computed inline in eval_content (Mimi encode->decode->ASR on the
# same probe audio), not via a separate tool.


# Checkpoint tag -> the bundle of judgments that must be reached at that stage
# (a human-readable documentation constant). The real gating logic does not exist
# yet (NotImplementedError). Maps to CURRICULUM §2 / RISKS §1·§2·§4.
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
    # separate (RISKS §4).
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
        Micro-averaged error rate. Returns 0.0 for an empty corpus.
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
        return 0.0
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
# `project_amnesty/datasets/` is real and used here; `project_amnesty/models/` exposes the
# real classes but its forward body is still TODO, so every model call below raises
# NotImplementedError at *runtime*. That is intended: the LOGIC here is complete and written
# against a documented contract, so once the model forward lands these evaluations run
# unchanged ("just works"). All heavy imports (torch / numpy / model / datasets) are kept LOCAL
# to each body so this entry-point module still imports with no heavy dependencies installed.
#
# Attribute names are kept IDENTICAL to what utils/train.py's diagnostics read, so
# training and evaluation see the same model:
#   model.role_emb        : (2, D) additive self/user Role Token            (ARCH §3.3)
#   model.audio_emb       : shared audio input embedding -- K per-codebook
#                           tables OR a single (K, C, D) tensor; the user-side
#                           init is copied from Moshi emb.8~15               (ARCH §5.4.2)
#   model.depth_role_emb  : (2, D_depth) Depth-side additive role embedding
#                           (optional)                                       (ARCH §5.4)
#   model(input_ids=, audio_codes=[, role_ids=]) -> outputs with
#       outputs.audio_logits (B, 2, K, T, C)  # axis 1 = [self, user] streams (ARCH §5.0.3/§5.4)
#       outputs.text_logits  (B, T, V)
#   model.generate(prefix, *, mode="q16", max_new_frames=...) -> obj/dict with
#       .codes (B, 2, K, T_gen)               # ALSO generates the user stream (ARCH §5.0.3)
# ---------------------------------------------------------------------------

# Mimi frame rate (ARCH §5.0.1): 1 text token per frame, 12.5 frames per second.
FRAME_RATE_HZ = 12.5

# ARCH §3.5.1 baseline (original Moshi self/user row cosine, tools/inspect_moshi_weights.py):
# semantic (codebook 0) ~= 0.501, acoustic (codebook 1) ~= 0.751. The PRIMARY check is the
# semantic level -- role differentiation concentrates there (ARCH §3.5.1).
ROLE_COS_BASELINE_SEMANTIC = 0.501
ROLE_COS_BASELINE_ACOUSTIC = 0.751


def _load_model(ckpt: str):
    """Construct the Haan model and load `ckpt` weights (model I/O contract above).

    Delegates to `project_amnesty.models.HaanForConditionalGeneration`. The class is real but
    its forward / warm-start body is still TODO, so this raises NotImplementedError at runtime
    -- intended (see module note). Heavy imports are local so the entry point stays importable
    without torch/transformers.
    """
    from project_amnesty.models import HaanForConditionalGeneration  # noqa: PLC0415 (lazy; forward TODO)

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
    """Normalize `model.audio_emb` into a list of per-codebook (C, D) numpy tables.

    Accepts the two documented layouts: a container (list / tuple / nn.ModuleList) of K
    per-codebook tables, or a single (K, C, D) tensor/parameter. A bare 2D (C, D) is treated as
    a single shared table (K == 1).
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
    raise ValueError(f"model.audio_emb has unsupported shape {getattr(arr, 'shape', None)}")


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


def _role_row_cosine(table, role0, role1) -> float:
    """Self/user row cosine on the SHARED audio table under the additive Role Token (ARCH §3.3).

    In Haan self and user index the SAME table and differ only by the additive role vector, so
    the §3.5.1 self-vs-user row cosine becomes cos(table_i + role0, table_i + role1). This
    reproduces the inspect_moshi_weights measurement on Haan's shared-table construction, so the
    result is directly comparable to the ARCH §3.5.1 baseline (semantic ~0.50, acoustic ~0.75).
    """
    import numpy as np  # noqa: PLC0415

    r0 = np.asarray(role0, dtype=np.float64).ravel()
    r1 = np.asarray(role1, dtype=np.float64).ravel()
    return _row_cosine(table + r0[None, :], table + r1[None, :])


# -- frame/code-level turn-taking primitives (eval_mechanism) -----------------

def _semantic_silence_from_payload(payload) -> set:
    """Extract the set of semantic (codebook-0) silence codes from a mimi_silence.json payload.

    Understands the derive_silence_codes.py schema -- {"silence_bank": [[...], ...],
    "num_codebooks": K, ...} where `silence_bank` is codebook-major (K rows, one silence-code
    sequence per codebook), so row 0 is the semantic level. Also accepts a frame-major bank
    (T rows of K codes -> column 0 is semantic), an explicit {"semantic": [...]}, or a bare list.
    """
    import numpy as np  # noqa: PLC0415

    if isinstance(payload, dict):
        if "semantic" in payload:
            return {int(c) for c in np.asarray(payload["semantic"]).ravel()}
        bank = payload.get("silence_bank")
        n_codebooks = payload.get("num_codebooks")
        if bank is None:
            return set()
        arr = np.asarray(bank)
        if arr.ndim == 2:
            if n_codebooks is not None and arr.shape[1] == n_codebooks and arr.shape[0] != n_codebooks:
                return {int(c) for c in arr[:, 0].ravel()}  # frame-major -> semantic column
            return {int(c) for c in arr[0].ravel()}         # codebook-major -> semantic row (default)
        return {int(c) for c in arr.ravel()}                # 1D bank == semantic sequence
    arr = np.asarray(payload)
    return {int(c) for c in (arr[0].ravel() if arr.ndim >= 2 else arr.ravel())}


def _silence_codes(semantic_streams, config_path: str = "configs/data/mimi_silence.json"):
    """Semantic (level-0) silence codes: the MEASURED bank if present, else a modal fallback.

    0 is a valid Mimi code, so silence must be the measured silence bank (ARCH: configs/data/
    mimi_silence.json / tools/derive_silence_codes.py), NOT zeros. Fallback when the bank is
    absent (documented, approximate): dialogue is mostly silence, so the single most frequent
    semantic code across the given streams stands in for the silence bank.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        codes = _semantic_silence_from_payload(payload)
        if codes:
            return codes
    stacked = [np.asarray(s).astype(int).ravel() for s in semantic_streams]
    allc = np.concatenate(stacked) if stacked else np.asarray([], dtype=int)
    if allc.size == 0:
        return set()
    vals, counts = np.unique(allc, return_counts=True)
    return {int(vals[int(counts.argmax())])}


def _speech_mask_from_semantic(semantic_codes, silence_codes) -> "np.ndarray":
    """Per-frame speech mask from Mimi semantic (level-0) codes: True where NOT a silence code."""
    import numpy as np  # noqa: PLC0415

    codes = np.asarray(semantic_codes).astype(int).ravel()
    silent = np.isin(codes, np.asarray(list(silence_codes), dtype=int))
    return ~silent


def _pad_token_ids(config_path: str = "configs/tokens.json") -> set:
    """Plain stream-PAD id(s) from the tokens config (ARCH §5.0.1/§7.6): between-word silence on
    the text stream (~65% of tokens). A non-PAD frame is a word or the EPAD onset trigger (both =
    the agent turn is active). Empty set if the config is absent -> caller uses semantic silence.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    if not os.path.exists(config_path):
        return set()
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = set()
    if isinstance(payload, dict):
        for key in ("PAD", "pad", "stream_pad", "STREAM_PAD"):
            if key in payload:
                try:
                    ids.add(int(payload[key]))
                except (TypeError, ValueError):
                    pass
    return ids


def _self_speech_mask(self_text, self_sem, silence_codes, pad_ids) -> "np.ndarray":
    """Self-stream per-frame speech presence.

    Continuous presence comes from the semantic-code silence bank (audio). When the generator
    also returns the self text stream and PAD ids are known, UNION the text utterance signal --
    a non-PAD frame is a word or the EPAD onset trigger (ARCH §5.0.1) -- which pins onsets to the
    EPAD trigger; the audio term fills the within-utterance PAD gaps (text is ~65% PAD even mid-
    speech, §5.0.1). Offsets fall on PAD/silence transitions.
    """
    import numpy as np  # noqa: PLC0415

    audio_speech = _speech_mask_from_semantic(self_sem, silence_codes)
    if self_text is None or not pad_ids:
        return audio_speech
    text_speaking = ~np.isin(np.asarray(self_text).astype(int), np.asarray(list(pad_ids), dtype=int))
    n = min(audio_speech.shape[0], text_speaking.shape[0])
    return audio_speech[:n] | text_speaking[:n]


def _runs(mask):
    """List of (start, end) speech runs (contiguous True regions) in a boolean frame mask."""
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
    gap = responder_onset - partner_offset. A positive gap = silence between turns (a clean turn
    switch); a non-positive gap = the responder started before the partner finished (overlap /
    barge-in). Callers split the two. One response is attributed per partner turn.
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
        gaps.append(int(cand.min()) - pe)  # >0 gap, <=0 overlap
    return gaps


def _bargein_backchannel(self_runs, user_speech, backchannel_max):
    """Classify self runs that begin during user speech into barge-ins vs backchannels.

    - barge-in: self takes the floor while the user is speaking (onset during user speech and the
      self run is not merely a short burst) -> interruption (full-duplex behavior, RISKS §1).
    - backchannel: a short self burst fully inside a user turn (user keeps the floor before and
      after) -> "uh-huh" (Full-Duplex-Bench family).
    Returns (n_bargein, n_backchannel, bargein_frame_mask). The mask marks the frames where a
    barge-in self run overlaps the ongoing user speech -- the §7.9 local speaker-similarity target.
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
        "n_gap_transitions": int((g > 0).sum()),      # clean turn switches (silence gap)
        "n_overlap_transitions": int((g <= 0).sum()),  # responder started before partner ended
    }


def _simulate_dialogue(model, prefix, max_new_frames):
    """q16 simulation generate (ARCH §5.0.3): produce BOTH self and user streams for offline eval.

    Returns three per-batch-element lists (self_semantic, user_semantic, self_text), each entry a
    frame-aligned 1D int array (self_text is None when the generator does not return it). Frame /
    code level only -- NO waveform, NO Mimi decode (keeps this off the content path, RISKS §4).
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        gen = model.generate(prefix, mode="q16", max_new_frames=max_new_frames)
    codes = gen["codes"] if isinstance(gen, dict) else getattr(gen, "codes", None)
    if codes is None:
        raise AttributeError(
            "model.generate(...) must return `codes` (B, 2, K, T_gen) -- the self+user streams of "
            "the q16 simulated dialogue (ARCH §5.0.3)."
        )
    codes = _as_ndarray(codes).astype(int)  # (B, 2, K, T)
    if codes.ndim != 4 or codes.shape[1] < 2:
        raise ValueError(f"generate codes must be (B, 2, K, T_gen); got shape {codes.shape}")
    batch = codes.shape[0]
    self_list = [codes[b, 0, 0, :] for b in range(batch)]   # semantic level-0, self stream
    user_list = [codes[b, 1, 0, :] for b in range(batch)]   # semantic level-0, user stream
    text = gen.get("text") if isinstance(gen, dict) else getattr(gen, "text", None)
    if text is not None:
        text_arr = _as_ndarray(text).astype(int)
        text_arr = text_arr.reshape(batch, -1)
        text_list = [text_arr[b] for b in range(batch)]
    else:
        text_list = [None] * batch
    return self_list, user_list, text_list


# -- representation-level primitives (probe_representations) -------------------

def _depth_role_divergence(model) -> dict:
    """Batch-2 self/user forward divergence probe for the shared Depth (ARCH §5.4 diagnostic ②).

    A single forward already emits both streams: outputs.audio_logits is (B, 2, K, T, C) with
    axis 1 = [self, user] (ARCH §5.0.3/§5.4). We forward a minimal synthetic frame-level prefix
    (identical content for both streams, so only the additive role embedding differs) and measure
    how far the self-role output diverges from the user-role output. If the additive Depth role
    embedding is diluted under other loss pressure the two collapse to the same output (role
    ignored) -> the ARCH §5.4 fallback is to promote the shared projection to two role-specific
    ones. Also reports the static cosine / L2 of `model.depth_role_emb` when present.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    n_codebooks = len(_audio_emb_tables(_require_model_attr(model, "audio_emb", "shared audio input embedding, ARCH §5.4.2")))
    n_frames = 8
    audio_codes = torch.zeros((1, n_codebooks, n_frames), dtype=torch.long)
    input_ids = torch.zeros((1, n_frames), dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, audio_codes=audio_codes)
    audio_logits = getattr(outputs, "audio_logits", None)
    if audio_logits is None and isinstance(outputs, dict):
        audio_logits = outputs.get("audio_logits")
    if audio_logits is None:
        raise AttributeError(
            "forward outputs must expose `audio_logits` (B, 2, K, T, C) for the Depth role "
            "divergence probe (ARCH §5.0.3/§5.4)."
        )
    logits = _as_ndarray(audio_logits)
    if logits.ndim < 2 or logits.shape[1] < 2:
        raise ValueError(f"audio_logits must have a self/user axis of size 2 at dim 1; got {logits.shape}")
    self_l = logits[:, 0, ...].ravel()
    user_l = logits[:, 1, ...].ravel()
    diff = float(np.linalg.norm(self_l - user_l))
    denom = float(np.linalg.norm(self_l) + np.linalg.norm(user_l) + 1e-8)
    batch_output_divergence = diff / denom  # matches train.py run_diagnostics' collapse metric

    depth_role_emb = getattr(model, "depth_role_emb", None)
    if depth_role_emb is not None:
        dre = _as_ndarray(depth_role_emb)
        depth_role_cos = _vec_cosine(dre[0], dre[1]) if dre.shape[0] >= 2 else float("nan")
        depth_role_l2 = float(np.linalg.norm(dre[0].ravel() - dre[1].ravel())) if dre.shape[0] >= 2 else float("nan")
    else:
        depth_role_cos = None
        depth_role_l2 = None

    return {
        "batch_output_divergence": batch_output_divergence,   # normalized L2 self vs user (ARCH §5.4 ②)
        "self_user_output_cos": _vec_cosine(self_l, user_l),  # cosine of the two stream outputs
        "collapsed": bool(batch_output_divergence < 1e-3),    # role-collapse alarm -> split projection
        "depth_role_cos": depth_role_cos,                     # static cos(depth_role_emb[0], [1])
        "depth_role_l2": depth_role_l2,
        "n_frames": n_frames,
    }


def _load_probe_set(probe_path: str):
    """Load the turn-taking causal-probe holdout (CURRICULUM §0, never trained on -- RISKS §1).

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
    """Run one holdout prefix through the model forward (documented I/O contract)."""
    return model(**prefix) if isinstance(prefix, dict) else model(prefix)


def _turntaking_features(model, prefixes):
    """Forward each holdout prefix and pull the turn-taking probe representation (one vector each).

    Uses the per-frame Temporal context vector z_s -- the representation the Depth turn-taking
    prediction reads (ARCH §5.0 eq.1) -- exposed as `outputs.hidden_states` (last layer) or
    `outputs.z`, mean-pooled over the frame axis. Raises until the models/ forward body lands.
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
    """Korean/English turn-taking causal-probe cross (RISKS §1).

    Fit a linear probe on one language's representations and test it on the OTHER language (both
    directions). A genuinely language-independent turn-taking circuit transfers across the cross;
    the "if Korean, one turn then stop" language shortcut (RISKS §1) does NOT. Returns
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
           failure. (The old `cross <= 0.60` test wrongly flagged cross ~= 0.0 -- strong INVERTED
           transfer -- as a shortcut.)
    """
    import numpy as np  # noqa: PLC0415

    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    langs = np.asarray(langs)
    en, ko = langs == "en", langs == "ko"

    # "Genuinely works" bar for the OUT-OF-SAMPLE in-language probe -- well above the 0.5 chance.
    in_lang_works = 0.70
    # cross "near chance" band: transfer_strength = |cross - 0.5| <= this counts as chance-level
    # (= transfer failure). 0.10 reproduces the old one-sided cross <= 0.60 band, symmetrically.
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
    from content. -- RISKS §4

    Measures transfer of the turn-taking mechanism (the "form") -- temporal /
    structural behaviors such as the timing of silence, backchannels, and
    interruptions (RISKS §8: this behavioral transfer is Haan's essential
    contribution). Uses Full-Duplex-Bench-family metrics and is **never summed with
    content accuracy (eval_content)** -- to catch the "partial failure" where the
    mechanism transfers but the content does not (RISKS §4), separate reporting is
    mandatory.

    Barge-in local-measurement caveat: when entering the acoustic prosody graft
    (Phase 3.5), drift monitoring must locally measure the **speaker similarity of
    barge-in / interruption frames**, not of ordinary speech (RISKS §7.9). This
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
             "turn_switch_timing": dict,          # transition-timing stats (EPAD-onset alignment, §5.0.1)
             "bargein_frame_mask_available": bool,  # for §7.9 local speaker-similarity measurement
             "split": str}
        Does not include content (WER/CER) keys (RISKS §4 separation principle).
    """
    # Enforce the holdout-only rule (RISKS §1): mechanism is measured only on the probe holdout.
    if split != "probe":
        raise ValueError(
            f"eval_mechanism only runs on the 'probe' holdout (training-unexposed, RISKS §1); "
            f"got split={split!r}."
        )

    import numpy as np  # noqa: PLC0415

    model = _load_model(ckpt)

    # Probe-holdout prefixes seed the simulated dialogues (never seen in training, RISKS §1).
    # Wire the REAL datasets stack. Imports stay LOCAL to this body so the entry-point
    # module still imports with no heavy deps (torch / datasets) at module top.
    # load_configs() reads configs/data/loader.yaml + tokens.yaml -> (DataConfig, LoaderConfig,
    # mix_cfg). For split="probe" build_dataloader uses a deterministic per-group sampler and
    # does NOT need mix_cfg (only training does); it returns a LoaderBundle. Pass a
    # KDCollatorConfig seeded with the config's token ids so the collator has them (the default
    # DelayConfig is fine -- delay does not affect the probe's mechanism metric positions).
    from project_amnesty.datasets.runtime.collator import KDCollatorConfig  # noqa: PLC0415 (lazy)
    from project_amnesty.datasets.runtime.loader import build_dataloader, load_configs  # noqa: PLC0415 (lazy)

    data_cfg, loader_cfg, _mix = load_configs()
    bundle = build_dataloader(
        data_cfg=data_cfg,
        loader_cfg=loader_cfg,
        split="probe",
        collator_cfg=KDCollatorConfig(tokens=data_cfg.tokens),
    )

    max_new_frames = 750  # 60 s at 12.5 Hz (matches the training context cap, train.py DataArgs)
    backchannel_max = max(1, int(round(0.6 * FRAME_RATE_HZ)))  # ~0.6 s short burst -> backchannel
    pad_ids = _pad_token_ids()

    # Accumulators over every simulated dialogue.
    total_frames = 0
    overlap_frames = 0
    union_frames = 0
    resp_gaps_s: list = []       # positive-gap turn latencies (seconds), both directions
    all_gaps_s: list = []        # signed transition gaps (seconds) for turn_switch_timing
    n_user_turns = 0
    n_bargein = 0
    n_backchannel = 0
    bargein_frames: list = []    # per-dialogue barge-in frame indices, for the §7.9 local measure

    # Iterate the bundle's DataLoader (a bare `for batch in bundle` would be wrong).
    for batch in bundle.loader:
        self_streams, user_streams, text_streams = _simulate_dialogue(model, batch, max_new_frames)
        for self_sem, user_sem, self_text in zip(self_streams, user_streams, text_streams):
            silence = _silence_codes([self_sem, user_sem])
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
                if g > 0:
                    resp_gaps_s.append(g / FRAME_RATE_HZ)

            # barge-in / backchannel classification on self runs vs the user speech mask.
            nb, nk, bmask = _bargein_backchannel(self_runs, user_speech, backchannel_max)
            n_bargein += nb
            n_backchannel += nk
            bargein_frames.append(np.where(bmask[:frames])[0].tolist())

    response_latency = float(np.mean(resp_gaps_s)) if resp_gaps_s else float("nan")
    overlap_rate = float(overlap_frames / union_frames) if union_frames else 0.0
    backchannel_rate = float(n_backchannel / n_user_turns) if n_user_turns else 0.0
    bargein_rate = float(n_bargein / n_user_turns) if n_user_turns else 0.0

    return {
        "response_latency": response_latency,      # seconds: partner utterance end -> response onset
        "overlap_rate": overlap_rate,              # simultaneous-speech frames / union speech frames
        "backchannel_rate": backchannel_rate,      # short backchannels per user turn
        "bargein_rate": bargein_rate,              # interruptions per user turn
        "turn_switch_timing": _timing_stats(all_gaps_s),  # transition-timing stats (EPAD-aligned, §5.0.1)
        "bargein_frame_mask_available": any(len(x) > 0 for x in bargein_frames),  # for §7.9 local
        "bargein_frame_mask": bargein_frames,      # per-dialogue barge-in frame indices (§7.9 target)
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

    1) **Role-vector separability** (ARCHITECTURE §3.5·§5.4): a learnable Role Token can
       have its role-distinction signal weakened under pressure from other losses. We
       measure the cosine similarity of the self/user role vectors and use the original
       Moshi measurements as the baseline (§3.5.1: semantic cos ~= 0.50, acoustic
       cos ~= 0.75 -- differentiation concentrates at the semantic level). The
       **primary check is the semantic level** (§3.5.1). The acoustic figure is
       reported at **codebook 1 only** (`role_cos_acoustic`) to match the §3.5.1
       baseline, which is a codebook-1 value; the per-level `role_cos_acoustic_by_level`
       exposes levels 1..K-1 so the §3.5.1 monotonic-1->7 confound stays visible. The
       Depth input-side role embedding (§5.4) is checked the same way -- whether two
       batch elements produce different outputs.

    2) **User-channel embedding shift** (RISKS §3): how far the shared embedding (or,
       in the original split structure, emb.8~15) has moved from its initialization
       (copied from Moshi's user-side emb.8~15, §5.4.2), measured by L2 distance and
       cosine similarity. A quantitative diagnostic of "was the simultaneous
       listen+speak channel trained under Korean conditions?".

    3) **Turn-taking causal probe** (RISKS §1): whether the turn-taking-related internal
       representation activates on a Korean audio prefix **regardless of language**,
       measured directly by causal probing -- distinguishing a genuine mechanism from a
       language-task shortcut of the form "if Korean, speak one turn then stop" (§1).
       The shortcut alarm (`probe_is_shortcut`) fires ONLY when the in-language probe
       genuinely works OUT-OF-SAMPLE (k-fold CV, so d>=n overfit accuracy cannot trip it)
       AND cross-lingual transfer sits at chance (measured by distance from 0.5, so a
       sign-inverted but transferred probe does not trip it); see `_causal_probe_cross`.
       Checkpoint A looks for the existence of this signal; B looks for a rise vs A
       (CURRICULUM §2).

    Args:
        ckpt: path to the checkpoint under evaluation. (The probe always uses the
            holdout prefixes, RISKS §1.)
        moshi_ckpt: Moshi warm-start source whose user-side tables (emb.8~15, §5.4.2)
            are the initialization the user-channel embedding shift is measured against
            (RISKS §3). Loaded exactly like tools/inspect_moshi_weights.py. Default matches
            utils/train.py's ModelArgs.moshi_ckpt ("kmhf/hf-moshiko").
        probe_path: path to the turn-taking causal-probe holdout set (CURRICULUM §0,
            never trained on -- RISKS §1). When None the probe-score keys are returned as
            NaN / None (the probe set was not provided).

    Returns:
        dict -- minimum keys (proposed):
            {"role_cos_semantic": float,                               # §3.5.1 codebook 0 (primary)
             "role_cos_acoustic": float,                               # §3.5.1 codebook 1 (baseline-comparable)
             "role_cos_acoustic_by_level": list[float],                # §3.5.1 acoustic levels 1..K-1
             "depth_role_divergence": dict,                             # §5.4 batch-2 divergence
             "user_emb_l2_shift": float, "user_emb_cos_to_init": float,  # RISKS §3
             "turntaking_probe_score": float,                           # RISKS §1
             "probe_is_shortcut": bool}                                 # language-task shortcut alarm
    """
    import numpy as np  # noqa: PLC0415

    model = _load_model(ckpt)

    # --- (1) role-vector separability (ARCH §3.5·§3.5.1·§5.4) --------------------------
    role_emb = _require_model_attr(model, "role_emb", "(2, D) additive self/user Role Token, ARCH §3.3")
    audio_emb = _require_model_attr(model, "audio_emb", "shared audio input embedding, ARCH §5.4.2")
    role_arr = _as_ndarray(role_emb)
    if role_arr.shape[0] < 2:
        raise ValueError(f"model.role_emb must have 2 rows (self/user); got shape {role_arr.shape}")
    role0, role1 = role_arr[0].ravel(), role_arr[1].ravel()

    cur_tables = _audio_emb_tables(audio_emb)  # list of (C, D), one per codebook
    n_codebooks = len(cur_tables)
    # §3.5.1-style self/user row cosine on Haan's SHARED table under the additive Role Token.
    # semantic = codebook 0 (PRIMARY check); acoustic = codebook 1 ONLY, to stay apples-to-apples
    # with ROLE_COS_BASELINE_ACOUSTIC (0.751), which ARCH §3.5.1 defines for codebook 1. A mean
    # over codebooks 1..K-1 is NOT comparable to that baseline: §3.5.1 notes the acoustic cosine
    # trends UP monotonically 1->7 (a down-weighting confound), so the levels are not interchangeable.
    # role_cos_acoustic_by_level exposes every acoustic level 1..K-1 so that monotonic trend (the
    # §3.5.1 confound check) stays visible in the report.
    if role0.shape[-1] == cur_tables[0].shape[-1]:
        role_cos_semantic = _role_row_cosine(cur_tables[0], role0, role1)
        role_cos_acoustic_by_level = [
            _role_row_cosine(cur_tables[k], role0, role1) for k in range(1, n_codebooks)
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

    # --- (2) user-channel embedding shift vs the Moshi init (RISKS §3, ARCH §5.4.2) ----
    # Load the Moshi user-side tables (emb.8~15) exactly like tools/inspect_moshi_weights.py --
    # this is the initialization the shared audio embedding was copied from (§5.4.2).
    from project_amnesty.tools.inspect_moshi_weights import load_embedding_tables  # noqa: PLC0415

    moshi_tables = load_embedding_tables(ckpt=moshi_ckpt, dep_q=n_codebooks)  # {0..2K-1}; user = [K, 2K)
    init_tables = [moshi_tables[n_codebooks + k] for k in range(n_codebooks)]
    l2s, coss = [], []
    for cur, init in zip(cur_tables, init_tables):
        if cur.shape[-1] != init.shape[-1]:
            raise ValueError(
                f"audio_emb dim {cur.shape[-1]} != Moshi init dim {init.shape[-1]} "
                f"(cannot measure the user-channel shift, RISKS §3)."
            )
        n = min(cur.shape[0], init.shape[0])  # compare the real audio-code rows shared by both
        l2s.append(_row_l2(cur[:n], init[:n]))
        coss.append(_row_cosine(cur[:n], init[:n]))
    user_emb_l2_shift = float(np.mean(l2s))
    user_emb_cos_to_init = float(np.mean(coss))

    # --- (3) Depth role divergence: batch-2 self/user forward collapse probe (ARCH §5.4) --
    depth_role_divergence = _depth_role_divergence(model)

    # --- (4) turn-taking causal probe, Korean/English cross (RISKS §1) -----------------
    if probe_path is None:
        # Probe set not provided (CURRICULUM §0 holdout absent): report the probe keys as unset.
        turntaking_probe_score = float("nan")
        probe_is_shortcut = None
    else:
        prefixes, labels, langs = _load_probe_set(probe_path)
        feats = _turntaking_features(model, prefixes)
        turntaking_probe_score, probe_is_shortcut = _causal_probe_cross(feats, labels, langs)

    return {
        "role_cos_semantic": role_cos_semantic,       # §3.5.1 (primary check)
        "role_cos_acoustic": role_cos_acoustic,       # §3.5.1 codebook 1 (matches ROLE_COS_BASELINE_ACOUSTIC)
        "role_cos_acoustic_by_level": role_cos_acoustic_by_level,  # per-level 1..K-1 (§3.5.1 monotonic-trend confound)
        "depth_role_divergence": depth_role_divergence,  # §5.4 batch-2 divergence
        "user_emb_l2_shift": user_emb_l2_shift,       # RISKS §3
        "user_emb_cos_to_init": user_emb_cos_to_init,  # RISKS §3
        "turntaking_probe_score": turntaking_probe_score,  # RISKS §1 (mean cross-lingual accuracy)
        "probe_is_shortcut": probe_is_shortcut,       # shortcut alarm: in-lang works out-of-sample
                                                      # + cross at chance; None if undecidable
    }


# ---------------------------------------------------------------------------
# Real dispatcher (model-free control flow; the model-dependent calls it makes
# are the TODO parts inside the sub-evaluations above).
# ---------------------------------------------------------------------------

def run_checkpoint(ckpt: str, tag: str) -> dict:
    """Run the judgment bundle for checkpoint tag A/B/C and assemble a report dict.

    Which sub-evaluations each tag calls (CURRICULUM §2 / RISKS §1·§2·§4):
      - A: probe_representations (early signal -- whether the turn-taking probe activates).
      - B: probe_representations + eval_content/eval_mechanism tracking English
           held-out interference (RISKS §2) + re-measuring the Korean-prefix probe
           (rise vs A).
      - C: eval_content + eval_mechanism (mechanism vs content emergence judgment) +
           probe_representations as support.

    The content section and the mechanism section are kept under **different keys**
    in the report and are NEVER summed into one scalar (RISKS §4). This function is
    a real dispatcher; the model-dependent work happens inside the sub-evaluations,
    which currently raise NotImplementedError.

    Args:
        ckpt: checkpoint path.
        tag: "A" | "B" | "C".

    Returns:
        dict -- the report, with keys:
            {"tag": str, "ckpt": str,
             "judgments": tuple[str, ...],   # CHECKPOINT_JUDGMENTS[tag]
             "probe": dict,                  # present for A, B, C
             "content": dict,                # present for B, C (separate key -- RISKS §4)
             "mechanism": dict}              # present for B, C (separate key -- RISKS §4)
        On a partial failure, the separate keys let us attribute where it broke
        (RISKS §4 · §8 negative result).
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
    report["probe"] = probe_representations(ckpt)

    # B and C add content and mechanism -- stored under separate keys and NEVER summed
    # into a single scalar (RISKS §4). At B these back the English held-out interference
    # tracking (RISKS §2); at C they back the Korean emergence judgment.
    if tag in ("B", "C"):
        report["content"] = eval_content(ckpt, split="probe")
        report["mechanism"] = eval_mechanism(ckpt, split="probe")

    return report


def main() -> None:
    """CLI entry point. Parse --ckpt / --checkpoint-tag (A|B|C) -> run run_checkpoint.

    Only the probe split is used (no training exposure, RISKS §1).
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
    args = parser.parse_args()

    # Enforce the holdout-only rule explicitly (RISKS §1) rather than relying on choices,
    # so a misuse produces a clear message.
    assert args.split == "probe", (
        f"Only the 'probe' split is allowed (training-unexposed holdout, RISKS §1); "
        f"got {args.split!r}."
    )

    # run_checkpoint is a real dispatcher; it raises NotImplementedError from within the
    # model-dependent sub-evaluations until the models/ forward lands (datasets/ is already real). The report
    # keeps content / mechanism / probing under separate keys -- never summed into one
    # scalar (RISKS §4).
    report = run_checkpoint(args.ckpt, args.checkpoint_tag)
    print(report)


if __name__ == "__main__":
    main()
