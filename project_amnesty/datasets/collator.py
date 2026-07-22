"""Batch assembly for the Moshi-KD audio path: delay, padding, loss weights.

Three things live here that cannot live in `__getitem__` because they need the
batch or a model hyperparameter:

1. **Delay.** Storage is delay-free (schema.py); the acoustic/semantic/text delay
   pattern is a model knob that changes between curriculum phases. `set_delay`
   exists so Phase 1 -> Phase 2 (tau=2 -> 1, text_delay -> 0) does not require
   rebuilding the DataLoader.
2. **Padding** to the batch max.
3. **Loss weights**, which are a function of zone/PAD/EPAD ids and of the
   post-delay validity ranges.

Teacher forcing (plan section 3.1): everything here is emitted in **target**
alignment. The 1-step autoregressive shift belongs to the model. We ship a
constant ``target_aligned=True`` key so a double shift is a grep-able tripwire
rather than an 80 ms timing distortion that trains happily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from project_amnesty.datasets.schema import FRAME_RATE_HZ, NUM_CODEBOOKS

from .config import TokenConfig
from .item import LANG_IDS, SAMPLE_TYPE_IDS, KDSample
from .kd_align import (
    assert_aligned,
    assert_kd_valid_subset,
    derive_kd_valid,
    shift_scan,
    teacher_offset,
    transition_weight,
)

# zone_ids values. 3 is batch padding, i.e. "not part of any sample".
ZONE_A, ZONE_B, ZONE_C, ZONE_PAD = 0, 1, 2, 3

_TEXT_ANCHOR_MSG = (
    "KDCollator received a text_anchor row ({uid!r}). text_anchor has T=0 and no "
    "codebook axis to hang a delay on -- it must go through "
    "project_amnesty.datasets.text_collator.TextAnchorCollator on its own micro-batch "
    "(plan section 5.4). Mixing happens at the grad_accum step level, not inside a batch."
)


def _frames_from_sec(sec: float) -> int:
    """Seconds -> 12.5 Hz frames, refusing to silently pick a side of a tie.

    0.6 s * 12.5 Hz = 7.5 frames. Rounding that quietly buries a ~40 ms
    turn-taking decision in the collator; the assert forces the team to write
    7 or 8 into the config.
    """
    exact = sec * FRAME_RATE_HZ
    n = int(round(exact))
    assert abs(exact - n) < 0.25, (
        f"text_delay_sec={sec} is {exact} frames at {FRAME_RATE_HZ} Hz, which is not "
        f"close enough to an integer. Set text_delay_frames explicitly (7 or 8 for 0.6 s) "
        f"instead of letting round() choose."
    )
    return n


@dataclass(frozen=True)
class DelayConfig:
    """The 9 raw per-stream delays: [text, cb0(semantic), cb1..cb7(acoustic)].

    Raw delays may be negative (text leading the audio is the pretraining
    regime). `offsets()` normalizes them; see the normalization note there.
    """

    acoustic_delay: int = 2               # tau, applied to codebooks 1..7
    semantic_delay: int = 0               # codebook 0
    text_delay_frames: int | None = None  # exactly one of these two
    text_delay_sec: float | None = None
    num_codebooks: int = NUM_CODEBOOKS

    def __post_init__(self) -> None:
        assert not (self.text_delay_frames is not None and self.text_delay_sec is not None), (
            "set text_delay_frames or text_delay_sec, not both"
        )
        if self.text_delay_sec is not None:
            object.__setattr__(self, "text_delay_frames", _frames_from_sec(self.text_delay_sec))
        if self.text_delay_frames is None:
            object.__setattr__(self, "text_delay_frames", 0)

    def raw(self) -> list[int]:
        """[text, semantic, acoustic * (K-1)] -- length K+1, pre-normalization."""
        return (
            [int(self.text_delay_frames), int(self.semantic_delay)]
            + [int(self.acoustic_delay)] * (self.num_codebooks - 1)
        )

    def offsets(self) -> list[int]:
        """Normalized, non-negative offsets. **This is plan section 7.8's home.**

        A negative text delay cannot push text before position 0, so it pushes
        *everything else forward* instead. The consequence that bites: with
        semantic_delay == 0 and text_delay == -1, codebook 0's offset is 1, not 0.
        Code that hardcodes ``o0 = 0`` and therefore leaves the teacher top-k
        unshifted is correct during finetuning and puts the KD target exactly one
        frame ahead of the student during pretraining -- which converges, never
        errors, and bakes in ~80 ms of turn-taking skew.
        """
        raw = self.raw()
        lo = min(raw)
        return [d - lo for d in raw]

    @property
    def max_offset(self) -> int:
        return max(self.offsets())


@dataclass
class KDCollatorConfig:
    tokens: TokenConfig = field(default_factory=TokenConfig)
    delay: DelayConfig = field(default_factory=DelayConfig)

    # "extend": the row occupies L + max_off positions, so no supervision is
    # discarded (the extra <=2 positions are free). "truncate" keeps output
    # length L and drops the tail; kept for reference-implementation diffing
    # and for the delay round-trip test.
    edge_mode: str = "extend"
    pad_to_multiple_of: int = 8

    # --- loss weights (plan section 3.7) ---
    w_nonsemantic_audio: float = 0.02   # codebooks 1..7
    w_stream_pad_text: float = 0.30     # class-imbalance correction, PAD is ~65%
    w_stream_epad_text: float = 1.00    # NOT down-weighted: speech-onset trigger
    w_zone_a: float = 0.0               # system prompt is context, not a target
    mask_synthetic_user: bool = True    # silence-filled user channel -> 0

    # --- Zone A / Zone B assembly (ARCHITECTURE section 7.1-7.4) ---
    # Zones are CONSTRUCTED here, not stored. Same principle that keeps delay out
    # of storage: a fixed config system prompt and a reference-that-is-another-row
    # are both recomputable, so storing them would bloat every row and freeze the
    # per-epoch variation Phase 2's "per-sample varied reference voices" wants.
    #
    # Empty system_prompt_ids means no Zone A, so the default is today's behaviour.
    system_prompt_ids: tuple[int, ...] = ()
    # Zone A goes on EVERY audio source, en_kd included. The teacher generated its
    # dialogues without a prefix, but that does not reach the loss: Zone A is fully
    # masked and kd_valid is False across it, and the teacher top-k is shifted by
    # the prefix length so it still lands on its own Zone C frames.
    #
    # Excluding en_kd is the actively dangerous option. ko_tts and en_solo would
    # carry a prefix while en_kd alone did not, making "no Zone A" perfectly
    # correlated with "do turn-taking" -- a cleaner shortcut cue than language
    # itself, and precisely the correlation RISKS_AND_DIAGNOSTICS section 1 calls the
    # project's most severe risk and that en_solo exists to break. It would also
    # train the turn-taking circuit in a context that never occurs at deployment,
    # where every sequence starts with the system prompt.
    zone_a_sources: tuple[str, ...] = ("ko_tts", "en_kd")
    # Zone B stays ko_tts-only: an en_kd row carries two voices, so no single
    # speaker identity is defined for it (the dataset asserts speaker == "").
    voice_prompt_sources: tuple[str, ...] = ("ko_tts",)

    # --- KD transition weighting (plan section 3.8) ---
    kd_transition_weight: float = 2.0
    kd_transition_halfwidth: int = 6    # +-6 frames ~ 0.5 s

    # --- ASR direction (DATA_STRATEGY section 4.2, ARCHITECTURE section 5.0.2) ---
    # Bidirectional reuse of the *same* (text, audio) pair. Section 5.0.2: "a single
    # delay hyper-parameter allows for switching from an ASR to a TTS model with no
    # changes in the loss, architecture, or training data". So nothing below
    # touches rows, channels, zones or losses -- only which delay this batch gets.
    # ASR reuses the **self (agent) audio channel**; it does not transcribe the
    # user stream, so the codes layout is identical to the TTS direction.
    asr_sources: tuple[str, ...] = ("ko_asr",)
    # None -> derived from `delay` by flipping the text term to +abs(text), which
    # makes text *lag* the audio. See _asr_delay.
    asr_delay: DelayConfig | None = None
    # NOT SPECIFIED BY THE DOCS. Moshi Table 1 gives text delay = +-0.6 s for
    # pretraining, and 0.6 * 12.5 = 7.5 frames is not integral -- _frames_from_sec
    # would fire its assert rather than pick a side. 8 is a rounding choice made
    # here and needs empirical confirmation; it is not a documented value.
    # Only used when `delay.text_delay_frames == 0` (the conversational setting,
    # which has no sign to flip).
    asr_text_delay_frames: int = 8

    # --- debug ---
    min_kd_hit_rate: float = 0.5
    shift_scan: tuple[int, ...] = (-2, -1, 0, 1, 2)

    def __post_init__(self) -> None:
        if isinstance(self.tokens, dict):
            self.tokens = TokenConfig(**self.tokens)
        if isinstance(self.delay, dict):
            self.delay = DelayConfig(**self.delay)
        if isinstance(self.asr_delay, dict):
            self.asr_delay = DelayConfig(**self.asr_delay)
        assert self.asr_text_delay_frames > 0, (
            "asr_text_delay_frames must be positive: a positive text delay is what "
            "places text after the audio (see DelayConfig.offsets)"
        )
        assert self.edge_mode in ("extend", "truncate"), (
            f"edge_mode must be extend|truncate, got {self.edge_mode!r}"
        )
        assert self.pad_to_multiple_of >= 1
        assert self.kd_transition_halfwidth >= 0
        # yaml hands these over as lists; downstream code does `in` and len() on
        # them and the config is meant to be hashable/frozen-ish.
        self.system_prompt_ids = tuple(int(t) for t in self.system_prompt_ids)
        self.zone_a_sources = tuple(str(s) for s in self.zone_a_sources)
        self.voice_prompt_sources = tuple(str(s) for s in self.voice_prompt_sources)
        self.asr_sources = tuple(str(s) for s in self.asr_sources)


def _place(dst: torch.Tensor, src: torch.Tensor, off: int, limit: int) -> int:
    """Write `src` into `dst` at time offset `off`, clipped to `limit`.

    The single write path for codes, text, validity, the teacher top-k and the
    KD frame weights. Sharing it is the point: plan section 3.6 requires the teacher for
    codebook 0 to land on exactly the same offset as codebook 0 itself, and the
    cheapest way to guarantee that is to make divergence impossible to express.

    Time is always the **last** axis. Returns the number of frames written.
    """
    n = min(src.shape[-1], limit - off, dst.shape[-1] - off)
    if n <= 0:
        return 0
    dst[..., off:off + n] = src[..., :n]
    return n


class KDCollator:
    """list[KDSample] -> the batch dict of plan section 3.3."""

    def __init__(self, cfg: KDCollatorConfig | None = None, **kwargs) -> None:
        self.cfg = cfg if cfg is not None else KDCollatorConfig(**kwargs)
        tok = self.cfg.tokens
        # Loud, named, before any I/O-shaped work. A wrong PAD id here does not
        # crash -- the model just never learns turn boundaries.
        tok.require(
            "text_pad_id", "text_epad_id", "batch_pad_id",
            "audio_init_id", "silence_bank", "mimi_ckpt_id",
        )
        assert tok.text_epad_id not in (tok.text_pad_id, tok.batch_pad_id), (
            "text_epad_id collides with text_pad_id or batch_pad_id"
        )
        self.K = tok.num_codebooks

    # -- delay knob, swappable without rebuilding the DataLoader (plan section 3.4) --
    def set_delay(self, delay: DelayConfig) -> None:
        assert isinstance(delay, DelayConfig)
        assert delay.num_codebooks == self.K, (
            f"DelayConfig has K={delay.num_codebooks}, collator has K={self.K}"
        )
        self.cfg.delay = delay

    @property
    def delay_offsets(self) -> list[int]:
        return self.cfg.delay.offsets()

    # ---------------------------------------------------- direction selection
    def asr_delay(self) -> DelayConfig:
        """The ASR-direction delay: same everything, text moved *after* the audio.

        Derived on demand rather than frozen at construction, so `set_delay`
        (Phase 1 -> Phase 2, tau=2 -> 1) keeps the two directions in step instead
        of leaving ASR on a stale acoustic delay.

        Sign convention, per `DelayConfig.offsets`: a **positive** text delay
        places text later, i.e. text lags audio -> ASR. Negative = text leads =
        TTS. Flipping to `+abs(text)` therefore mirrors the pretraining +-0.6 s
        pair exactly; when the main text delay is 0 there is no sign to flip and
        `asr_text_delay_frames` supplies the magnitude.
        """
        cfg = self.cfg
        if cfg.asr_delay is not None:
            return cfg.asr_delay
        base = cfg.delay
        t = int(base.text_delay_frames or 0)
        return DelayConfig(
            acoustic_delay=base.acoustic_delay,
            semantic_delay=base.semantic_delay,
            text_delay_frames=abs(t) if t != 0 else int(cfg.asr_text_delay_frames),
            num_codebooks=base.num_codebooks,
        )

    def _delay_for(self, rows: list[KDSample]) -> DelayConfig:
        """Pick the batch's delay from its group. Homogeneity is *asserted*, not
        assumed.

        The sampler emits single-group batches and RoutingCollator asserts it, but
        `delay_offsets` is one tensor for the whole batch, so it physically cannot
        express two directions. A mixed batch would silently apply the TTS delay to
        rows that need the ASR one -- which trains happily and teaches the wrong
        circuit. The check is on the *direction*, not on the source, so batches
        that legitimately mix ko_tts and en_kd rows are unaffected.
        """
        cfg = self.cfg
        directions = {r["source"] in cfg.asr_sources for r in rows}
        assert len(directions) == 1, (
            "batch mixes ASR-direction and TTS-direction rows "
            f"(sources={sorted({str(r['source']) for r in rows})}, "
            f"asr_sources={cfg.asr_sources}). delay_offsets is a single per-batch "
            "tensor and cannot express both directions."
        )
        return self.asr_delay() if directions.pop() else cfg.delay

    # ------------------------------------------------------------------ main
    def __call__(self, rows: list[KDSample]) -> dict:
        cfg, tok, K = self.cfg, self.cfg.tokens, self.K
        assert len(rows) > 0, "empty batch"

        for r in rows:
            if r["sample_type"] == "text_anchor" or bool(r["is_text_only"]):
                raise ValueError(_TEXT_ANCHOR_MSG.format(uid=r["sample_uid"]))

        delay = self._delay_for(rows)
        off = delay.offsets()
        assert len(off) == K + 1 and min(off) == 0 and all(o >= 0 for o in off)
        max_off = max(off)
        off_text, off_audio = off[0], off[1:]
        off_cb0 = off[1]

        B = len(rows)
        lens = [int(r["num_frames"]) for r in rows]
        # Zone A / Zone B are built here, so the assembled length is known only
        # after asking each row what prefix it gets.
        prefixes = [self._prefix_lens(r) for r in rows]
        totals = [na + nb + L for (na, nb), L in zip(prefixes, lens)]
        # Per-row output extent. "extend" keeps every supervised frame; the delay
        # only ever adds <= tau positions.
        extents = [t + max_off if cfg.edge_mode == "extend" else t for t in totals]
        T = max(extents)
        m = cfg.pad_to_multiple_of
        T = ((T + m - 1) // m) * m

        topks = [int(r["topk"]) for r in rows if bool(r["has_teacher"])]
        kmax = max(topks) if topks else 0
        if topks:
            assert len(set(topks)) == 1, f"mixed topk in one batch: {sorted(set(topks))}"

        dev = rows[0]["codes_self"].device

        codes = torch.full((B, 2, K, T), int(tok.audio_init_id), dtype=torch.int64, device=dev)
        text_tokens = torch.full((B, T), int(tok.batch_pad_id), dtype=torch.int64, device=dev)
        stream_valid = torch.zeros((B, 2, K, T), dtype=torch.bool, device=dev)
        text_valid = torch.zeros((B, T), dtype=torch.bool, device=dev)
        zone_ids = torch.full((B, T), ZONE_PAD, dtype=torch.uint8, device=dev)

        t_val = torch.zeros((B, 2, T, kmax), dtype=torch.float16, device=dev)
        t_idx = torch.full((B, 2, T, kmax), -1, dtype=torch.int64, device=dev)
        kd_frame_weight = torch.ones((B, 2, T), dtype=torch.float32, device=dev)
        # Only role 0 (self) carries a teacher: the other side is input, not a
        # prediction target, and dropping it halves the dominant tensor.
        teacher_row = torch.zeros((B, 2), dtype=torch.bool, device=dev)
        synthetic_user = torch.zeros((B,), dtype=torch.bool, device=dev)

        sample_uid: list[str] = []
        sample_type_id = torch.zeros((B,), dtype=torch.int64, device=dev)
        lang_id = torch.zeros((B,), dtype=torch.int64, device=dev)
        has_teacher = torch.zeros((B,), dtype=torch.bool, device=dev)
        num_frames = torch.zeros((B,), dtype=torch.int64, device=dev)

        # kd_valid is not just "cb0 is valid": Zone A and Zone B carry no teacher.
        kd_zone_c = torch.zeros((B, T), dtype=torch.bool, device=dev)

        for b, r in enumerate(rows):
            L, extent = lens[b], extents[b]
            na, nb = prefixes[b]
            assert L > 0, f"{r['sample_uid']}: frame sample with T=0"
            self._assert_row_shapes(r, L, K)

            sample_uid.append(r["sample_uid"])
            sample_type_id[b] = SAMPLE_TYPE_IDS[r["sample_type"]]
            lang_id[b] = LANG_IDS[r["lang"]]
            has_teacher[b] = bool(r["has_teacher"])
            num_frames[b] = L
            # ko_tts / en_solo arrive with the user channel already silence-filled
            # by the Dataset (plan section 2.4). We do not re-fill it; we only refuse to
            # train on fabricated audio.
            synthetic_user[b] = r["sample_type"] != "en_kd"

            # --- Zone A + Zone B + Zone C, assembled BEFORE the delay. Delaying
            # Zone C on its own and prepending afterwards would put the prefix and
            # the conversation on two different time bases (RISKS section 7.8).
            asm = self._assemble(r, na, nb, dev)
            P = na + nb
            ones = torch.ones((P + L,), dtype=torch.bool, device=dev)

            # --- audio: per-codebook offsets, per-codebook validity ---
            for k in range(K):
                o = off_audio[k]
                for role, key in ((0, "codes_self"), (1, "codes_other")):
                    _place(codes[b, role, k], asm[key][k], o, extent)
                    _place(stream_valid[b, role, k], ones, o, extent)

            # --- text: agent-only stream ---
            _place(text_tokens[b], asm["text_self"], off_text, extent)
            _place(text_valid[b], ones, off_text, extent)

            # --- teacher top-k for codebook 0. The offset rule is kd_align's, not
            # ours: the online path has to reproduce it against live logits and a
            # second copy here is how the two silently diverge.
            off_teacher = teacher_offset(off_cb0, na, nb)
            if bool(r["has_teacher"]):
                teacher_row[b, 0] = True
                # (T, k) -> (k, T): _place always works on the last axis, which is
                # what makes "same helper as codes" literally true.
                _place(t_val[b, 0].transpose(0, 1),
                       r["teacher_val"][0].transpose(0, 1).to(torch.float16),
                       off_teacher, extent)
                _place(t_idx[b, 0].transpose(0, 1),
                       r["teacher_idx"][0].transpose(0, 1).to(torch.int64),
                       off_teacher, extent)

            # --- KD transition weights: computed pre-delay on **Zone C text only**
            # (Zone A is dense, so every frame there would read as a speech onset),
            # then shifted by the same offset as the teacher it weights.
            for role, key in ((0, "text_self"), (1, "text_other")):
                w = self._transition_weight(r[key])
                _place(kd_frame_weight[b, role], w, off_teacher, extent)

            # The teacher's own footprint, in output coordinates. Used to keep
            # kd_valid off Zone A/B without re-deriving cb0's validity.
            _place(kd_zone_c[b],
                   torch.ones((L,), dtype=torch.bool, device=dev), off_teacher, extent)

            # --- zones. Positions, not streams: Zone A/B are prefix regions of the
            # assembled sequence, so they are indexed from output position 0.
            zone_ids[b, :extent] = ZONE_C
            zone_ids[b, :min(na, extent)] = ZONE_A
            zone_ids[b, min(na, extent):min(na + nb, extent)] = ZONE_B

        # kd_valid is DERIVED from codebook 0's validity, never recomputed --
        # a second derivation of the same quantity is where plan section 7.8 lives.
        # The extra terms are a restriction, not a re-derivation: there is no
        # teacher over Zone A (silence + system prompt) or Zone B (a reference
        # utterance). Shared with the online path; see kd_align.
        kd_valid = derive_kd_valid(stream_valid[:, :, 0], teacher_row, kd_zone_c)

        attention_mask = stream_valid.any(dim=2).any(dim=1) | text_valid

        # Zone A in TEXT coordinates. `zone_ids` marks Zone A at output position 0, but the text
        # stream (and its leading system prompt) is placed at off_text; in the ASR direction
        # (off_text > 0) the two part ways. Shift the Zone A mask by the batch's off_text so the text
        # loss masks the ACTUAL system-prompt positions, not the empty text delay head -- otherwise
        # the real prompt tokens land in ZONE_C and train as weight-1.0 targets (ARCH §7.2). Audio
        # zones stay positional: their Zone A is silence, harmless wherever the delay puts it.
        text_zone_a = zone_ids == ZONE_A
        if off_text:
            shifted = torch.zeros_like(text_zone_a)
            shifted[:, off_text:] = text_zone_a[:, : T - off_text]
            text_zone_a = shifted

        audio_w, text_w = self._loss_weights(
            stream_valid, text_valid, text_tokens, zone_ids, synthetic_user, text_zone_a
        )

        return {
            "codes": codes,
            "role_ids": torch.arange(2, dtype=torch.int64, device=dev).expand(B, 2).contiguous(),
            "text_tokens": text_tokens,
            "stream_valid": stream_valid,
            "text_valid": text_valid,
            "attention_mask": attention_mask,
            "zone_ids": zone_ids,
            "audio_loss_weight": audio_w,
            "text_loss_weight": text_w,
            "teacher_topk_val": t_val,
            "teacher_topk_idx": t_idx,
            "kd_valid": kd_valid,
            "kd_frame_weight": kd_frame_weight,
            "sample_type_id": sample_type_id,
            "lang_id": lang_id,
            "has_teacher": has_teacher,
            "num_frames": num_frames,          # Zone C frames only
            "zone_a_frames": torch.tensor([p[0] for p in prefixes],
                                          dtype=torch.int64, device=dev),
            "zone_b_frames": torch.tensor([p[1] for p in prefixes],
                                          dtype=torch.int64, device=dev),
            "sample_uid": sample_uid,
            "delay_offsets": torch.tensor(off, dtype=torch.int64, device=dev),
            # Tripwire: the data is frame-aligned (input_ids[t] / codes[...,t] are frame t), so the
            # 1-step autoregressive shift is applied at LOSS time, not baked into the data. For text,
            # `train.py:build_loss_fn` owns that shift (logits[:-1] vs input_ids[1:], mirroring
            # ForCausalLMLoss, since forward_with_contract bypasses the model's loss_function). For
            # audio the depth decoder predicts the current delay-patterned frame from hidden[t], so no
            # 1-step shift there -- the acoustic delay pattern already carries audio's time alignment.
            "target_aligned": True,
            # Aliases for the MODEL I/O + BATCH contract (utils/train.py section 3), which names
            # the same two tensors `audio_codes` and `input_ids`. Both consumers of a batch --
            # `forward_with_contract` and `build_loss_fn` -- gate on those names, so without these
            # a real collator batch raises KeyError at the first forward. The alias is added here
            # rather than renaming the originals because `codes`/`text_tokens` are what the whole
            # datasets package and its tests speak, and the two names now denote the SAME tensor
            # object (no copy, no divergence possible).
            "audio_codes": codes,
            "input_ids": text_tokens,
        }

    # ------------------------------------------------------- zone A/B assembly
    def _prefix_lens(self, r: KDSample) -> tuple[int, int]:
        """(zone_a_frames, zone_b_frames) for one row, from config + the row.

        Nothing is read from the row's own keys here beyond `sample_type` and the
        reference the Dataset looked up: the zones are a property of the *run*, not
        of the corpus, which is why nothing about them is stored.
        """
        cfg = self.cfg
        st = r["sample_type"]
        na = len(cfg.system_prompt_ids) if st in cfg.zone_a_sources else 0
        nb = 0
        if st in cfg.voice_prompt_sources and bool(r["has_ref"]):
            nb = int(r["ref_codes"].shape[1])
        return na, nb

    def _silence(self, n: int, dev) -> torch.Tensor:
        """(K, n) int64 Mimi silence, tiled from the bank the Dataset uses.

        Phase 0, deterministically: the Dataset randomizes the phase of the *user*
        fill so "no user" is not a memorizable constant, but Zone A is a fixed
        prefix on both channels and there is nothing for the model to shortcut.
        """
        bank = torch.as_tensor(
            self.cfg.tokens.silence_bank_array(), dtype=torch.int64, device=dev
        )                                                  # (K, P)
        if n <= 0:
            return bank[:, :0]
        idx = torch.arange(n, device=dev) % bank.shape[1]
        return bank[:, idx]

    def _assemble(self, r: KDSample, na: int, nb: int, dev) -> dict[str, torch.Tensor]:
        """[Zone A | Zone B | Zone C] per stream, still delay-free.

        Zone A (ARCHITECTURE section 7.2): system prompt text one token per frame
        -- section 7.1's "dense" region -- over silence on BOTH channels.
        Zone B (section 7.4): the reference utterance on the agent channel *with*
        its aligned transcript, silence on the user channel. Including the
        transcript is the whole point: it makes Zone B structurally identical to
        generation, so Zone C is a continuation rather than a mode switch.
        """
        cs = r["codes_self"].to(torch.int64)
        co = r["codes_other"].to(torch.int64)
        ts = r["text_self"].to(torch.int64)
        if na == 0 and nb == 0:
            return {"codes_self": cs, "codes_other": co, "text_self": ts}

        sil_a = self._silence(na, dev)
        pre_self = [sil_a]
        pre_text = [torch.as_tensor(self.cfg.system_prompt_ids, dtype=torch.int64, device=dev)]
        if nb:
            pre_self.append(r["ref_codes"].to(torch.int64).to(dev))
            pre_text.append(r["ref_text"].to(torch.int64).to(dev))
        return {
            "codes_self": torch.cat(pre_self + [cs], dim=1),
            # user channel is silence across BOTH prefix zones
            "codes_other": torch.cat([self._silence(na + nb, dev), co], dim=1),
            "text_self": torch.cat(pre_text + [ts]),
        }

    # ------------------------------------------------------------------ parts
    @staticmethod
    def _assert_row_shapes(r: KDSample, L: int, K: int) -> None:
        uid = r["sample_uid"]
        assert tuple(r["codes_self"].shape) == (K, L), (
            f"{uid}: codes_self {tuple(r['codes_self'].shape)} != {(K, L)}"
        )
        # The Dataset silence-fills the absent user channel; we assert the shape
        # rather than filling again, so a regression there fails here loudly.
        assert tuple(r["codes_other"].shape) == (K, L), (
            f"{uid}: codes_other {tuple(r['codes_other'].shape)} != {(K, L)} -- the "
            f"Dataset is responsible for silence-filling the absent user channel"
        )
        assert tuple(r["text_self"].shape) == (L,), f"{uid}: text_self != (T,)"
        assert tuple(r["text_other"].shape) == (L,), f"{uid}: text_other != (T,)"
        # Zone B: the reference must be frame-aligned to its own transcript, or the
        # voice prompt and its inner monologue drift against each other (section 7.4).
        R = int(r["ref_codes"].shape[-1])
        assert tuple(r["ref_codes"].shape) == (K, R), (
            f"{uid}: ref_codes {tuple(r['ref_codes'].shape)} != {(K, R)}"
        )
        assert tuple(r["ref_text"].shape) == (R,), (
            f"{uid}: ref_text {tuple(r['ref_text'].shape)} is not frame-aligned to "
            f"ref_codes (R={R})"
        )
        if bool(r["has_teacher"]):
            kk = int(r["topk"])
            assert tuple(r["teacher_val"].shape) == (K, L, kk), f"{uid}: teacher_val shape"
            assert tuple(r["teacher_idx"].shape) == (K, L, kk), f"{uid}: teacher_idx shape"

    def _transition_weight(self, text: torch.Tensor) -> torch.Tensor:
        """(L,) fp32 KD weight in **pre-delay** coordinates. See kd_align.

        Config binding only -- the rule itself is shared with the online path, so
        it lives in kd_align rather than here.
        """
        cfg, tok = self.cfg, self.cfg.tokens
        return transition_weight(
            text,
            pad_id=tok.text_pad_id,
            epad_id=tok.text_epad_id,
            weight=cfg.kd_transition_weight,
            halfwidth=cfg.kd_transition_halfwidth,
        )

    def _loss_weights(self, stream_valid, text_valid, text_tokens, zone_ids, synthetic_user,
                      text_zone_a):
        """Multiplicative, fixed order, floored by the validity mask.

        Batch padding is not a separate weight term -- it is stream_valid's 0.0
        floor, so no downstream factor can accidentally resurrect a pad position.
        Normalization (sum(w*ce)/sum(w)) belongs to the loss: doing it here makes
        per-signal gradient-norm monitoring uninterpretable.
        """
        cfg = self.cfg
        not_zone_a = (zone_ids != ZONE_A).float()          # (B, T), position-0 coords -> audio only

        w_audio = stream_valid.float()
        w_audio *= not_zone_a[:, None, None, :]
        w_audio[:, :, 1:, :] *= cfg.w_nonsemantic_audio
        if cfg.mask_synthetic_user:
            w_audio[:, 1] *= (~synthetic_user).float()[:, None, None]

        # The text stream sits at off_text, so its Zone A (the system prompt -- REAL tokens, unlike the
        # audio's silence) is masked with `text_zone_a`, the Zone A mask shifted into text coordinates,
        # NOT the position-0 `zone_ids` above. In the ASR direction (off_text > 0) they disagree, and
        # the position-0 mask would leave the prompt as a weight-1.0 target (ARCH §7.2). In the TTS
        # direction off_text == 0, so `text_zone_a` equals `zone_ids == ZONE_A` and nothing changes.
        w_text = text_valid.float() * (~text_zone_a).float()
        tok = cfg.tokens
        w_text = torch.where(text_tokens == tok.text_pad_id, w_text * cfg.w_stream_pad_text, w_text)
        w_text = torch.where(text_tokens == tok.text_epad_id, w_text * cfg.w_stream_epad_text, w_text)
        return w_audio, w_text

    # -------------------------------------------------------------- debug
    def debug_alignment_check(self, batch: dict, rows: list[KDSample]) -> dict[int, float]:
        """Plan section 3.9: structural round-trip + the semantic shift scan.

        The shift scan is the real plan section 7.8 detector. A stored sample token was
        drawn from that frame's logits, so it must appear in that frame's teacher
        top-k. topk_dump=32 against gen_top_k=250 means the hit rate is not 1.0,
        hence the assert is on the **argmax over relative shift**, not on an
        absolute threshold. It catches a hardcoded o0, a normalization skipped
        under negative text delay, a crop off-by-one, and a double shift.
        """
        off = [int(x) for x in batch["delay_offsets"]]
        codes, sv = batch["codes"], batch["stream_valid"]
        K = self.K

        # (a) structural round trip. Deliberately re-derives the offset from
        # `off` instead of calling kd_align.teacher_offset: a check that reuses
        # the implementation it checks cannot catch that implementation.
        for b, r in enumerate(rows):
            L = int(r["num_frames"])
            na, nb = self._prefix_lens(r)
            P = na + nb
            total = P + L
            extent = total + max(off) if self.cfg.edge_mode == "extend" else total
            for k in range(K):
                o = off[k + 1] + P                    # Zone C starts after the prefix
                n = min(L, extent - o)
                for role, key in ((0, "codes_self"), (1, "codes_other")):
                    got = codes[b, role, k, o:o + n]
                    want = r[key][k, :n].to(torch.int64)
                    assert torch.equal(got, want), (
                        f"round-trip failed at b={b} role={role} k={k} off={o}, offsets={off}"
                    )
                # validity spans the whole assembled row, i.e. from the delay head
                # to the end of Zone C -- the prefix is real data, not padding.
                assert not sv[b, :, k, :off[k + 1]].any(), f"head of k={k} must be invalid"
                assert not sv[b, :, k, o + n:].any(), f"tail of k={k} must be invalid"
        assert_kd_valid_subset(batch["kd_valid"], sv)

        # (b) semantic shift scan, self side only. Batch-dict-only by design, so
        # the online path runs the identical detector on live-scored batches.
        rates = shift_scan(
            codes[:, 0, 0],
            batch["teacher_topk_idx"][:, 0],
            batch["kd_valid"][:, 0],
            shifts=self.cfg.shift_scan,
        )
        assert_aligned(rates, self.cfg.min_kd_hit_rate, context=f"delay_offsets={off}")
        return rates
