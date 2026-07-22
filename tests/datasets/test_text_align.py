"""Tests for the inner-monologue text alignment (project_amnesty/datasets/mixins.py).

These exist because there were none, and a real bug lived here undetected: the
old `align_uniform` spread tokens with `np.linspace(0, T-1, n)`, so `pos[-1]` was
always `T-1`, and its EPAD guard `if last + 1 < num_frames` was therefore
unsatisfiable for any utterance with two or more tokens. EPAD -- the speech-onset
trigger the whole turn-taking transfer rides on -- was never emitted, while the
collator carried a dedicated `w_stream_epad_text = 1.0` to protect it and the KD
transition detector special-cased it. The machinery ran on a token that was not
in the data.

The semantics being pinned (ARCHITECTURE.md 5.0.1 and 7.6):
  * EPAD is inserted one frame BEFORE a word's onset -- an onset trigger, not an
    end-of-utterance marker.
  * It is inserted per word, which is why its frequency is unlike a once-per-turn
    token such as <|im_start|>.
  * "Utterance complete" needs no marker: the stream just returns to PAD.
"""

from __future__ import annotations

import numpy as np
import pytest

from project_amnesty.datasets.offline.mixins import TextTokCfg, align_timestamps, align_uniform

PAD, EPAD = 900, 901
CFG = TextTokCfg(tokenizer_name="dummy", text_pad_id=PAD, text_epad_id=EPAD)


def words(*sizes: int) -> list[list[int]]:
    """Per-word token id lists with distinct, non-special ids."""
    out, nxt = [], 1
    for n in sizes:
        out.append(list(range(nxt, nxt + n)))
        nxt += n
    return out


def runs_of_tokens(stream: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) spans of real (non-PAD, non-EPAD) tokens."""
    spans, i = [], 0
    while i < len(stream):
        if stream[i] not in (PAD, EPAD):
            j = i
            while j < len(stream) and stream[j] not in (PAD, EPAD):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


# ------------------------------------------------------------- the regression


@pytest.mark.parametrize("n_words,T", [(2, 10), (5, 20), (18, 111), (3, 100), (8, 40)])
def test_epad_is_emitted_at_all(n_words: int, T: int):
    """The exact bug: with >=2 words the old code emitted zero EPADs, always."""
    s = align_uniform(words(*([2] * n_words)), T, CFG)
    assert (s == EPAD).any(), "no EPAD emitted -- the onset trigger is missing from the data"


def test_epad_count_matches_word_count():
    s = align_uniform(words(2, 3, 1, 2), 40, CFG)
    assert int((s == EPAD).sum()) == 4


# ------------------------------------------------------------ onset semantics


def test_every_epad_is_immediately_followed_by_a_token():
    """EPAD means 'a word starts next'. An EPAD followed by PAD would be a
    terminator -- the semantics the old code implemented."""
    s = align_uniform(words(2, 3, 1, 2, 4), 60, CFG)
    for i in np.flatnonzero(s == EPAD):
        assert i + 1 < len(s), "EPAD must not be the final frame"
        assert s[i + 1] not in (PAD, EPAD), f"EPAD at {i} is not followed by a token"


def test_every_word_onset_is_preceded_by_epad():
    s = align_uniform(words(2, 3, 1, 2, 4), 60, CFG)
    spans = runs_of_tokens(s)
    assert len(spans) == 5
    for start, _ in spans:
        assert start > 0 and s[start - 1] == EPAD, f"word at {start} has no EPAD before it"


def test_no_terminal_epad_after_the_last_word():
    """Completion is expressed by returning to PAD, not by a trailing EPAD."""
    s = align_uniform(words(2, 2), 30, CFG)
    last_end = runs_of_tokens(s)[-1][1]
    assert not (s[last_end:] == EPAD).any()
    assert (s[last_end:] == PAD).all()


# ------------------------------------------------------------------- ordering


def test_tokens_keep_their_order_and_are_contiguous_within_a_word():
    ws = words(3, 2)
    s = align_uniform(ws, 30, CFG)
    spans = runs_of_tokens(s)
    for (start, end), w in zip(spans, ws):
        assert s[start:end].tolist() == w


def test_words_do_not_overlap_or_reorder():
    s = align_uniform(words(2, 2, 2, 2), 40, CFG)
    spans = runs_of_tokens(s)
    for (_, e), (s2, _) in zip(spans, spans[1:]):
        assert e <= s2 - 1, "words must stay separated by at least the EPAD frame"


# ------------------------------------------------------------- edge behaviour


def test_empty_input_is_all_pad():
    assert (align_uniform([], 10, CFG) == PAD).all()
    assert (align_uniform([[], []], 10, CFG) == PAD).all()


def test_single_word():
    s = align_uniform(words(3), 10, CFG)
    (start, end), = runs_of_tokens(s)
    assert s[start - 1] == EPAD
    assert s[start:end].tolist() == [1, 2, 3]


def test_last_word_is_not_truncated_when_it_fits():
    """The onset spread must reserve room for the final word's tokens."""
    ws = words(2, 2, 4)
    s = align_uniform(ws, 24, CFG)
    assert runs_of_tokens(s)[-1][1] - runs_of_tokens(s)[-1][0] == 4


def test_overflow_drops_trailing_words_rather_than_corrupting():
    """More speech than frames: keep a valid prefix, never write out of range."""
    s = align_uniform(words(*([3] * 20)), 12, CFG)
    assert len(s) == 12
    assert set(np.unique(s)) - {PAD, EPAD}, "some tokens should still land"
    for i in np.flatnonzero(s == EPAD):
        assert i + 1 >= len(s) or s[i + 1] not in (PAD,)


def test_dtype_and_length():
    s = align_uniform(words(2, 2), 17, CFG)
    assert s.dtype == np.int32 and s.shape == (17,)


# --------------------------------------------------------------- timestamps


class _FakeTok:
    """Splits on nothing; one id per character, offset so ids never hit PAD/EPAD."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text.strip()]


def test_timestamps_puts_epad_before_onset_not_at_word_end():
    """The old code wrote EPAD at int(w['end'] * 12.5) -- a terminator. With a gap
    between words those are different frames, and only the onset one is correct."""
    ws = [{"word": "aa", "start": 0.4, "end": 0.6}, {"word": "bb", "start": 1.6, "end": 1.8}]
    s = align_timestamps(ws, 40, _FakeTok(), CFG)
    spans = runs_of_tokens(s)
    assert len(spans) == 2
    for start, _ in spans:
        assert s[start - 1] == EPAD
    # and nothing sits at the first word's end frame, which is where it used to go
    assert s[int(0.6 * 12.5)] != EPAD


def test_timestamps_respects_frame_bounds():
    ws = [{"word": "aa", "start": 0.0, "end": 0.2}, {"word": "bb", "start": 9.0, "end": 9.5}]
    s = align_timestamps(ws, 10, _FakeTok(), CFG)
    assert len(s) == 10


# ------------------------------------------------- SeqKD retokenization (7.2)
#
# en_kd text arrives in the teacher's Helium vocabulary (PAD=3), and every
# downstream consumer compares against the *student's* PAD. Passing the teacher
# ids through fails silently: PAD never matches, so the quality filter scores
# every dialogue `never_silent` and rejects the entire corpus, en_solo finds zero
# solo windows, and the collator's PAD down-weighting never fires.

SRC_PAD = 3


WORD_INITIAL = 10_000   # id offset marking a word-initial piece (the real '▁')


class _FakeSrcTok:
    """SentencePiece-shaped source tokenizer: one id per character.

    Word-initial characters get +WORD_INITIAL, mirroring the real '▁' prefix
    without needing a 32k vocab. Ids stay positive: negative ids are the absent
    marker the real streams use, and the implementation skips them.
    """

    def convert_ids_to_tokens(self, tid):
        return "▁" + chr(tid - WORD_INITIAL) if tid >= WORD_INITIAL else chr(tid)

    def decode(self, ids):
        return "".join(chr(i - WORD_INITIAL if i >= WORD_INITIAL else i) for i in ids)


def src_stream(words_at: list[tuple[str, int]], num_frames: int) -> np.ndarray:
    """Build a teacher-vocab stream with each word's chars at consecutive frames."""
    s = np.full(num_frames, SRC_PAD, dtype=np.int32)
    for word, onset in words_at:
        ids = [ord(word[0]) + WORD_INITIAL] + [ord(c) for c in word[1:]]
        s[onset:onset + len(ids)] = ids
    return s


def test_retokenize_preserves_length_and_word_onsets():
    from project_amnesty.datasets.offline.mixins import retokenize_frame_aligned

    placed = [("abc", 5), ("de", 20)]
    s = src_stream(placed, 40)
    out = retokenize_frame_aligned(s, _FakeSrcTok(), SRC_PAD, _FakeTok(), CFG)

    assert len(out) == 40
    for word, onset in placed:
        assert out[onset:onset + len(word)].tolist() == [ord(c) for c in word]
        assert out[onset - 1] == EPAD


def test_retokenize_maps_teacher_pad_onto_student_pad():
    """The bug this whole path exists for: without it pad_ratio is 0.0 and the
    quality filter rejects every dialogue as `never_silent`."""
    from project_amnesty.datasets.offline.mixins import retokenize_frame_aligned

    s = src_stream([("hi", 4), ("there", 12)], 40)
    out = retokenize_frame_aligned(s, _FakeSrcTok(), SRC_PAD, _FakeTok(), CFG)

    assert (out == SRC_PAD).sum() == 0, "teacher PAD id leaked into the student stream"
    assert 0.15 < float((out == PAD).mean()) < 0.97, "filter would reject this row"


def test_retokenize_keeps_a_silent_speaker_silent():
    """A collapsed speaker must stay all-PAD, or `always_silent` stops detecting it."""
    from project_amnesty.datasets.offline.mixins import retokenize_frame_aligned

    out = retokenize_frame_aligned(
        np.full(30, SRC_PAD, dtype=np.int32), _FakeSrcTok(), SRC_PAD, _FakeTok(), CFG
    )
    assert (out == PAD).all()


def test_word_grouping_splits_on_word_prefix_not_on_frame_gaps():
    """Multi-token words must survive as one word: splitting them would move the
    later fragments' onsets and fabricate EPAD triggers mid-word."""
    from project_amnesty.datasets.offline.mixins import words_from_frame_stream

    s = src_stream([("abc", 5), ("de", 20)], 40)
    assert words_from_frame_stream(s, _FakeSrcTok(), SRC_PAD) == [("abc", 5), ("de", 20)]


def test_place_words_at_frames_has_no_float_roundtrip_drift():
    """align_timestamps goes through seconds; frame 7 is 6.999... on the way back.
    The frame-indexed entry point must not inherit that."""
    from project_amnesty.datasets.offline.mixins import place_words_at_frames

    for f in range(1, 30):
        s = place_words_at_frames([("a", f)], 40, _FakeTok(), CFG)
        assert s[f] == ord("a"), f"word landed off frame {f}"
