"""The PAD/EPAD ids are written down in three files. This pins them together.

configs/data/text_tok.yaml already claims this test exists:

    # MIRROR of configs/tokens.yaml. Kept literal because this file is loaded by a
    # bare yaml.safe_load into a dataclass, which cannot resolve interpolation.
    # tests/test_config_tokens.py asserts the two agree -- drift here means the id
    # used to BAKE the data differs from the id used to WEIGHT the loss, which is
    # invisible: the model just never learns turn boundaries.

It did not. The duplication is deliberate -- text_tok.yaml is read by a plain
yaml.safe_load into a dataclass with no interpolation, and filter.yaml by
another -- but duplication without a pin is just three chances to be wrong.

Why each copy exists, and what a mismatch silently breaks:

  configs/tokens.yaml        the training stack (TokenConfig). Decides which
                             frames get the PAD loss down-weight.
  configs/data/text_tok.yaml the baker (TextTokCfg). Decides which id is written
                             into ko_tts streams and emitted by the SeqKD
                             retokenizer for en_kd.
  configs/data/filter.yaml   the en_kd quality filter. Decides what counts as
                             silence when scoring a dialogue for collapse.

Bake with one id and weight with another and nothing raises: the model just
never learns turn boundaries. Filter with a third and every dialogue scores
`never_silent`, so the en_kd corpus comes out empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TOKENS = ROOT / "configs" / "tokens.yaml"
TEXT_TOK = ROOT / "configs" / "data" / "text_tok.yaml"
FILTER = ROOT / "configs" / "data" / "filter.yaml"


def load(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def tokens() -> dict:
    return load(TOKENS)


@pytest.mark.parametrize("key", ["text_pad_id", "text_epad_id"])
def test_text_tok_mirrors_tokens(tokens: dict, key: str):
    """The baker and the loss must agree, or turn boundaries are never learned."""
    assert load(TEXT_TOK)[key] == tokens[key], (
        f"{key} drifted between {TOKENS.name} and {TEXT_TOK.name}"
    )


def test_filter_pad_mirrors_tokens(tokens: dict):
    """The filter scores the *retokenized* stream, so it needs the student's PAD.

    The teacher's PAD is 3 (Helium). Leaving 3 here -- or any id the retokenizer
    does not emit -- makes pad_ratio 0.0 for every speaker, which trips
    `never_silent` and rejects the entire corpus with a healthy-looking report.
    """
    assert load(FILTER)["text_pad_id"] == tokens["text_pad_id"], (
        f"text_pad_id drifted between {TOKENS.name} and {FILTER.name}"
    )


def test_teacher_pad_is_not_used_as_the_student_pad(tokens: dict):
    """Guards the specific confusion this whole SeqKD path exists to fix."""
    from project_amnesty.datasets.mixins import MOSHI_TEXT_PAD_ID

    assert tokens["text_pad_id"] != MOSHI_TEXT_PAD_ID
    assert load(FILTER)["text_pad_id"] != MOSHI_TEXT_PAD_ID


def test_pad_and_epad_are_distinct(tokens: dict):
    """align_uniform writes EPAD only where the frame still reads PAD; if the two
    ids collide it overwrites word tokens instead of filling silence."""
    assert tokens["text_pad_id"] != tokens["text_epad_id"]


def test_filter_config_loads():
    """filter.yaml is consumed by a bare dataclass ctor -- a stray key raises
    TypeError at ingest time, after the dialogues have been read. Fail here."""
    from project_amnesty.datasets.en_kd_dataset import FilterConfig

    filt = FilterConfig.from_yaml(str(FILTER))
    assert 0.0 < filt.silence_ratio_min < filt.silence_ratio_max < 1.0
    assert filt.min_frames > 0


def test_no_generation_config_lingers():
    """Generation left this package; a stale generation.yaml would read as a
    supported in-repo path and send someone looking for the generator."""
    assert not (ROOT / "configs" / "data" / "generation.yaml").exists()
