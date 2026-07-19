"""Tests for the silence-bank derivation tool (plan section 2.5).

The real Mimi is never loaded here: a 20 s encode plus a checkpoint download is
not a unit test, and the parts that can actually be *wrong* -- edge trimming, the
per-codebook mode, the run-length diagnostic, which probe's frames get shipped --
are pure array code. Encoding is injected through `derive_silence_codes(encode=)`,
which is exactly the seam `--dry-run` uses, so these tests exercise the shipped
path rather than a parallel one.

The tool used to *fail* when a codebook's silence was not a single constant code.
Against the real codec that fires on all 8 codebooks, so the gate is now opt-in
(`--require-constant`) and the shipped artifact is a (K, P) bank of real frames.
Several tests below exist specifically to pin that inversion down.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from data_pipeline.schema import CODEBOOK_SIZE, FRAME_RATE_HZ, NUM_CODEBOOKS, SAMPLE_RATE
from training.tools.derive_silence_codes import (
    DEFAULT_MIMI_CKPT,
    ProbeParams,
    assert_modal_shares,
    codebook_max_runs,
    codebook_modes,
    derive_silence_codes,
    main,
    synthesize_probes,
    trim_edges,
)

K = NUM_CODEBOOKS
TRUE_SILENCE = (11, 22, 33, 44, 55, 66, 77, 88)


# ------------------------------------------------------------- fake encoders


def fake_encode(
    codes_per_cb=TRUE_SILENCE,
    *,
    params: ProbeParams | None = None,
    edge_junk: bool = True,
    noise_frac: float = 0.0,
    seed: int = 0,
):
    """A stub Mimi: constant codes, junk in the frames `trim_edges` must remove.

    `noise_frac` replaces that fraction of the *interior* frames with a different
    code, which is how the modal-share gate gets exercised without a model.
    """
    params = params or ProbeParams()
    rng = np.random.default_rng(seed)

    def encode(wav: np.ndarray) -> np.ndarray:
        n = int(round(len(wav) / params.sample_rate * FRAME_RATE_HZ))
        out = np.repeat(np.asarray(codes_per_cb, dtype=np.int64)[:, None], n, axis=1)
        if noise_frac > 0:
            m = max(1, int(round(n * noise_frac)))
            for k in range(out.shape[0]):
                idx = rng.choice(n, size=m, replace=False)
                out[k, idx] = (out[k, 0] + 1) % CODEBOOK_SIZE
        if edge_junk:
            # Values that would move every mode if trimming silently stopped.
            out[:, : params.trim_frames] = 7
            out[:, -params.trim_frames :] = 9
        return out

    return encode


# ------------------------------------------------------------- probe signals


def test_probes_are_silence_and_quiet_room_tone():
    p = ProbeParams(seconds=2.0)
    probes = synthesize_probes(p)

    assert set(probes) == {"digital_silence", "room_tone"}
    n = int(round(p.seconds * p.sample_rate))
    for name, wav in probes.items():
        assert wav.dtype == np.float32, name
        assert wav.shape == (n,), name
        assert np.abs(wav).max() <= 1.0, name

    assert not probes["digital_silence"].any(), "digital silence must be exactly zero"

    # -60 dBFS is an RMS target; a pure-zero room tone would defeat the point of
    # having a second probe at all.
    rms = float(np.sqrt(np.mean(probes["room_tone"].astype(np.float64) ** 2)))
    assert rms > 0.0
    assert 20 * np.log10(rms) == pytest.approx(-60.0, abs=0.5)


def test_probes_are_deterministic_in_the_seed():
    a = synthesize_probes(ProbeParams(seconds=5.0, seed=3))["room_tone"]
    b = synthesize_probes(ProbeParams(seconds=5.0, seed=3))["room_tone"]
    c = synthesize_probes(ProbeParams(seconds=5.0, seed=4))["room_tone"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_probe_params_rejects_a_probe_too_short_to_trim():
    # 1 s = 12 frames, which cannot survive 10 frames off each end.
    with pytest.raises(AssertionError, match="does not survive trimming"):
        ProbeParams(seconds=1.0, trim_frames=10)


# ------------------------------------------------------------- edge trimming


def test_trim_edges_drops_both_ends():
    codes = np.arange(K * 50, dtype=np.int64).reshape(K, 50)
    out = trim_edges(codes, 10)
    assert out.shape == (K, 30)
    assert np.array_equal(out, codes[:, 10:40])


def test_trim_edges_zero_is_identity():
    codes = np.zeros((K, 5), dtype=np.int64)
    assert np.array_equal(trim_edges(codes, 0), codes)


def test_trim_edges_refuses_to_empty_the_array():
    # 20 frames, 10 off each end -> nothing left. Silently returning an empty
    # array would make codebook_modes' mode a division by zero one frame later.
    with pytest.raises(AssertionError, match="cannot trim"):
        trim_edges(np.zeros((K, 20), dtype=np.int64), 10)
    with pytest.raises(AssertionError, match="cannot trim"):
        trim_edges(np.zeros((K, 19), dtype=np.int64), 10)


def test_trim_edges_rejects_non_2d():
    with pytest.raises(AssertionError, match=r"\(K, T\)"):
        trim_edges(np.zeros((1, K, 40), dtype=np.int64), 5)


# -------------------------------------------------------------------- modes


def test_codebook_modes_constant_input():
    codes = np.repeat(np.asarray(TRUE_SILENCE, dtype=np.int64)[:, None], 100, axis=1)
    modes, shares = codebook_modes(codes)
    assert modes.tolist() == list(TRUE_SILENCE)
    assert shares.tolist() == [1.0] * K


def test_codebook_modes_is_per_codebook_not_global():
    """The mode must be taken along time, independently per codebook.

    A global mode (or an axis mix-up) would return one value for all 8 rows --
    and since the bank is written straight into every solo sample's user
    channel, that mistake is invisible downstream.
    """
    codes = np.zeros((K, 100), dtype=np.int64)
    for k in range(K):
        codes[k] = TRUE_SILENCE[k]
    codes[3, :10] = 999  # a minority intruder in one codebook only
    modes, shares = codebook_modes(codes)
    assert modes.tolist() == list(TRUE_SILENCE)
    assert shares[3] == pytest.approx(0.9)
    assert all(shares[k] == 1.0 for k in range(K) if k != 3)


def test_codebook_modes_share_is_a_fraction_of_frames():
    codes = np.zeros((K, 10), dtype=np.int64)
    codes[:, :7] = 5
    codes[:, 7:] = 6
    modes, shares = codebook_modes(codes)
    assert modes.tolist() == [5] * K
    assert shares.tolist() == [0.7] * K


def test_codebook_modes_rejects_out_of_range_codes():
    codes = np.full((K, 10), CODEBOOK_SIZE, dtype=np.int64)
    with pytest.raises(AssertionError, match="out of range"):
        codebook_modes(codes)


# ---------------------------------------------------------------- run lengths


def test_max_runs_measures_the_longest_plateau_per_codebook():
    """The number that justifies P: a bank shorter than the longest plateau
    chops it and re-injects it at period P."""
    codes = np.zeros((K, 12), dtype=np.int64)
    codes[0] = [1] * 12                       # one run of 12
    codes[1] = [1, 2] * 6                     # never longer than 1
    codes[2] = [3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6]   # longest run 5
    runs = codebook_max_runs(codes)
    assert runs[0] == 12
    assert runs[1] == 1
    assert runs[2] == 5


def test_max_runs_rejects_non_2d():
    with pytest.raises(AssertionError, match=r"\(K, T\)"):
        codebook_max_runs(np.zeros((K,), dtype=np.int64))


# ------------------------------------------------------- modal-share gate


def test_assert_modal_shares_passes_above_threshold():
    assert_modal_shares(np.full(K, 0.95), np.arange(K), 0.9)


def test_assert_modal_shares_names_the_failing_codebook():
    """A low share means silence is a *loop*, not a code. Naming which codebook
    is the whole value of the message: it tells the reader whether one codebook
    needs a stored loop or the checkpoint is simply wrong."""
    shares = np.full(K, 0.99)
    shares[5] = 0.42
    modes = np.asarray(TRUE_SILENCE)

    with pytest.raises(AssertionError) as ei:
        assert_modal_shares(shares, modes, 0.9)

    msg = str(ei.value)
    assert "codebook 5" in msg
    assert "0.420" in msg
    assert f"mode={TRUE_SILENCE[5]}" in msg
    assert "loop" in msg.lower()
    # And it must not name a codebook that passed.
    assert "codebook 4" not in msg


def test_assert_modal_shares_threshold_is_strict():
    # The plan says "> 0.9". Exactly 0.9 is not > 0.9.
    with pytest.raises(AssertionError):
        assert_modal_shares(np.full(K, 0.9), np.arange(K), 0.9)
    assert_modal_shares(np.full(K, 0.9000001), np.arange(K), 0.9)


def test_assert_modal_shares_reports_every_failure_at_once():
    shares = np.full(K, 0.99)
    shares[[0, 7]] = 0.1
    with pytest.raises(AssertionError) as ei:
        assert_modal_shares(shares, np.arange(K), 0.9)
    assert "codebook 0" in str(ei.value) and "codebook 7" in str(ei.value)


# ---------------------------------------------------------- end-to-end (stub)


def bank_encode(pattern, *, params: ProbeParams | None = None, edge_junk: bool = True):
    """A stub Mimi that emits a repeating (K, P0) `pattern` over the interior.

    Unlike `fake_encode` this is *not* constant per codebook, which is the whole
    point: a derivation that collapsed the frames to their mode would still pass
    every constant-input test.
    """
    params = params or ProbeParams()
    pattern = np.asarray(pattern, dtype=np.int64)

    def encode(wav: np.ndarray) -> np.ndarray:
        n = int(round(len(wav) / params.sample_rate * FRAME_RATE_HZ))
        reps = -(-n // pattern.shape[1])
        out = np.tile(pattern, (1, reps))[:, :n].copy()
        if edge_junk:
            out[:, : params.trim_frames] = 7
            out[:, -params.trim_frames :] = 9
        return out

    return encode


def test_derive_ships_the_frames_themselves_not_the_mode():
    """The core inversion: the payload is a bank of real frames.

    With a 3-cycle pattern the mode covers only ~1/3 of the frames, so the old
    contract would have refused to write anything at all. The bank must reproduce
    the interior frames exactly, in order.
    """
    params = ProbeParams(seconds=20.0)
    pattern = np.asarray([[c, c + 1, c + 2] for c in TRUE_SILENCE], dtype=np.int64)
    out = derive_silence_codes(params=params, encode=bank_encode(pattern, params=params))

    bank = np.asarray(out["silence_bank"])
    n_total = int(round(params.seconds * FRAME_RATE_HZ))
    P = n_total - 2 * params.trim_frames
    assert bank.shape == (K, P)
    assert out["bank_period"] == P
    # The interior is the pattern tiled starting at frame trim_frames.
    expected = np.tile(pattern, (1, -(-n_total // 3)))[:, params.trim_frames : n_total - params.trim_frames]
    assert np.array_equal(bank, expected)
    json.dumps(out)  # the payload must actually be serializable


def test_derive_p_is_the_whole_trimmed_probe():
    """P is deliberately not a short window: it must cover the longest plateau.

    A 40-frame plateau inside a 230-frame probe has to survive intact, otherwise
    tiling re-injects it as a sawtooth at period P.
    """
    params = ProbeParams(seconds=20.0)
    n_total = int(round(params.seconds * FRAME_RATE_HZ))
    pattern = np.zeros((K, 80), dtype=np.int64)
    pattern[:, :40] = np.asarray(TRUE_SILENCE)[:, None]
    pattern[:, 40:] = np.asarray(TRUE_SILENCE)[:, None] + 1
    out = derive_silence_codes(params=params, encode=bank_encode(pattern, params=params))

    assert out["bank_period"] == n_total - 2 * params.trim_frames
    assert max(out["max_runs"]) >= 40
    assert out["bank_period"] > max(out["max_runs"]), (
        "P must exceed the longest measured plateau, or tiling creates a sawtooth"
    )


def test_derive_no_longer_fails_on_a_low_modal_share():
    """The measured codec has shares of 0.24-0.55 on every acoustic codebook.

    Failing there is what forced the bank in the first place, so the default path
    must record the share and carry on.
    """
    params = ProbeParams(seconds=20.0)
    encode = fake_encode(params=params, noise_frac=0.4)
    out = derive_silence_codes(params=params, encode=encode)

    assert min(out["modal_shares"]) < params.min_modal_share
    assert out["require_constant"] is False
    assert np.asarray(out["silence_bank"]).shape[0] == K


def test_require_constant_restores_the_old_gate():
    params = ProbeParams(seconds=20.0)
    encode = fake_encode(params=params, noise_frac=0.4)
    with pytest.raises(AssertionError, match="modal share"):
        derive_silence_codes(params=params, encode=encode, require_constant=True)


def test_derive_records_modes_and_runs_as_diagnostics():
    params = ProbeParams(seconds=20.0)
    out = derive_silence_codes(params=params, encode=fake_encode(params=params))

    assert out["modes"] == list(TRUE_SILENCE)
    assert out["modal_shares"] == [1.0] * K
    assert out["max_runs"] == [out["frames_used"]] * K
    assert out["num_codebooks"] == K
    assert out["codebook_size"] == CODEBOOK_SIZE
    assert out["mimi_ckpt_id"] == DEFAULT_MIMI_CKPT
    assert out["probe"]["sample_rate"] == SAMPLE_RATE
    assert out["probe"]["trim_frames"] == 10
    assert set(out["per_probe"]) == {"digital_silence", "room_tone"}
    # Both probes are pooled before the diagnostics are taken.
    assert out["frames_used"] == sum(p["frames_used"] for p in out["per_probe"].values())


def test_derive_writes_both_probes_banks_and_defaults_to_digital_silence():
    """room_tone is kept for comparison, but the shipped fill is digital silence:
    a solo sample's user channel is an absent microphone, not a quiet room."""
    params = ProbeParams(seconds=20.0)
    a = bank_encode(np.asarray(TRUE_SILENCE)[:, None] + np.arange(3), params=params)
    b = bank_encode(np.asarray(TRUE_SILENCE)[:, None] + np.arange(3) + 100, params=params)
    calls: list[int] = []

    def encode(wav):
        calls.append(1)
        return a(wav) if len(calls) == 1 else b(wav)

    out = derive_silence_codes(params=params, encode=encode)
    assert len(calls) == 2, "both probes must be encoded"
    assert out["bank_probe"] == "digital_silence"
    for name in ("digital_silence", "room_tone"):
        assert "silence_bank" in out["per_probe"][name]
    assert out["silence_bank"] == out["per_probe"]["digital_silence"]["silence_bank"]
    assert (
        out["per_probe"]["room_tone"]["silence_bank"]
        != out["per_probe"]["digital_silence"]["silence_bank"]
    )


def test_derive_can_ship_the_room_tone_bank_instead():
    params = ProbeParams(seconds=20.0)
    calls: list[int] = []

    def encode(wav):
        calls.append(1)
        base = 0 if len(calls) == 1 else 100
        return bank_encode(
            np.asarray(TRUE_SILENCE)[:, None] + np.arange(3) + base, params=params
        )(wav)

    out = derive_silence_codes(params=params, bank_probe="room_tone", encode=encode)
    assert out["bank_probe"] == "room_tone"
    assert out["silence_bank"] == out["per_probe"]["room_tone"]["silence_bank"]


def test_derive_rejects_an_unknown_bank_probe():
    with pytest.raises(AssertionError, match="bank_probe"):
        derive_silence_codes(
            params=ProbeParams(seconds=20.0), bank_probe="pink_noise",
            encode=fake_encode(),
        )


def test_derive_trims_before_building_the_bank():
    """The stub writes junk into exactly the frames trimming removes.

    Under the bank contract this is directly observable: the junk values would
    appear at the ends of the bank itself.
    """
    params = ProbeParams(seconds=20.0)
    out = derive_silence_codes(params=params, encode=fake_encode(params=params, edge_junk=True))

    bank = np.asarray(out["silence_bank"])
    assert not np.isin(bank, [7, 9]).any(), "edge junk survived into the bank"
    n_total = int(round(params.seconds * FRAME_RATE_HZ))
    assert out["per_probe"]["digital_silence"]["frames_used"] == n_total - 2 * params.trim_frames


def test_derive_rejects_a_checkpoint_with_the_wrong_codebook_count():
    params = ProbeParams(seconds=20.0)
    encode = fake_encode(tuple(range(K + 4)), params=params)
    with pytest.raises(AssertionError, match="codebooks, expected"):
        derive_silence_codes(params=params, encode=encode)


def test_derive_records_the_ckpt_id_it_was_given():
    """mimi_ckpt_id is the corpus/codec cross-check; it must be the id actually
    used, not the default constant."""
    params = ProbeParams(seconds=20.0)
    out = derive_silence_codes(
        params=params, ckpt_id="someone/other-mimi", encode=fake_encode(params=params)
    )
    assert out["mimi_ckpt_id"] == "someone/other-mimi"


# ------------------------------------------------------------------- the cli


def test_dry_run_loads_no_model_and_writes_nothing(tmp_path, capsys, monkeypatch):
    import training.tools.derive_silence_codes as mod

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("--dry-run must not load Mimi")

    monkeypatch.setattr(mod, "load_mimi", boom)

    out = tmp_path / "mimi_silence.json"
    assert main(["--dry-run", "--output", str(out)]) == 0
    assert not out.exists(), "--dry-run must not write the config"

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["silence_bank"]) == K
    assert payload["modal_shares"] == [1.0] * K


def test_cli_writes_the_payload_to_a_nested_path(tmp_path, capsys, monkeypatch):
    """The real (non-dry-run) path, with only the encoder stubbed.

    `configs/data/` may not exist yet on a fresh clone, so the writer has to
    create it -- otherwise the tool fails after the expensive part.
    """
    import training.tools.derive_silence_codes as mod

    # Stub the model boundary and nothing else: argument parsing, ProbeParams
    # construction, derivation and writing all run for real.
    monkeypatch.setattr(mod, "load_mimi", lambda *a, **k: object())
    monkeypatch.setattr(
        mod, "encode_probe", lambda mimi, wav, device="cpu": fake_encode()(wav)
    )

    out = tmp_path / "does" / "not" / "exist" / "mimi_silence.json"
    assert main(["--output", str(out)]) == 0

    written = json.loads(out.read_text())
    assert [row[0] for row in written["silence_bank"]] == list(TRUE_SILENCE)
    assert written["mimi_ckpt_id"] == DEFAULT_MIMI_CKPT
    # Stdout must be the same payload, so `... | jq` works.
    assert json.loads(capsys.readouterr().out) == written


def test_cli_require_constant_propagates_the_failure(tmp_path, monkeypatch):
    """The gate is opt-in now, but when asked for it must still abort before
    anything is written."""
    import training.tools.derive_silence_codes as mod

    params = ProbeParams()
    monkeypatch.setattr(mod, "load_mimi", lambda *a, **k: object())
    monkeypatch.setattr(
        mod, "encode_probe",
        lambda mimi, wav, device="cpu": fake_encode(params=params, noise_frac=0.4)(wav),
    )

    out = tmp_path / "mimi_silence.json"
    with pytest.raises(AssertionError, match="modal share"):
        main(["--require-constant", "--output", str(out)])
    assert not out.exists()


def test_cli_without_require_constant_writes_a_noisy_bank(tmp_path, monkeypatch):
    """Same input, no flag: it must succeed. This is the real-codec case."""
    import training.tools.derive_silence_codes as mod

    params = ProbeParams()
    monkeypatch.setattr(mod, "load_mimi", lambda *a, **k: object())
    monkeypatch.setattr(
        mod, "encode_probe",
        lambda mimi, wav, device="cpu": fake_encode(params=params, noise_frac=0.4)(wav),
    )

    out = tmp_path / "mimi_silence.json"
    assert main(["--output", str(out)]) == 0
    written = json.loads(out.read_text())
    assert min(written["modal_shares"]) < 0.9
    assert len(written["silence_bank"]) == K


# ------------------------------------------------- the committed artifact


def test_committed_config_matches_the_shipped_contract():
    """configs/data/mimi_silence.json is a checked-in derived artifact.

    It is loaded by TokenConfig at import of any real training config, so a
    shape/codec regression in it is a training-time failure, not a tool failure.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    payload = json.loads((repo / "configs/data/mimi_silence.json").read_text())

    bank = np.asarray(payload["silence_bank"])
    assert bank.shape == (K, payload["bank_period"])
    assert bank.min() >= 0 and bank.max() < CODEBOOK_SIZE
    assert payload["bank_probe"] == "digital_silence"
    assert payload["mimi_ckpt_id"] == "kmhf/hf-moshiko"
    # P must cover the longest plateau actually measured, or tiling is periodic
    # in a way the fill was supposed to avoid.
    assert payload["bank_period"] >= max(payload["max_runs"])
