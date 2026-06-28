"""Tests for win-probability calibration (src/calibration.py + app wiring)."""
import numpy as np

from src import calibration as cal


def test_identity_calibrator_is_passthrough():
    c = cal.PlattCalibrator(a=1.0, b=0.0)
    assert abs(c(0.5) - 0.5) < 1e-9
    assert abs(c(0.72) - 0.72) < 1e-6


def test_fit_shrinks_overconfident_probs_toward_half():
    # Synthetic overconfident model: it says p but the truth is closer to 0.5.
    rng = np.random.RandomState(0)
    raw = rng.uniform(0.05, 0.95, 4000)
    true = 0.5 + (raw - 0.5) * 0.5          # half as extreme as claimed
    outcomes = (rng.uniform(size=raw.size) < true).astype(int)

    c = cal.fit(raw, outcomes)
    assert c.a < 1.0                          # slope < 1 => shrink toward 0.5
    assert c(0.80) < 0.80 and c(0.80) > 0.5   # pulled in, still a favorite
    assert c(0.20) > 0.20 and c(0.20) < 0.5   # symmetric on the dog side


def test_calibration_is_monotonic_through_half():
    c = cal.fit(*_synthetic())
    # passes 0.5 through (no side flip) and preserves ordering
    assert abs(c(0.5) - 0.5) < 0.03
    assert c(0.45) < 0.5 < c(0.55)
    assert c(0.60) < c(0.70) < c(0.80)


def test_calibrate_pct_handles_none_and_rounds():
    assert cal.calibrate_pct(None, 72.0) == 72.0          # no calibrator -> identity
    c = cal.PlattCalibrator(a=0.5, b=0.0)
    out = cal.calibrate_pct(c, 72.0)
    assert 60.0 < out < 72.0
    assert round(out, 1) == out                            # already rounded


def test_save_load_roundtrip(tmp_path):
    c = cal.PlattCalibrator(a=0.479, b=0.116)
    cal.save(c, tmp_path)
    loaded = cal.load(tmp_path)
    assert loaded is not None
    assert abs(loaded.a - 0.479) < 1e-9 and abs(loaded.b - 0.116) < 1e-9
    assert abs(loaded(0.7) - c(0.7)) < 1e-12


def test_load_missing_returns_none(tmp_path):
    assert cal.load(tmp_path) is None


def test_enrich_applies_calibration_and_keeps_pick(mocker):
    """_calibrate_core shrinks the displayed win% but leaves the favored side."""
    import src.app as app
    mocker.patch('src.app._get_calibrator',
                 return_value=cal.PlattCalibrator(a=0.479, b=0.116))
    core = {'home_win_pct': 72.0, 'away_win_pct': 28.0, 'median_home_score': 5.0}
    out = app._calibrate_core(core)
    assert out['raw_home_win_pct'] == 72.0
    assert out['home_win_pct'] < 72.0                      # calibrated down
    assert out['home_win_pct'] > 50.0                      # still favored -> pick unchanged
    assert round(out['home_win_pct'] + out['away_win_pct'], 1) == 100.0


def _synthetic():
    rng = np.random.RandomState(1)
    raw = rng.uniform(0.05, 0.95, 4000)
    true = 0.5 + (raw - 0.5) * 0.5
    return raw, (rng.uniform(size=raw.size) < true).astype(int)
