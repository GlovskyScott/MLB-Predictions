import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.features import (
    build_game_features, FEATURE_COLUMNS,
    _shrink_pitcher_stats, _PITCHER_DEFAULTS, _PITCHER_SHRINK_IP,
)

_NEW_FETCHER_MOCKS = {
    'src.features.get_pitcher_splits': lambda name, year: {},
    'src.features.get_game_lineup': lambda gid: {},
    'src.features.get_pitcher_days_rest': lambda name, date, year: 5,
    'src.features.get_team_recent_runs': lambda tid, date, year: 4.5,
    'src.features.get_bullpen_stress_l3': lambda tid, date, year, cache_only=False: 3.0,
    'src.features.get_pitcher_handedness': lambda name: 'R',
    'src.features.get_team_batting_vs_hand': lambda tid, year: {'vs_lhp': 0.730, 'vs_rhp': 0.730},
}


def _all_mocks(sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather):
    return {
        'src.features.get_pitching_stats': lambda year: sample_pitching_stats,
        'src.features.get_batting_stats': lambda year, **kw: sample_batting_stats,
        'src.features.get_team_batting_stats': lambda year, **kw: sample_team_batting,
        'src.features.get_bullpen_stats': lambda year, **kw: sample_pitching_stats.head(1),
        'src.features.get_weather_for_game': lambda **kw: sample_weather,
        'src.features.get_stadium': lambda tid: {'lat': 40.83, 'lon': -73.93, 'roof': 'open'},
        **_NEW_FETCHER_MOCKS,
    }


def _run(sample_game, sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather):
    mocks = _all_mocks(sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather)
    with patch.multiple('src.features', **{k.replace('src.features.', ''): v for k, v in mocks.items()}):
        return build_game_features(sample_game, year=2026, weather=sample_weather)


def test_build_game_features_returns_dict(sample_game, sample_pitching_stats,
                                          sample_batting_stats, sample_team_batting,
                                          sample_weather):
    result = _run(sample_game, sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather)
    assert isinstance(result, dict)


def test_build_game_features_has_all_feature_columns(sample_game, sample_pitching_stats,
                                                      sample_batting_stats, sample_team_batting,
                                                      sample_weather):
    result = _run(sample_game, sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather)
    for col in FEATURE_COLUMNS:
        assert col in result, f"Missing feature: {col}"


def test_build_game_features_values_are_numeric(sample_game, sample_pitching_stats,
                                                sample_batting_stats, sample_team_batting,
                                                sample_weather):
    result = _run(sample_game, sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather)
    for col in FEATURE_COLUMNS:
        assert isinstance(result[col], (int, float, np.floating, np.integer)), \
            f"Feature {col} is not numeric: {type(result[col])}"


def test_build_game_features_no_nans(sample_game, sample_pitching_stats,
                                     sample_batting_stats, sample_team_batting,
                                     sample_weather):
    result = _run(sample_game, sample_pitching_stats, sample_batting_stats, sample_team_batting, sample_weather)
    for col in FEATURE_COLUMNS:
        assert not np.isnan(result[col]), f"Feature {col} is NaN"


# ---- pitcher-stat shrinkage (regress rate stats to league avg by IP) --------

def test_shrink_zero_ip_collapses_to_league_average():
    # A TBD / no-data starter (ip=0) must read as exactly the league-average prior,
    # not whatever placeholder rate stats came along.
    sp = {'era': 9.82, 'fip': 12.0, 'xfip': 8.0, 'whip': 2.46,
          'k9': 4.9, 'bb9': 6.0, 'hr9': 8.4, 'ip': 0.0}
    out = _shrink_pitcher_stats(dict(sp))
    for key in ('era', 'fip', 'xfip', 'whip', 'k9', 'bb9', 'hr9'):
        assert out[key] == pytest.approx(_PITCHER_DEFAULTS[key])
    assert out['ip'] == 0.0  # IP itself is not a rate; left untouched


def test_shrink_small_sample_pulled_toward_prior():
    # 3.2-IP spot starter with a 12.00 FIP must not be taken at face value.
    sp = {'era': 9.82, 'fip': 12.0, 'xfip': 8.0, 'whip': 2.46,
          'k9': 4.9, 'bb9': 6.0, 'hr9': 8.4, 'ip': 3.2}
    out = _shrink_pitcher_stats(dict(sp))
    # exact empirical-Bayes formula: (ip*raw + K*prior) / (ip + K)
    K = _PITCHER_SHRINK_IP
    expected_fip = (3.2 * 12.0 + K * _PITCHER_DEFAULTS['fip']) / (3.2 + K)
    assert out['fip'] == pytest.approx(expected_fip)
    assert out['fip'] < 6.0                      # strongly regressed
    assert out['fip'] > _PITCHER_DEFAULTS['fip']  # but still above average


def test_shrink_large_sample_stays_near_raw():
    # A full-season workload (190 IP) keeps most of its own signal.
    sp = {'era': 2.50, 'fip': 2.60, 'xfip': 3.0, 'whip': 0.95,
          'k9': 11.0, 'bb9': 1.5, 'hr9': 0.6, 'ip': 190.0}
    out = _shrink_pitcher_stats(dict(sp))
    assert out['fip'] == pytest.approx(2.6, abs=0.4)   # close to raw 2.60
    assert out['fip'] < _PITCHER_DEFAULTS['fip']       # ace stays below average
