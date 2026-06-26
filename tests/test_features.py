import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.features import build_game_features, FEATURE_COLUMNS

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
