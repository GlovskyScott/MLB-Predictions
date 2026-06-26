import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.features import build_game_features, FEATURE_COLUMNS

def test_build_game_features_returns_dict(sample_game, sample_pitching_stats,
                                           sample_batting_stats, sample_team_batting,
                                           sample_weather):
    with patch('src.features.get_pitching_stats', return_value=sample_pitching_stats), \
         patch('src.features.get_batting_stats', return_value=sample_batting_stats), \
         patch('src.features.get_team_batting_stats', return_value=sample_team_batting), \
         patch('src.features.get_bullpen_stats', return_value=sample_pitching_stats.head(1)), \
         patch('src.features.get_weather_for_game', return_value=sample_weather), \
         patch('src.features.get_stadium', return_value={'lat': 40.83, 'lon': -73.93, 'roof': 'open'}):
        result = build_game_features(sample_game, year=2026)
    assert isinstance(result, dict)

def test_build_game_features_has_all_feature_columns(sample_game, sample_pitching_stats,
                                                      sample_batting_stats, sample_team_batting,
                                                      sample_weather):
    with patch('src.features.get_pitching_stats', return_value=sample_pitching_stats), \
         patch('src.features.get_batting_stats', return_value=sample_batting_stats), \
         patch('src.features.get_team_batting_stats', return_value=sample_team_batting), \
         patch('src.features.get_bullpen_stats', return_value=sample_pitching_stats.head(1)), \
         patch('src.features.get_weather_for_game', return_value=sample_weather), \
         patch('src.features.get_stadium', return_value={'lat': 40.83, 'lon': -73.93, 'roof': 'open'}):
        result = build_game_features(sample_game, year=2026)
    for col in FEATURE_COLUMNS:
        assert col in result, f"Missing feature: {col}"

def test_build_game_features_values_are_numeric(sample_game, sample_pitching_stats,
                                                 sample_batting_stats, sample_team_batting,
                                                 sample_weather):
    with patch('src.features.get_pitching_stats', return_value=sample_pitching_stats), \
         patch('src.features.get_batting_stats', return_value=sample_batting_stats), \
         patch('src.features.get_team_batting_stats', return_value=sample_team_batting), \
         patch('src.features.get_bullpen_stats', return_value=sample_pitching_stats.head(1)), \
         patch('src.features.get_weather_for_game', return_value=sample_weather), \
         patch('src.features.get_stadium', return_value={'lat': 40.83, 'lon': -73.93, 'roof': 'open'}):
        result = build_game_features(sample_game, year=2026)
    for col in FEATURE_COLUMNS:
        assert isinstance(result[col], (int, float, np.floating, np.integer)), \
            f"Feature {col} is not numeric: {type(result[col])}"

def test_build_game_features_no_nans(sample_game, sample_pitching_stats,
                                      sample_batting_stats, sample_team_batting,
                                      sample_weather):
    with patch('src.features.get_pitching_stats', return_value=sample_pitching_stats), \
         patch('src.features.get_batting_stats', return_value=sample_batting_stats), \
         patch('src.features.get_team_batting_stats', return_value=sample_team_batting), \
         patch('src.features.get_bullpen_stats', return_value=sample_pitching_stats.head(1)), \
         patch('src.features.get_weather_for_game', return_value=sample_weather), \
         patch('src.features.get_stadium', return_value={'lat': 40.83, 'lon': -73.93, 'roof': 'open'}):
        result = build_game_features(sample_game, year=2026)
    for col in FEATURE_COLUMNS:
        assert not np.isnan(result[col]), f"Feature {col} is NaN"
