import pytest
import numpy as np
from src.simulator import simulate_game, INNING_WEIGHTS

def make_prediction():
    return {
        'home_win_prob': 0.55,
        'away_win_prob': 0.45,
        'predicted_home_runs': 4.5,
        'predicted_away_runs': 3.8,
    }

def test_simulate_game_returns_expected_keys():
    result = simulate_game(make_prediction(), n_simulations=100, seed=42)
    required = {
        'home_win_pct', 'away_win_pct',
        'median_home_score', 'median_away_score',
        'home_innings', 'away_innings',
        'score_distribution', 'predicted_score',
    }
    assert required.issubset(result.keys())

def test_simulate_game_win_pcts_sum_to_one():
    result = simulate_game(make_prediction(), n_simulations=500, seed=42)
    assert abs(result['home_win_pct'] + result['away_win_pct'] - 100.0) < 0.1

def test_simulate_game_innings_has_nine_entries():
    result = simulate_game(make_prediction(), n_simulations=100, seed=42)
    assert len(result['home_innings']) == 9
    assert len(result['away_innings']) == 9

def test_simulate_game_innings_are_non_negative():
    result = simulate_game(make_prediction(), n_simulations=100, seed=42)
    assert all(v >= 0 for v in result['home_innings'])
    assert all(v >= 0 for v in result['away_innings'])

def test_simulate_game_scores_are_positive():
    result = simulate_game(make_prediction(), n_simulations=200, seed=42)
    assert result['median_home_score'] >= 0
    assert result['median_away_score'] >= 0

def test_simulate_game_score_distribution_is_dict():
    result = simulate_game(make_prediction(), n_simulations=100, seed=42)
    dist = result['score_distribution']
    assert isinstance(dist, dict)
    assert 'home' in dist
    assert 'away' in dist
    assert isinstance(dist['home'], list)
    assert isinstance(dist['away'], list)

def test_inning_weights_sum_to_one():
    assert abs(sum(INNING_WEIGHTS) - 1.0) < 1e-6

def test_simulate_game_predicted_score_format():
    result = simulate_game(make_prediction(), n_simulations=100, seed=42)
    score = result['predicted_score']
    assert isinstance(score, str)
    assert '-' in score

def test_simulate_game_is_reproducible():
    r1 = simulate_game(make_prediction(), n_simulations=100, seed=99)
    r2 = simulate_game(make_prediction(), n_simulations=100, seed=99)
    assert r1['home_win_pct'] == r2['home_win_pct']

def test_predicted_score_is_never_a_tie():
    # The headline predicted score must be a real (decided) game, never N-N.
    for seed in range(20):
        s = simulate_game(make_prediction(), n_simulations=200, seed=seed)['predicted_score']
        a, h = s.split('-')
        assert a != h, f"predicted_score was a tie: {s} (seed {seed})"

def test_simulate_game_output_covers_prediction_fields():
    # extract_core silently drops keys it can't find; guard against drift between
    # the simulator output and the persisted PREDICTION_FIELDS.
    from src.predictions import PREDICTION_FIELDS
    out = simulate_game(make_prediction(), n_simulations=100, seed=1)
    game_identity = {'game_id', 'game_date', 'home_id', 'away_id', 'home_name',
                     'away_name', 'venue_id', 'venue_name'}  # supplied by the schedule, not the sim
    sim_fields = set(PREDICTION_FIELDS) - game_identity
    assert sim_fields.issubset(out.keys()), sim_fields - set(out.keys())
