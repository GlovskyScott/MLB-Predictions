import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.model import (
    build_training_data, train_models, predict_game, load_models,
    RUN_REGRESSOR_PARAMS,
)
from src.features import FEATURE_COLUMNS

def make_synthetic_training_df(n=100):
    np.random.seed(42)
    data = {col: np.random.uniform(0, 5, n) for col in FEATURE_COLUMNS}
    data['home_score'] = np.random.poisson(4.5, n).astype(float)
    data['away_score'] = np.random.poisson(4.2, n).astype(float)
    data['home_win'] = (data['home_score'] > data['away_score']).astype(int)
    data['game_date'] = ['2026-04-01'] * n
    data['status'] = ['Final'] * n
    return pd.DataFrame(data)

def test_train_models_returns_dict_of_models(tmp_path):
    df = make_synthetic_training_df(100)
    models = train_models(df, model_dir=tmp_path)
    assert 'win' in models
    assert 'home_runs' in models
    assert 'away_runs' in models

def test_train_models_saves_pickle_files(tmp_path):
    df = make_synthetic_training_df(100)
    train_models(df, model_dir=tmp_path)
    assert (tmp_path / 'model_win.pkl').exists()
    assert (tmp_path / 'model_runs_home.pkl').exists()
    assert (tmp_path / 'model_runs_away.pkl').exists()

def test_predict_game_returns_probabilities(tmp_path):
    df = make_synthetic_training_df(100)
    models = train_models(df, model_dir=tmp_path)
    features = {col: 1.0 for col in FEATURE_COLUMNS}
    result = predict_game(features, models)
    assert 'home_win_prob' in result
    assert 'away_win_prob' in result
    assert 'predicted_home_runs' in result
    assert 'predicted_away_runs' in result

def test_predict_game_win_probs_sum_to_one(tmp_path):
    df = make_synthetic_training_df(100)
    models = train_models(df, model_dir=tmp_path)
    features = {col: 1.0 for col in FEATURE_COLUMNS}
    result = predict_game(features, models)
    assert abs(result['home_win_prob'] + result['away_win_prob'] - 1.0) < 0.001

def test_predict_game_run_predictions_are_positive(tmp_path):
    df = make_synthetic_training_df(100)
    models = train_models(df, model_dir=tmp_path)
    features = {col: 1.0 for col in FEATURE_COLUMNS}
    result = predict_game(features, models)
    assert result['predicted_home_runs'] >= 0
    assert result['predicted_away_runs'] >= 0

def test_load_models_returns_same_predictions(tmp_path):
    df = make_synthetic_training_df(100)
    models = train_models(df, model_dir=tmp_path)
    features = {col: 1.0 for col in FEATURE_COLUMNS}
    pred1 = predict_game(features, models)
    loaded = load_models(model_dir=tmp_path)
    pred2 = predict_game(features, loaded)
    assert abs(pred1['home_win_prob'] - pred2['home_win_prob']) < 0.001

def test_run_regressor_params_reflects_walk_forward_tuning():
    # max_depth=3 was selected by scripts/tune_regressors.py: it beat depth-4 on
    # both the walk-forward search and the locked 2026 holdout. Guards against an
    # accidental revert to the un-tuned depth-4 config.
    assert RUN_REGRESSOR_PARAMS['n_estimators'] == 200
    assert RUN_REGRESSOR_PARAMS['max_depth'] == 3
    assert RUN_REGRESSOR_PARAMS['learning_rate'] == 0.05


def test_train_models_honors_custom_params(tmp_path):
    # A custom param set must actually reach the run regressors: a 1-tree stump
    # predicts a near-constant, so its run predictions differ sharply from the
    # 200-tree default on the same data.
    df = make_synthetic_training_df(200)
    feats = {col: 3.0 for col in FEATURE_COLUMNS}

    default_models = train_models(df, model_dir=tmp_path / 'a')
    stump_models = train_models(
        df, model_dir=tmp_path / 'b',
        params={'n_estimators': 1, 'max_depth': 1, 'learning_rate': 1.0},
    )

    default_pred = predict_game(feats, default_models)['predicted_home_runs']
    stump_pred = predict_game(feats, stump_models)['predicted_home_runs']
    assert abs(default_pred - stump_pred) > 0.01


def test_build_training_data_filters_completed_games():
    df = pd.DataFrame({
        'status': ['Final', 'Preview', 'Final', 'In Progress'],
        'home_score': [5.0, None, 3.0, 2.0],
        'away_score': [3.0, None, 4.0, 1.0],
        **{col: [1.0] * 4 for col in FEATURE_COLUMNS}
    })
    result = build_training_data(df)
    assert len(result) == 2
    assert all(result['status'] == 'Final')
