import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from src.features import FEATURE_COLUMNS

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "data"


def build_training_data(games_with_features: pd.DataFrame) -> pd.DataFrame:
    """Filter to completed games and ensure target columns exist."""
    df = games_with_features.copy()
    completed = df[df['status'] == 'Final'].copy()
    completed = completed.dropna(subset=['home_score', 'away_score'])
    completed['home_win'] = (completed['home_score'] > completed['away_score']).astype(int)
    return completed


def train_models(training_df: pd.DataFrame, model_dir: Path = None) -> dict:
    """Train XGBoost win probability + run total models. Returns dict of models."""
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = build_training_data(training_df) if 'home_win' not in training_df.columns else training_df
    X = df[FEATURE_COLUMNS].fillna(0).values

    win_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1,
    )
    win_model.fit(X, df['home_win'].values)

    runs_home_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    runs_home_model.fit(X, df['home_score'].values.astype(float))

    runs_away_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    runs_away_model.fit(X, df['away_score'].values.astype(float))

    models = {'win': win_model, 'home_runs': runs_home_model, 'away_runs': runs_away_model}

    # Safety: pickle files are written and read only by this codebase — not sourced from
    # user input or network. Joblib serialization is acceptable here.
    joblib.dump(win_model, model_dir / 'model_win.pkl')
    joblib.dump(runs_home_model, model_dir / 'model_runs_home.pkl')
    joblib.dump(runs_away_model, model_dir / 'model_runs_away.pkl')
    return models


def load_models(model_dir: Path = None) -> dict:
    """Load trained models from disk.

    Safety: these pkl files are written by train_models() in this same codebase,
    never sourced from user input or network. Joblib/pickle is acceptable here.
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)
    return {
        'win': joblib.load(model_dir / 'model_win.pkl'),
        'home_runs': joblib.load(model_dir / 'model_runs_home.pkl'),
        'away_runs': joblib.load(model_dir / 'model_runs_away.pkl'),
    }


def models_exist(model_dir: Path = None) -> bool:
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)
    return all(
        (model_dir / f).exists()
        for f in ['model_win.pkl', 'model_runs_home.pkl', 'model_runs_away.pkl']
    )


def predict_game(features: dict, models: dict) -> dict:
    """Predict win probability and expected run totals for one game."""
    X = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
    home_win_prob = float(models['win'].predict_proba(X)[0][1])
    pred_home_runs = float(max(0.0, models['home_runs'].predict(X)[0]))
    pred_away_runs = float(max(0.0, models['away_runs'].predict(X)[0]))
    return {
        'home_win_prob': home_win_prob,
        'away_win_prob': 1.0 - home_win_prob,
        'predicted_home_runs': round(pred_home_runs, 2),
        'predicted_away_runs': round(pred_away_runs, 2),
    }
