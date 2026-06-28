import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from src.features import FEATURE_COLUMNS, INNING_FEATURE_COLUMNS, build_inning_feature_row

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "data"

# Hyperparameters for the two run regressors (home_runs / away_runs). These feed
# the simulator, which produces every served number (win%, scores, moneyline
# edge), so they are the highest-impact knob in the system. Selected by
# walk-forward (past-only) cross-validation in scripts/tune_regressors.py against
# out-of-sample run-prediction error, with a locked final-season holdout to guard
# against overfitting the backtest. Changing these changes the pkl bytes and so
# forks a new model version (see CLAUDE.md "versioned prediction store").
#
# max_depth=3 (was 4): walk-forward over 2024-25 cut out-of-sample run MAE from
# 2.488 -> 2.465 (-0.9%), and the same config independently beat depth-4 on the
# untouched 2026 holdout (2.474 -> 2.448, -1.0%). The shallower tree is the more
# regularized — and simpler — model, so it both improves accuracy and reduces
# overfitting. Re-run scripts/tune_regressors.py after any feature change.
RUN_REGRESSOR_PARAMS = {
    'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
}


def build_training_data(games_with_features: pd.DataFrame) -> pd.DataFrame:
    """Filter to completed games and ensure target columns exist."""
    df = games_with_features.copy()
    completed = df[df['status'] == 'Final'].copy()
    completed = completed.dropna(subset=['home_score', 'away_score'])
    completed['home_win'] = (completed['home_score'] > completed['away_score']).astype(int)
    return completed


def train_models(training_df: pd.DataFrame, model_dir: Path = None,
                 params: dict = None) -> dict:
    """Train XGBoost win probability + run total models. Returns dict of models.

    ``params`` overrides the run-regressor hyperparameters (RUN_REGRESSOR_PARAMS
    by default). The win classifier keeps fixed hyperparameters — it is not used
    to serve predictions (the simulator derives win% from the run regressors).
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_params = dict(params) if params is not None else dict(RUN_REGRESSOR_PARAMS)

    df = build_training_data(training_df) if 'home_win' not in training_df.columns else training_df
    X = df[FEATURE_COLUMNS].fillna(0).values

    win_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1,
    )
    win_model.fit(X, df['home_win'].values)

    runs_home_model = XGBRegressor(
        random_state=42, n_jobs=-1, **run_params,
    )
    runs_home_model.fit(X, df['home_score'].values.astype(float))

    runs_away_model = XGBRegressor(
        random_state=42, n_jobs=-1, **run_params,
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


def inning_model_exists(model_dir: Path = None) -> bool:
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    return (Path(model_dir) / 'model_inning.pkl').exists()


def runs_to_bucket(n) -> int:
    """Per-inning run bucket: 0 runs -> 0, exactly 1 -> 1, 2 or more -> 2."""
    n = int(n)
    return 0 if n == 0 else (1 if n == 1 else 2)


def train_inning_model(df: pd.DataFrame, model_dir: Path = None) -> XGBClassifier:
    """Train a per-inning run-bucket classifier (3-class: 0 / 1 / 2+ runs).

    Label column: 'runs_bucket' (0/1/2). Falls back to deriving it from a binary
    'scored' column only if 'runs_bucket' is absent (legacy callers)."""
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = df[INNING_FEATURE_COLUMNS].fillna(df[INNING_FEATURE_COLUMNS].median()).values
    if 'runs_bucket' in df.columns:
        y = df['runs_bucket'].astype(int).values
    else:
        y = df['scored'].astype(int).values  # legacy binary fallback

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=3,
        eval_metric='mlogloss', random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    joblib.dump(model, model_dir / 'model_inning.pkl')
    return model


def load_inning_model(model_dir: Path = None) -> XGBClassifier:
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    return joblib.load(Path(model_dir) / 'model_inning.pkl')


def _combine_inning_dist(home: list, away: list) -> list:
    """Combine two teams' per-inning [P0, P1, P2+] (percent) into the inning's
    total-runs distribution [P(0 total), P(exactly 1), P(2+ total)] under
    independence — "P(any team scores)" framed by combined runs."""
    h = [x / 100.0 for x in home]
    a = [x / 100.0 for x in away]
    c0 = h[0] * a[0]
    c1 = h[0] * a[1] + h[1] * a[0]
    c2 = max(0.0, 1.0 - c0 - c1)
    return [round(c0 * 100, 1), round(c1 * 100, 1), round(c2 * 100, 1)]


def predict_inning_probs(game_feats: dict, inning_model: XGBClassifier) -> dict:
    """Per-inning run-bucket distribution [P0, P1, P2+] (percent) for each team,
    plus the combined per-inning total-runs distribution.

    Returns {'home': 9x[3], 'away': 9x[3], 'combined': 9x[3]}."""
    home_rows = [build_inning_feature_row(game_feats, i, True) for i in range(1, 10)]
    away_rows = [build_inning_feature_row(game_feats, i, False) for i in range(1, 10)]
    X = pd.DataFrame(home_rows + away_rows)[INNING_FEATURE_COLUMNS].fillna(0).values
    probs = inning_model.predict_proba(X)  # (18, 3)
    if probs.shape[1] != 3:
        # A legacy binary inning model — let the caller fall back to the simulator's
        # 3-class distribution rather than emit malformed cells.
        raise ValueError("inning model is not 3-class (0/1/2+); retrain required")
    to_pct = lambda row: [round(float(p) * 100, 1) for p in row]
    home = [to_pct(probs[i]) for i in range(9)]
    away = [to_pct(probs[i]) for i in range(9, 18)]
    combined = [_combine_inning_dist(home[i], away[i]) for i in range(9)]
    return {'home': home, 'away': away, 'combined': combined}


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
