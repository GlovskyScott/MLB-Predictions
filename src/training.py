"""Model training orchestration: build training matrices, train/load, and the
model metadata + retrain trigger. Holds the in-process model caches."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.fetcher import get_season_schedule, get_game_linescore
from src.features import build_game_features, build_inning_feature_row, _NEUTRAL_WEATHER, FEATURE_VERSION
from src.model import (
    train_models, load_models, models_exist,
    train_inning_model, load_inning_model, inning_model_exists,
)

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODEL_META_FILE = _DATA_DIR / "model_meta.json"
_TRAINING_YEARS = [2024, 2025, 2026]

_models_cache: dict = {}
_inning_model_cache = None


def reset_model_caches() -> None:
    """Drop the in-process model objects so the next access reloads/retrains."""
    global _models_cache, _inning_model_cache
    _models_cache = {}
    _inning_model_cache = None


def read_model_meta() -> dict:
    if _MODEL_META_FILE.exists():
        return json.loads(_MODEL_META_FILE.read_text())
    return {}


def write_model_meta(n_games: int) -> None:
    _MODEL_META_FILE.write_text(json.dumps({
        'training_years': _TRAINING_YEARS,
        'n_games': n_games,
        'feature_version': FEATURE_VERSION,
    }))


def needs_retrain() -> bool:
    if not models_exist():
        return True
    meta = read_model_meta()
    if meta.get('training_years') != _TRAINING_YEARS:
        return True
    if meta.get('feature_version') != FEATURE_VERSION:
        return True
    return False


def build_training_df(years: list[int] = None) -> pd.DataFrame:
    if years is None:
        years = _TRAINING_YEARS
    rows = []
    for year in years:
        all_games = get_season_schedule(year)
        completed = [
            g for g in all_games
            if g.get('status') == 'Final' and g.get('home_score') is not None
        ]
        for game in completed:
            try:
                features = build_game_features(game, year=year, weather=_NEUTRAL_WEATHER, for_training=True)
                features['home_score'] = float(game['home_score'])
                features['away_score'] = float(game['away_score'])
                features['home_win'] = 1 if float(game['home_score']) > float(game['away_score']) else 0
                features['status'] = 'Final'
                features['game_date'] = game['game_date']
                rows.append(features)
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_models(force_retrain: bool = False) -> dict | None:
    global _models_cache
    if _models_cache and not force_retrain:
        return _models_cache
    if force_retrain or needs_retrain():
        training_df = build_training_df()
        if training_df.empty:
            return None
        models = train_models(training_df)
        write_model_meta(len(training_df))
        _models_cache = models
        return models
    _models_cache = load_models()
    return _models_cache


def build_inning_training_df(years: list[int] = None) -> pd.DataFrame:
    if years is None:
        years = _TRAINING_YEARS
    rows = []
    for year in years:
        all_games = get_season_schedule(year)
        completed = [
            g for g in all_games
            if g.get('status') == 'Final' and g.get('home_score') is not None
        ]
        # Parallel-fetch all linescores first (populates disk cache)
        game_ids = [int(g['game_id']) for g in completed]
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(get_game_linescore, gid): gid for gid in game_ids}
            for fut in as_completed(futures):
                fut.result()
        for game in completed:
            linescore = get_game_linescore(int(game['game_id']))
            if not linescore:
                continue
            try:
                game_feats = build_game_features(game, year=year, weather=_NEUTRAL_WEATHER, for_training=True)
            except Exception:
                continue
            for inning in range(1, 10):
                for batting_is_home in (True, False):
                    key = 'home' if batting_is_home else 'away'
                    inn_runs = linescore[key][inning - 1]
                    row = build_inning_feature_row(game_feats, inning, batting_is_home)
                    row['scored'] = 1 if inn_runs >= 1 else 0
                    rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_inning_model(force_retrain: bool = False):
    global _inning_model_cache
    if _inning_model_cache and not force_retrain:
        return _inning_model_cache
    if not force_retrain and inning_model_exists():
        _inning_model_cache = load_inning_model()
        return _inning_model_cache
    df = build_inning_training_df()
    if df.empty or len(df) < 1000:
        return None
    _inning_model_cache = train_inning_model(df)
    return _inning_model_cache
