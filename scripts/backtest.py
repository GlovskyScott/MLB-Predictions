#!/usr/bin/env python3
"""Honest, out-of-sample evaluation of the game models.

The dashboard's "90-day accuracy" is **in-sample**: the models train on every
completed 2024-2026 game and are then graded on those same games. This script
instead does proper temporal validation so you can see the model's real skill:

  * in-sample        train on all games, grade on all games  (the inflated number)
  * temporal-holdout train on earlier years, grade on a held-out test year
  * walk-forward     expanding window: for each month of the test year, train on
                     everything strictly before it, then grade that month

It reports winner accuracy and run-total MAE for each, plus the naive
home-team-always baseline, so the leakage gap is visible and quantified.

Usage:
    python scripts/backtest.py [--test-year 2026] [--rebuild]

Note: features remain season-aggregate (see build_game_features), so a residual
within-season look-ahead persists on the feature side even in the held-out
splits — the model-level leakage (training on the test games) is what this fixes.
Point-in-time features are the next step; this makes their value measurable.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training import build_training_df, _TRAINING_YEARS  # noqa: E402
from src.model import train_models                  # noqa: E402
from src.features import FEATURE_COLUMNS            # noqa: E402
from src.fetcher import get_season_schedule         # noqa: E402
from xgboost import XGBClassifier                   # noqa: E402

_CACHE = _REPO / "data" / "backtest_features.parquet"  # git-ignored

# Leak-free, schedule-only features computed strictly from games before each date.
_PIT_COLUMNS = [
    'home_rs_pg', 'home_ra_pg', 'home_win_pct', 'home_run_diff_pg',
    'away_rs_pg', 'away_ra_pg', 'away_win_pct', 'away_run_diff_pg',
]


def point_in_time_matrix() -> pd.DataFrame:
    """Build a strictly point-in-time feature matrix from the schedule cache.

    For every completed game, the features are each team's season-to-date runs
    scored/allowed per game, win%, and run differential — computed in a single
    forward pass that snapshots the running totals BEFORE the game is added, so
    no game ever sees its own result or any future game. This is the fully
    leak-free baseline (no season-aggregate stats anywhere).
    """
    games = []
    for year in _TRAINING_YEARS:
        for g in get_season_schedule(year):
            if g.get('status') == 'Final' and g.get('home_score') is not None and g.get('away_score') is not None:
                games.append({**g, 'year': year})
    games.sort(key=lambda g: str(g['game_date']))

    agg: dict = {}  # (year, team_id) -> [runs_scored, runs_allowed, wins, games]

    def snap(year, tid):
        rs, ra, w, n = agg.get((year, tid), [0, 0, 0, 0])
        if n == 0:
            return 4.5, 4.5, 0.5, 0.0, 0  # league-ish priors before any games
        return rs / n, ra / n, w / n, (rs - ra) / n, n

    rows = []
    for g in games:
        y, h, a = g['year'], g['home_id'], g['away_id']
        hs, as_ = int(g['home_score']), int(g['away_score'])
        h_rs, h_ra, h_wp, h_rd, h_n = snap(y, h)
        a_rs, a_ra, a_wp, a_rd, a_n = snap(y, a)
        if h_n >= 5 and a_n >= 5:   # need a little history for a fair feature
            rows.append({
                'game_date': str(g['game_date']), 'year': y,
                'home_win': 1 if hs > as_ else 0,
                'home_rs_pg': h_rs, 'home_ra_pg': h_ra, 'home_win_pct': h_wp, 'home_run_diff_pg': h_rd,
                'away_rs_pg': a_rs, 'away_ra_pg': a_ra, 'away_win_pct': a_wp, 'away_run_diff_pg': a_rd,
            })
        # update running totals AFTER snapshotting
        for tid, rs_for, rs_against, won in ((h, hs, as_, hs > as_), (a, as_, hs, as_ > hs)):
            cur = agg.setdefault((y, tid), [0, 0, 0, 0])
            cur[0] += rs_for; cur[1] += rs_against; cur[2] += 1 if won else 0; cur[3] += 1
    return pd.DataFrame(rows)


def walk_forward_pit(df: pd.DataFrame, test_year: int) -> dict:
    """Walk-forward winner accuracy on the leak-free point-in-time features."""
    test = df[df['year'] == test_year].copy()
    if test.empty:
        return {'n': 0}
    test['month'] = test['game_date'].str[:7]
    correct = total = 0
    for month in sorted(test['month'].unique()):
        train = df[df['game_date'] < f'{month}-01']
        month_df = test[test['month'] == month]
        if len(train) < 500 or month_df.empty:
            continue
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric='logloss', random_state=42, n_jobs=-1)
        clf.fit(train[_PIT_COLUMNS].values, train['home_win'].values)
        pred = clf.predict_proba(month_df[_PIT_COLUMNS].values)[:, 1] > 0.5
        correct += int((pred == (month_df['home_win'].values == 1)).sum())
        total += len(month_df)
    if not total:
        return {'n': 0}
    return {'n': total, 'winner_accuracy': round(correct / total * 100, 1), 'runs_mae': None}


def _featurize() -> pd.DataFrame:
    """Build (or load cached) the featurized + labeled game matrix."""
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    print("Building feature matrix for all completed games (slow, ~1 min)...")
    df = build_training_df()
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['year'] = df['game_date'].str[:4].astype(int)
    df.to_parquet(_CACHE)
    return df


def _fit(train_df: pd.DataFrame) -> dict:
    with tempfile.TemporaryDirectory() as tmp:        # don't clobber data/*.pkl
        return train_models(train_df, model_dir=Path(tmp))


def _evaluate(models: dict, test_df: pd.DataFrame) -> dict:
    if test_df.empty:
        return {'n': 0}
    X = test_df[FEATURE_COLUMNS].fillna(0).values
    home_win_prob = models['win'].predict_proba(X)[:, 1]
    pred_home = np.maximum(0.0, models['home_runs'].predict(X))
    pred_away = np.maximum(0.0, models['away_runs'].predict(X))
    actual_win = test_df['home_win'].values
    predicted_home_won = home_win_prob > 0.5
    acc = float((predicted_home_won == (actual_win == 1)).mean()) * 100
    mae = float((np.abs(pred_home - test_df['home_score'].values)
                 + np.abs(pred_away - test_df['away_score'].values)).mean() / 2)
    return {'n': len(test_df), 'winner_accuracy': round(acc, 1), 'runs_mae': round(mae, 2)}


def in_sample(df: pd.DataFrame) -> dict:
    return _evaluate(_fit(df), df)


def temporal_holdout(df: pd.DataFrame, test_year: int) -> dict:
    train = df[df['year'] < test_year]
    test = df[df['year'] == test_year]
    if train.empty or test.empty:
        return {'n': 0, 'note': 'need >=1 earlier year + the test year'}
    return _evaluate(_fit(train), test)


def walk_forward(df: pd.DataFrame, test_year: int) -> dict:
    """Expanding window by month within the test year: train on everything
    strictly before each month, grade that month, then pool the results."""
    test = df[df['year'] == test_year].copy()
    if test.empty:
        return {'n': 0}
    test['month'] = test['game_date'].str[:7]
    correct = total = 0
    abs_err = 0.0
    for month in sorted(test['month'].unique()):
        train = df[df['game_date'] < f'{month}-01']
        month_df = test[test['month'] == month]
        if len(train) < 500 or month_df.empty:
            continue
        r = _evaluate(_fit(train), month_df)
        n = r['n']
        correct += round(r['winner_accuracy'] / 100 * n)
        total += n
        abs_err += r['runs_mae'] * n
    if not total:
        return {'n': 0}
    return {'n': total, 'winner_accuracy': round(correct / total * 100, 1),
            'runs_mae': round(abs_err / total, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-year', type=int, default=2026)
    ap.add_argument('--rebuild', action='store_true', help='rebuild the cached feature matrix')
    args = ap.parse_args()
    if args.rebuild and _CACHE.exists():
        _CACHE.unlink()

    df = _featurize()
    baseline = float((df['home_win'] == 1).mean()) * 100

    print(f"\nGames: {len(df)}  |  years: {sorted(df['year'].unique())}")
    print(f"Naive baseline (home team always): {baseline:.1f}%\n")

    print("Building leak-free point-in-time matrix from the schedule...")
    pit = point_in_time_matrix()

    rows = [
        ("in-sample (what the dashboard shows)", in_sample(df)),
        (f"temporal holdout (test {args.test_year})", temporal_holdout(df, args.test_year)),
        (f"walk-forward (test {args.test_year}, monthly)", walk_forward(df, args.test_year)),
        (f"walk-forward, point-in-time features", walk_forward_pit(pit, args.test_year)),
    ]
    print(f"\n{'split':42} {'n':>6} {'winner acc':>11} {'runs MAE':>9}")
    print("-" * 72)
    for name, r in rows:
        if r.get('n'):
            mae = '—' if r.get('runs_mae') is None else r['runs_mae']
            print(f"{name:42} {r['n']:>6} {r['winner_accuracy']:>10}% {mae:>9}")
        else:
            print(f"{name:42} {'—':>6}  {r.get('note', 'no data')}")
    print("\nReading: in-sample is inflated by training on the test games; the holdout/"
          "\nwalk-forward rows are the model's real skill (features still season-aggregate);"
          "\nthe point-in-time row uses ZERO season-aggregate stats — the fully-honest floor.\n")


if __name__ == '__main__':
    main()
