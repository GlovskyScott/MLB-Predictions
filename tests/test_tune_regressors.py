"""Tests for the walk-forward run-regressor tuner (scripts/tune_regressors.py).

The heavy orchestration (real XGBoost fits over months of data) is exercised by
a small synthetic smoke test; the decision logic that protects against
overfitting the backtest (the holdout split + the margin-gated selection) is
tested directly because that is where correctness actually matters.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.tune_regressors import (
    REGRESSOR_SEARCH_SPACE,
    time_holdout_split,
    select_best,
    walk_forward_run_mae,
)
from src.features import FEATURE_COLUMNS


def _make_monthly_df(months, per_month=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mo in months:
        for _ in range(per_month):
            row = {col: rng.uniform(0, 5) for col in FEATURE_COLUMNS}
            row['home_score'] = float(rng.poisson(4.5))
            row['away_score'] = float(rng.poisson(4.2))
            row['home_win'] = int(row['home_score'] > row['away_score'])
            row['game_date'] = f'{mo}-15'
            rows.append(row)
    return pd.DataFrame(rows)


# ---- search space -----------------------------------------------------------

def test_search_space_includes_baseline_first():
    names = [name for name, _ in REGRESSOR_SEARCH_SPACE]
    assert names[0] == 'baseline'


def test_search_space_is_small_and_well_formed():
    # Small on purpose: a giant grid would let us overfit the backtest noise.
    assert 2 <= len(REGRESSOR_SEARCH_SPACE) <= 16
    for name, params in REGRESSOR_SEARCH_SPACE:
        assert isinstance(name, str) and name
        assert params['n_estimators'] >= 1
        assert params['max_depth'] >= 1


# ---- holdout split ----------------------------------------------------------

def test_time_holdout_split_partitions_by_date():
    df = _make_monthly_df(['2024-04', '2024-05', '2026-04'])
    search, holdout = time_holdout_split(df, holdout_start='2026-01-01')
    assert (search['game_date'] < '2026-01-01').all()
    assert (holdout['game_date'] >= '2026-01-01').all()
    assert len(search) + len(holdout) == len(df)


def test_time_holdout_split_holdout_can_be_empty():
    df = _make_monthly_df(['2024-04'])
    search, holdout = time_holdout_split(df, holdout_start='2026-01-01')
    assert len(holdout) == 0
    assert len(search) == len(df)


# ---- margin-gated selection (the anti-overfitting guard) --------------------

def test_select_best_keeps_baseline_when_no_config_clears_margin():
    # 'b' is better but only by 0.5%, under a 2% required margin -> keep baseline.
    results = {'baseline': 1.000, 'b': 0.995}
    assert select_best(results, baseline_key='baseline', rel_margin=0.02) == 'baseline'


def test_select_best_adopts_clear_winner():
    results = {'baseline': 1.000, 'b': 0.900, 'c': 0.950}
    assert select_best(results, baseline_key='baseline', rel_margin=0.02) == 'b'


def test_select_best_returns_baseline_when_it_is_outright_best():
    results = {'baseline': 0.900, 'b': 1.000}
    assert select_best(results, baseline_key='baseline', rel_margin=0.02) == 'baseline'


# ---- walk-forward evaluator (smoke) -----------------------------------------

def test_walk_forward_run_mae_is_a_positive_oos_average(tmp_path):
    df = _make_monthly_df(['2024-04', '2024-05', '2024-06', '2024-07'])
    mae = walk_forward_run_mae(
        df, params={'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.1},
        min_train=100,
    )
    # Poisson(4.5)/Poisson(4.2): absolute error per team is order ~1.7 runs.
    assert 0.5 < mae < 4.0
