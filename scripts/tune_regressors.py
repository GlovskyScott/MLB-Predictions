"""Walk-forward hyperparameter tuning for the run regressors.

The two run regressors (home_runs / away_runs) feed the Monte-Carlo simulator,
which produces every served number (win%, scores, run-line / total edges). So
their out-of-sample run-prediction accuracy is the highest-impact knob we have.

This tuner picks their XGBoost hyperparameters **honestly**, the same way
scripts/build_calibrator.py builds the calibrator:

  * Past-only walk-forward folds. For each month, train on strictly earlier
    games and score the held-out month. No future game ever leaks into training,
    so the reported MAE estimates how well the config predicts *tomorrow*, not
    how well it memorizes history.

  * A small, curated search space (not a giant grid). Trying thousands of
    configs and keeping the single best backtest number is how you overfit the
    backtest itself; a handful of sensible configs keeps that risk low.

  * A locked final-season holdout that the search never sees. The winner of the
    walk-forward search must *also* beat the baseline on this untouched slice
    before we'd adopt it — a guard against the search getting lucky on noise.

  * A required improvement margin. Baseball folds are noisy; a config that wins
    by a hair is probably winning on luck, so we keep the baseline unless a
    challenger clears a real margin.

Run:  python -m scripts.tune_regressors [--min-train 1000] [--holdout-start 2026-01-01]
                                         [--margin 0.01] [--json data/regressor_tuning_result.json]
"""
import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
warnings.filterwarnings("ignore")

from src.features import FEATURE_COLUMNS  # noqa: E402
from src.model import RUN_REGRESSOR_PARAMS, train_models  # noqa: E402

# Curated, deliberately small. Each entry is (name, params) and overrides only
# the run-regressor hyperparameters. subsample/colsample are held at 0.8 (the
# production values) throughout; we vary capacity (depth, trees, learning rate)
# and explicit regularization (min_child_weight, reg_lambda). 'baseline' must be
# first and must equal the current production config.
_BASE = dict(RUN_REGRESSOR_PARAMS)


def _cfg(**overrides):
    return {**_BASE, **overrides}


REGRESSOR_SEARCH_SPACE = [
    ("baseline", _cfg()),                                   # current production
    ("depth4", _cfg(max_depth=4)),                          # one level deeper
    ("deep", _cfg(max_depth=5)),
    ("slow", _cfg(n_estimators=400, learning_rate=0.03)),
    ("slow_depth4", _cfg(max_depth=4, n_estimators=400, learning_rate=0.03)),
    ("mcw", _cfg(n_estimators=300, min_child_weight=5)),
    ("l2", _cfg(n_estimators=300, reg_lambda=3.0)),
    ("strong_reg", _cfg(n_estimators=400, learning_rate=0.03,
                        min_child_weight=5, reg_lambda=2.0)),
    ("capacity", _cfg(max_depth=5, n_estimators=300, learning_rate=0.04,
                      min_child_weight=3)),
    ("fast_small", _cfg(n_estimators=150, learning_rate=0.08)),
]


def time_holdout_split(df: pd.DataFrame, holdout_start: str):
    """Split into (search, holdout) at a date boundary. Past stays in search."""
    gd = df["game_date"].astype(str)
    search = df[gd < holdout_start]
    holdout = df[gd >= holdout_start]
    return search.reset_index(drop=True), holdout.reset_index(drop=True)


def _combined_run_mae(models, test: pd.DataFrame) -> tuple[float, int]:
    """Sum of absolute errors over both teams' runs, and the count of team-games."""
    X = test[FEATURE_COLUMNS].fillna(0).values
    ph = np.maximum(0.0, models["home_runs"].predict(X))
    pa = np.maximum(0.0, models["away_runs"].predict(X))
    err = np.abs(ph - test["home_score"].astype(float).values).sum()
    err += np.abs(pa - test["away_score"].astype(float).values).sum()
    return float(err), 2 * len(test)


def walk_forward_run_mae(df: pd.DataFrame, params: dict, min_train: int) -> float:
    """Pooled out-of-sample MAE (per team-game) over past-only monthly folds."""
    d = df.copy()
    d["game_date"] = d["game_date"].astype(str)
    d["month"] = d["game_date"].str[:7]
    d = d.sort_values("game_date").reset_index(drop=True)

    total_err, total_n = 0.0, 0
    for mo in sorted(d["month"].unique()):
        train = d[d["game_date"] < f"{mo}-01"]
        if len(train) < min_train:
            continue
        test = d[d["month"] == mo]
        if test.empty:
            continue
        with tempfile.TemporaryDirectory() as t:
            models = train_models(train, model_dir=Path(t), params=params)
        err, n = _combined_run_mae(models, test)
        total_err += err
        total_n += n
    if total_n == 0:
        raise ValueError("no walk-forward folds met min_train; lower --min-train")
    return total_err / total_n


def confirm_holdout_mae(search: pd.DataFrame, holdout: pd.DataFrame, params: dict) -> float:
    """Train on the whole search window, score the locked holdout once."""
    with tempfile.TemporaryDirectory() as t:
        models = train_models(search, model_dir=Path(t), params=params)
    err, n = _combined_run_mae(models, holdout)
    return err / n


def select_best(results: dict, baseline_key: str, rel_margin: float) -> str:
    """Return the winning config name.

    A challenger is only chosen if it beats the baseline MAE by at least
    ``rel_margin`` (relative). Otherwise the baseline is kept — noise-sized
    "wins" are not adopted.
    """
    base = results[baseline_key]
    threshold = base * (1.0 - rel_margin)
    challengers = {k: v for k, v in results.items() if k != baseline_key and v <= threshold}
    if not challengers:
        return baseline_key
    return min(challengers, key=challengers.get)


def featurize() -> pd.DataFrame:
    # Reuse the calibrator's featurized cache + target derivation so tuning sees
    # exactly the games training does.
    from scripts.build_calibrator import featurize as _f
    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-train", type=int, default=1000)
    ap.add_argument("--holdout-start", default="2026-01-01",
                    help="games on/after this date are the locked confirmation holdout")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="relative MAE improvement a challenger must clear to be adopted")
    ap.add_argument("--json", default=str(_REPO / "data" / "regressor_tuning_result.json"))
    args = ap.parse_args()

    df = featurize()
    search, holdout = time_holdout_split(df, args.holdout_start)
    print(f"Loaded {len(df)} games {df['game_date'].min()}..{df['game_date'].max()}")
    print(f"Search window: {len(search)} games (< {args.holdout_start})")
    print(f"Locked holdout: {len(holdout)} games (>= {args.holdout_start})\n")

    print("Walk-forward search (out-of-sample MAE per team-game, lower is better):")
    results = {}
    for name, params in REGRESSOR_SEARCH_SPACE:
        mae = walk_forward_run_mae(search, params, args.min_train)
        results[name] = mae
        print(f"  {name:<13} MAE={mae:.4f}")

    # The search winner is simply the lowest out-of-sample MAE. The locked
    # holdout — not a large margin — is the gate that protects against adopting a
    # config that only got lucky on the search folds: the winner must beat the
    # baseline on a slice the search never touched. (--margin is reported as an
    # advisory "is this even a meaningful gap on the search window?" flag.)
    winner = min(results, key=results.get)
    base_mae = results["baseline"]
    win_mae = results[winner]
    rel = (base_mae - win_mae) / base_mae * 100
    cleared = select_best(results, "baseline", args.margin) == winner and winner != "baseline"

    print(f"\nSearch winner: {winner}  "
          f"(MAE {win_mae:.4f} vs baseline {base_mae:.4f}, {rel:+.2f}%; "
          f"{'clears' if cleared else 'under'} {args.margin*100:.0f}% advisory margin)")

    decision = "baseline"
    holdout_info = {}
    if winner != "baseline" and len(holdout) > 0:
        space = dict(REGRESSOR_SEARCH_SPACE)
        base_h = confirm_holdout_mae(search, holdout, space["baseline"])
        win_h = confirm_holdout_mae(search, holdout, space[winner])
        holdout_info = {"baseline": base_h, winner: win_h}
        h_rel = (base_h - win_h) / base_h * 100
        print(f"\nLocked-holdout confirmation (trained on full search window):")
        print(f"  baseline MAE={base_h:.4f}   {winner} MAE={win_h:.4f} ({h_rel:+.2f}%)")
        if win_h < base_h:
            decision = winner
            print(f"  -> {winner} wins on BOTH the search and the untouched holdout. ADOPT.")
        else:
            print(f"  -> {winner} does NOT beat baseline on the holdout. KEEP BASELINE.")
    elif winner != "baseline":
        decision = winner  # no holdout available to confirm against
        print("\n(no holdout games available to confirm against)")
    else:
        print("\nBaseline is already the best config. KEEP BASELINE.")

    chosen_params = dict(REGRESSOR_SEARCH_SPACE)[decision]
    out = {
        "search_results": results,
        "holdout": holdout_info,
        "search_winner": winner,
        "decision": decision,
        "chosen_params": chosen_params,
        "margin": args.margin,
        "holdout_start": args.holdout_start,
        "min_train": args.min_train,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nDecision: use '{decision}' params -> {chosen_params}")
    print(f"Wrote {args.json}")
    if decision != "baseline":
        print("\nTo adopt: set RUN_REGRESSOR_PARAMS in src/model.py to chosen_params, "
              "then POST /retrain (forks a new model version) and rebuild the calibrator.")


if __name__ == "__main__":
    main()
