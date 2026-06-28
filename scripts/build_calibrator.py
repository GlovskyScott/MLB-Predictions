"""Build the win-probability calibrator from walk-forward predictions.

Trains the run regressors on a past-only expanding window, predicts run totals
for the next month of games, simulates them through the *same* Monte-Carlo the
app serves, and pairs the raw simulated win% with the actual outcome. Those
honest out-of-sample pairs are fit with a Platt calibrator (src/calibration.py)
and saved to data/model_calibrator.pkl.

Run:  python -m scripts.build_calibrator [--nsim 500] [--min-train 1000]
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
warnings.filterwarnings("ignore")

from src import calibration as cal  # noqa: E402
from src.features import FEATURE_COLUMNS, INNING_FEATURE_COLUMNS  # noqa: E402
from src.model import train_models, train_inning_model  # noqa: E402
from src.simulator import simulate_game  # noqa: E402
from src.training import build_training_df, build_inning_training_df  # noqa: E402

_CACHE = _REPO / "data" / "backtest_features.parquet"
_INNING_CACHE = _REPO / "data" / "inning_training.parquet"
_DATA = _REPO / "data"


def featurize() -> pd.DataFrame:
    if _CACHE.exists():
        df = pd.read_parquet(_CACHE)
    else:
        df = build_training_df()
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["game_date"] = df["game_date"].astype(str)
    df["month"] = df["game_date"].str[:7]
    return df.sort_values("game_date").reset_index(drop=True)


def _fit_regressors(train: pd.DataFrame):
    with tempfile.TemporaryDirectory() as t:
        return train_models(train, model_dir=Path(t))


def walk_forward_pairs(df: pd.DataFrame, nsim: int, min_train: int):
    """Yield honest (raw_sim_home_win_prob, actual_home_win) over all months."""
    months = sorted(df["month"].unique())
    raw, out = [], []
    for mo in months:
        train = df[df["game_date"] < f"{mo}-01"]
        if len(train) < min_train:
            continue
        test = df[df["month"] == mo]
        if test.empty:
            continue
        models = _fit_regressors(train)
        X = test[FEATURE_COLUMNS].fillna(0).values
        ph = np.maximum(0.0, models["home_runs"].predict(X))
        pa = np.maximum(0.0, models["away_runs"].predict(X))
        for k, (idx, row) in enumerate(test.iterrows()):
            sim = simulate_game(
                {"predicted_home_runs": float(ph[k]), "predicted_away_runs": float(pa[k])},
                n_simulations=nsim, seed=int(idx) % 100000,
            )
            raw.append(sim["home_win_pct"] / 100.0)
            out.append(int(row["home_win"]))
        print(f"  {mo}: train={len(train):>5}  games={len(test):>3}  cumulative pairs={len(raw)}")
    return np.array(raw), np.array(out)


def inning_featurize() -> pd.DataFrame:
    # Rebuild the cache if it predates the 3-class 'runs_bucket' label.
    if _INNING_CACHE.exists():
        df = pd.read_parquet(_INNING_CACHE)
        if "runs_bucket" not in df.columns:
            df = build_inning_training_df()
            df.to_parquet(_INNING_CACHE)
    else:
        df = build_inning_training_df()
        df.to_parquet(_INNING_CACHE)
    df["game_date"] = df["game_date"].astype(str)
    df["month"] = df["game_date"].str[:7]
    return df.sort_values("game_date").reset_index(drop=True)


def inning_walk_forward_pairs(df: pd.DataFrame, min_train: int):
    """Yield honest (raw [P0,P1,P2+], actual bucket 0/1/2) over all months."""
    months = sorted(df["month"].unique())
    raw, out = [], []
    label_col = "runs_bucket" if "runs_bucket" in df.columns else "scored"
    for mo in months:
        train = df[df["game_date"] < f"{mo}-01"]
        if len(train) < min_train:
            continue
        test = df[df["month"] == mo]
        if test.empty:
            continue
        with tempfile.TemporaryDirectory() as t:
            m = train_inning_model(train, model_dir=Path(t))
        X = test[INNING_FEATURE_COLUMNS].fillna(0).values
        raw.extend(m.predict_proba(X).tolist())   # (n, 3)
        out.extend(test[label_col].astype(int).values)
        print(f"  {mo}: train={len(train):>6}  innings={len(test):>4}  cumulative pairs={len(raw)}")
    return np.array(raw), np.array(out)


def reliability(p, y, label):
    print(f"\n{label}  (n={len(p)}, Brier={np.mean((p - y) ** 2):.3f})")
    print(f"  {'model says':>12} {'n':>6} {'actual win':>11}")
    for lo in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        hi = lo + 0.1 if lo >= 0.3 else 0.3
        m = (p >= lo) & (p < hi)
        if m.sum() > 5:
            print(f"  {lo*100:>4.0f}-{hi*100:<3.0f}% {m.sum():>6} {y[m].mean()*100:>10.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=500)
    ap.add_argument("--min-train", type=int, default=1000)
    args = ap.parse_args()

    df = featurize()
    print(f"Loaded {len(df)} games {df['game_date'].min()}..{df['game_date'].max()}")
    raw, out = walk_forward_pairs(df, args.nsim, args.min_train)

    c = cal.fit(raw, out)
    calibrated = np.array([c(p) for p in raw])

    reliability(raw, out, "BEFORE — raw simulated win% (overconfident)")
    reliability(calibrated, out, "AFTER  — calibrated win%")
    print(f"\nPlatt fit: a={c.a:.3f} (slope; <1 = shrink toward 50%), b={c.b:+.3f}")
    print(f"Example: raw 72% -> {c(0.72)*100:.0f}% | raw 65% -> {c(0.65)*100:.0f}% | "
          f"raw 55% -> {c(0.55)*100:.0f}%")

    path = cal.save(c, _DATA)
    print(f"\nSaved win calibrator -> {path}")

    # ---- 3-class inning run-bucket calibrator (P0 / P1 / P2+) ----
    print("\n" + "=" * 60 + "\nINNING CALIBRATOR (per-inning P 0 / 1 / 2+ runs)\n" + "=" * 60)
    idf = inning_featurize()
    print(f"Loaded {len(idf)} inning rows {idf['game_date'].min()}..{idf['game_date'].max()}")
    iraw, iout = inning_walk_forward_pairs(idf, max(args.min_train * 4, 20000))
    mc = cal.fit_multiclass(iraw, iout)
    for c_idx, label in enumerate(("0 runs", "1 run", "2+ runs")):
        col = iraw[:, c_idx]
        m = mc.maps[c_idx]
        reliability(col, (iout == c_idx).astype(int), f"P({label}) — raw")
        print(f"  Platt fit class {label}: a={m.a:.3f}, b={m.b:+.3f}")
    ipath = cal.save_multiclass(mc, _DATA)
    print(f"Saved 3-class inning calibrator -> {ipath}")


if __name__ == "__main__":
    main()
