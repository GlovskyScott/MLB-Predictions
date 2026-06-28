"""Real-closing-line CLV backtest on reconstructed past seasons.

Walk-forward trains+sims the model WITHIN a past era (so it's in-distribution),
fits an in-era win calibrator on those honest pairs, then bets the calibrated
model vs the de-vigged REAL closing line (SportsbookReviewsOnline) and reports
CLV/ROI. This is the methodologically correct test of "does the model beat a true
MLB close" — on a large sample the current store can't give us.

Prereqs:
  python -m scripts.build_historical_features 2019 2021   # -> backtest_features_2019_2021.parquet
  python -m scripts.load_sbr_closing 2019 2021            # -> sbr_closing_2019_2021.parquet

Run:  python -m scripts.clv_backtest_historical --years 2019 2021 [--nsim 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src import calibration as cal  # noqa: E402
from src.fetcher import devig_home_prob  # noqa: E402
from scripts.build_calibrator import walk_forward_pairs  # noqa: E402
from scripts.clv_backtest import evaluate  # noqa: E402

_DATA = _REPO / "data"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["game_date"] = df["game_date"].astype(str)
    df["month"] = df["game_date"].str[:7]
    return df.sort_values("game_date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2019, 2021])
    ap.add_argument("--nsim", type=int, default=200)
    ap.add_argument("--min-train", type=int, default=1000)
    ap.add_argument("--edge", type=float, default=0.03)
    args = ap.parse_args()
    tag = "_".join(str(y) for y in args.years)

    feats = _prep(pd.read_parquet(_DATA / f"backtest_features_{tag}.parquet"))
    print(f"Loaded {len(feats)} games {feats['game_date'].min()}..{feats['game_date'].max()}")

    raw, out, gids, dates = walk_forward_pairs(feats, args.nsim, args.min_train)

    # In-era calibration (fit on the honest walk-forward pairs, like build_calibrator).
    calib = cal.fit(raw, out)
    print(f"\nIn-era win calibrator: a={calib.a:.3f} (slope), b={calib.b:+.3f}")

    # Real closing lines, keyed by game_id.
    sbr = pd.read_parquet(_DATA / f"sbr_closing_{tag}.parquet")
    close = {int(r.game_id): (r.ml_home, r.ml_away) for r in sbr.itertuples(index=False)}

    rows, n_match = [], 0
    for r, y, g in zip(raw, out, gids):
        if g is None:
            continue
        c = close.get(int(g))
        if not c:
            continue
        mkt = devig_home_prob(c[0], c[1])
        if mkt is None:
            continue
        n_match += 1
        rows.append({"model": float(calib(float(r))), "market": float(mkt),
                     "ml_home": c[0], "ml_away": c[1], "home_win": int(y)})

    print(f"\nWalk-forward OOS games: {len(raw)}   joined to a REAL closing line: {n_match} "
          f"({100*n_match/max(len(raw),1):.0f}%)")
    if not rows:
        print("No games joined a closing line."); return

    mwin = np.mean([(x["model"] > 0.5) == (x["home_win"] == 1) for x in rows])
    kwin = np.mean([(x["market"] > 0.5) == (x["home_win"] == 1) for x in rows])
    print(f"Straight-up pick accuracy:  model {mwin:.1%}   CLOSING-favorite {kwin:.1%}")
    print("(a real close grades ~57-58%; if 'closing-favorite' is in that band the "
          "benchmark is legit)")

    for e in sorted({args.edge, 0.0, 0.05, 0.08}):
        evaluate(rows, e)


if __name__ == "__main__":
    main()
