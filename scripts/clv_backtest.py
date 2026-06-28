"""CLV / edge backtest: does the calibrated model beat the market moneyline?

This grades the model the only way that matters for "beating the book": it takes
the **walk-forward, out-of-sample** model win% (the same honest pairs the
calibrator is fit on, via ``build_calibrator.walk_forward_pairs``), applies the
served calibration, and bets it against the de-vigged market line whenever the
disagreement exceeds an edge threshold. It reports realized ROI (flat and
fractional-Kelly), hit-rate, and the mean model-vs-line edge ("CLV proxy"),
bucketed by edge.

Data sources (both already on disk, no network):
  * model probs : the calibrator's walk-forward OOS sim (2024-2026)
  * market line : data/market_odds/<date>.json  (predictions.load_market_odds)

LINE QUALITY CAVEAT — the market_odds store currently holds ESPN historical
lines, which behave like an opening/early number (they grade ~54%, below a true
MLB close ~57-58%). So a positive ROI here is the *weaker* claim "model beats the
early line", not "model beats the close". The line source is pluggable: drop true
closing prices into data/market_odds/<date>.json (same {game_id: {ml_home,
ml_away}} schema) and rerun — nothing else changes. open->close CLV needs two
prices per game, which the current free store does not have.

Run:  python -m scripts.clv_backtest [--nsim 300] [--edge 0.03] [--min-train 1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src import calibration as cal  # noqa: E402
from src import predictions as _pred  # noqa: E402
from src.fetcher import devig_home_prob  # noqa: E402
from scripts.build_calibrator import featurize, walk_forward_pairs  # noqa: E402

_DATA = _REPO / "data"


def _profit(american: float, won: bool) -> float:
    """Profit on a 1-unit stake at American odds (negative stake on a loss)."""
    if won:
        return american / 100.0 if american > 0 else 100.0 / abs(american)
    return -1.0


def _kelly_fraction(p: float, american: float) -> float:
    """Full-Kelly stake fraction for prob p at American odds (0 if no edge)."""
    b = american / 100.0 if american > 0 else 100.0 / abs(american)  # net decimal
    f = (p * (b + 1.0) - 1.0) / b
    return max(0.0, f)


def collect(nsim: int, min_train: int, closing: bool = False):
    """Return list of per-game dicts joining calibrated model prob + market line.

    closing=True reads the real-closing-line store (market_closing/, from
    backfill/capture_closing_odds) instead of the opener-grade market_odds store.
    """
    load_lines = _pred.load_closing_odds if closing else _pred.load_market_odds
    print(f"Line source: {'market_closing (REAL closes)' if closing else 'market_odds (ESPN opener)'}")
    df = featurize()
    print(f"Loaded {len(df)} games {df['game_date'].min()}..{df['game_date'].max()}")
    raw, out, gids, dates = walk_forward_pairs(df, nsim, min_train)

    calib = cal.load(_DATA)
    if calib is None:
        print("WARNING: no calibrator on disk -> using RAW (overconfident) probs")
        cfn = lambda p: p  # noqa: E731
    else:
        cfn = calib
        print(f"Loaded win calibrator: a={calib.a:.3f}, b={calib.b:+.3f}")

    rows, n_oos, n_matched = [], 0, 0
    odds_cache: dict = {}
    for r, y, g, d in zip(raw, out, gids, dates):
        n_oos += 1
        if g is None:
            continue
        try:
            gid = int(g)
        except (TypeError, ValueError):
            continue
        if d not in odds_cache:
            odds_cache[d] = load_lines(_DATA, d)
        o = odds_cache[d].get(str(gid))
        if not o:
            continue
        mkt = devig_home_prob(o.get("ml_home"), o.get("ml_away"))
        if mkt is None:
            continue
        n_matched += 1
        rows.append({
            "date": d, "gid": gid,
            "model": float(cfn(float(r))),   # calibrated home win prob
            "market": float(mkt),            # de-vigged home win prob
            "ml_home": o.get("ml_home"), "ml_away": o.get("ml_away"),
            "home_win": int(y),
        })
    print(f"\nWalk-forward OOS games: {n_oos}   joined to a market line: {n_matched} "
          f"({100*n_matched/max(n_oos,1):.0f}%)")
    return rows


def evaluate(rows, edge_thresh: float, kelly_cap: float = 0.25):
    """Place a bet on whichever side the model favors past the edge threshold."""
    flat_staked = flat_profit = 0.0
    k_staked = k_profit = 0.0
    n_bets = wins = 0
    clv_sum = 0.0
    buckets: dict = {}

    for r in rows:
        # Home-side edge; bet the side the model likes if it clears the threshold.
        edge_home = r["model"] - r["market"]
        if edge_home >= edge_thresh:
            side, p, american, won = "H", r["model"], r["ml_home"], r["home_win"] == 1
            edge = edge_home
        elif -edge_home >= edge_thresh:
            side, p, american, won = "A", 1 - r["model"], r["ml_away"], r["home_win"] == 0
            edge = -edge_home
        else:
            continue
        if american is None:
            continue
        n_bets += 1
        wins += int(won)
        clv_sum += edge
        pr = _profit(float(american), won)
        flat_staked += 1.0
        flat_profit += pr
        kf = min(kelly_cap, _kelly_fraction(p, float(american)) / 4.0)  # quarter-Kelly, capped
        k_staked += kf
        k_profit += kf * pr
        b = buckets.setdefault(round(np.floor(edge * 20) / 20, 3), [0, 0, 0.0])
        b[0] += 1
        b[1] += int(won)
        b[2] += pr

    print(f"\n{'='*64}\nEDGE BACKTEST  (threshold |model-market| >= {edge_thresh:.0%})\n{'='*64}")
    if n_bets == 0:
        print("No bets cleared the edge threshold.")
        return
    print(f"Bets: {n_bets}   Win rate: {wins/n_bets:.1%}   "
          f"Mean model-vs-line edge (CLV proxy): {clv_sum/n_bets:+.1%}")
    print(f"Flat $1 stake:     ROI {flat_profit/flat_staked:+.2%}   "
          f"(profit {flat_profit:+.1f}u on {flat_staked:.0f}u)")
    if k_staked > 0:
        print(f"Quarter-Kelly:     ROI {k_profit/k_staked:+.2%}   "
              f"(profit {k_profit:+.2f}u on {k_staked:.1f}u)")
    print(f"\n  {'edge>=':>7} {'bets':>5} {'win%':>6} {'ROI':>8}")
    for lo in sorted(buckets):
        n, w, pf = buckets[lo]
        print(f"  {lo:>6.0%} {n:>5} {w/n:>5.0%} {pf/n:>+7.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=300, help="sims per game (lower=faster)")
    ap.add_argument("--min-train", type=int, default=1000)
    ap.add_argument("--edge", type=float, default=0.03, help="min |model-market| to bet")
    ap.add_argument("--closing", action="store_true",
                    help="grade vs real closes (market_closing/) instead of the ESPN opener store")
    args = ap.parse_args()

    rows = collect(args.nsim, args.min_train, closing=args.closing)
    if not rows:
        print("\nNo games joined a market line — nothing to grade. "
              "Is data/market_odds/ populated for the OOS dates?")
        return

    # Baseline: how often does the model's favored side simply win?
    agree = sum(1 for r in rows if (r["model"] > 0.5) == (r["market"] > 0.5))
    mwin = np.mean([(r["model"] > 0.5) == (r["home_win"] == 1) for r in rows])
    kwin = np.mean([(r["market"] > 0.5) == (r["home_win"] == 1) for r in rows])
    print(f"\nAll {len(rows)} matched games:")
    print(f"  model picks the market favorite on {agree/len(rows):.0%} of games")
    print(f"  straight-up pick accuracy:  model {mwin:.1%}   market-favorite {kwin:.1%}")

    for e in sorted({args.edge, 0.0, 0.05, 0.08}):
        evaluate(rows, e)


if __name__ == "__main__":
    main()
