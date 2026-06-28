"""Backfill TRUE closing moneylines for past MLB games from The Odds API
historical endpoint, into data/market_closing/ — credit-optimized for a single
paid month, then cancel. Thin CLI over ``src.closing_backfill`` (shared with the
app's launch-time self-heal).

Credit model (verified): historical = 10 credits per region per market, one
request = a whole snapshot. We request only h2h + us, cluster each day's games by
start time (one snapshot per start-block near its close, capped per day), are
resumable (manifest of finished dates), and stop on --max-credits or the live
x-requests-remaining header.

ALWAYS run --estimate first (free; statsapi schedule, no Odds API calls).

  python -m scripts.backfill_closing_odds --start 2024-03-01 --end 2026-06-28 --estimate
  ODDS_API_KEY=... python -m scripts.backfill_closing_odds --start 2024-03-01 --end 2026-06-28
"""
import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src import closing_backfill as cb  # noqa: E402

_DATA = _REPO / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tolerance-min", type=int, default=60)
    ap.add_argument("--max-snapshots-per-day", type=int, default=4)
    ap.add_argument("--max-credits", type=int, default=19000)
    ap.add_argument("--regions", default="us")
    ap.add_argument("--estimate", action="store_true", help="dry run; no API calls / credits")
    args = ap.parse_args()

    if args.estimate:
        e = cb.estimate(args.start, args.end, args.tolerance_min, args.max_snapshots_per_day)
        print(f"Game-days: {e['game_days']}   snapshots: {e['snapshots']}   "
              f"credits: {e['credits']}  (cap {args.max_snapshots_per_day}/day, "
              f"tol {args.tolerance_min}m)")
        for mo in sorted(e["months"]):
            print(f"  {mo}: {e['months'][mo]:>4} snapshots = {e['months'][mo]*10:>5} credits")
        print(f"\nFits in 20,000/mo? {'YES' if e['credits'] <= 20000 else 'NO — raise tolerance or lower cap/range'}")
        return

    if not os.environ.get("ODDS_API_KEY"):
        print("ODDS_API_KEY not set — refusing to run live. Set the key, or use "
              "--estimate. Free key: https://the-odds-api.com")
        return
    cb.backfill_range(_DATA, args.start, args.end, tolerance_min=args.tolerance_min,
                      max_snapshots_per_day=args.max_snapshots_per_day,
                      max_credits=args.max_credits, regions=args.regions, log=print)


if __name__ == "__main__":
    main()
