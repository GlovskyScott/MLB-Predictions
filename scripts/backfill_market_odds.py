"""Backfill the write-once market-odds (moneyline) store from ESPN's public API.

ESPN's scoreboard/summary returns closing moneylines for completed games, so we
can snapshot history once and grade the blended ("Consensus") line reproducibly.
Run after a retrain (before scripts/build_calibrator.py, which needs these lines
to fit the blender) and as a daily top-up.

Usage:
  python -m scripts.backfill_market_odds [--days N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse
import datetime as dt
from pathlib import Path

from src.fetcher import get_schedule, get_market_odds, market_key
from src import predictions as P

_DATA = Path(__file__).resolve().parent.parent / "data"


def collect_market_odds(date: str, data_dir=_DATA) -> int:
    """Fetch and persist (write-once) the moneylines for one date.

    Returns the number of new games written (0 if already captured or no odds)."""
    odds = get_market_odds(date)
    if not odds:
        return 0
    by_game = {}
    for g in get_schedule(date):
        mk = odds.get(market_key(g.get('away_name', ''), g.get('home_name', '')))
        if mk and mk.get('ml_home') is not None and mk.get('ml_away') is not None:
            by_game[g['game_id']] = {'ml_home': mk['ml_home'], 'ml_away': mk['ml_away']}
    before = len(P.load_market_odds(data_dir, date))
    P.save_market_odds(data_dir, date, by_game)
    return len(P.load_market_odds(data_dir, date)) - before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    if args.start and args.end:
        d0 = dt.date.fromisoformat(args.start)
        d1 = dt.date.fromisoformat(args.end)
    else:
        d1 = dt.date.today()
        d0 = d1 - dt.timedelta(days=args.days)
    total = 0
    d = d0
    while d <= d1:
        ds = d.isoformat()
        try:
            n = collect_market_odds(ds)
            total += n
            print(f"{ds}: +{n} games")
        except Exception as e:
            print(f"{ds}: ERROR {e}")
        d += dt.timedelta(days=1)
    print(f"Done. {total} game lines captured.")


if __name__ == "__main__":
    main()
