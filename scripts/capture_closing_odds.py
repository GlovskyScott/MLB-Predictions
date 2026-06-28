"""Capture today's CLOSING moneylines from The Odds API into the closing store.

Run this NEAR first pitch (e.g. a cron at the time of the earliest game) so the
recorded line is a closing-ish number. It upserts — running again later overwrites
each game with the fresher line. Accumulates forward into data/market_closing/;
the real-closing-line backtest reads it when enough games pile up.

Setup (one-time): get a free key (no card) at https://the-odds-api.com and export
  ODDS_API_KEY=...   Without it this is a graceful no-op.

Cron example (run hourly during the afternoon/evening so each game gets a
near-first-pitch snapshot):
  0 12-23 * * *  cd /path/to/repo && .venv/bin/python -m scripts.capture_closing_odds

Run:  python -m scripts.capture_closing_odds [YYYY-MM-DD]
"""
import os
import sys
from datetime import date as _date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src import predictions as _pred  # noqa: E402
from src.fetcher import get_closing_moneylines, get_schedule, market_key  # noqa: E402

_DATA = _REPO / "data"


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else _date.today().strftime("%Y-%m-%d")
    if not os.environ.get("ODDS_API_KEY"):
        print("ODDS_API_KEY not set — no-op. Get a free key at https://the-odds-api.com")
        return
    lines = get_closing_moneylines()
    if not lines:
        print("No lines returned (no upcoming games, or API error/limit).")
        return
    schedule = get_schedule(day)
    snap, matched = {}, 0
    for g in schedule:
        mk = lines.get(market_key(g.get("away_name", ""), g.get("home_name", "")))
        if mk:
            snap[g["game_id"]] = {"ml_home": mk["ml_home"], "ml_away": mk["ml_away"],
                                  "book": mk.get("book")}
            matched += 1
    _pred.save_closing_odds(_DATA, day, snap)
    books = {v.get("book") for v in snap.values()}
    print(f"{day}: {len(lines)} games priced, {matched} matched to schedule, "
          f"saved/updated {len(snap)} (books: {sorted(b for b in books if b)})")
    print(f"  store: {_pred._closing_odds_dir(_DATA) / (day + '.json')}")


if __name__ == "__main__":
    main()
