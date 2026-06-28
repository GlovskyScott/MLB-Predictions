"""Backfill TRUE closing moneylines for past MLB games from The Odds API
historical endpoint, into the data/market_closing/ store — credit-optimized for a
single paid month, then cancel and use the free forward capture.

Credit model (verified): historical = 10 credits per region per market, and one
request returns a whole snapshot of every game at that timestamp. So we:
  * request ONLY h2h + us  -> 10 credits/snapshot
  * CLUSTER each day's games by start time and take one snapshot per cluster
    (it captures every game in that start-block near its close), capped per day
  * are RESUMABLE (a manifest of finished dates) -> re-runs never re-spend
  * stop on a hard --max-credits cap AND on the live x-requests-remaining header

ALWAYS run --estimate first (free; uses the statsapi schedule, no Odds API calls)
to see the exact credit cost before spending.

  python -m scripts.backfill_closing_odds --start 2024-03-01 --end 2026-06-28 --estimate
  ODDS_API_KEY=... python -m scripts.backfill_closing_odds --start 2024-03-01 --end 2026-06-28
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import statsapi

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src import predictions as _pred  # noqa: E402
from src.fetcher import get_historical_moneylines, market_key  # noqa: E402

_DATA = _REPO / "data"
_MANIFEST = _pred._closing_odds_dir(_DATA) / "_backfilled.json"


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def games_by_date(start: str, end: str) -> dict:
    """{date: [(commence_iso, game_id, away_name, home_name), ...]} via statsapi (free)."""
    out: dict = {}
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    cur = s
    while cur <= e:                                   # month-chunked to cut calls
        chunk_end = min(e, (cur.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1))
        for g in statsapi.schedule(start_date=cur.strftime("%Y-%m-%d"),
                                   end_date=chunk_end.strftime("%Y-%m-%d")):
            ct = g.get("game_datetime")
            if not ct or not g.get("game_id"):
                continue
            if g.get("game_type") != "R":         # regular season only (skip spring/exhibition)
                continue
            out.setdefault(g["game_date"], []).append(
                (ct, g["game_id"], g.get("away_name", ""), g.get("home_name", "")))
        cur = chunk_end + timedelta(days=1)
    return out


def cluster(games: list, tol_min: int, cap: int) -> list:
    """Greedy-cluster games by start time; widen tolerance until <= cap clusters.
    Returns [(snapshot_iso, [games_in_cluster]), ...] — snapshot = cluster's earliest start."""
    games = sorted(games, key=lambda x: x[0])
    tol = tol_min
    while True:
        clusters, cur = [], []
        anchor = None
        for g in games:
            t = _parse(g[0])
            if anchor is None or (t - anchor) <= timedelta(minutes=tol):
                if anchor is None:
                    anchor = t
                cur.append(g)
            else:
                clusters.append(cur)
                cur, anchor = [g], t
        if cur:
            clusters.append(cur)
        if len(clusters) <= cap or tol > 24 * 60:
            return [(c[0][0], c) for c in clusters]   # snapshot ts = earliest commence
        tol = int(tol * 1.5)


def _load_manifest() -> set:
    try:
        return set(json.loads(_MANIFEST.read_text()))
    except Exception:
        return set()


def _save_manifest(done: set) -> None:
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(sorted(done)))


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

    by_date = games_by_date(args.start, args.end)
    done = set() if args.estimate else _load_manifest()
    game_days = sorted(d for d, g in by_date.items() if g)

    # ---- estimate (free) ----
    if args.estimate:
        months: dict = {}
        total = 0
        for d in game_days:
            cl = cluster(by_date[d], args.tolerance_min, args.max_snapshots_per_day)
            total += len(cl)
            months[d[:7]] = months.get(d[:7], 0) + len(cl)
        print(f"Game-days: {len(game_days)}   snapshots: {total}   "
              f"credits: {total*10}  (cap {args.max_snapshots_per_day}/day, tol {args.tolerance_min}m)")
        for mo in sorted(months):
            print(f"  {mo}: {months[mo]:>4} snapshots = {months[mo]*10:>5} credits")
        print(f"\nFits in 20,000/mo? {'YES' if total*10 <= 20000 else 'NO — raise tolerance or lower cap/range'}")
        return

    # ---- live backfill ----
    import os
    if not os.environ.get("ODDS_API_KEY"):
        print("ODDS_API_KEY not set — refusing to run live (would mark dates done with no "
              "data). Set the key, or use --estimate. Free key: https://the-odds-api.com")
        return
    used = 0
    for d in game_days:
        if d in done:
            continue
        cl = cluster(by_date[d], args.tolerance_min, args.max_snapshots_per_day)
        day_snap, day_failed = {}, False
        for snap_ts, members in cl:
            if used + 10 > args.max_credits:
                print(f"Hit --max-credits ({args.max_credits}). Stopping at {d}. Resumable.")
                if day_snap:
                    _pred.save_closing_odds(_DATA, d, day_snap)
                _save_manifest(done)
                return
            lines, remaining, ok = get_historical_moneylines(snap_ts, regions=args.regions)
            used += 10
            if not ok:
                day_failed = True
                print(f"  WARN {d} snapshot {snap_ts} failed after retries — will retry on rerun.")
                continue
            for ct, gid, away, home in members:
                mk = lines.get(market_key(away, home))
                if mk:
                    day_snap[gid] = {"ml_home": mk["ml_home"], "ml_away": mk["ml_away"],
                                     "book": mk.get("book")}
            if remaining is not None and remaining < 50:
                print(f"x-requests-remaining={remaining} < 50. Stopping at {d}. Resumable.")
                _pred.save_closing_odds(_DATA, d, day_snap)
                if not day_failed:
                    done.add(d)
                _save_manifest(done)
                return
        if day_snap:
            _pred.save_closing_odds(_DATA, d, day_snap)
        # Mark done only if the whole day succeeded AND captured something — a fully
        # failed day stays un-done so a rerun retries it (no permanent gaps).
        if not day_failed and (day_snap or not by_date[d]):
            done.add(d)
        if len(done) % 20 == 0:
            _save_manifest(done)
            print(f"  {d}: {len(day_snap)} lines  | credits used {used} | dates done {len(done)}")
    _save_manifest(done)
    print(f"DONE. credits used {used}, dates captured {len(done)}.")


if __name__ == "__main__":
    main()
