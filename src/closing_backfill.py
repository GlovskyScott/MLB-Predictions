"""Reusable historical closing-line backfill (paid Odds API *historical* endpoint).

Shared by ``scripts/backfill_closing_odds.py`` (CLI, with ``--estimate``) and the
app's launch-time self-heal, which fetches the *previous days'* closes that aren't
captured yet (you can only get a true close after the game ends, and the free
``/odds`` feed drops finished games — so a past close needs the paid historical
endpoint). Manifest + store live under the passed ``data_dir`` so it stays
test-isolatable, and a resumable manifest means re-runs never re-spend credits.
"""
import json
import os
from datetime import datetime, timedelta

import statsapi

from src import predictions as _pred
from src.fetcher import get_historical_moneylines, market_key


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
            if g.get("game_type") != "R":             # regular season only
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
        clusters, cur, anchor = [], [], None
        for g in games:
            t = _parse(g[0])
            if anchor is None or (t - anchor) <= timedelta(minutes=tol):
                anchor = anchor or t
                cur.append(g)
            else:
                clusters.append(cur)
                cur, anchor = [g], t
        if cur:
            clusters.append(cur)
        if len(clusters) <= cap or tol > 24 * 60:
            return [(c[0][0], c) for c in clusters]   # snapshot ts = earliest commence
        tol = int(tol * 1.5)


def _manifest_path(data_dir):
    return _pred._closing_odds_dir(data_dir) / "_backfilled.json"


def load_manifest(data_dir) -> set:
    try:
        return set(json.loads(_manifest_path(data_dir).read_text()))
    except Exception:
        return set()


def save_manifest(data_dir, done: set) -> None:
    p = _manifest_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(done)))


def estimate(start: str, end: str, tolerance_min: int = 60,
             max_snapshots_per_day: int = 4) -> dict:
    """Free dry run: snapshots/credits the range would cost (no API calls)."""
    by_date = games_by_date(start, end)
    game_days = sorted(d for d, g in by_date.items() if g)
    months, total = {}, 0
    for d in game_days:
        n = len(cluster(by_date[d], tolerance_min, max_snapshots_per_day))
        total += n
        months[d[:7]] = months.get(d[:7], 0) + n
    return {"game_days": len(game_days), "snapshots": total,
            "credits": total * 10, "months": months}


def backfill_range(data_dir, start: str, end: str, *, tolerance_min: int = 60,
                   max_snapshots_per_day: int = 4, max_credits: int = 19000,
                   regions: str = "us", log=lambda *a: None) -> dict:
    """Backfill real closing lines for [start, end] into data_dir's closing store.

    Needs ODDS_API_KEY (paid historical endpoint) — a no-op without it. Resumable
    via the manifest (done dates skipped → re-runs only fetch new/missing days, so
    steady state is ~yesterday for the launch-time self-heal). Returns
    {'used': credits, 'captured': dates_with_lines}.
    """
    if not os.environ.get("ODDS_API_KEY"):
        log("ODDS_API_KEY not set — skipping closing-line backfill.")
        return {"used": 0, "captured": 0}
    by_date = games_by_date(start, end)
    done = load_manifest(data_dir)
    used = captured = 0
    for d in sorted(dd for dd, g in by_date.items() if g):
        if d in done:
            continue
        day_snap, day_failed = {}, False
        for snap_ts, members in cluster(by_date[d], tolerance_min, max_snapshots_per_day):
            if used + 10 > max_credits:
                if day_snap:
                    _pred.save_closing_odds(data_dir, d, day_snap)
                save_manifest(data_dir, done)
                log(f"hit max_credits={max_credits}; stopping at {d} (resumable)")
                return {"used": used, "captured": captured}
            lines, remaining, ok = get_historical_moneylines(snap_ts, regions=regions)
            used += 10
            if not ok:
                day_failed = True
                continue
            for _ct, gid, away, home in members:
                mk = lines.get(market_key(away, home))
                if mk:
                    day_snap[gid] = {"ml_home": mk["ml_home"], "ml_away": mk["ml_away"],
                                     "book": mk.get("book")}
            if remaining is not None and remaining < 50:
                if day_snap:
                    _pred.save_closing_odds(data_dir, d, day_snap)
                if not day_failed:
                    done.add(d)
                save_manifest(data_dir, done)
                log(f"quota low (remaining={remaining}); stopping at {d} (resumable)")
                return {"used": used, "captured": captured}
        if day_snap:
            _pred.save_closing_odds(data_dir, d, day_snap)
            captured += 1
        if not day_failed and (day_snap or not by_date[d]):
            done.add(d)
    save_manifest(data_dir, done)
    log(f"closing backfill done: used {used} credits, captured {captured} dates")
    return {"used": used, "captured": captured}
