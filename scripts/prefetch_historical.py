"""Pre-fetch pitcher handedness and team batting-vs-hand splits for all
historical starters (2026 → 2025 → 2024), saving to disk cache as we go.

Run once before retraining so the model has real values for these features
instead of defaults.
"""
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetcher import (
    get_season_schedule,
    get_pitcher_handedness,
    get_team_batting_vs_hand,
    _pitcher_hand_cache,
    _PITCHER_HAND_CACHE_FILE,
    _save_json_cache,
    _load_json_cache,
)

BATCH_SIZE = 10
YEARS = [2026, 2025, 2024]


def flush_hand_cache():
    """Write entire in-memory hand cache to disk in one shot."""
    _save_json_cache(_PITCHER_HAND_CACHE_FILE, dict(_pitcher_hand_cache))


def prefetch_year(year: int):
    print(f"\n=== {year} ===")
    schedule = get_season_schedule(year)
    completed = [g for g in schedule if g.get('status') == 'Final']

    # Unique pitchers (home + away), preserving order most-recent-first
    seen_pitchers: set = set()
    pitchers: list[str] = []
    for g in sorted(completed, key=lambda x: x['game_date'], reverse=True):
        for side in ('home_probable_pitcher', 'away_probable_pitcher'):
            raw = g.get(side, '')
            name = str(raw).strip() if raw and str(raw) != 'nan' else ''
            if name and name.lower() not in seen_pitchers:
                seen_pitchers.add(name.lower())
                pitchers.append(name)

    # Unique team IDs
    team_ids: set[int] = set()
    for g in completed:
        if g.get('home_id'):
            team_ids.add(int(g['home_id']))
        if g.get('away_id'):
            team_ids.add(int(g['away_id']))

    print(f"  {len(pitchers)} unique starters, {len(team_ids)} teams")

    # --- Pitcher handedness ---
    already_cached = sum(1 for p in pitchers if p.lower().strip() in _pitcher_hand_cache)
    to_fetch = [p for p in pitchers if p.lower().strip() not in _pitcher_hand_cache]
    print(f"  Handedness: {already_cached} cached, {len(to_fetch)} to fetch")

    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i:i + BATCH_SIZE]
        for name in batch:
            result = get_pitcher_handedness(name)
            sys.stdout.write(f"    {name}: {result}\n")
            sys.stdout.flush()
        flush_hand_cache()
        print(f"  [{i + len(batch)}/{len(to_fetch)}] hand cache saved")
        if i + BATCH_SIZE < len(to_fetch):
            time.sleep(0.5)  # gentle rate-limit pause between batches

    # --- Team batting vs hand ---
    from src.fetcher import _team_batting_splits_cache, _TEAM_BATTING_SPLITS_CACHE_FILE, Path as _Path
    disk_file = _Path(str(_TEAM_BATTING_SPLITS_CACHE_FILE).replace('{year}', str(year)))

    already_team = sum(1 for tid in team_ids if (tid, year) in _team_batting_splits_cache)
    to_fetch_teams = [tid for tid in sorted(team_ids) if (tid, year) not in _team_batting_splits_cache]
    print(f"  Batting splits: {already_team} cached, {len(to_fetch_teams)} to fetch")

    for i in range(0, len(to_fetch_teams), BATCH_SIZE):
        batch = to_fetch_teams[i:i + BATCH_SIZE]
        for tid in batch:
            result = get_team_batting_vs_hand(tid, year)
            sys.stdout.write(f"    team {tid}: vs_lhp={result['vs_lhp']} vs_rhp={result['vs_rhp']}\n")
            sys.stdout.flush()
        # Flush team splits to disk
        all_disk = _load_json_cache(disk_file)
        for tid in batch:
            key = (tid, year)
            if key in _team_batting_splits_cache:
                all_disk[str(tid)] = _team_batting_splits_cache[key]
        _save_json_cache(disk_file, all_disk)
        print(f"  [{i + len(batch)}/{len(to_fetch_teams)}] team splits saved")
        if i + BATCH_SIZE < len(to_fetch_teams):
            time.sleep(0.5)


if __name__ == '__main__':
    t0 = time.time()
    for year in YEARS:
        prefetch_year(year)
    print(f"\nDone in {time.time() - t0:.1f}s")
    print("Re-run src/app.py retrain to use the new cached values.")
