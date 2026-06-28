import statsapi
import pybaseball
import pandas as pd
import numpy as np
import requests
import threading
from pathlib import Path
from datetime import date

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

_DATA_RELEASE_TAG = "data-cache"
_DATA_RELEASE_ASSET = "data_cache.tar.gz"
_MODEL_RELEASE_TAG = "latest"
_MODEL_PKLS = ["model_win.pkl", "model_runs_home.pkl", "model_runs_away.pkl", "model_inning.pkl"]
_PREDICTIONS_RELEASE_TAG = "prediction-archive"
_PREDICTIONS_RELEASE_ASSET = "predictions.tar.gz"


def bootstrap_model_cache(repo: str = "jackleh/MLB-Predictions") -> None:
    """Download model pkl files from the latest release into data/ if missing.

    Safe to call repeatedly — no-op if all pkls already exist.

    Security note: the pkls are unpickled (model.load_models -> joblib.load),
    which executes code on load. Only ever point this at the project's own
    trusted releases — never a third-party `repo`.
    """
    if all((_DATA_DIR / p).exists() for p in _MODEL_PKLS):
        return
    import subprocess
    print("Model files not found — downloading from GitHub release...")
    try:
        for pkl in _MODEL_PKLS:
            if not (_DATA_DIR / pkl).exists():
                subprocess.run(
                    ["gh", "release", "download", _MODEL_RELEASE_TAG,
                     "-R", repo, "-D", str(_DATA_DIR), "--pattern", pkl],
                    check=True, capture_output=True,
                )
        print("Model files restored.")
    except Exception as e:
        print(f"Could not download model files ({e}). Will retrain.")


def bootstrap_data_cache(repo: str = "jackleh/MLB-Predictions") -> None:
    """Download and extract the data release if the cache is empty.

    Call this once at startup before any data fetching. Safe to call repeatedly —
    it's a no-op if cached CSVs already exist.
    """
    sentinel = _DATA_DIR / "schedule_2024.csv"
    if sentinel.exists():
        return
    import subprocess, tempfile, tarfile
    print("Data cache not found — downloading from GitHub release...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["gh", "release", "download", _DATA_RELEASE_TAG,
                 "-R", repo, "-D", tmp, "--pattern", _DATA_RELEASE_ASSET],
                check=True, capture_output=True,
            )
            asset_path = Path(tmp) / _DATA_RELEASE_ASSET
            with tarfile.open(asset_path, "r:gz") as tf:
                tf.extractall(_DATA_DIR, filter="data")  # block path traversal
        print("Data cache restored.")
    except Exception as e:
        print(f"Could not download data cache ({e}). Will fetch fresh data instead.")


def bootstrap_predictions_cache(repo: str = "jackleh/MLB-Predictions") -> None:
    """Download and extract the versioned prediction archive if absent.

    Restores data/predictions/ (all model versions + versions.json) so the
    archive pages and historical accuracy work without re-simulating. Safe to
    call repeatedly — a no-op once the local store exists.
    """
    sentinel = _DATA_DIR / "predictions" / "versions.json"
    if sentinel.exists():
        return
    import subprocess, tempfile, tarfile
    print("Prediction archive not found — downloading from GitHub release...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["gh", "release", "download", _PREDICTIONS_RELEASE_TAG,
                 "-R", repo, "-D", tmp, "--pattern", _PREDICTIONS_RELEASE_ASSET],
                check=True, capture_output=True,
            )
            asset_path = Path(tmp) / _PREDICTIONS_RELEASE_ASSET
            with tarfile.open(asset_path, "r:gz") as tf:
                tf.extractall(_DATA_DIR / "predictions", filter="data")  # block path traversal
        print("Prediction archive restored.")
    except Exception as e:
        print(f"Could not download prediction archive ({e}). Predictions will be generated fresh.")


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

_WEATHER_CACHE_FILE = _DATA_DIR / "weather_cache.csv"
_weather_cache: dict = {}  # (date, lat, lon) → weather dict, in-memory layer

_LINESCORE_CACHE_FILE = _DATA_DIR / "linescore_cache.csv"
_linescore_cache: dict = {}  # game_pk → {home: [9 ints], away: [9 ints]}
_linescore_cache_lock = threading.Lock()

_season_schedule_memory: dict = {}  # year → list[dict], prevents repeated CSV reads

_pitcher_splits_cache: dict = {}  # (player_name_lower, year) → splits dict
_lineup_cache: dict = {}  # game_pk → {'home': [names], 'away': [names]}
_days_rest_cache: dict = {}  # (pitcher_name_lower, game_date, year) → int
_recent_runs_cache: dict = {}  # (team_id, game_date, year, n_games) → float
_bullpen_stress_cache: dict = {}  # (team_id, game_date, year) → float
_pitcher_hand_cache: dict = {}  # player_name_lower → 'L' or 'R'
_team_batting_splits_cache: dict = {}  # (team_id, year) → {'vs_lhp': ops, 'vs_rhp': ops}

# Disk cache files for new API lookups (avoids re-fetching on every restart/retrain)
_PITCHER_SPLITS_CACHE_FILE = _DATA_DIR / "pitcher_splits_{year}.json"
_PITCHER_HAND_CACHE_FILE = _DATA_DIR / "pitcher_hand.json"
_TEAM_BATTING_SPLITS_CACHE_FILE = _DATA_DIR / "team_batting_splits_{year}.json"


def _load_json_cache(path: Path) -> dict:
    try:
        import json
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _save_json_cache(path: Path, data: dict) -> None:
    try:
        import json
        path.write_text(json.dumps(data))
    except Exception:
        pass

def _load_weather_cache() -> None:
    global _weather_cache
    if _weather_cache or not _WEATHER_CACHE_FILE.exists():
        return
    df = pd.read_csv(_WEATHER_CACHE_FILE)
    for _, row in df.iterrows():
        # Skip entries missing humidity — they were cached before this field was added;
        # they will be re-fetched and saved with humidity on next access.
        if 'humidity_pct' not in df.columns or pd.isna(row.get('humidity_pct')):
            continue
        key = (str(row['date']), round(float(row['lat']), 2), round(float(row['lon']), 2))
        _weather_cache[key] = {
            'temperature_f': float(row['temperature_f']),
            'wind_speed_mph': float(row['wind_speed_mph']),
            'wind_direction_deg': float(row['wind_direction_deg']),
            'precipitation_mm': float(row['precipitation_mm']),
            'humidity_pct': float(row['humidity_pct']),
            'is_dome': False,
        }


def _save_weather_cache_entry(date: str, lat: float, lon: float, weather: dict) -> None:
    row = pd.DataFrame([{
        'date': date, 'lat': round(lat, 2), 'lon': round(lon, 2),
        'temperature_f': weather['temperature_f'],
        'wind_speed_mph': weather['wind_speed_mph'],
        'wind_direction_deg': weather['wind_direction_deg'],
        'precipitation_mm': weather['precipitation_mm'],
        'humidity_pct': weather.get('humidity_pct', 50.0),
    }])
    row.to_csv(_WEATHER_CACHE_FILE, mode='a', header=not _WEATHER_CACHE_FILE.exists(), index=False)


# ---------------------------------------------------------------------------
# Schedule / lineups
# ---------------------------------------------------------------------------

def get_schedule(game_date: str) -> list[dict]:
    """Fetch regular season MLB schedule for a given date (YYYY-MM-DD)."""
    games_raw = statsapi.schedule(date=game_date)
    games = []
    for g in games_raw:
        if g.get('game_type') != 'R':
            continue
        games.append({
            'game_id': g['game_id'],
            'game_date': g['game_date'],
            'home_id': g['home_id'],
            'away_id': g['away_id'],
            'home_name': g['home_name'],
            'away_name': g['away_name'],
            'venue_id': g.get('venue_id', 0),
            'venue_name': g.get('venue_name', ''),
            'game_datetime': g.get('game_datetime', ''),
            'status': g.get('status', ''),
            'home_probable_pitcher': g.get('home_probable_pitcher', ''),
            'away_probable_pitcher': g.get('away_probable_pitcher', ''),
            'home_score': g.get('home_score'),
            'away_score': g.get('away_score'),
        })
    return games


def get_probable_pitchers(game_id: int, game: dict) -> dict:
    """Extract probable pitcher names from a game dict."""
    return {
        'home_pitcher_name': game.get('home_probable_pitcher', ''),
        'away_pitcher_name': game.get('away_probable_pitcher', ''),
    }


def get_pitcher_splits(player_name: str, year: int) -> dict:
    """Return home/away pitching splits for a pitcher via the MLB Stats API.

    Returns {'home': {era, whip, k9, bb9, hr9, ip}, 'away': {...}}.
    Either or both keys may be absent if data is unavailable.
    """
    if not player_name:
        return {}
    cache_key = (player_name.lower().strip(), year)
    if cache_key in _pitcher_splits_cache:
        return _pitcher_splits_cache[cache_key]

    # Load disk cache on first access for this year
    disk_cache_file = Path(str(_PITCHER_SPLITS_CACHE_FILE).replace('{year}', str(year)))
    disk_key = player_name.lower().strip()
    if not _pitcher_splits_cache:
        disk_data = _load_json_cache(disk_cache_file)
        for k, v in disk_data.items():
            _pitcher_splits_cache[(k, year)] = v
    if cache_key in _pitcher_splits_cache:
        return _pitcher_splits_cache[cache_key]

    def _sf(val):
        try:
            return float(val) if val and str(val) not in ('-.--', '--', '') else None
        except (TypeError, ValueError):
            return None

    try:
        results = statsapi.lookup_player(player_name)
        if not results:
            last = player_name.split()[-1] if ' ' in player_name else ''
            results = statsapi.lookup_player(last) if last else []
        if not results:
            _pitcher_splits_cache[cache_key] = {}
            return {}
        player_id = results[0]['id']
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
            params={'stats': 'homeAndAway', 'group': 'pitching', 'season': year},
            timeout=10,
        )
        resp.raise_for_status()
        out: dict = {}
        for block in resp.json().get('stats', []):
            for s in block.get('splits', []):
                code = s.get('split', {}).get('code', '')
                st = s.get('stat', {})
                key = 'home' if code == 'H' else 'away' if code == 'A' else None
                if not key:
                    continue
                out[key] = {
                    'era': _sf(st.get('era')),
                    'whip': _sf(st.get('whip')),
                    'k9': _sf(st.get('strikeoutsPer9Inn')),
                    'bb9': _sf(st.get('walksPer9Inn')),
                    'hr9': _sf(st.get('homeRunsPer9')),
                    'ip': _sf(st.get('inningsPitched')) or 0.0,
                }
        _pitcher_splits_cache[cache_key] = out
        # Persist to disk so retrain doesn't re-fetch
        disk_cache_file = Path(str(_PITCHER_SPLITS_CACHE_FILE).replace('{year}', str(year)))
        all_disk = _load_json_cache(disk_cache_file)
        all_disk[player_name.lower().strip()] = out
        _save_json_cache(disk_cache_file, all_disk)
        return out
    except Exception:
        _pitcher_splits_cache[cache_key] = {}
        return {}


def get_game_lineup(game_pk: int) -> dict:
    """Return the batting order for a game.

    Returns {'home': [full_names], 'away': [full_names]} or {} if not yet posted.
    """
    if not game_pk:
        return {}
    if game_pk in _lineup_cache:
        return _lineup_cache[game_pk]
    try:
        data = statsapi.get('game', {
            'gamePk': game_pk,
            'fields': 'gameData,players,fullName,liveData,boxscore,teams,home,away,battingOrder',
        })
        box = data.get('liveData', {}).get('boxscore', {}).get('teams', {})
        home_order = box.get('home', {}).get('battingOrder', [])
        away_order = box.get('away', {}).get('battingOrder', [])
        if not home_order and not away_order:
            return {}
        players_data = data.get('gameData', {}).get('players', {})

        def _name(pid):
            return players_data.get(f'ID{pid}', {}).get('fullName', '')

        result = {
            'home': [n for pid in home_order if (n := _name(pid))],
            'away': [n for pid in away_order if (n := _name(pid))],
        }
        _lineup_cache[game_pk] = result
        return result
    except Exception:
        return {}


def get_pitcher_days_rest(pitcher_name: str, game_date: str, year: int) -> int:
    """Return days since pitcher's last start. Returns 5 (normal rest) if unknown."""
    if not pitcher_name or not game_date:
        return 5
    cache_key = (pitcher_name.lower().strip(), game_date, year)
    if cache_key in _days_rest_cache:
        return _days_rest_cache[cache_key]
    try:
        schedule = get_season_schedule(year)
        pitcher_lower = pitcher_name.lower().strip()
        prev_starts = [
            g['game_date'] for g in schedule
            if g.get('game_date', '') < game_date
            and (g.get('home_probable_pitcher', '').lower().strip() == pitcher_lower
                 or g.get('away_probable_pitcher', '').lower().strip() == pitcher_lower)
        ]
        if not prev_starts:
            _days_rest_cache[cache_key] = 5
            return 5
        from datetime import date as _d
        rest = (_d.fromisoformat(game_date) - _d.fromisoformat(max(prev_starts))).days
        result = max(1, min(rest, 15))
        _days_rest_cache[cache_key] = result
        return result
    except Exception:
        _days_rest_cache[cache_key] = 5
        return 5


def get_team_recent_runs(team_id: int, game_date: str, year: int, n_games: int = 15) -> float:
    """Return average runs scored per game across the last n_games completed games."""
    if not game_date:
        return 4.5
    cache_key = (team_id, game_date, year, n_games)
    if cache_key in _recent_runs_cache:
        return _recent_runs_cache[cache_key]
    try:
        schedule = get_season_schedule(year)
        recent = sorted(
            [g for g in schedule
             if (g.get('home_id') == team_id or g.get('away_id') == team_id)
             and g.get('game_date', '') < game_date
             and g.get('status') == 'Final'
             and g.get('home_score') is not None],
            key=lambda x: x['game_date'], reverse=True,
        )[:n_games]
        if not recent:
            result = 4.5
        else:
            runs = [
                float(g['home_score'] if g.get('home_id') == team_id else g['away_score'])
                for g in recent
            ]
            result = round(sum(runs) / len(runs), 3)
    except Exception:
        result = 4.5
    # Cache every outcome (incl. the default) so repeated lookups don't recompute.
    _recent_runs_cache[cache_key] = result
    return result


def get_bullpen_stress_l3(team_id: int, game_date: str, year: int,
                          cache_only: bool = False) -> float:
    """Return total late-inning (7-9) runs allowed in the last 3 games.

    Proxy for bullpen workload: a high value means the bullpen was used heavily.
    Pass cache_only=True to avoid network calls (for training).
    """
    if not game_date:
        return 3.0
    cache_key = (team_id, game_date, year)
    if cache_key in _bullpen_stress_cache:
        return _bullpen_stress_cache[cache_key]
    try:
        schedule = get_season_schedule(year)
        recent = sorted(
            [g for g in schedule
             if (g.get('home_id') == team_id or g.get('away_id') == team_id)
             and g.get('game_date', '') < game_date
             and g.get('status') == 'Final'],
            key=lambda x: x['game_date'], reverse=True,
        )[:3]
        total = 0.0
        for g in recent:
            linescore = get_game_linescore(int(g['game_id']), cache_only=cache_only)
            if not linescore:
                total += 1.0
                continue
            is_home = g.get('home_id') == team_id
            opp_runs = linescore['away'] if is_home else linescore['home']
            total += sum(opp_runs[i] for i in range(6, 9) if i < len(opp_runs))
        result = round(total, 2)
        _bullpen_stress_cache[cache_key] = result
        return result
    except Exception:
        return 3.0


def get_pitcher_handedness(player_name: str) -> str:
    """Return 'L' or 'R' for the pitcher's throwing hand. Defaults to 'R'."""
    if not player_name:
        return 'R'
    cache_key = player_name.lower().strip()
    if cache_key in _pitcher_hand_cache:
        return _pitcher_hand_cache[cache_key]
    # Load disk cache on first miss
    if not _pitcher_hand_cache:
        for k, v in _load_json_cache(_PITCHER_HAND_CACHE_FILE).items():
            _pitcher_hand_cache[k] = v
    if cache_key in _pitcher_hand_cache:
        return _pitcher_hand_cache[cache_key]
    try:
        results = statsapi.lookup_player(player_name)
        if not results:
            last = player_name.split()[-1] if ' ' in player_name else ''
            results = statsapi.lookup_player(last) if last else []
        if not results:
            _pitcher_hand_cache[cache_key] = 'R'
            return 'R'
        player_id = results[0]['id']
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}",
            timeout=10,
        )
        resp.raise_for_status()
        hand = resp.json().get('people', [{}])[0].get('pitchHand', {}).get('code', 'R')
        _pitcher_hand_cache[cache_key] = hand
        # Persist to disk
        disk_data = _load_json_cache(_PITCHER_HAND_CACHE_FILE)
        disk_data[cache_key] = hand
        _save_json_cache(_PITCHER_HAND_CACHE_FILE, disk_data)
        return hand
    except Exception:
        _pitcher_hand_cache[cache_key] = 'R'
        return 'R'


def get_team_batting_vs_hand(team_id: int, year: int) -> dict:
    """Return team OPS vs left-handed and right-handed pitchers."""
    cache_key = (team_id, year)
    if cache_key in _team_batting_splits_cache:
        return _team_batting_splits_cache[cache_key]
    # Load disk cache on first miss
    disk_cache_file = Path(str(_TEAM_BATTING_SPLITS_CACHE_FILE).replace('{year}', str(year)))
    if not _team_batting_splits_cache:
        for k, v in _load_json_cache(disk_cache_file).items():
            _team_batting_splits_cache[(int(k), year)] = v
    if cache_key in _team_batting_splits_cache:
        return _team_batting_splits_cache[cache_key]
    defaults = {'vs_lhp': 0.730, 'vs_rhp': 0.730}
    try:
        out: dict = {}
        for hand_key, sit_code in [('vs_lhp', 'vl'), ('vs_rhp', 'vr')]:
            resp = requests.get(
                f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats",
                params={'stats': 'statSplits', 'group': 'hitting', 'season': year,
                        'gameType': 'R', 'sitCodes': sit_code},
                timeout=10,
            )
            resp.raise_for_status()
            for block in resp.json().get('stats', []):
                for s in block.get('splits', []):
                    stat = s.get('stat', {})
                    obp = float(stat.get('obp') or 0)
                    slg = float(stat.get('slg') or 0)
                    if obp + slg > 0:
                        out[hand_key] = round(obp + slg, 3)
        result = {**defaults, **out}
        _team_batting_splits_cache[cache_key] = result
        # Persist to disk
        all_disk = _load_json_cache(disk_cache_file)
        all_disk[str(team_id)] = result
        _save_json_cache(disk_cache_file, all_disk)
        return result
    except Exception:
        _team_batting_splits_cache[cache_key] = defaults
        return defaults


def get_team_roster(team_id: int) -> list[dict]:
    """Return active roster for a team."""
    try:
        roster_data = statsapi.roster(team_id, rosterType='active')
        players = []
        for line in roster_data.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 3:
                players.append({'name': ' '.join(parts[2:]), 'number': parts[0], 'position': parts[1]})
        return players
    except Exception:
        return []


def get_season_schedule(year: int) -> list[dict]:
    """Fetch all regular season games for a year. Memory-cached after first load."""
    if year in _season_schedule_memory:
        return _season_schedule_memory[year]
    cache_file = _DATA_DIR / f"schedule_{year}.csv"
    if cache_file.exists():
        result = pd.read_csv(cache_file).to_dict('records')
        _season_schedule_memory[year] = result
        return result

    all_games = []
    for month in range(3, 11):
        try:
            month_str = f"{year}-{month:02d}-01"
            end_str = f"{year}-{month:02d}-30"
            games_raw = statsapi.schedule(start_date=month_str, end_date=end_str)
            for g in games_raw:
                if g.get('game_type') != 'R':
                    continue
                all_games.append({
                    'game_id': g['game_id'],
                    'game_date': g['game_date'],
                    'home_id': g['home_id'],
                    'away_id': g['away_id'],
                    'home_name': g['home_name'],
                    'away_name': g['away_name'],
                    'venue_id': g.get('venue_id', 0),
                    'venue_name': g.get('venue_name', ''),
                    'status': g.get('status', ''),
                    'home_score': g.get('home_score'),
                    'away_score': g.get('away_score'),
                    'home_probable_pitcher': g.get('home_probable_pitcher', ''),
                    'away_probable_pitcher': g.get('away_probable_pitcher', ''),
                })
        except Exception:
            continue

    df = pd.DataFrame(all_games)
    df.to_csv(cache_file, index=False)
    _season_schedule_memory[year] = all_games
    return all_games


_schedule_cache_lock = threading.Lock()


def refresh_schedule_date(year: int, game_date: str) -> int:
    """Re-fetch one date from the live API and merge updated status/scores into
    the cached season schedule (CSV + memory).

    The bundled schedule cache may be snapshotted before a day's games finish,
    leaving recent dates with games that have no final score. Calling this fills
    them in so the results comparison can find completed games. Returns the
    number of games updated; a no-op returning 0 if the live API or cache file
    is unavailable. Thread-safe (the results backfill runs parallel workers).
    """
    try:
        live = get_schedule(game_date)
    except Exception:
        return 0
    if not live:
        return 0
    cache_file = _DATA_DIR / f"schedule_{year}.csv"
    if not cache_file.exists():
        return 0
    live_by_id = {g['game_id']: g for g in live}
    fields = ('status', 'home_score', 'away_score',
              'home_probable_pitcher', 'away_probable_pitcher')
    with _schedule_cache_lock:
        df = pd.read_csv(cache_file)
        # An all-empty cached column loads as float64; pandas refuses to store a
        # string into it. Coerce the columns we write to object first.
        for f in fields:
            if f in df.columns:
                df[f] = df[f].astype(object)
        mask = df['game_date'].astype(str).str[:10] == game_date
        updated = 0
        for idx in df[mask].index:
            g = live_by_id.get(df.at[idx, 'game_id'])
            if not g:
                continue
            for f in fields:
                if f in df.columns and g.get(f) is not None:
                    df.at[idx, f] = g.get(f)
            updated += 1
        if updated:
            df.to_csv(cache_file, index=False)
            _season_schedule_memory.pop(year, None)  # force reload from CSV
    return updated


def _load_linescore_cache() -> None:
    global _linescore_cache
    if _linescore_cache or not _LINESCORE_CACHE_FILE.exists():
        return
    try:
        df = pd.read_csv(_LINESCORE_CACHE_FILE)
        for _, row in df.iterrows():
            gid = int(row['game_pk'])
            _linescore_cache[gid] = {
                'home': [int(row.get(f'home_inn{i}', 0)) for i in range(1, 10)],
                'away': [int(row.get(f'away_inn{i}', 0)) for i in range(1, 10)],
            }
    except Exception:
        pass


def get_game_linescore(game_pk: int, cache_only: bool = False) -> dict | None:
    """Fetch per-inning run totals for a completed game, with disk cache.

    Pass cache_only=True to return None rather than making a network call when
    the game isn't in the disk cache — useful during model training.
    """
    _load_linescore_cache()
    if game_pk in _linescore_cache:
        return _linescore_cache[game_pk]
    if cache_only:
        return None
    try:
        data = statsapi.get('game', {
            'gamePk': game_pk,
            'fields': 'liveData,linescore,innings,num,home,away,runs',
        })
        innings_data = data['liveData']['linescore']['innings']
        home_runs = [0] * 9
        away_runs = [0] * 9
        for inn in innings_data:
            idx = int(inn['num']) - 1
            if 0 <= idx < 9:
                home_runs[idx] = int(inn.get('home', {}).get('runs', 0) or 0)
                away_runs[idx] = int(inn.get('away', {}).get('runs', 0) or 0)
        result = {'home': home_runs, 'away': away_runs}
        with _linescore_cache_lock:
            _linescore_cache[game_pk] = result
            row_data = {'game_pk': game_pk}
            for i in range(9):
                row_data[f'home_inn{i + 1}'] = home_runs[i]
                row_data[f'away_inn{i + 1}'] = away_runs[i]
            pd.DataFrame([row_data]).to_csv(
                _LINESCORE_CACHE_FILE, mode='a',
                header=not _LINESCORE_CACHE_FILE.exists(), index=False,
            )
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Player and team stats (pybaseball)
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    return a.div(b).replace([np.inf, -np.inf], default).fillna(default)


def get_pitching_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch pitching stats from Baseball Reference. Caches to CSV.

    Returns columns: Name, Team, ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9, IP, GS
    FIP and xFIP are approximated from ERA (not available from bref).
    """
    cache_file = _DATA_DIR / f"pitching_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    pybaseball.cache.enable()
    df = pybaseball.pitching_stats_bref(year)
    df = df[df['Lev'].str.startswith('Maj', na=False)].copy()
    df = df.rename(columns={'Tm': 'Team', 'SO9': 'K/9'})

    ip = df['IP'].fillna(0)
    df['BB/9'] = _safe_div(df['BB'].fillna(0) * 9, ip, default=3.0)
    df['HR/9'] = _safe_div(df['HR'].fillna(0) * 9, ip, default=1.2)
    # FIP and xFIP not in bref — use ERA as a reasonable proxy
    df['FIP'] = df['ERA'].fillna(4.50)
    df['xFIP'] = df['ERA'].fillna(4.50)

    out = df[['Name', 'Team', 'ERA', 'FIP', 'xFIP', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'IP', 'GS']].copy()
    out.to_csv(cache_file, index=False)
    return out


def _calc_woba(df: pd.DataFrame) -> pd.Series:
    """Calculate wOBA from raw batting counting stats using 2024 linear weights."""
    uBB = (df.get('BB', 0) - df.get('IBB', 0)).clip(lower=0)
    hbp = df.get('HBP', pd.Series(0, index=df.index)).fillna(0)
    singles = (df['H'] - df.get('2B', 0) - df.get('3B', 0) - df.get('HR', 0)).clip(lower=0)
    doubles = df.get('2B', pd.Series(0, index=df.index)).fillna(0)
    triples = df.get('3B', pd.Series(0, index=df.index)).fillna(0)
    hrs = df.get('HR', pd.Series(0, index=df.index)).fillna(0)
    sf = df.get('SF', pd.Series(0, index=df.index)).fillna(0)
    ab = df.get('AB', pd.Series(1, index=df.index)).fillna(1)

    numerator = (0.690 * uBB + 0.722 * hbp + 0.888 * singles
                 + 1.271 * doubles + 1.616 * triples + 2.101 * hrs)
    denominator = (ab + uBB + hbp + sf).replace(0, np.nan)
    return (numerator / denominator).fillna(0.315)


def get_batting_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch batting stats from Baseball Reference. Caches to CSV."""
    cache_file = _DATA_DIR / f"batting_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    pybaseball.cache.enable()
    df = pybaseball.batting_stats_bref(year)
    df = df[df['Lev'].str.startswith('Maj', na=False)].copy()
    df = df.rename(columns={'Tm': 'Team', 'BA': 'AVG'})

    df['wOBA'] = _calc_woba(df)
    df['ISO'] = (df['SLG'] - df['AVG']).fillna(0.150)

    out = df[['Name', 'Team', 'wOBA', 'OPS', 'ISO', 'PA', 'AB']].copy()
    out.to_csv(cache_file, index=False)
    return out


def get_team_batting_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Aggregate team-level batting stats from player data."""
    cache_file = _DATA_DIR / f"team_batting_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    players = get_batting_stats(year, force_refresh=force_refresh)
    # Weighted average by PA
    players = players[players['PA'] > 0].copy()
    teams = players.groupby('Team').apply(
        lambda g: pd.Series({
            'wOBA': np.average(g['wOBA'], weights=g['PA']),
            'OPS': np.average(g['OPS'].fillna(0.730), weights=g['PA']),
            'PA': g['PA'].sum(),
        }), include_groups=False
    ).reset_index()
    teams.to_csv(cache_file, index=False)
    return teams


def get_bullpen_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Return pitchers who are primarily relievers (GS < 5)."""
    df = get_pitching_stats(year, force_refresh=force_refresh)
    if 'GS' in df.columns:
        return df[df['GS'] < 5].copy()
    return df.copy()


def get_team_pitching_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Aggregate team-level pitching stats from player data."""
    cache_file = _DATA_DIR / f"team_pitching_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    players = get_pitching_stats(year, force_refresh=force_refresh)
    players = players[players['IP'] > 0].copy()
    teams = players.groupby('Team').apply(
        lambda g: pd.Series({
            'ERA': np.average(g['ERA'].fillna(4.5), weights=g['IP']),
            'WHIP': np.average(g['WHIP'].fillna(1.35), weights=g['IP']),
            'IP': g['IP'].sum(),
        }), include_groups=False
    ).reset_index()
    teams.to_csv(cache_file, index=False)
    return teams


# ---------------------------------------------------------------------------
# Weather (Open-Meteo — free, no API key required)
# ---------------------------------------------------------------------------

def _hour_value(series: list, hour: int, default: float) -> float:
    """Return series[hour] clamped to the series' own length.

    Each Open-Meteo hourly array is indexed independently — a missing field
    falls back to a short default list, so indexing every array with one shared
    index (derived from a longer array) would overflow.
    """
    if not series:
        return default
    return float(series[min(hour, len(series) - 1)])


def get_weather_forecast(lat: float, lon: float, game_datetime: str) -> dict:
    """Fetch weather forecast from Open-Meteo for a future game."""
    game_date = game_datetime[:10]
    game_hour = int(game_datetime[11:13]) if 'T' in game_datetime and len(game_datetime) > 13 else 19

    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,windspeed_10m,winddirection_10m,precipitation,relative_humidity_2m',
        'temperature_unit': 'fahrenheit',
        'windspeed_unit': 'mph',
        'start_date': game_date,
        'end_date': game_date,
        'timezone': 'UTC',
    }
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get('hourly', {})

    return {
        'temperature_f': _hour_value(hourly.get('temperature_2m'), game_hour, 72.0),
        'wind_speed_mph': _hour_value(hourly.get('windspeed_10m'), game_hour, 0.0),
        'wind_direction_deg': _hour_value(hourly.get('winddirection_10m'), game_hour, 0.0),
        'precipitation_mm': _hour_value(hourly.get('precipitation'), game_hour, 0.0),
        'humidity_pct': _hour_value(hourly.get('relative_humidity_2m'), game_hour, 50.0),
        'is_dome': False,
    }


def get_weather_historical(lat: float, lon: float, game_date: str, game_hour: int = 19) -> dict:
    """Fetch historical weather from Open-Meteo for a past game."""
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,windspeed_10m,winddirection_10m,precipitation,relative_humidity_2m',
        'temperature_unit': 'fahrenheit',
        'windspeed_unit': 'mph',
        'start_date': game_date,
        'end_date': game_date,
        'timezone': 'UTC',
    }
    resp = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get('hourly', {})

    return {
        'temperature_f': _hour_value(hourly.get('temperature_2m'), game_hour, 72.0),
        'wind_speed_mph': _hour_value(hourly.get('windspeed_10m'), game_hour, 0.0),
        'wind_direction_deg': _hour_value(hourly.get('winddirection_10m'), game_hour, 0.0),
        'precipitation_mm': _hour_value(hourly.get('precipitation'), game_hour, 0.0),
        'humidity_pct': _hour_value(hourly.get('relative_humidity_2m'), game_hour, 50.0),
        'is_dome': False,
    }


def get_weather_for_game(lat: float, lon: float, game_datetime: str, is_dome: bool = False) -> dict:
    """Get weather for a game. Returns neutral conditions for domes. Caches historical lookups."""
    if is_dome:
        return {
            'temperature_f': 72.0,
            'wind_speed_mph': 0.0,
            'wind_direction_deg': 0.0,
            'precipitation_mm': 0.0,
            'humidity_pct': 50.0,
            'is_dome': True,
        }
    game_date = game_datetime[:10] if game_datetime else ''
    today = date.today().strftime('%Y-%m-%d')

    if game_date and game_date <= today:
        # Check in-memory + disk cache before hitting the API
        _load_weather_cache()
        cache_key = (game_date, round(lat, 2), round(lon, 2))
        if cache_key in _weather_cache:
            return _weather_cache[cache_key]

        hour = int(game_datetime[11:13]) if game_datetime and 'T' in game_datetime else 19
        try:
            result = get_weather_historical(lat, lon, game_date, game_hour=hour)
            _weather_cache[cache_key] = result
            _save_weather_cache_entry(game_date, lat, lon, result)
            return result
        except Exception:
            pass

    try:
        return get_weather_forecast(lat, lon, game_datetime or f"{today}T19:00:00Z")
    except Exception:
        return {
            'temperature_f': 72.0,
            'wind_speed_mph': 0.0,
            'wind_direction_deg': 0.0,
            'precipitation_mm': 0.0,
            'humidity_pct': 50.0,
            'is_dome': False,
        }
