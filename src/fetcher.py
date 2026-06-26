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

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

_WEATHER_CACHE_FILE = _DATA_DIR / "weather_cache.csv"
_weather_cache: dict = {}  # (date, lat, lon) → weather dict, in-memory layer

_LINESCORE_CACHE_FILE = _DATA_DIR / "linescore_cache.csv"
_linescore_cache: dict = {}  # game_pk → {home: [9 ints], away: [9 ints]}
_linescore_cache_lock = threading.Lock()

def _load_weather_cache() -> None:
    global _weather_cache
    if _weather_cache or not _WEATHER_CACHE_FILE.exists():
        return
    df = pd.read_csv(_WEATHER_CACHE_FILE)
    for _, row in df.iterrows():
        key = (str(row['date']), round(float(row['lat']), 2), round(float(row['lon']), 2))
        _weather_cache[key] = {
            'temperature_f': float(row['temperature_f']),
            'wind_speed_mph': float(row['wind_speed_mph']),
            'wind_direction_deg': float(row['wind_direction_deg']),
            'precipitation_mm': float(row['precipitation_mm']),
            'is_dome': False,
        }

def _save_weather_cache_entry(date: str, lat: float, lon: float, weather: dict) -> None:
    row = pd.DataFrame([{
        'date': date, 'lat': round(lat, 2), 'lon': round(lon, 2),
        'temperature_f': weather['temperature_f'],
        'wind_speed_mph': weather['wind_speed_mph'],
        'wind_direction_deg': weather['wind_direction_deg'],
        'precipitation_mm': weather['precipitation_mm'],
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
    """Fetch all regular season games for a year. Caches to CSV."""
    cache_file = _DATA_DIR / f"schedule_{year}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file).to_dict('records')

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
    return all_games


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


def get_game_linescore(game_pk: int) -> dict | None:
    """Fetch per-inning run totals for a completed game, with disk cache."""
    _load_linescore_cache()
    if game_pk in _linescore_cache:
        return _linescore_cache[game_pk]
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

def get_weather_forecast(lat: float, lon: float, game_datetime: str) -> dict:
    """Fetch weather forecast from Open-Meteo for a future game."""
    game_date = game_datetime[:10]
    game_hour = int(game_datetime[11:13]) if 'T' in game_datetime and len(game_datetime) > 13 else 19

    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,windspeed_10m,winddirection_10m,precipitation',
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

    temp = hourly.get('temperature_2m', [72.0])
    wind_s = hourly.get('windspeed_10m', [0.0])
    wind_d = hourly.get('winddirection_10m', [0.0])
    precip = hourly.get('precipitation', [0.0])
    idx = min(game_hour, len(temp) - 1) if temp else 0

    return {
        'temperature_f': float(temp[idx]) if temp else 72.0,
        'wind_speed_mph': float(wind_s[idx]) if wind_s else 0.0,
        'wind_direction_deg': float(wind_d[idx]) if wind_d else 0.0,
        'precipitation_mm': float(precip[idx]) if precip else 0.0,
        'is_dome': False,
    }


def get_weather_historical(lat: float, lon: float, game_date: str, game_hour: int = 19) -> dict:
    """Fetch historical weather from Open-Meteo for a past game."""
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,windspeed_10m,winddirection_10m,precipitation',
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

    temp = hourly.get('temperature_2m', [72.0])
    wind_s = hourly.get('windspeed_10m', [0.0])
    wind_d = hourly.get('winddirection_10m', [0.0])
    precip = hourly.get('precipitation', [0.0])
    idx = min(game_hour, len(temp) - 1) if temp else 0

    return {
        'temperature_f': float(temp[idx]) if temp else 72.0,
        'wind_speed_mph': float(wind_s[idx]) if wind_s else 0.0,
        'wind_direction_deg': float(wind_d[idx]) if wind_d else 0.0,
        'precipitation_mm': float(precip[idx]) if precip else 0.0,
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
            'is_dome': False,
        }
