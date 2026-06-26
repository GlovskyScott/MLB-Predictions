import statsapi
import pybaseball
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import date

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


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


# ---------------------------------------------------------------------------
# Player and team stats (pybaseball)
# ---------------------------------------------------------------------------

def get_pitching_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch FanGraphs pitching stats for a season. Caches to CSV."""
    cache_file = _DATA_DIR / f"pitching_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    pybaseball.cache.enable()
    df = pybaseball.pitching_stats(year, qual=0)
    required_cols = ['Name', 'Team', 'ERA', 'FIP', 'xFIP', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'IP', 'GS']
    available = [c for c in required_cols if c in df.columns]
    df = df[available].copy()
    df.to_csv(cache_file, index=False)
    return df


def get_batting_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch FanGraphs batting stats for a season. Caches to CSV."""
    cache_file = _DATA_DIR / f"batting_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    pybaseball.cache.enable()
    df = pybaseball.batting_stats(year, qual=0)
    required_cols = ['Name', 'Team', 'wOBA', 'OPS', 'ISO', 'PA', 'AB', 'AVG', 'OBP', 'SLG']
    available = [c for c in required_cols if c in df.columns]
    df = df[available].copy()
    df.to_csv(cache_file, index=False)
    return df


def get_team_batting_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch team-level batting stats."""
    cache_file = _DATA_DIR / f"team_batting_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    pybaseball.cache.enable()
    df = pybaseball.team_batting(year)
    df.to_csv(cache_file, index=False)
    return df


def get_bullpen_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Return pitchers who are primarily relievers (GS < 5)."""
    df = get_pitching_stats(year, force_refresh=force_refresh)
    if 'GS' in df.columns:
        return df[df['GS'] < 5].copy()
    return df.copy()


def get_team_pitching_stats(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch team-level pitching stats."""
    cache_file = _DATA_DIR / f"team_pitching_{year}.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file)

    pybaseball.cache.enable()
    df = pybaseball.team_pitching(year)
    df.to_csv(cache_file, index=False)
    return df


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
    """Get weather for a game. Returns neutral conditions for domes."""
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
        hour = int(game_datetime[11:13]) if game_datetime and 'T' in game_datetime else 19
        try:
            return get_weather_historical(lat, lon, game_date, game_hour=hour)
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
