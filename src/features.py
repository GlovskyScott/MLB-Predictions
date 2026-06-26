import pandas as pd
import numpy as np
from pathlib import Path
from src.fetcher import (
    get_pitching_stats, get_batting_stats, get_team_batting_stats,
    get_bullpen_stats, get_weather_for_game
)
from src.stadiums import get_stadium, classify_wind

_DATA_DIR = Path(__file__).parent.parent / "data"

# Canonical ordered list of all features fed to XGBoost
FEATURE_COLUMNS = [
    # Away starter
    'away_sp_era', 'away_sp_fip', 'away_sp_xfip', 'away_sp_whip',
    'away_sp_k9', 'away_sp_bb9', 'away_sp_hr9', 'away_sp_ip',
    # Home starter
    'home_sp_era', 'home_sp_fip', 'home_sp_xfip', 'home_sp_whip',
    'home_sp_k9', 'home_sp_bb9', 'home_sp_hr9', 'home_sp_ip',
    # Away team batting
    'away_team_woba', 'away_team_ops',
    # Home team batting
    'home_team_woba', 'home_team_ops',
    # Away bullpen
    'away_bullpen_era', 'away_bullpen_whip',
    # Home bullpen
    'home_bullpen_era', 'home_bullpen_whip',
    # Park factors
    'park_runs_factor', 'park_hr_factor',
    # Weather
    'temperature_f', 'wind_speed_mph', 'wind_out', 'wind_in', 'precipitation_flag',
    # Context
    'is_dome',
]

# MLB Stats API team_id → FanGraphs abbreviation
TEAM_ID_TO_FG = {
    110: 'BAL', 111: 'BOS', 133: 'OAK', 136: 'SEA', 108: 'LAA',
    117: 'HOU', 140: 'TEX', 141: 'TOR', 139: 'TBR', 142: 'MIN',
    145: 'CWS', 116: 'DET', 118: 'KCR', 114: 'CLE', 147: 'NYY',
    144: 'ATL', 146: 'MIA', 121: 'NYM', 143: 'PHI', 120: 'WSN',
    112: 'CHC', 113: 'CIN', 158: 'MIL', 134: 'PIT', 138: 'STL',
    109: 'ARI', 115: 'COL', 119: 'LAD', 135: 'SDP', 137: 'SFG',
}

_PITCHER_DEFAULTS = {
    'era': 4.50, 'fip': 4.30, 'xfip': 4.30, 'whip': 1.35,
    'k9': 8.0, 'bb9': 3.0, 'hr9': 1.2, 'ip': 0.0,
}
_TEAM_BATTING_DEFAULTS = {'woba': 0.315, 'ops': 0.730}
_BULLPEN_DEFAULTS = {'era': 4.00, 'whip': 1.30}


def _get_pitcher_stats(pitcher_name: str, pitching_df: pd.DataFrame) -> dict:
    if not pitcher_name or pitching_df.empty:
        return _PITCHER_DEFAULTS.copy()
    mask = pitching_df['Name'].str.lower().str.strip() == pitcher_name.lower().strip()
    matches = pitching_df[mask]
    if matches.empty:
        last_name = pitcher_name.split()[-1].lower() if pitcher_name.split() else ''
        if last_name:
            mask2 = pitching_df['Name'].str.lower().str.contains(last_name, na=False)
            matches = pitching_df[mask2]
    if matches.empty:
        return _PITCHER_DEFAULTS.copy()
    row = matches.iloc[0]
    return {
        'era': float(row.get('ERA', _PITCHER_DEFAULTS['era']) or _PITCHER_DEFAULTS['era']),
        'fip': float(row.get('FIP', _PITCHER_DEFAULTS['fip']) or _PITCHER_DEFAULTS['fip']),
        'xfip': float(row.get('xFIP', _PITCHER_DEFAULTS['xfip']) or _PITCHER_DEFAULTS['xfip']),
        'whip': float(row.get('WHIP', _PITCHER_DEFAULTS['whip']) or _PITCHER_DEFAULTS['whip']),
        'k9': float(row.get('K/9', _PITCHER_DEFAULTS['k9']) or _PITCHER_DEFAULTS['k9']),
        'bb9': float(row.get('BB/9', _PITCHER_DEFAULTS['bb9']) or _PITCHER_DEFAULTS['bb9']),
        'hr9': float(row.get('HR/9', _PITCHER_DEFAULTS['hr9']) or _PITCHER_DEFAULTS['hr9']),
        'ip': float(row.get('IP', 0.0) or 0.0),
    }


def _get_team_batting(team_id: int, batting_df: pd.DataFrame) -> dict:
    fg_abbr = TEAM_ID_TO_FG.get(team_id, '')
    if batting_df.empty or not fg_abbr:
        return _TEAM_BATTING_DEFAULTS.copy()
    col = 'Team' if 'Team' in batting_df.columns else batting_df.columns[0]
    mask = batting_df[col].str.upper() == fg_abbr.upper()
    matches = batting_df[mask]
    if matches.empty:
        return _TEAM_BATTING_DEFAULTS.copy()
    row = matches.iloc[0]
    return {
        'woba': float(row.get('wOBA', _TEAM_BATTING_DEFAULTS['woba']) or _TEAM_BATTING_DEFAULTS['woba']),
        'ops': float(row.get('OPS', _TEAM_BATTING_DEFAULTS['ops']) or _TEAM_BATTING_DEFAULTS['ops']),
    }


def _get_bullpen_stats(team_id: int, bullpen_df: pd.DataFrame) -> dict:
    fg_abbr = TEAM_ID_TO_FG.get(team_id, '')
    if bullpen_df.empty or not fg_abbr:
        return _BULLPEN_DEFAULTS.copy()
    mask = bullpen_df['Team'].str.upper() == fg_abbr.upper()
    team_bp = bullpen_df[mask]
    if team_bp.empty:
        return _BULLPEN_DEFAULTS.copy()
    era = team_bp['ERA'].dropna()
    whip = team_bp['WHIP'].dropna()
    return {
        'era': float(era.mean()) if not era.empty else _BULLPEN_DEFAULTS['era'],
        'whip': float(whip.mean()) if not whip.empty else _BULLPEN_DEFAULTS['whip'],
    }


def _get_park_factors(home_team_id: int) -> dict:
    pf_file = _DATA_DIR / "park_factors.csv"
    defaults = {'runs_factor': 1.0, 'hr_factor': 1.0}
    if not pf_file.exists():
        return defaults
    df = pd.read_csv(pf_file)
    row = df[df['team_id'] == home_team_id]
    if row.empty:
        return defaults
    r = row.iloc[0]
    return {
        'runs_factor': float(r.get('runs_factor', 1.0)),
        'hr_factor': float(r.get('hr_factor', 1.0)),
    }


_NEUTRAL_WEATHER = {
    'temperature_f': 72.0, 'wind_speed_mph': 0.0,
    'wind_direction_deg': 0.0, 'precipitation_mm': 0.0, 'is_dome': False,
}


def build_game_features(game: dict, year: int = 2026, weather: dict = None) -> dict:
    """Build a numeric feature vector for one game.

    Pass weather=None to fetch live weather, or supply a pre-built dict to skip the API call.
    """
    pitching_df = get_pitching_stats(year)
    batting_df = get_team_batting_stats(year)
    bullpen_df = get_bullpen_stats(year)

    home_id = game['home_id']
    away_id = game['away_id']
    stadium = get_stadium(home_id) or {}
    is_dome = stadium.get('roof') == 'dome'
    lat = stadium.get('lat', 39.0)
    lon = stadium.get('lon', -95.0)

    if weather is None:
        weather = get_weather_for_game(
            lat=lat, lon=lon,
            game_datetime=game.get('game_datetime', ''),
            is_dome=is_dome,
        )

    home_sp = _get_pitcher_stats(game.get('home_probable_pitcher', ''), pitching_df)
    away_sp = _get_pitcher_stats(game.get('away_probable_pitcher', ''), pitching_df)
    home_bat = _get_team_batting(home_id, batting_df)
    away_bat = _get_team_batting(away_id, batting_df)
    home_bp = _get_bullpen_stats(home_id, bullpen_df)
    away_bp = _get_bullpen_stats(away_id, bullpen_df)
    park = _get_park_factors(home_id)

    wind_dir = weather.get('wind_direction_deg', 0.0)
    wind_label = classify_wind(wind_dir, home_id)

    return {
        'away_sp_era': away_sp['era'],
        'away_sp_fip': away_sp['fip'],
        'away_sp_xfip': away_sp['xfip'],
        'away_sp_whip': away_sp['whip'],
        'away_sp_k9': away_sp['k9'],
        'away_sp_bb9': away_sp['bb9'],
        'away_sp_hr9': away_sp['hr9'],
        'away_sp_ip': away_sp['ip'],
        'home_sp_era': home_sp['era'],
        'home_sp_fip': home_sp['fip'],
        'home_sp_xfip': home_sp['xfip'],
        'home_sp_whip': home_sp['whip'],
        'home_sp_k9': home_sp['k9'],
        'home_sp_bb9': home_sp['bb9'],
        'home_sp_hr9': home_sp['hr9'],
        'home_sp_ip': home_sp['ip'],
        'away_team_woba': away_bat['woba'],
        'away_team_ops': away_bat['ops'],
        'home_team_woba': home_bat['woba'],
        'home_team_ops': home_bat['ops'],
        'away_bullpen_era': away_bp['era'],
        'away_bullpen_whip': away_bp['whip'],
        'home_bullpen_era': home_bp['era'],
        'home_bullpen_whip': home_bp['whip'],
        'park_runs_factor': park['runs_factor'],
        'park_hr_factor': park['hr_factor'],
        'temperature_f': float(weather.get('temperature_f', 72.0)),
        'wind_speed_mph': float(weather.get('wind_speed_mph', 0.0)),
        'wind_out': 1.0 if wind_label == 'out_to_cf' else 0.0,
        'wind_in': 1.0 if wind_label == 'in_from_cf' else 0.0,
        'precipitation_flag': 1.0 if weather.get('precipitation_mm', 0) > 0.1 else 0.0,
        'is_dome': 1.0 if is_dome else 0.0,
    }
