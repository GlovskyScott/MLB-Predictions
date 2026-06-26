import pandas as pd
import numpy as np
from pathlib import Path
from src.fetcher import (
    get_pitching_stats, get_batting_stats, get_team_batting_stats,
    get_bullpen_stats, get_weather_for_game, get_pitcher_splits, get_game_lineup,
    get_pitcher_days_rest, get_team_recent_runs, get_bullpen_stress_l3,
    get_pitcher_handedness, get_team_batting_vs_hand,
)
from src.stadiums import get_stadium, classify_wind

_DATA_DIR = Path(__file__).parent.parent / "data"

FEATURE_VERSION = 3  # Increment whenever FEATURE_COLUMNS changes

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
    'temperature_f', 'wind_speed_mph', 'wind_out', 'wind_in', 'precipitation_flag', 'humidity_pct',
    # Context
    'is_dome',
    # Pitcher rest (days since last start)
    'away_sp_days_rest', 'home_sp_days_rest',
    # Recent team offensive form (runs per game, last 15 games)
    'away_runs_l15', 'home_runs_l15',
    # Bullpen workload (late-inning runs allowed, last 3 games — proxy for fatigue)
    'away_bullpen_stress_l3', 'home_bullpen_stress_l3',
    # Handedness matchup
    'away_sp_is_lhp', 'home_sp_is_lhp',
    'away_bat_ops_vs_sp_hand', 'home_bat_ops_vs_sp_hand',
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


def _get_lineup_batting(player_names: list, batting_df: pd.DataFrame) -> dict | None:
    """Compute aggregate wOBA/OPS for a specific lineup.

    Returns None when fewer than 3 players are matched, signalling the caller to
    fall back to team-level batting stats.
    """
    if not player_names or batting_df.empty:
        return None
    rows = []
    for name in player_names:
        mask = batting_df['Name'].str.lower().str.strip() == name.lower().strip()
        matches = batting_df[mask]
        if matches.empty:
            last = name.split()[-1].lower() if name.split() else ''
            if last:
                mask2 = batting_df['Name'].str.lower().str.contains(last, na=False)
                matches = batting_df[mask2]
        if not matches.empty:
            rows.append(matches.iloc[0])
    if len(rows) < 3:
        return None
    df = pd.DataFrame(rows)
    pa = df['PA'].fillna(1).replace(0, 1)
    return {
        'woba': float(np.average(df['wOBA'].fillna(0.315), weights=pa)),
        'ops': float(np.average(df['OPS'].fillna(0.730), weights=pa)),
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


INNING_FEATURE_COLUMNS = [
    'inning', 'is_home',
    'batting_woba', 'batting_ops', 'batting_runs_l15',
    'pitching_era', 'pitching_whip',
    'sp_days_rest', 'bullpen_stress_l3',
    'park_runs_factor',
]

_NEUTRAL_WEATHER = {
    'temperature_f': 72.0, 'wind_speed_mph': 0.0,
    'wind_direction_deg': 0.0, 'precipitation_mm': 0.0,
    'humidity_pct': 50.0, 'is_dome': False,
}


def build_inning_feature_row(game_feats: dict, inning: int, batting_is_home: bool) -> dict:
    """Build a single feature row for predicting whether a team scores in a given inning."""
    pfx = 'home' if batting_is_home else 'away'
    opp = 'away' if batting_is_home else 'home'
    if inning <= 6:
        era_key, whip_key = f'{opp}_sp_era', f'{opp}_sp_whip'
        sp_days_rest = float(game_feats.get(f'{opp}_sp_days_rest', 5))
    else:
        era_key, whip_key = f'{opp}_bullpen_era', f'{opp}_bullpen_whip'
        sp_days_rest = 0.0  # bullpen doesn't have a single starter's rest
    return {
        'inning': inning,
        'is_home': int(batting_is_home),
        'batting_woba': game_feats.get(f'{pfx}_team_woba', 0.315),
        'batting_ops': game_feats.get(f'{pfx}_team_ops', 0.730),
        'batting_runs_l15': game_feats.get(f'{pfx}_runs_l15', 4.5),
        'pitching_era': game_feats.get(era_key, 4.00),
        'pitching_whip': game_feats.get(whip_key, 1.30),
        'sp_days_rest': sp_days_rest,
        'bullpen_stress_l3': game_feats.get(f'{opp}_bullpen_stress_l3', 3.0),
        'park_runs_factor': game_feats.get('park_runs_factor', 1.0),
    }


def build_game_features(game: dict, year: int = 2026, weather: dict = None,
                        for_training: bool = False) -> dict:
    """Build a numeric feature vector for one game.

    Pass weather=None to fetch live weather, or supply a pre-built dict to skip the API call.
    Pass for_training=True to skip slow external API calls (pitcher splits, handedness,
    lineup, team batting splits) and use defaults — keeps training fast.
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

    if not for_training:
        # Override aggregate pitcher stats with context-specific home/away splits
        home_pitcher_splits = get_pitcher_splits(game.get('home_probable_pitcher', ''), year)
        away_pitcher_splits = get_pitcher_splits(game.get('away_probable_pitcher', ''), year)

        def _apply_splits(sp: dict, splits: dict, side: str) -> None:
            s = splits.get(side, {})
            for sp_key, api_key in [('era', 'era'), ('whip', 'whip'), ('k9', 'k9'),
                                      ('bb9', 'bb9'), ('hr9', 'hr9'), ('ip', 'ip')]:
                if s.get(api_key) is not None:
                    sp[sp_key] = s[api_key]
            if s.get('era') is not None:
                sp['fip'] = s['era']
                sp['xfip'] = s['era']

        _apply_splits(home_sp, home_pitcher_splits, 'home')
        _apply_splits(away_sp, away_pitcher_splits, 'away')

        # Use actual lineup batting stats when the day's batting order is available
        batting_df_individual = get_batting_stats(year)
        lineup = get_game_lineup(game.get('game_id', 0))
        home_lineup_bat = _get_lineup_batting(lineup.get('home', []), batting_df_individual)
        away_lineup_bat = _get_lineup_batting(lineup.get('away', []), batting_df_individual)
        if home_lineup_bat is not None:
            home_bat = home_lineup_bat
        if away_lineup_bat is not None:
            away_bat = away_lineup_bat

    home_bp = _get_bullpen_stats(home_id, bullpen_df)
    away_bp = _get_bullpen_stats(away_id, bullpen_df)
    park = _get_park_factors(home_id)

    game_date = game.get('game_date', '')

    # Pitcher days of rest — computed from cached schedule, fast for both training and inference
    home_days_rest = get_pitcher_days_rest(game.get('home_probable_pitcher', ''), game_date, year)
    away_days_rest = get_pitcher_days_rest(game.get('away_probable_pitcher', ''), game_date, year)

    # Recent team offensive form — computed from cached schedule, fast for both
    home_runs_l15 = get_team_recent_runs(home_id, game_date, year)
    away_runs_l15 = get_team_recent_runs(away_id, game_date, year)

    # Bullpen workload — from schedule + linescores; cache_only avoids API calls during training
    home_bp_stress = get_bullpen_stress_l3(home_id, game_date, year, cache_only=for_training)
    away_bp_stress = get_bullpen_stress_l3(away_id, game_date, year, cache_only=for_training)

    # Handedness matchup — disk-cached after prefetch_historical.py, fast for both training and inference
    home_sp_hand = get_pitcher_handedness(game.get('home_probable_pitcher', ''))
    away_sp_hand = get_pitcher_handedness(game.get('away_probable_pitcher', ''))
    home_bat_splits = get_team_batting_vs_hand(home_id, year)
    away_bat_splits = get_team_batting_vs_hand(away_id, year)

    home_sp_is_lhp = 1.0 if home_sp_hand == 'L' else 0.0
    away_sp_is_lhp = 1.0 if away_sp_hand == 'L' else 0.0
    away_bat_ops_vs_sp = away_bat_splits['vs_lhp'] if home_sp_hand == 'L' else away_bat_splits['vs_rhp']
    home_bat_ops_vs_sp = home_bat_splits['vs_lhp'] if away_sp_hand == 'L' else home_bat_splits['vs_rhp']

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
        'humidity_pct': float(weather.get('humidity_pct', 50.0)),
        'is_dome': 1.0 if is_dome else 0.0,
        'away_sp_days_rest': float(away_days_rest),
        'home_sp_days_rest': float(home_days_rest),
        'away_runs_l15': away_runs_l15,
        'home_runs_l15': home_runs_l15,
        'away_bullpen_stress_l3': away_bp_stress,
        'home_bullpen_stress_l3': home_bp_stress,
        'away_sp_is_lhp': away_sp_is_lhp,
        'home_sp_is_lhp': home_sp_is_lhp,
        'away_bat_ops_vs_sp_hand': away_bat_ops_vs_sp,
        'home_bat_ops_vs_sp_hand': home_bat_ops_vs_sp,
    }
