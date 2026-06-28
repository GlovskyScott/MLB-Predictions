import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.fetcher import (
    get_schedule, get_probable_pitchers, get_team_roster,
    get_pitching_stats, get_batting_stats, get_bullpen_stats,
    get_team_batting_stats, get_weather_forecast, get_weather_historical,
    get_weather_for_game, refresh_schedule_date,
)
import src.fetcher as fetcher

MOCK_SCHEDULE = [
    {
        'game_id': 745003,
        'game_date': '2026-06-25',
        'game_type': 'R',
        'status': 'Preview',
        'away_name': 'Boston Red Sox',
        'home_name': 'New York Yankees',
        'away_id': 111,
        'home_id': 147,
        'venue_id': 3313,
        'venue_name': 'Yankee Stadium',
        'game_datetime': '2026-06-25T23:05:00Z',
        'away_probable_pitcher': 'Shane Bieber',
        'home_probable_pitcher': 'Gerrit Cole',
    }
]

# --- Schedule tests ---

def test_get_schedule_returns_list(mocker):
    mocker.patch('statsapi.schedule', return_value=MOCK_SCHEDULE)
    result = get_schedule('2026-06-25')
    assert isinstance(result, list)
    assert len(result) == 1

def test_get_schedule_filters_non_regular_season(mocker):
    mock_data = MOCK_SCHEDULE + [{**MOCK_SCHEDULE[0], 'game_type': 'S', 'game_id': 99}]
    mocker.patch('statsapi.schedule', return_value=mock_data)
    result = get_schedule('2026-06-25')
    assert len(result) == 1

def test_get_schedule_game_has_required_fields(mocker):
    mocker.patch('statsapi.schedule', return_value=MOCK_SCHEDULE)
    games = get_schedule('2026-06-25')
    game = games[0]
    required = {'game_id', 'game_date', 'home_id', 'away_id', 'home_name', 'away_name', 'venue_id', 'game_datetime'}
    assert required.issubset(game.keys())

def test_get_schedule_handles_empty_date(mocker):
    mocker.patch('statsapi.schedule', return_value=[])
    result = get_schedule('2026-06-25')
    assert result == []

def test_refresh_schedule_date_fills_final_scores(mocker, tmp_path):
    # Cached schedule has the game as not-yet-final with no scores
    cache = tmp_path / "schedule_2026.csv"
    pd.DataFrame([{
        'game_id': 745003, 'game_date': '2026-06-26',
        'home_id': 147, 'away_id': 111,
        'home_name': 'New York Yankees', 'away_name': 'Boston Red Sox',
        'venue_id': 3313, 'venue_name': 'Yankee Stadium',
        'status': 'Scheduled', 'home_score': None, 'away_score': None,
        'home_probable_pitcher': '', 'away_probable_pitcher': '',
    }]).to_csv(cache, index=False)
    mocker.patch('src.fetcher._DATA_DIR', tmp_path)
    fetcher._season_schedule_memory.pop(2026, None)
    # Live API now reports the game as Final with scores
    final_game = {**MOCK_SCHEDULE[0], 'game_id': 745003, 'game_date': '2026-06-26',
                  'status': 'Final', 'home_score': 8, 'away_score': 0}
    mocker.patch('statsapi.schedule', return_value=[final_game])

    n = refresh_schedule_date(2026, '2026-06-26')

    assert n == 1
    df = pd.read_csv(cache)
    row = df[df['game_id'] == 745003].iloc[0]
    assert row['status'] == 'Final'
    assert int(row['home_score']) == 8 and int(row['away_score']) == 0
    assert 2026 not in fetcher._season_schedule_memory  # memory invalidated


def test_refresh_schedule_date_noop_when_no_live_games(mocker, tmp_path):
    cache = tmp_path / "schedule_2026.csv"
    pd.DataFrame([{'game_id': 1, 'game_date': '2026-06-26', 'status': 'Scheduled',
                   'home_score': None, 'away_score': None}]).to_csv(cache, index=False)
    mocker.patch('src.fetcher._DATA_DIR', tmp_path)
    mocker.patch('statsapi.schedule', return_value=[])
    assert refresh_schedule_date(2026, '2026-06-26') == 0


def test_get_probable_pitchers_returns_dict(mocker):
    mocker.patch('statsapi.schedule', return_value=MOCK_SCHEDULE)
    result = get_probable_pitchers(745003, MOCK_SCHEDULE[0])
    assert isinstance(result, dict)
    assert 'home_pitcher_name' in result
    assert 'away_pitcher_name' in result

# --- pybaseball stats tests ---

def test_get_pitching_stats_returns_dataframe(mocker):
    mock_df = pd.DataFrame({
        'Name': ['Gerrit Cole'], 'Team': ['NYY'], 'ERA': [3.2],
        'FIP': [3.1], 'xFIP': [3.3], 'WHIP': [1.1],
        'K/9': [10.5], 'BB/9': [2.1], 'HR/9': [1.1], 'IP': [120.0], 'GS': [20],
    })
    mocker.patch('pybaseball.pitching_stats', return_value=mock_df)
    mocker.patch('pybaseball.cache.enable')
    result = get_pitching_stats(2026, force_refresh=True)
    assert isinstance(result, pd.DataFrame)
    assert 'ERA' in result.columns
    assert 'Name' in result.columns

def test_get_pitching_stats_caches_to_csv(mocker, tmp_path):
    mocker.patch('src.fetcher._DATA_DIR', tmp_path)
    # bref mock needs Lev + Tm columns (what pitching_stats_bref returns)
    mock_df = pd.DataFrame({
        'Name': ['Cole'], 'Tm': ['NYY'], 'Lev': ['Maj-AL'],
        'ERA': [3.2], 'WHIP': [1.1], 'SO9': [10.5], 'BB': [30.0],
        'HR': [15.0], 'IP': [120.0], 'GS': [20], 'G': [20],
    })
    mock_pb = mocker.patch('pybaseball.pitching_stats_bref', return_value=mock_df)
    mocker.patch('pybaseball.cache.enable')
    get_pitching_stats(2026, force_refresh=True)
    get_pitching_stats(2026)  # second call should use cache
    assert mock_pb.call_count == 1

def test_get_batting_stats_returns_dataframe(mocker):
    mock_df = pd.DataFrame({
        'Name': ['Judge'], 'Team': ['NYY'], 'wOBA': [0.42],
        'OPS': [1.05], 'ISO': [0.32], 'PA': [300],
    })
    mocker.patch('pybaseball.batting_stats', return_value=mock_df)
    mocker.patch('pybaseball.cache.enable')
    result = get_batting_stats(2026, force_refresh=True)
    assert isinstance(result, pd.DataFrame)
    assert 'wOBA' in result.columns

def test_get_bullpen_stats_excludes_starters(mocker, tmp_path):
    mocker.patch('src.fetcher._DATA_DIR', tmp_path)
    # bref format: only starters (GS >= 18)
    mock_df = pd.DataFrame({
        'Name': ['Cole', 'Bieber'], 'Tm': ['NYY', 'CLE'], 'Lev': ['Maj-AL', 'Maj-AL'],
        'ERA': [3.2, 2.9], 'WHIP': [1.1, 1.05], 'SO9': [10.5, 9.8],
        'BB': [50.0, 45.0], 'HR': [15.0, 12.0], 'IP': [120.0, 110.0],
        'GS': [20, 18], 'G': [20, 18],
    })
    mocker.patch('pybaseball.pitching_stats_bref', return_value=mock_df)
    mocker.patch('pybaseball.cache.enable')
    result = get_bullpen_stats(2026, force_refresh=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0  # both have GS >= 5, so no relievers

def test_get_team_batting_stats_returns_per_team(mocker):
    mock_df = pd.DataFrame({
        'Team': ['NYY', 'BOS'], 'wOBA': [0.34, 0.32],
        'OPS': [0.82, 0.79], 'R': [380, 340], 'H': [700, 670],
    })
    mocker.patch('pybaseball.team_batting', return_value=mock_df)
    mocker.patch('pybaseball.cache.enable')
    result = get_team_batting_stats(2026, force_refresh=True)
    assert isinstance(result, pd.DataFrame)
    assert 'wOBA' in result.columns

# --- Weather tests ---

MOCK_FORECAST_RESPONSE = {
    "hourly": {
        "time": ["2026-06-25T22:00", "2026-06-25T23:00"],
        "temperature_2m": [75.2, 74.0],
        "windspeed_10m": [12.5, 11.0],
        "winddirection_10m": [180.0, 185.0],
        "precipitation": [0.0, 0.0],
    }
}

def test_get_weather_forecast_returns_dict(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_FORECAST_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    mocker.patch('requests.get', return_value=mock_resp)
    result = get_weather_forecast(lat=40.8296, lon=-73.9262, game_datetime='2026-06-25T23:05:00Z')
    assert isinstance(result, dict)
    assert 'temperature_f' in result
    assert 'wind_speed_mph' in result
    assert 'wind_direction_deg' in result
    assert 'precipitation_mm' in result

def test_get_weather_forecast_temperature_in_range(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_FORECAST_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    mocker.patch('requests.get', return_value=mock_resp)
    result = get_weather_forecast(lat=40.8296, lon=-73.9262, game_datetime='2026-06-25T23:05:00Z')
    assert 0 < result['temperature_f'] < 130

def test_get_weather_dome_returns_neutral():
    result = get_weather_for_game(lat=27.7683, lon=-82.6534, game_datetime='2026-06-25T23:05:00Z', is_dome=True)
    assert result['temperature_f'] == 72.0
    assert result['wind_speed_mph'] == 0.0
    assert result['is_dome'] is True

def test_get_weather_historical_returns_dict(mocker):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hourly": {
            "time": ["2026-04-01T19:00", "2026-04-01T20:00"],
            "temperature_2m": [65.0, 64.0],
            "windspeed_10m": [8.0, 7.5],
            "winddirection_10m": [220.0, 225.0],
            "precipitation": [0.0, 0.0],
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mocker.patch('requests.get', return_value=mock_resp)
    result = get_weather_historical(lat=40.8296, lon=-73.9262, game_date='2026-04-01', game_hour=19)
    assert isinstance(result, dict)
    assert 'temperature_f' in result


def test_bootstrap_predictions_cache_noop_when_present(tmp_path, mocker):
    mocker.patch('src.fetcher._DATA_DIR', tmp_path)
    (tmp_path / 'predictions').mkdir()
    (tmp_path / 'predictions' / 'versions.json').write_text('[]')
    run = mocker.patch('subprocess.run')
    fetcher.bootstrap_predictions_cache()
    run.assert_not_called()  # store present -> no download


def test_get_market_odds_averages_books(mocker):
    import src.fetcher as f
    f._market_odds_cache.clear()
    sb = {'events': [{'id': '1', 'competitions': [{'competitors': [
        {'homeAway': 'home', 'team': {'displayName': 'Boston Red Sox'}},
        {'homeAway': 'away', 'team': {'displayName': 'New York Yankees'}}]}]}]}
    summary = {'pickcenter': [
        {'overUnder': 8.0, 'overOdds': -120, 'underOdds': -100,
         'homeTeamOdds': {'moneyLine': 100}, 'awayTeamOdds': {'moneyLine': -120}},
        {'overUnder': 9.0, 'overOdds': -100, 'underOdds': -120,
         'homeTeamOdds': {'moneyLine': 120}, 'awayTeamOdds': {'moneyLine': -140}}]}

    def fake_get(url, params=None, timeout=None):
        m = MagicMock()
        m.json.return_value = sb if 'scoreboard' in url else summary
        return m
    mocker.patch('src.fetcher.requests.get', side_effect=fake_get)

    out = f.get_market_odds('2026-06-27')
    k = f.market_key('New York Yankees', 'Boston Red Sox')
    assert k in out
    assert out[k]['total'] == 8.5           # (8 + 9)/2
    assert out[k]['ml_away'] == -130.0      # (-120 + -140)/2
    assert out[k]['n_books'] == 2
