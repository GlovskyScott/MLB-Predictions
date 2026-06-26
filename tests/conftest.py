import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_pitching_stats():
    return pd.DataFrame({
        'Name': ['Gerrit Cole', 'Shane Bieber', 'Sandy Alcantara'],
        'Team': ['NYY', 'CLE', 'MIA'],
        'ERA': [3.20, 2.90, 2.50],
        'FIP': [3.10, 2.80, 2.40],
        'xFIP': [3.30, 3.00, 2.60],
        'WHIP': [1.10, 1.05, 1.00],
        'K/9': [10.5, 9.8, 9.2],
        'BB/9': [2.1, 2.0, 1.8],
        'HR/9': [1.1, 0.9, 0.7],
        'IP': [120.0, 110.0, 115.0],
        'GS': [20, 18, 19],
        'playerid': [1001, 1002, 1003],
    })

@pytest.fixture
def sample_batting_stats():
    return pd.DataFrame({
        'Name': ['Aaron Judge', 'Shohei Ohtani', 'Mookie Betts'],
        'Team': ['NYY', 'LAD', 'LAD'],
        'wOBA': [0.420, 0.410, 0.380],
        'OPS': [1.050, 1.020, 0.950],
        'ISO': [0.320, 0.300, 0.250],
        'PA': [300, 310, 290],
        'AB': [270, 280, 260],
        'playerid': [2001, 2002, 2003],
    })

@pytest.fixture
def sample_game():
    return {
        'game_id': 745003,
        'game_date': '2026-06-25',
        'home_id': 147,
        'away_id': 111,
        'home_name': 'New York Yankees',
        'away_name': 'Boston Red Sox',
        'venue_id': 3313,
        'venue_name': 'Yankee Stadium',
        'game_datetime': '2026-06-25T23:05:00Z',
        'home_pitcher_id': 1001,
        'home_pitcher_name': 'Gerrit Cole',
        'away_pitcher_id': 1002,
        'away_pitcher_name': 'Shane Bieber',
        'home_pitcher_hand': 'R',
        'away_pitcher_hand': 'R',
        'status': 'Preview',
        'home_probable_pitcher': 'Gerrit Cole',
        'away_probable_pitcher': 'Shane Bieber',
    }

@pytest.fixture
def sample_team_batting():
    return pd.DataFrame({
        'Team': ['NYY', 'BOS'],
        'wOBA': [0.340, 0.325],
        'OPS': [0.820, 0.790],
        'R': [380, 340],
        'H': [700, 670],
    })

@pytest.fixture
def sample_weather():
    return {
        'temperature_f': 75.0,
        'wind_speed_mph': 12.0,
        'wind_direction_deg': 180.0,
        'wind_label': 'out_to_cf',
        'precipitation_mm': 0.0,
        'is_dome': False,
    }
