import pytest
from unittest.mock import patch, MagicMock
from src.app import create_app

MOCK_SIM_RESULT = {
    'game_id': 745003,
    'game_date': '2026-06-25',
    'home_name': 'New York Yankees',
    'away_name': 'Boston Red Sox',
    'venue_name': 'Yankee Stadium',
    'home_win_pct': 58.2,
    'away_win_pct': 41.8,
    'median_home_score': 5.0,
    'median_away_score': 3.0,
    'predicted_score': '5-3',
    'home_innings': [0.3, 0.2, 0.5, 0.6, 0.4, 0.3, 0.5, 0.7, 0.5],
    'away_innings': [0.2, 0.3, 0.4, 0.3, 0.5, 0.3, 0.4, 0.4, 0.3],
    'home_innings_scoring_pct': [30.0, 20.0, 50.0, 60.0, 40.0, 30.0, 50.0, 70.0, 50.0],
    'away_innings_scoring_pct': [20.0, 30.0, 40.0, 30.0, 50.0, 30.0, 40.0, 40.0, 30.0],
    'score_distribution': {
        'home': [10, 20, 40, 60, 80],
        'away': [15, 25, 45, 55, 70],
        'labels': [0, 1, 2, 3, 4],
    },
    'n_simulations': 1000,
    'game_datetime': '2026-06-25T23:05:00Z',
    'weather': {'temperature_f': 75.0, 'wind_speed_mph': 12.0, 'is_dome': False,
                'wind_direction_deg': 180.0, 'precipitation_mm': 0.0},
    'home_pitcher': 'Gerrit Cole',
    'away_pitcher': 'Shane Bieber',
    'status': 'Preview',
    'home_id': 147,
    'away_id': 111,
    'venue_id': 3313,
    'home_probable_pitcher': 'Gerrit Cole',
    'away_probable_pitcher': 'Shane Bieber',
    'home_logo': 'https://www.mlbstatic.com/team-logos/147.svg',
    'away_logo': 'https://www.mlbstatic.com/team-logos/111.svg',
    'home_color': '#132448', 'home_color2': '#C4CED4',
    'away_color': '#BD3039', 'away_color2': '#0C2340',
    'home_abbr': 'NYY', 'away_abbr': 'BOS',
    'home_color_rgb': '19,36,72', 'away_color_rgb': '189,48,57',
    'home_bar_color': '#132448', 'away_bar_color': '#BD3039',
}

MOCK_RESULTS_DATA = {
    'result_date': '2026-06-24',
    'n_completed': 1,
    'winner_accuracy': 75.0,
    'avg_score_err': 1.5,
    'games': [{
        **MOCK_SIM_RESULT,
        'status': 'Final',
        'actual_home_score': 6,
        'actual_away_score': 3,
        'actual_home_won': True,
        'predicted_home_won': True,
        'home_score_err': 1.0,
        'away_score_err': 0.5,
        'winner_correct': True,
    }],
}

MOCK_AGGREGATE = {
    'total_games': 10,
    'days_with_games': 7,
    'winner_accuracy': 58.0,
    'avg_score_err': 1.8,
    'daily': [],
}


@pytest.fixture
def app():
    return create_app(testing=True)


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_returns_200(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    response = client.get('/')
    assert response.status_code == 200


def test_index_contains_team_names(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    response = client.get('/')
    assert b'Yankees' in response.data or b'New York' in response.data


def test_index_contains_win_probability(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    response = client.get('/')
    assert b'58' in response.data


def test_refresh_redirects(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    response = client.post('/refresh')
    assert response.status_code in (302, 200)


def test_index_handles_no_games(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    response = client.get('/')
    assert response.status_code == 200


def test_retrain_redirects(client, mocker):
    mocker.patch('src.app._get_models', return_value={})
    mocker.patch('src.app._get_inning_model', return_value=None)
    response = client.post('/retrain')
    assert response.status_code in (302, 200)
