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

@pytest.fixture
def app():
    return create_app(testing=True)

@pytest.fixture
def client(app):
    return app.test_client()

def test_index_returns_200(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    response = client.get('/')
    assert response.status_code == 200

def test_index_contains_team_names(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    response = client.get('/')
    assert b'Yankees' in response.data or b'New York' in response.data

def test_index_contains_win_probability(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    response = client.get('/')
    assert b'58' in response.data

def test_refresh_redirects(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    response = client.post('/refresh')
    assert response.status_code in (302, 200)

def test_game_detail_returns_200(client, mocker):
    import src.app as app_module
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    app_module._simulation_cache = [MOCK_SIM_RESULT]
    response = client.get('/game/745003')
    assert response.status_code == 200

def test_index_handles_no_games(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[])
    response = client.get('/')
    assert response.status_code == 200

def test_results_route_returns_200(client, mocker):
    mocker.patch('src.app.run_results_comparison', return_value=MOCK_RESULTS_DATA)
    response = client.get('/results/2026-06-24')
    assert response.status_code == 200

def test_results_contains_accuracy(client, mocker):
    mocker.patch('src.app.run_results_comparison', return_value=MOCK_RESULTS_DATA)
    response = client.get('/results/2026-06-24')
    assert b'75' in response.data

def test_results_shows_correct_wrong(client, mocker):
    mocker.patch('src.app.run_results_comparison', return_value=MOCK_RESULTS_DATA)
    response = client.get('/results/2026-06-24')
    assert b'Correct' in response.data

def test_results_no_games(client, mocker):
    mocker.patch('src.app.run_results_comparison', return_value={
        'result_date': '2026-06-24', 'n_completed': 0,
        'winner_accuracy': 0, 'avg_score_err': None, 'games': [],
    })
    response = client.get('/results/2026-06-24')
    assert response.status_code == 200
    assert b'No completed games' in response.data

def test_retrain_redirects(client, mocker):
    mocker.patch('src.app._get_models', return_value={})
    response = client.post('/retrain')
    assert response.status_code in (302, 200)
