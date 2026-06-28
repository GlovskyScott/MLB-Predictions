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
    mocker.patch('src.app._current_version', return_value='v')
    mocker.patch('src.app._backfill_results_cache')  # don't spawn real re-sim
    response = client.post('/retrain')
    assert response.status_code in (302, 200)


# --- Versioned prediction store ---

def test_get_prediction_uses_store_no_resim(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='vTEST')
    P.save_prediction(tmp_path, 'vTEST', '2026-06-26',
                      [{'game_id': 1, 'home_win_pct': 55.0}])
    sim = mocker.patch('src.app.simulate_game')
    out = app.get_prediction('2026-06-26')
    assert out[0]['game_id'] == 1
    sim.assert_not_called()


def test_get_prediction_generates_and_persists_on_miss(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='vGEN')
    mocker.patch.object(app, '_get_models', return_value={'win': 1})
    mocker.patch.object(app, '_get_inning_model', return_value=None)
    mocker.patch('src.app.get_schedule', return_value=[
        {'game_id': 7, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
         'home_name': 'NYY', 'away_name': 'BOS', 'venue_name': 'YS', 'venue_id': 1,
         'game_datetime': '2026-06-26T23:05:00Z'}])
    mocker.patch('src.app.build_game_features', return_value={'elevation_ft': 10})
    mocker.patch('src.app.get_stadium', return_value={'roof': 'open', 'lat': 40.0, 'lon': -73.0})
    mocker.patch('src.app.get_weather_for_game', return_value={'is_dome': False})
    mocker.patch('src.app.predict_game', return_value={
        'home_win_prob': 0.6, 'away_win_prob': 0.4,
        'predicted_home_runs': 5.0, 'predicted_away_runs': 3.0})
    mocker.patch('src.app.simulate_game', return_value={
        'home_win_pct': 60.0, 'away_win_pct': 40.0,
        'median_home_score': 5.0, 'median_away_score': 3.0,
        'modal_home_score': 5, 'modal_away_score': 3, 'predicted_score': '5-3',
        'score_distribution': {'home': [], 'away': [], 'labels': []},
        'home_innings_scoring_pct': [], 'away_innings_scoring_pct': [],
        'home_innings': [], 'away_innings': [], 'n_simulations': 1000})

    out = app.get_prediction('2026-06-26')
    assert out[0]['game_id'] == 7 and out[0]['home_win_pct'] == 60.0
    # persisted under the current version
    assert P.load_prediction(tmp_path, 'vGEN', '2026-06-26')[0]['game_id'] == 7


def test_run_daily_simulation_enriches_stored_core(mocker):
    import src.app as app
    mocker.patch('src.app.get_prediction', return_value=[{
        'game_id': 7, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'NYY', 'away_name': 'BOS', 'home_win_pct': 60.0, 'away_win_pct': 40.0,
        'predicted_score': '5-3', 'median_home_score': 5.0, 'median_away_score': 3.0}])
    mocker.patch('src.app.get_schedule', return_value=[{
        'game_id': 7, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'NYY', 'away_name': 'BOS', 'venue_id': 1, 'venue_name': 'YS',
        'game_datetime': '2026-06-26T23:05:00Z',
        'home_probable_pitcher': 'A', 'away_probable_pitcher': 'B'}])
    mocker.patch('src.app.build_game_features', return_value={'elevation_ft': 10})
    mocker.patch('src.app.get_stadium', return_value={'roof': 'open', 'lat': 40.0, 'lon': -73.0})
    mocker.patch('src.app.get_weather_for_game', return_value={'is_dome': False})
    mocker.patch('src.app.get_game_lineup', return_value={})
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': 'L', 'primary': '#132448', 'secondary': '#C4CED4', 'abbr': 'X'})

    out = app.run_daily_simulation('2026-06-26')
    g = out[0]
    assert g['home_win_pct'] == 60.0          # from frozen core
    assert g['home_logo'] == 'L'              # enrichment
    assert g['weather']['is_dome'] is False   # enrichment


def test_compare_date_joins_without_resim(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'NYY', 'away_name': 'BOS', 'home_win_pct': 60.0, 'away_win_pct': 40.0,
        'median_home_score': 5.0, 'median_away_score': 3.0, 'predicted_score': '5-3'}])
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 6, 'away_score': 2, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    sim = mocker.patch('src.app.simulate_game')
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    data = app.compare_date('2026-06-26')
    assert data['n_completed'] == 1
    assert data['games'][0]['winner_correct'] is True   # predicted home win, home won
    assert data['games'][0]['home_score_err'] == 1.0    # |5 - 6|
    sim.assert_not_called()


def test_compare_date_archived_version_no_generate(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='vCURRENT')
    # archived version has its own stored prediction
    P.save_prediction(tmp_path, 'vOLD', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_win_pct': 30.0, 'median_home_score': 2.0, 'median_away_score': 5.0}])
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 6, 'away_score': 2, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    gp = mocker.patch('src.app.get_prediction')
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    data = app.compare_date('2026-06-26', version='vOLD')
    assert data['games'][0]['winner_correct'] is False  # predicted away (30%<50), home won
    gp.assert_not_called()  # archived version is frozen — no generate-on-miss


def test_retrain_forks_version_and_keeps_archive(client, tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    # live pkls present so _current_version() can hash them
    for n in ['model_win.pkl', 'model_runs_home.pkl', 'model_runs_away.pkl', 'model_inning.pkl']:
        (tmp_path / n).write_bytes(b'NEWMODEL')
    # an archived prior version with a stored prediction
    P.save_prediction(tmp_path, 'vOLD', '2026-06-26', [{'game_id': 1, 'home_win_pct': 30.0}])
    mocker.patch('src.app._get_models', return_value={'win': 1})
    mocker.patch('src.app._get_inning_model', return_value=None)
    backfill = mocker.patch('src.app._backfill_results_cache')

    resp = client.post('/retrain')
    assert resp.status_code in (302, 200)

    new_v = P.model_version(tmp_path)
    versions = [v['version'] for v in P.read_versions(tmp_path)]
    assert new_v in versions                                   # new version registered
    assert (tmp_path / 'predictions' / 'vOLD' / '2026-06-26.json').exists()  # archive intact
    backfill.assert_called_once()                              # re-sim kicked off


def test_archive_pages_render(client, tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='vCUR')
    P.append_version(tmp_path, {'version': 'vCUR', 'created_at': '2026-06-27T00:00:00+00:00',
                                'feature_version': 4, 'n_games': 5921})
    P.save_prediction(tmp_path, 'vCUR', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'NYY', 'away_name': 'BOS', 'home_win_pct': 60.0,
        'median_home_score': 5.0, 'median_away_score': 3.0, 'predicted_score': '5-3'}])
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 6, 'away_score': 2, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    r1 = client.get('/archive')
    assert r1.status_code == 200
    assert b'vCUR' in r1.data
    r2 = client.get('/archive/vCUR')
    assert r2.status_code == 200
    assert b'2026-06-26' in r2.data


def test_archive_unknown_version_404(client, tmp_path, mocker):
    import src.app as app
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='vCUR')
    # no versions.json -> unknown version must not touch the filesystem path
    assert client.get('/archive/..%2f..').status_code in (404, 308, 400)
    assert client.get('/archive/bogus').status_code == 404


def test_compare_date_win_pct_boundary_50(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    app._actuals_cache.clear()
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_win_pct': 50.0, 'median_home_score': 4.0, 'median_away_score': 4.0}])
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 5, 'away_score': 3, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})
    g = app.compare_date('2026-06-26')['games'][0]
    # 50.0 is not > 50.0 -> predicts away; home actually won -> incorrect (deterministic)
    assert g['predicted_home_won'] is False
    assert g['winner_correct'] is False


def test_actuals_for_date_memoizes_settled_dates(mocker):
    import src.app as app
    app._actuals_cache.clear()
    sched = mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-03-15', 'status': 'Final',
        'home_score': 5, 'away_score': 3, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    a1 = app._actuals_for_date('2026-03-15')   # old/settled -> caches
    a2 = app._actuals_for_date('2026-03-15')   # served from memo
    assert a1 == a2 and 1 in a1
    assert '2026-03-15' in app._actuals_cache
    assert sched.call_count == 1               # second call didn't recompute


def test_market_lines_runline_total_and_moneyline():
    import src.app as app
    # home runs ~ {4: .5, 5: .5} (mean 4.5); away runs ~ {3: .5, 4: .5} (mean 3.5)
    core = {
        'home_win_pct': 60.0, 'away_win_pct': 40.0,
        'score_distribution': {'labels': [0, 1, 2, 3, 4, 5, 6],
                               'home': [0, 0, 0, 0, 50, 50, 0],
                               'away': [0, 0, 0, 50, 50, 0, 0]},
    }
    L = app._market_lines(core)
    # moneyline: fair odds from win prob
    assert L['ml_home'] == '-150' and L['ml_away'] == '+150'
    # run line is the fixed 1.5 with odds; home is the favorite
    assert L['spread_home'].startswith('-1.5 ') and L['spread_away'].startswith('+1.5 ')
    # P(home wins by >=2) = 0.25 -> +300 ; the other side -300
    assert L['spread_home'] == '-1.5 +300' and L['spread_away'] == '+1.5 -300'
    # total line ends in .0/.5; here expected total 8.0
    assert L['total_line'] == '8.0'
    assert L['total_line'].endswith(('.0', '.5'))
    assert L['total_over'] == '-100' and L['total_under'] == '-100'


def test_market_lines_blank_without_distribution():
    import src.app as app
    L = app._market_lines({'home_win_pct': 55.0, 'away_win_pct': 45.0})
    assert L['ml_home'] == '-122' and L['spread_home'] == '—' and L['total_line'] == '—'
