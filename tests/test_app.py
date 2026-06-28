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
    'home_innings_dist': [[70.0, 22.0, 8.0]] * 9,
    'away_innings_dist': [[78.0, 16.0, 6.0]] * 9,
    'combined_innings_dist': [[55.0, 30.0, 15.0]] * 9,
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
        'ml_correct': True, 'ml_pick': 'NYY',
        'spread_correct': False, 'spread_pick': 'NYY -1.5',
        'total_correct': None, 'total_pick': None,
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


def test_index_renders_inning_distribution_bars(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    response = client.get('/')
    assert b'inn-bar' in response.data        # 3-class stacked bars rendered
    assert b'Both' in response.data           # combined (any team scores) row


def test_results_row_shows_ml_and_spread_picks(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    html = client.get('/').data.decode()
    assert 'NYY -1.5' in html          # spread pick shown for the previous prediction
    assert '>ML<' in html and '>RL<' in html
    # Total pick is hidden for past games with no captured market line.
    assert '>Tot<' not in html


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
    mocker.patch('src.app.get_market_odds', return_value={})  # no live ESPN call
    mocker.patch('src.app._get_calibrator', return_value=None)  # identity: test enrichment only
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


def test_compare_date_grades_ml_total_spread(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    # Deterministic core: home always 5, away always 3 -> total 8, home covers -1.5.
    dist = {'labels': [0, 1, 2, 3, 4, 5], 'home': [0, 0, 0, 0, 0, 1], 'away': [0, 0, 0, 1, 0, 0]}
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'NYY', 'away_name': 'BOS', 'home_win_pct': 70.0, 'away_win_pct': 30.0,
        'median_home_score': 5.0, 'median_away_score': 3.0, 'predicted_score': '5-3',
        'score_distribution': dist}])
    P.save_market_lines(tmp_path, '2026-06-26', {1: 7.5})   # market total line
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 5, 'away_score': 3, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    sim = mocker.patch('src.app.simulate_game')
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    data = app.compare_date('2026-06-26')
    g = data['games'][0]
    assert g['ml_correct'] is True          # picked home (70%), home won
    assert g['spread_correct'] is True      # home won by 2 -> covers -1.5
    assert g['total_correct'] is True       # total 8 > 7.5, over picked
    assert data['ml_accuracy'] == 100.0
    assert data['spread_accuracy'] == 100.0
    assert data['total_accuracy'] == 100.0
    assert data['total_graded'] == 1
    sim.assert_not_called()


def test_compare_date_total_na_without_line(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    dist = {'labels': [0, 1, 2, 3, 4, 5], 'home': [0, 0, 0, 0, 0, 1], 'away': [0, 0, 0, 1, 0, 0]}
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_win_pct': 70.0, 'away_win_pct': 30.0, 'median_home_score': 5.0,
        'median_away_score': 3.0, 'score_distribution': dist}])
    # No market line saved for this date -> Total is N/A.
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 5, 'away_score': 3, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    mocker.patch('src.app.simulate_game')
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    data = app.compare_date('2026-06-26')
    assert data['games'][0]['total_correct'] is None
    assert data['total_graded'] == 0
    assert data['total_accuracy'] is None   # nothing graded
    assert data['spread_accuracy'] == 100.0  # spread still graded


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


def test_market_block_edges_and_default_odds():
    import src.app as app
    # model: home runs {4:.5,5:.5}, away {3:.5,4:.5}; home_win 55%
    game = {'home_win_pct': 55.0, 'away_win_pct': 45.0, 'home_abbr': 'BOS', 'away_abbr': 'NYY',
            'score_distribution': {'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   'home': [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
                                   'away': [0, 0, 0, 50, 50, 0, 0, 0, 0, 0]}}
    mk = {'total': 7.5, 'over_odds': None, 'under_odds': None,   # missing -> default -110
          'ml_home': 130, 'ml_away': -150, 'n_books': 2}
    b = app._market_block(game, mk)
    # run line + total prices default to -110 when the book doesn't list them
    assert b['runline_odds'] == '-110'
    assert b['total_over'] == '-110' and b['total_under'] == '-110'
    assert b['total'] == '7.5' and b['ml_home'] == '+130' and b['ml_away'] == '-150'
    # edges (model prob - market implied), shown on the side the model favors
    assert b['edge_ml_side'] == 'BOS' and b['edge_ml_pct'] == 13     # model 55% vs ~42%
    assert b['edge_total_side'] == 'Over' and b['edge_total_pct'] == 25  # model O 75% vs 50%
    assert b['edge_rl_side'] == 'NYY +1.5' and b['edge_rl_pct'] == 25   # fav covers 25% vs 50%


def test_calibrate_core_applies_inning_dist_calibrator(tmp_path, mocker):
    import src.app as app
    from src import calibration as cal
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._calibrator_cache.clear()
    # A shrinking per-class map -> calibrated dist differs but each inning still sums to 100.
    mc = cal.MulticlassInningCalibrator(
        [cal.PlattCalibrator(a=0.5), cal.PlattCalibrator(a=0.5), cal.PlattCalibrator(a=0.5)])
    cal.save_multiclass(mc, tmp_path)
    core = {
        'home_win_pct': 55.0,
        'home_innings_dist': [[70.0, 22.0, 8.0]] * 9,
        'away_innings_dist': [[80.0, 15.0, 5.0]] * 9,
        'combined_innings_dist': [[56.0, 30.0, 14.0]] * 9,
    }
    out = app._calibrate_core(core)
    app._calibrator_cache.clear()
    assert out['home_innings_dist'][0] != [70.0, 22.0, 8.0]      # calibrated
    assert abs(sum(out['home_innings_dist'][0]) - 100.0) < 0.5    # renormalized
    assert abs(sum(out['combined_innings_dist'][0]) - 100.0) < 0.5  # recomputed + valid


def test_weighted_market_accuracy_pools_by_graded_count():
    import src.app as app
    daily = [
        {'spread_accuracy': 50.0, 'spread_graded': 10},   # 5 correct
        {'spread_accuracy': 100.0, 'spread_graded': 10},  # 10 correct
        {'spread_accuracy': None, 'spread_graded': 0},    # ignored
    ]
    acc, n = app._weighted_market_accuracy(daily, 'spread_accuracy', 'spread_graded')
    assert n == 20 and acc == 75.0   # 15/20


def test_weighted_market_accuracy_none_when_nothing_graded():
    import src.app as app
    daily = [{'total_accuracy': None, 'total_graded': 0}]
    acc, n = app._weighted_market_accuracy(daily, 'total_accuracy', 'total_graded')
    assert acc is None and n == 0


def test_market_total_lines_maps_priced_games_only():
    import src.app as app
    from src.fetcher import market_key
    games = [
        {'game_id': 1, 'away_name': 'Boston Red Sox', 'home_name': 'New York Yankees'},
        {'game_id': 2, 'away_name': 'Chicago Cubs', 'home_name': 'St. Louis Cardinals'},
        {'game_id': 3, 'away_name': 'No Odds', 'home_name': 'Team'},
    ]
    market = {
        market_key('Boston Red Sox', 'New York Yankees'): {'total': 8.5, 'n_books': 3},
        market_key('Chicago Cubs', 'St. Louis Cardinals'): {'total': None, 'n_books': 2},
    }
    out = app._market_total_lines(games, market)
    assert out == {1: 8.5}   # game 2 has no total; game 3 has no market


def test_best_edge_picks_largest():
    import src.app as app
    assert app._best_edge(None) is None
    assert app._best_edge({}) is None
    # only ML priced -> ML wins by default
    assert app._best_edge({'edge_ml_side': 'BOS', 'edge_ml_pct': 4,
                           'edge_total_side': '', 'edge_total_pct': None,
                           'edge_rl_side': '', 'edge_rl_pct': None}) == \
        {'market': 'ML', 'side': 'BOS', 'pct': 4}
    # picks the biggest across the three markets
    best = app._best_edge({'edge_ml_side': 'BOS', 'edge_ml_pct': 3,
                           'edge_total_side': 'Over', 'edge_total_pct': 7,
                           'edge_rl_side': 'NYY +1.5', 'edge_rl_pct': 2})
    assert best == {'market': 'Total', 'side': 'Over', 'pct': 7}


def test_top_edges_ranks_and_caps():
    import src.app as app
    games = [
        {'game_id': 1, 'home_abbr': 'BAL', 'away_abbr': 'WSN',
         'market': {'edge_ml_side': 'WSN', 'edge_ml_pct': 3,
                    'edge_total_side': 'Under', 'edge_total_pct': 9,
                    'edge_rl_side': '', 'edge_rl_pct': None}},
        {'game_id': 2, 'home_abbr': 'NYY', 'away_abbr': 'BOS', 'market': None},  # skipped
        {'game_id': 3, 'home_abbr': 'LAD', 'away_abbr': 'SDP',
         'market': {'edge_ml_side': 'LAD', 'edge_ml_pct': 5,
                    'edge_total_side': '', 'edge_total_pct': None,
                    'edge_rl_side': 'LAD -1.5', 'edge_rl_pct': 1}},
    ]
    out = app._top_edges(games, n=2)
    assert len(out) == 2                                  # capped
    assert [e['pct'] for e in out] == [9, 5]              # ranked desc
    assert out[0] == {'game_id': 1, 'matchup': 'WSN @ BAL',
                      'market': 'Total', 'side': 'Under', 'pct': 9}
    assert all(e['game_id'] != 2 for e in out)            # no-market game skipped


def test_chat_context_includes_games_and_model(mocker):
    import src.app as app
    mocker.patch.object(app, '_simulation_cache', [{
        'game_id': 7, 'away_abbr': 'NYY', 'home_abbr': 'BOS',
        'away_name': 'New York Yankees', 'home_name': 'Boston Red Sox',
        'away_win_pct': 45.0, 'home_win_pct': 55.0, 'modal_away_score': 3, 'modal_home_score': 5,
        'away_pitcher': 'Cole', 'home_pitcher': 'Bello',
        'lines': {'total_line': '9.5', 'ml_away': '+120', 'ml_home': '-130'},
        'market': {'total': 8.5, 'ml_away': '+130', 'ml_home': '-150',
                   'edge_ml_side': 'BOS', 'edge_ml_pct': 6, 'edge_total_side': 'Over',
                   'edge_total_pct': 5, 'edge_rl_side': 'NYY +1.5', 'edge_rl_pct': 4},
    }])
    mocker.patch.object(app, '_aggregate_days', return_value={
        'winner_accuracy': 60.0, 'total_games': 100, 'avg_score_err': 2.1})
    ctx = app._build_chat_context(focus_game_id=7)
    assert 'NYY @ BOS' in ctx and 'model win%' in ctx
    assert 'edges' in ctx and 'BOS +6%' in ctx
    assert 'HISTORICAL ACCURACY' in ctx and 'MODEL:' in ctx
    assert 'FOCUS GAME' in ctx and 'Boston Red Sox' in ctx


def test_chat_route_streams(client, mocker):
    import src.app as app
    mocker.patch.object(app, '_build_chat_context', return_value='ctx')
    mocker.patch('src.app.stream_chat', return_value=iter(['Hello', ' there']))
    r = client.post('/chat', json={'messages': [{'role': 'user', 'content': 'hi'}]})
    assert r.status_code == 200 and r.get_data(as_text=True) == 'Hello there'
    # empty conversation rejected
    assert client.post('/chat', json={'messages': []}).status_code == 400
