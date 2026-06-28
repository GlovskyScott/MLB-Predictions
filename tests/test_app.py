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


def test_index_delivers_inning_distribution(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    html = client.get('/').data.decode()
    # Per-inning P(0/1/2+ runs) is delivered to the client (game-detail view).
    assert 'innings' in html
    assert '2+ runs' in html      # inning-moneyline buckets
    assert 'Both' in html         # combined (any team) toggle


def test_results_row_shows_ml_pick_only(client, mocker):
    mocker.patch('src.app.run_daily_simulation', return_value=[MOCK_SIM_RESULT])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value=MOCK_AGGREGATE)
    html = client.get('/').data.decode()
    assert 'NYY' in html          # yesterday's graded game present
    # moneyline only — run-line / total markets were removed entirely
    assert '>RL<' not in html and '>Tot<' not in html and '-1.5' not in html


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


def test_compare_date_grades_ml(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'NYY', 'away_name': 'BOS', 'home_win_pct': 70.0, 'away_win_pct': 30.0,
        'median_home_score': 5.0, 'median_away_score': 3.0, 'predicted_score': '5-3'}])
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
    assert g['winner_correct'] is True
    assert data['ml_accuracy'] == 100.0
    # spread/total markets were removed entirely
    assert 'spread_accuracy' not in data and 'total_accuracy' not in data
    assert 'spread_correct' not in g and 'total_correct' not in g
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


def test_market_block_applies_edge_threshold():
    import src.app as app
    mk = {'ml_home': -110, 'ml_away': -110}            # de-vigged market = 50%
    small = app._market_block({'home_win_pct': 52.0, 'home_abbr': 'H', 'away_abbr': 'A'}, mk)
    big = app._market_block({'home_win_pct': 62.0, 'home_abbr': 'H', 'away_abbr': 'A'}, mk)
    assert small['edge_ml_pct'] == 0      # 2% gap < 5% floor -> not an edge
    assert small['edge_ml_raw_pct'] == 2  # raw gap preserved
    assert big['edge_ml_pct'] == 12       # 12% gap >= floor -> shown
    assert big['edge_ml_side'] == 'H'


def test_history_page_shows_results_and_closing_line(client, tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='vCUR')
    P.append_version(tmp_path, {'version': 'vCUR', 'created_at': '2026-06-27T00:00:00+00:00',
                                'feature_version': 5, 'n_games': 5921})
    P.save_prediction(tmp_path, 'vCUR', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_name': 'Yankees', 'away_name': 'Red Sox', 'home_win_pct': 60.0,
        'median_home_score': 5.0, 'median_away_score': 3.0, 'predicted_score': '5-3'}])
    P.save_closing_odds(tmp_path, '2026-06-26',
                        {1: {'ml_home': -150, 'ml_away': 130, 'book': 'pinnacle'}})
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 6, 'away_score': 2, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'NYY'})

    r = client.get('/history')
    assert r.status_code == 200
    html = r.data.decode()
    assert '2026-06-26' in html          # date
    assert '2–' in html or '2-' in html  # final score (away 2)
    assert 'NYY' in html                 # the model's pick (home favored 60%)
    assert '-150' in html                # the real closing line


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


def test_market_lines_moneyline_only():
    import src.app as app
    L = app._market_lines({'home_win_pct': 60.0, 'away_win_pct': 40.0})
    # moneyline: fair odds from win prob
    assert L['ml_home'] == '-150' and L['ml_away'] == '+150'
    # run-line / total markets were removed
    assert 'spread_home' not in L and 'total_line' not in L


def test_market_block_moneyline_edge():
    import src.app as app
    game = {'home_win_pct': 55.0, 'away_win_pct': 45.0, 'home_abbr': 'BOS', 'away_abbr': 'NYY'}
    mk = {'ml_home': 130, 'ml_away': -150, 'n_books': 2}
    b = app._market_block(game, mk)
    assert b['ml_home'] == '+130' and b['ml_away'] == '-150'
    # moneyline edge (model prob - market implied), on the side the model favors
    assert b['edge_ml_side'] == 'BOS' and b['edge_ml_pct'] == 13     # model 55% vs ~42%
    # run-line / total edges were removed
    assert 'edge_total_pct' not in b and 'edge_rl_pct' not in b and 'runline_odds' not in b


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


def test_best_edge_moneyline_only():
    import src.app as app
    assert app._best_edge(None) is None
    assert app._best_edge({}) is None
    assert app._best_edge({'edge_ml_side': 'BOS', 'edge_ml_pct': 4}) == \
        {'market': 'ML', 'side': 'BOS', 'pct': 4}


def test_top_edges_ranks_and_caps():
    import src.app as app
    games = [
        {'game_id': 1, 'home_abbr': 'BAL', 'away_abbr': 'WSN',
         'market': {'edge_ml_side': 'WSN', 'edge_ml_pct': 9}},
        {'game_id': 2, 'home_abbr': 'NYY', 'away_abbr': 'BOS', 'market': None},  # skipped
        {'game_id': 3, 'home_abbr': 'LAD', 'away_abbr': 'SDP',
         'market': {'edge_ml_side': 'LAD', 'edge_ml_pct': 5}},
    ]
    out = app._top_edges(games, n=2)
    assert len(out) == 2                                  # capped
    assert [e['pct'] for e in out] == [9, 5]              # ranked desc
    assert out[0] == {'game_id': 1, 'matchup': 'WSN @ BAL',
                      'market': 'ML', 'side': 'WSN', 'pct': 9}
    assert all(e['game_id'] != 2 for e in out)            # no-market game skipped


def test_chat_context_includes_games_and_model(mocker):
    import src.app as app
    mocker.patch.object(app, '_simulation_cache', [{
        'game_id': 7, 'away_abbr': 'NYY', 'home_abbr': 'BOS',
        'away_name': 'New York Yankees', 'home_name': 'Boston Red Sox',
        'away_win_pct': 45.0, 'home_win_pct': 55.0, 'modal_away_score': 3, 'modal_home_score': 5,
        'away_pitcher': 'Cole', 'home_pitcher': 'Bello',
        'lines': {'ml_away': '+120', 'ml_home': '-130'},
        'market': {'ml_away': '+130', 'ml_home': '-150',
                   'edge_ml_side': 'BOS', 'edge_ml_pct': 6},
    }])
    mocker.patch.object(app, '_aggregate_days', return_value={
        'winner_accuracy': 60.0, 'total_games': 100, 'avg_score_err': 2.1})
    ctx = app._build_chat_context(focus_game_id=7)
    assert 'NYY @ BOS' in ctx and 'model win%' in ctx
    # computed verdict pins the value side to its EXACT market price (so the LLM
    # can't invent odds or reverse the edge)
    assert 'COMPUTED VERDICT' in ctx and 'BOS at -150' in ctx and '+6%' in ctx
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


# ---- market blend (Consensus moneyline) ------------------------------------

def test_blend_core_replaces_winpct_with_blend(tmp_path, mocker):
    import src.app as app
    from src import blend
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._blender_cache.clear()
    blend.save(blend.MarketBlender(a=0.5, b=0.5, c=0.0), tmp_path)
    core = {'home_win_pct': 80.0, 'away_win_pct': 20.0, 'raw_home_win_pct': 80.0}
    out = app._blend_core(core, market_home_prob=0.50)
    assert 50.0 < out['home_win_pct'] < 80.0          # pulled toward market
    assert out['away_win_pct'] == round(100 - out['home_win_pct'], 1)
    assert out['model_home_win_pct'] == 80.0          # model-only preserved
    app._blender_cache.clear()


def test_blend_core_falls_back_without_market(tmp_path, mocker):
    import src.app as app
    from src import blend
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._blender_cache.clear()
    blend.save(blend.MarketBlender(a=0.5, b=0.5, c=0.0), tmp_path)
    core = {'home_win_pct': 64.0, 'away_win_pct': 36.0, 'raw_home_win_pct': 64.0}
    out = app._blend_core(core, market_home_prob=None)
    assert out['home_win_pct'] == 64.0                # unchanged
    assert out['model_home_win_pct'] == 64.0
    app._blender_cache.clear()


def test_blend_core_identity_without_blender(tmp_path, mocker):
    import src.app as app
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._blender_cache.clear()                         # no blender file
    core = {'home_win_pct': 64.0, 'away_win_pct': 36.0, 'raw_home_win_pct': 64.0}
    out = app._blend_core(core, market_home_prob=0.50)
    assert out['home_win_pct'] == 64.0


def test_compare_date_grades_blended_pick(tmp_path, mocker):
    import src.app as app
    from src import predictions as P, blend
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    app._blender_cache.clear(); app._calibrator_cache.clear()
    # Blender leans hard on the market (a=0, b=3): the market decides the pick.
    blend.save(blend.MarketBlender(a=0.0, b=3.0, c=0.0), tmp_path)
    # Model favors HOME (70%); market favors AWAY (home +200 / away -240).
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_win_pct': 70.0, 'away_win_pct': 30.0, 'raw_home_win_pct': 70.0,
        'median_home_score': 4.0, 'median_away_score': 5.0}])
    P.save_market_odds(tmp_path, '2026-06-26', {1: {'ml_home': 200, 'ml_away': -240}})
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 3, 'away_score': 6, 'home_id': 147, 'away_id': 111}])  # away won
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    sim = mocker.patch('src.app.simulate_game')
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    data = app.compare_date('2026-06-26')
    g = data['games'][0]
    assert g['winner_correct'] is True          # blended followed market -> AWAY -> correct
    assert g['model_winner_correct'] is False    # model picked HOME -> wrong
    assert data['consensus_accuracy'] == 100.0
    assert data['model_accuracy'] == 0.0
    sim.assert_not_called()
    app._blender_cache.clear()


def test_index_headline_is_consensus(client, mocker):
    sim = dict(MOCK_SIM_RESULT)
    sim['home_win_pct'] = 58.0; sim['away_win_pct'] = 42.0
    sim['model_home_win_pct'] = 64.0   # model-only, must NOT be surfaced
    mocker.patch('src.app.run_daily_simulation', return_value=[sim])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value={
        'winner_accuracy': 58.0, 'consensus_accuracy': 58.0, 'model_accuracy': 55.0,
        'total_games': 100, 'avg_score_err': 2.1, 'daily': []})
    html = client.get('/').data.decode()
    assert 'Consensus' in html
    assert '>64<' not in html   # model-only number not surfaced as a standalone value
