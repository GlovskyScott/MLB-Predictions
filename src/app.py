import pandas as pd
from flask import Flask, render_template, redirect, url_for
from datetime import date
from pathlib import Path
from src.fetcher import (
    get_schedule, get_pitching_stats, get_team_batting_stats,
    get_bullpen_stats, get_season_schedule, get_weather_for_game,
)
from src.features import build_game_features
from src.model import train_models, load_models, models_exist, predict_game, build_training_data
from src.simulator import simulate_game
from src.stadiums import get_stadium

_DATA_DIR = Path(__file__).parent.parent / "data"
_simulation_cache: list[dict] = []
_last_simulated_date: str = ""


def _build_training_df(year: int = 2026) -> pd.DataFrame:
    """Build training DataFrame from completed season games."""
    all_games = get_season_schedule(year)
    completed = [
        g for g in all_games
        if g.get('status') == 'Final' and g.get('home_score') is not None
    ]

    rows = []
    for game in completed:
        try:
            features = build_game_features(game, year=year)
            features['home_score'] = float(game['home_score'])
            features['away_score'] = float(game['away_score'])
            features['home_win'] = 1 if float(game['home_score']) > float(game['away_score']) else 0
            features['status'] = 'Final'
            features['game_date'] = game['game_date']
            rows.append(features)
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_daily_simulation(sim_date: str = None, n_simulations: int = 1000) -> list[dict]:
    """Fetch today's games, train/load model, run simulations."""
    if sim_date is None:
        sim_date = date.today().strftime('%Y-%m-%d')

    if not models_exist():
        training_df = _build_training_df(2026)
        models = train_models(training_df) if not training_df.empty else None
    else:
        models = load_models()

    games = get_schedule(sim_date)
    results = []

    for game in games:
        try:
            features = build_game_features(game, year=2026)
            stadium = get_stadium(game['home_id']) or {}
            is_dome = stadium.get('roof') == 'dome'

            weather = get_weather_for_game(
                lat=stadium.get('lat', 39.0),
                lon=stadium.get('lon', -95.0),
                game_datetime=game.get('game_datetime', ''),
                is_dome=is_dome,
            )

            if models:
                prediction = predict_game(features, models)
            else:
                prediction = {
                    'home_win_prob': 0.5, 'away_win_prob': 0.5,
                    'predicted_home_runs': 4.5, 'predicted_away_runs': 4.2,
                }

            sim = simulate_game(prediction, n_simulations=n_simulations)
            results.append({
                **game,
                **sim,
                'weather': weather,
                'home_pitcher': game.get('home_probable_pitcher', 'TBD'),
                'away_pitcher': game.get('away_probable_pitcher', 'TBD'),
            })
        except Exception as e:
            results.append({**game, 'error': str(e)})

    return results


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['TESTING'] = testing

    @app.route('/')
    def index():
        global _simulation_cache, _last_simulated_date
        today = date.today().strftime('%Y-%m-%d')
        if not _simulation_cache or _last_simulated_date != today:
            _simulation_cache = run_daily_simulation(today)
            _last_simulated_date = today
        return render_template('index.html', results=_simulation_cache, sim_date=today)

    @app.route('/refresh', methods=['POST'])
    def refresh():
        global _simulation_cache, _last_simulated_date
        today = date.today().strftime('%Y-%m-%d')
        _simulation_cache = run_daily_simulation(today)
        _last_simulated_date = today
        return redirect(url_for('index'))

    @app.route('/game/<int:game_id>')
    def game_detail(game_id: int):
        global _simulation_cache
        game = next((g for g in _simulation_cache if g.get('game_id') == game_id), None)
        if game is None:
            return "Game not found", 404
        return render_template('game.html', game=game)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
