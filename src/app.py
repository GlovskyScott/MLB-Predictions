import json
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.fetcher import (
    get_schedule, get_pitching_stats, get_team_batting_stats,
    get_bullpen_stats, get_season_schedule, get_weather_for_game,
)
from src.features import build_game_features, _NEUTRAL_WEATHER
from src.model import train_models, load_models, models_exist, predict_game, build_training_data
from src.simulator import simulate_game
from src.stadiums import get_stadium
from src.teams import get_team_meta

_DATA_DIR = Path(__file__).parent.parent / "data"


def _hex_to_rgb_str(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _bar_color(primary: str, secondary: str) -> str:
    """Return a visible bar color for dark backgrounds — uses secondary when primary is too dark."""
    h = primary.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return secondary if luminance < 0.25 else primary


_MODEL_META_FILE = _DATA_DIR / "model_meta.json"
_RESULTS_CACHE_DIR = _DATA_DIR / "results_cache"
_TRAINING_YEARS = [2024, 2025, 2026]

_simulation_cache: list[dict] = []
_last_simulated_date: str = ""
_models_cache: dict = {}
_results_cache: dict = {}
_last_results_date: str = ""


def _read_model_meta() -> dict:
    if _MODEL_META_FILE.exists():
        return json.loads(_MODEL_META_FILE.read_text())
    return {}


def _write_model_meta(n_games: int) -> None:
    _MODEL_META_FILE.write_text(json.dumps({
        'training_years': _TRAINING_YEARS,
        'n_games': n_games,
    }))


def _needs_retrain() -> bool:
    if not models_exist():
        return True
    meta = _read_model_meta()
    if meta.get('training_years') != _TRAINING_YEARS:
        return True
    return False


def _build_training_df(years: list[int] = None) -> pd.DataFrame:
    if years is None:
        years = _TRAINING_YEARS

    rows = []
    for year in years:
        all_games = get_season_schedule(year)
        completed = [
            g for g in all_games
            if g.get('status') == 'Final' and g.get('home_score') is not None
        ]
        for game in completed:
            try:
                features = build_game_features(game, year=year, weather=_NEUTRAL_WEATHER)
                features['home_score'] = float(game['home_score'])
                features['away_score'] = float(game['away_score'])
                features['home_win'] = 1 if float(game['home_score']) > float(game['away_score']) else 0
                features['status'] = 'Final'
                features['game_date'] = game['game_date']
                rows.append(features)
            except Exception:
                continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _get_models(force_retrain: bool = False) -> dict | None:
    global _models_cache
    if _models_cache and not force_retrain:
        return _models_cache

    if force_retrain or _needs_retrain():
        training_df = _build_training_df()
        if training_df.empty:
            return None
        models = train_models(training_df)
        _write_model_meta(len(training_df))
        _models_cache = models
        return models

    _models_cache = load_models()
    return _models_cache


def run_daily_simulation(sim_date: str = None, n_simulations: int = 1000) -> list[dict]:
    if sim_date is None:
        sim_date = date.today().strftime('%Y-%m-%d')

    models = _get_models()
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
            home_meta = get_team_meta(game['home_id'])
            away_meta = get_team_meta(game['away_id'])
            results.append({
                **game,
                **sim,
                'weather': weather,
                'home_pitcher': game.get('home_probable_pitcher', 'TBD'),
                'away_pitcher': game.get('away_probable_pitcher', 'TBD'),
                'home_logo': home_meta['logo_url'],
                'away_logo': away_meta['logo_url'],
                'home_color': home_meta['primary'],
                'away_color': away_meta['primary'],
                'home_color2': home_meta['secondary'],
                'away_color2': away_meta['secondary'],
                'home_abbr': home_meta['abbr'],
                'away_abbr': away_meta['abbr'],
                'home_color_rgb': _hex_to_rgb_str(home_meta['primary']),
                'away_color_rgb': _hex_to_rgb_str(away_meta['primary']),
                'home_bar_color': _bar_color(home_meta['primary'], home_meta['secondary']),
                'away_bar_color': _bar_color(away_meta['primary'], away_meta['secondary']),
            })
        except Exception as e:
            results.append({**game, 'error': str(e)})

    return results


def run_results_comparison(result_date: str, n_simulations: int = 500) -> dict:
    models = _get_models()
    games = get_schedule(result_date)
    completed = [g for g in games if g.get('status') == 'Final'
                 and g.get('home_score') is not None and g.get('away_score') is not None]

    results = []
    correct_winner = 0

    for game in completed:
        try:
            features = build_game_features(game, year=2026)
            if models:
                prediction = predict_game(features, models)
            else:
                prediction = {
                    'home_win_prob': 0.5, 'away_win_prob': 0.5,
                    'predicted_home_runs': 4.5, 'predicted_away_runs': 4.2,
                }

            sim = simulate_game(prediction, n_simulations=n_simulations, seed=game['game_id'] % 10000)
            home_meta = get_team_meta(game['home_id'])
            away_meta = get_team_meta(game['away_id'])

            actual_home = int(game['home_score'])
            actual_away = int(game['away_score'])
            actual_home_won = actual_home > actual_away
            predicted_home_won = sim['home_win_pct'] > 50.0

            home_score_err = abs(sim['median_home_score'] - actual_home)
            away_score_err = abs(sim['median_away_score'] - actual_away)

            if actual_home_won == predicted_home_won:
                correct_winner += 1

            results.append({
                **game,
                **sim,
                'actual_home_score': actual_home,
                'actual_away_score': actual_away,
                'actual_home_won': actual_home_won,
                'predicted_home_won': predicted_home_won,
                'home_score_err': round(home_score_err, 1),
                'away_score_err': round(away_score_err, 1),
                'winner_correct': actual_home_won == predicted_home_won,
                'home_logo': home_meta['logo_url'],
                'away_logo': away_meta['logo_url'],
                'home_color': home_meta['primary'],
                'away_color': away_meta['primary'],
                'home_color2': home_meta['secondary'],
                'away_color2': away_meta['secondary'],
            })
        except Exception as e:
            results.append({**game, 'error': str(e),
                            'actual_home_score': game.get('home_score'),
                            'actual_away_score': game.get('away_score')})

    accuracy = round(correct_winner / len(completed) * 100, 1) if completed else 0
    scored = [r for r in results if not r.get('error') and 'home_score_err' in r]
    avg_err = round(
        sum(r['home_score_err'] + r['away_score_err'] for r in scored) / (2 * len(scored)), 2
    ) if scored else None

    return {
        'result_date': result_date,
        'games': results,
        'n_completed': len(completed),
        'winner_accuracy': accuracy,
        'avg_score_err': avg_err,
    }


def _get_results_for_date(date_str: str) -> dict:
    """Return results for a past date, using disk cache to avoid re-simulation."""
    _RESULTS_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _RESULTS_CACHE_DIR / f"{date_str}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    data = run_results_comparison(date_str, n_simulations=200)
    if data.get('n_completed', 0) > 0:
        cache_file.write_text(json.dumps(data, default=str))
    return data


def _aggregate_days(n_days: int) -> dict:
    """Aggregate prediction accuracy across the past n_days using parallel disk-cached fetches."""
    dates = [
        (date.today() - timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(1, n_days + 1)
    ]

    daily = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_get_results_for_date, d): d for d in dates}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r.get('n_completed', 0) > 0:
                    daily.append(r)
            except Exception:
                pass

    daily.sort(key=lambda x: x['result_date'], reverse=True)

    total_games = sum(r['n_completed'] for r in daily)
    total_correct = sum(
        round(r['winner_accuracy'] / 100 * r['n_completed']) for r in daily
    )
    accuracy = round(total_correct / total_games * 100, 1) if total_games else 0
    scored = [r for r in daily if r.get('avg_score_err') is not None]
    avg_err = round(
        sum(r['avg_score_err'] for r in scored) / len(scored), 2
    ) if scored else None

    return {
        'total_games': total_games,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'avg_score_err': avg_err,
        'daily': daily,
    }


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['TESTING'] = testing

    @app.route('/')
    def index():
        global _simulation_cache, _last_simulated_date, _results_cache, _last_results_date
        today = date.today().strftime('%Y-%m-%d')
        yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

        if not _simulation_cache or _last_simulated_date != today:
            _simulation_cache = run_daily_simulation(today)
            _last_simulated_date = today

        if not _results_cache or _last_results_date != yesterday:
            try:
                _results_cache = _get_results_for_date(yesterday)
                _last_results_date = yesterday
            except Exception:
                _results_cache = {}

        last_7 = _aggregate_days(7)
        last_30 = _aggregate_days(30)

        return render_template('index.html',
                               results=_simulation_cache, sim_date=today,
                               yesterday=_results_cache, yesterday_date=yesterday,
                               last_7=last_7, last_30=last_30)

    @app.route('/refresh', methods=['POST'])
    def refresh():
        global _simulation_cache, _last_simulated_date
        today = date.today().strftime('%Y-%m-%d')
        _simulation_cache = run_daily_simulation(today)
        _last_simulated_date = today
        return redirect(url_for('index'))

    @app.route('/retrain', methods=['POST'])
    def retrain():
        global _models_cache
        _models_cache = {}
        _get_models(force_retrain=True)
        return redirect(url_for('index'))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
