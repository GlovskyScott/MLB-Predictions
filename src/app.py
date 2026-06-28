import json
import threading
import requests as _requests
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, Response, stream_with_context
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.fetcher import (
    get_schedule, get_pitching_stats, get_team_batting_stats,
    get_bullpen_stats, get_season_schedule, get_weather_for_game, get_game_linescore,
    get_game_lineup, bootstrap_data_cache, bootstrap_model_cache,
    refresh_schedule_date,
)
from src.features import build_game_features, build_inning_feature_row, _NEUTRAL_WEATHER, FEATURE_VERSION
from src.model import (
    train_models, load_models, models_exist, predict_game, build_training_data,
    train_inning_model, load_inning_model, inning_model_exists, predict_inning_probs,
)
from src.simulator import simulate_game
from src.stadiums import get_stadium
from src.teams import get_team_meta
from src import predictions as _pred
from datetime import datetime, timezone

_DATA_DIR = Path(__file__).parent.parent / "data"


def _hex_to_rgb_str(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _bar_color(primary: str, secondary: str) -> str:
    """Return a legible team color for dark backgrounds.
    Picks the brighter of primary/secondary, then blends toward white until
    the result meets the minimum readable luminance.
    """
    MIN_LUM = 0.28

    def _lum(hex_color: str) -> float:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
        return 0.299 * r + 0.587 * g + 0.114 * b

    color = primary if _lum(primary) >= _lum(secondary) else secondary

    if _lum(color) >= MIN_LUM:
        return color

    # Blend toward white in 5% steps until readable
    h = color.lstrip('#')
    r0, g0, b0 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    for step in range(5, 100, 5):
        t = step / 100.0
        r = min(int(r0 + (255 - r0) * t), 255)
        g = min(int(g0 + (255 - g0) * t), 255)
        b = min(int(b0 + (255 - b0) * t), 255)
        if 0.299 * (r / 255) + 0.587 * (g / 255) + 0.114 * (b / 255) >= MIN_LUM:
            return f"#{r:02x}{g:02x}{b:02x}"

    return '#888888'


_MODEL_META_FILE = _DATA_DIR / "model_meta.json"
_EXPLANATIONS_DIR = _DATA_DIR / "explanations"
_TRAINING_YEARS = [2024, 2025, 2026]
N_SIMULATIONS = 1000  # fixed sim count for the prediction of record (live + backfill)

_simulation_cache: list[dict] = []
_last_simulated_date: str = ""
_models_cache: dict = {}
_inning_model_cache = None
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
        'feature_version': FEATURE_VERSION,
    }))


def _needs_retrain() -> bool:
    if not models_exist():
        return True
    meta = _read_model_meta()
    if meta.get('training_years') != _TRAINING_YEARS:
        return True
    if meta.get('feature_version') != FEATURE_VERSION:
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
                features = build_game_features(game, year=year, weather=_NEUTRAL_WEATHER, for_training=True)
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


def _build_inning_training_df(years: list[int] = None) -> pd.DataFrame:
    if years is None:
        years = _TRAINING_YEARS
    rows = []
    for year in years:
        all_games = get_season_schedule(year)
        completed = [
            g for g in all_games
            if g.get('status') == 'Final' and g.get('home_score') is not None
        ]
        # Parallel-fetch all linescores first (populates disk cache)
        game_ids = [int(g['game_id']) for g in completed]
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(get_game_linescore, gid): gid for gid in game_ids}
            for fut in as_completed(futures):
                fut.result()
        for game in completed:
            linescore = get_game_linescore(int(game['game_id']))
            if not linescore:
                continue
            try:
                game_feats = build_game_features(game, year=year, weather=_NEUTRAL_WEATHER, for_training=True)
            except Exception:
                continue
            for inning in range(1, 10):
                for batting_is_home in (True, False):
                    key = 'home' if batting_is_home else 'away'
                    inn_runs = linescore[key][inning - 1]
                    row = build_inning_feature_row(game_feats, inning, batting_is_home)
                    row['scored'] = 1 if inn_runs >= 1 else 0
                    rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _get_inning_model(force_retrain: bool = False):
    global _inning_model_cache
    if _inning_model_cache and not force_retrain:
        return _inning_model_cache
    if not force_retrain and inning_model_exists():
        _inning_model_cache = load_inning_model()
        return _inning_model_cache
    df = _build_inning_training_df()
    if df.empty or len(df) < 1000:
        return None
    _inning_model_cache = train_inning_model(df)
    return _inning_model_cache


def _current_version() -> str | None:
    """Version id of the live model pkls. Registers it on first sight."""
    v = _pred.model_version(_DATA_DIR)
    if v and not any(e.get('version') == v for e in _pred.read_versions(_DATA_DIR)):
        _pred.append_version(_DATA_DIR, {
            'version': v,
            'created_at': datetime.now(timezone.utc).isoformat(),
            **_read_model_meta(),
        })
    return v


def _simulate_core_for_date(sim_date: str, models, inning_model) -> list[dict]:
    """Generate the deterministic prediction core for every game on sim_date.

    Used by both the live "today" path and the backfill, so the artifact shown
    live is byte-for-byte what later appears under "yesterday".
    """
    cores = []
    for game in get_schedule(sim_date):
        try:
            features = build_game_features(game, year=2026)
            stadium = get_stadium(game['home_id']) or {}
            is_dome = stadium.get('roof') == 'dome'
            weather = get_weather_for_game(
                lat=stadium.get('lat', 39.0), lon=stadium.get('lon', -95.0),
                game_datetime=game.get('game_datetime', ''), is_dome=is_dome,
            )
            if models:
                prediction = predict_game(features, models)
            else:
                prediction = {'home_win_prob': 0.5, 'away_win_prob': 0.5,
                              'predicted_home_runs': 4.5, 'predicted_away_runs': 4.2}
            sim = simulate_game(prediction, n_simulations=N_SIMULATIONS,
                                seed=game['game_id'] % 100000)
            if inning_model:
                try:
                    inning_probs = predict_inning_probs(features, inning_model)
                    sim['home_innings_scoring_pct'] = inning_probs['home']
                    sim['away_innings_scoring_pct'] = inning_probs['away']
                except Exception:
                    pass
            cores.append(_pred.extract_core({**game, **sim}))
        except Exception:
            continue
    return cores


def get_prediction(sim_date: str) -> list[dict]:
    """Return the frozen prediction core for sim_date under the current version.

    Served verbatim if it exists; otherwise simulated once, persisted, returned.
    Never re-simulates a date that already has a stored prediction.
    """
    version = _current_version()
    if version:
        stored = _pred.load_prediction(_DATA_DIR, version, sim_date)
        if stored is not None:
            return stored
    cores = _simulate_core_for_date(sim_date, _get_models(), _get_inning_model())
    if version:
        _pred.save_prediction(_DATA_DIR, version, sim_date, cores)
    return cores


def _enrich_game(game: dict, core: dict) -> dict:
    """Combine a schedule game shell with its frozen prediction core and the
    presentation/context fields (weather, lineup, features, logos, colors) that
    are re-derived at render time rather than persisted in the artifact."""
    features = build_game_features(game, year=2026)
    stadium = get_stadium(game['home_id']) or {}
    is_dome = stadium.get('roof') == 'dome'
    weather = get_weather_for_game(
        lat=stadium.get('lat', 39.0), lon=stadium.get('lon', -95.0),
        game_datetime=game.get('game_datetime', ''), is_dome=is_dome,
    )
    home_meta = get_team_meta(game['home_id'])
    away_meta = get_team_meta(game['away_id'])
    lineup = get_game_lineup(game.get('game_id'))
    return {
        **game,
        **core,
        'weather': weather,
        'lineup': lineup,
        'features': features,
        'elevation_ft': int(features.get('elevation_ft', 0)),
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
    }


def run_daily_simulation(sim_date: str = None, n_simulations: int = 1000) -> list[dict]:
    if sim_date is None:
        sim_date = date.today().strftime('%Y-%m-%d')

    cores = {c['game_id']: c for c in get_prediction(sim_date)}
    results = []
    for game in get_schedule(sim_date):
        try:
            results.append(_enrich_game(game, cores.get(game['game_id'], {})))
        except Exception as e:
            results.append({**game, 'error': str(e)})

    return results


def _is_final_game(g) -> bool:
    return (g.get('status') == 'Final'
            and g.get('home_score') is not None and not pd.isna(g.get('home_score'))
            and g.get('away_score') is not None and not pd.isna(g.get('away_score')))


def _actuals_for_date(result_date: str) -> dict:
    """Return {game_id: game} for FINAL games on result_date, refreshing the
    cached schedule from the live API when it isn't fully final yet."""
    year = int(result_date[:4])
    all_games = get_season_schedule(year)
    games = [g for g in all_games if g.get('game_date') == result_date]
    if not games or not all(_is_final_game(g) for g in games):
        if refresh_schedule_date(year, result_date) > 0:
            all_games = get_season_schedule(year)
            games = [g for g in all_games if g.get('game_date') == result_date]
        if not games:
            games = get_schedule(result_date)  # fallback for today/future
    return {g['game_id']: g for g in games if _is_final_game(g)}


def compare_date(result_date: str, version: str = None) -> dict:
    """Join the frozen prediction for result_date with actual finals — no sim.

    For the current version, generates-on-miss; for an archived version, reads
    only what is stored (frozen). Accuracy is computed from the stored core's
    win% and median scores, so it never changes unless the model version does.
    """
    current = _current_version()
    if version is None:
        version = current
    if version == current:
        cores = get_prediction(result_date)
    else:
        cores = _pred.load_prediction(_DATA_DIR, version, result_date) or []

    actuals = _actuals_for_date(result_date)
    results = []
    correct = 0
    for core in cores:
        game = actuals.get(core.get('game_id'))
        if not game:
            continue  # not final yet — no comparison row
        actual_home = int(game['home_score'])
        actual_away = int(game['away_score'])
        actual_home_won = actual_home > actual_away
        predicted_home_won = core.get('home_win_pct', 50.0) > 50.0
        home_err = abs(core.get('median_home_score', 0) - actual_home)
        away_err = abs(core.get('median_away_score', 0) - actual_away)
        if actual_home_won == predicted_home_won:
            correct += 1
        home_meta = get_team_meta(game['home_id'])
        away_meta = get_team_meta(game['away_id'])
        results.append({
            **game,
            **core,
            'actual_home_score': actual_home,
            'actual_away_score': actual_away,
            'actual_home_won': actual_home_won,
            'predicted_home_won': predicted_home_won,
            'home_score_err': round(home_err, 1),
            'away_score_err': round(away_err, 1),
            'winner_correct': actual_home_won == predicted_home_won,
            'home_logo': home_meta['logo_url'],
            'away_logo': away_meta['logo_url'],
            'home_color': home_meta['primary'],
            'away_color': away_meta['primary'],
            'home_color2': home_meta['secondary'],
            'away_color2': away_meta['secondary'],
        })

    n = len(results)
    accuracy = round(correct / n * 100, 1) if n else 0
    scored = [r for r in results if 'home_score_err' in r]
    avg_err = round(
        sum(r['home_score_err'] + r['away_score_err'] for r in scored) / (2 * len(scored)), 2
    ) if scored else None

    return {
        'result_date': result_date,
        'games': results,
        'n_completed': n,
        'winner_accuracy': accuracy,
        'avg_score_err': avg_err,
    }


def _get_results_for_date(date_str: str) -> dict:
    """Results for a past date by joining the frozen prediction with actuals."""
    return compare_date(date_str)


def _aggregate_days(n_days: int, version: str = None) -> dict:
    """Aggregate accuracy across the past n_days for a model version.

    Only joins dates that already have a stored prediction for the version, so
    the page never triggers simulation on load — generation happens in the
    backfill thread.
    """
    if version is None:
        version = _current_version()
    dates = [
        (date.today() - timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(1, n_days + 1)
    ]
    ready = (
        [d for d in dates if _pred.load_prediction(_DATA_DIR, version, d) is not None]
        if version else []
    )

    daily = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(compare_date, d, version): d for d in ready}
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


def _backfill_one(d: str) -> None:
    """Ensure a stored prediction exists for date d under the current version."""
    try:
        get_prediction(d)  # no-op if already stored; generates + persists on miss
    except Exception:
        pass


def _backfill_results_cache(n_days: int = 90) -> None:
    """Generate-and-persist predictions for the last n_days under the current
    model version, using parallel workers. Runs after startup (and after a
    retrain) so the index loads immediately; already-stored dates are skipped.
    """
    version = _current_version()
    dates = [
        (date.today() - timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(1, n_days + 1)
    ]
    todo = [
        d for d in dates
        if not version or _pred.load_prediction(_DATA_DIR, version, d) is None
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pool.map(_backfill_one, todo)


_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.1:8b"
_explanation_cache: dict = {}  # game_id → full explanation text


def _load_disk_explanations(game_date: str) -> None:
    """Load any saved explanations for game_date from disk into memory cache."""
    day_dir = _EXPLANATIONS_DIR / game_date
    if not day_dir.exists():
        return
    for f in day_dir.glob("*.txt"):
        try:
            gid = int(f.stem)
            if gid not in _explanation_cache:
                _explanation_cache[gid] = f.read_text()
        except (ValueError, OSError):
            pass


def _save_disk_explanation(game_date: str, game_id: int, text: str) -> None:
    """Persist one explanation to disk under data/explanations/{date}/{game_id}.txt."""
    day_dir = _EXPLANATIONS_DIR / game_date
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / f"{game_id}.txt").write_text(text)


def _build_explain_prompt(game: dict) -> str:
    f = game.get('features', {})
    away = game.get('away_name', 'Away')
    home = game.get('home_name', 'Home')
    venue = game.get('venue_name', 'the ballpark')
    away_p = game.get('away_pitcher', 'TBD')
    home_p = game.get('home_pitcher', 'TBD')
    away_win = game.get('away_win_pct', 50)
    home_win = game.get('home_win_pct', 50)
    modal_away = game.get('modal_away_score', '?')
    modal_home = game.get('modal_home_score', '?')

    away_hand = 'LHP' if f.get('away_sp_is_lhp', 0) > 0.5 else 'RHP'
    home_hand = 'LHP' if f.get('home_sp_is_lhp', 0) > 0.5 else 'RHP'

    weather = game.get('weather') or {}
    is_dome = f.get('is_dome', 0) > 0.5
    if is_dome:
        wx = 'indoor dome — weather not a factor'
    else:
        wx = f"{weather.get('temperature_f', '?'):.0f}°F, {weather.get('wind_speed_mph', 0):.0f} mph"
        if f.get('wind_out', 0) > 0.5:
            wx += ' blowing out (hitter-friendly)'
        elif f.get('wind_in', 0) > 0.5:
            wx += ' blowing in (pitcher-friendly)'

    return f"""You are a sharp baseball analyst. Write 2-3 tight paragraphs explaining why the model predicts this outcome. Be specific, cite the numbers, and lead with the most decisive factors. No bullet points. Confident, present-tense analyst voice. Keep it under 200 words.

{away} @ {home} — {venue}
Win probability: {away} {away_win}% | {home} {home_win}%
Most likely score: {away} {modal_away} – {home} {modal_home}

Starters:
  {away}: {away_p} ({away_hand}) ERA {f.get('away_sp_era',0):.2f} FIP {f.get('away_sp_fip',0):.2f} WHIP {f.get('away_sp_whip',0):.2f} — {f.get('away_sp_days_rest',5):.0f}d rest
  {home}: {home_p} ({home_hand}) ERA {f.get('home_sp_era',0):.2f} FIP {f.get('home_sp_fip',0):.2f} WHIP {f.get('home_sp_whip',0):.2f} — {f.get('home_sp_days_rest',5):.0f}d rest

Offense (wOBA / OPS / R/G last 15):
  {away}: {f.get('away_team_woba',0):.3f} / {f.get('away_team_ops',0):.3f} / {f.get('away_runs_l15',0):.1f}
  {home}: {f.get('home_team_woba',0):.3f} / {f.get('home_team_ops',0):.3f} / {f.get('home_runs_l15',0):.1f}

Bullpen (ERA / L3 stress):
  {away}: {f.get('away_bullpen_era',0):.2f} ERA / {f.get('away_bullpen_stress_l3',0):.1f} stress
  {home}: {f.get('home_bullpen_era',0):.2f} ERA / {f.get('home_bullpen_stress_l3',0):.1f} stress

Handedness OPS edge:
  {away} vs {home_hand}: {f.get('away_bat_ops_vs_sp_hand',0):.3f}
  {home} vs {away_hand}: {f.get('home_bat_ops_vs_sp_hand',0):.3f}

Park runs factor: {f.get('park_runs_factor',1.0):.3f}  Elevation: {f.get('elevation_ft',500):.0f} ft  Weather: {wx}  Humidity: {f.get('humidity_pct',50):.0f}%

Analysis:"""


def _stream_ollama(prompt: str, game_id: int = None, game_date: str = None):
    """Stream Ollama response as SSE chunks, caching the full text to memory and disk when done."""
    buf = []
    try:
        resp = _requests.post(
            _OLLAMA_URL,
            json={'model': _OLLAMA_MODEL, 'prompt': prompt, 'stream': True,
                  'options': {'num_predict': 350, 'temperature': 0.7}},
            stream=True,
            timeout=90,
        )
        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            text = chunk.get('response', '')
            if text:
                buf.append(text)
                yield f"data: {json.dumps({'text': text})}\n\n"
            if chunk.get('done'):
                if game_id is not None and buf:
                    full = ''.join(buf)
                    _explanation_cache[game_id] = full
                    if game_date:
                        _save_disk_explanation(game_date, game_id, full)
                yield "data: [DONE]\n\n"
                return
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


def _generate_explanation_sync(game: dict) -> str:
    """Call Ollama synchronously (no streaming). Used by the background pre-generator."""
    prompt = _build_explain_prompt(game)
    try:
        resp = _requests.post(
            _OLLAMA_URL,
            json={'model': _OLLAMA_MODEL, 'prompt': prompt, 'stream': False,
                  'options': {'num_predict': 350, 'temperature': 0.7}},
            timeout=120,
        )
        return resp.json().get('response', '').strip()
    except Exception:
        return ''


def _pregenerate_explanations(games: list, game_date: str) -> None:
    """Background: generate and persist explanations for all valid games sequentially."""
    for game in games:
        gid = game.get('game_id')
        if not gid or game.get('error') or gid in _explanation_cache:
            continue
        text = _generate_explanation_sync(game)
        if text:
            _explanation_cache[gid] = text
            _save_disk_explanation(game_date, gid, text)


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['TESTING'] = testing
    if not testing:
        bootstrap_data_cache()
        bootstrap_model_cache()
        threading.Thread(target=_backfill_results_cache, daemon=True).start()

    @app.route('/')
    def index():
        global _simulation_cache, _last_simulated_date, _results_cache, _last_results_date
        today = date.today().strftime('%Y-%m-%d')
        yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

        if not _simulation_cache or _last_simulated_date != today:
            _simulation_cache = run_daily_simulation(today)
            _last_simulated_date = today
            _load_disk_explanations(today)
            threading.Thread(target=_pregenerate_explanations, args=(_simulation_cache, today), daemon=True).start()

        if not _results_cache or _last_results_date != yesterday:
            try:
                _results_cache = _get_results_for_date(yesterday)
                _last_results_date = yesterday
            except Exception:
                _results_cache = {}

        last_7 = _aggregate_days(7)
        last_30 = _aggregate_days(90)
        meta = _read_model_meta()

        return render_template('index.html',
                               results=_simulation_cache, sim_date=today,
                               yesterday=_results_cache, yesterday_date=yesterday,
                               last_7=last_7, last_30=last_30,
                               model_version=meta.get('feature_version', '?'),
                               model_n_games=meta.get('n_games', '?'),
                               model_years=meta.get('training_years', []))

    @app.route('/refresh', methods=['POST'])
    def refresh():
        global _simulation_cache, _last_simulated_date
        today = date.today().strftime('%Y-%m-%d')
        _simulation_cache = run_daily_simulation(today)
        _last_simulated_date = today
        _load_disk_explanations(today)
        threading.Thread(target=_pregenerate_explanations, args=(_simulation_cache, today), daemon=True).start()
        return redirect(url_for('index'))

    @app.route('/retrain', methods=['POST'])
    def retrain():
        global _models_cache, _inning_model_cache
        _models_cache = {}
        _inning_model_cache = None
        _get_models(force_retrain=True)
        _get_inning_model(force_retrain=True)
        return redirect(url_for('index'))

    @app.route('/explain/<int:game_id>')
    def explain(game_id: int):
        game = next((g for g in _simulation_cache if g.get('game_id') == game_id), None)
        if not game:
            return Response(
                f"data: {json.dumps({'error': 'Game not found — try refreshing the page.'})}\n\n",
                mimetype='text/event-stream',
            )

        headers = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}

        if game_id in _explanation_cache:
            cached = _explanation_cache[game_id]
            def _from_cache():
                yield f"data: {json.dumps({'text': cached})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_from_cache()), mimetype='text/event-stream', headers=headers)

        prompt = _build_explain_prompt(game)
        game_date = game.get('game_date', date.today().strftime('%Y-%m-%d'))
        return Response(
            stream_with_context(_stream_ollama(prompt, game_id=game_id, game_date=game_date)),
            mimetype='text/event-stream',
            headers=headers,
        )

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
