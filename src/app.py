import json
import threading
import requests as _requests
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, Response, stream_with_context, abort
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.fetcher import (
    get_schedule, get_pitching_stats, get_team_batting_stats,
    get_bullpen_stats, get_season_schedule, get_weather_for_game, get_game_linescore,
    get_game_lineup, bootstrap_data_cache, bootstrap_model_cache,
    bootstrap_predictions_cache, refresh_schedule_date,
    get_market_odds, market_key,
)
from src.features import build_game_features, build_inning_feature_row, _NEUTRAL_WEATHER, FEATURE_VERSION, FEATURE_COLUMNS
from src.model import (
    train_models, load_models, models_exist, predict_game, build_training_data,
    train_inning_model, load_inning_model, inning_model_exists, predict_inning_probs,
)
from src.simulator import simulate_game
from src.stadiums import get_stadium
from src.teams import get_team_meta
from src import predictions as _pred
from src.colors import hex_to_rgb_str as _hex_to_rgb_str, bar_color as _bar_color
from src.explanations import (
    _explanation_cache, _load_disk_explanations, _build_explain_prompt,
    _stream_ollama, _pregenerate_explanations,
    _stream_picks, _pregenerate_picks, _load_disk_picks,
)
from src.training import (
    get_models as _get_models, get_inning_model as _get_inning_model,
    read_model_meta as _read_model_meta, reset_model_caches as _reset_model_caches,
)
from datetime import datetime, timezone

_DATA_DIR = Path(__file__).parent.parent / "data"

N_SIMULATIONS = 1000  # fixed sim count for the prediction of record (live + backfill)
_REFRESH_WINDOW_DAYS = 4  # only re-fetch finals from the live API for dates this recent

_simulation_cache: list[dict] = []
_last_simulated_date: str = ""
_results_cache: dict = {}
_last_results_date: str = ""
_actuals_cache: dict = {}  # settled date -> {game_id: final game}; avoids redundant live refreshes


def _current_version() -> str | None:
    """Version id of the live model pkls. Registers it on first sight.

    Naming rule: a model's name is v<major>.<build>, where major is the feature
    generation it trains on (FEATURE_VERSION) and build increments from 0 for
    each new model at that generation. So a retrain bumps the build (v4.0 ->
    v4.1) and a new feature set bumps the major and resets the build (-> v5.0).
    """
    v = _pred.model_version(_DATA_DIR)
    if v and not any(e.get('version') == v for e in _pred.read_versions(_DATA_DIR)):
        major = FEATURE_VERSION
        build = _pred.next_build(_DATA_DIR, major)
        _pred.append_version(_DATA_DIR, {
            'version': v,
            'name': f'v{major}.{build}',
            'major': major,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'features': len(FEATURE_COLUMNS),
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


def _american(p: float) -> str:
    """Fair (no-vig) American odds for a probability p."""
    p = min(max(p, 0.01), 0.99)
    odds = round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
    return f"{odds:+d}"


def _market_lines(game: dict) -> dict:
    """Derive model-implied, sportsbook-style betting lines from a prediction core.

    Real lines are fixed numbers with odds attached, not continuous expectations:
    - moneyline: fair American odds from the win probability
    - run line:  fixed ±1.5 (MLB standard), odds = P(favorite wins by ≥2)
    - total:     a .5/.0 line near the expected total, with fair over/under odds

    The run-margin and total distributions are reconstructed by convolving the
    stored per-team score histograms (the simulator draws the teams' runs
    independently, so the convolution matches its joint distribution). All values
    are display-ready strings.
    """
    from collections import defaultdict

    ml_home = _american(game.get('home_win_pct', 50.0) / 100.0)
    ml_away = _american(game.get('away_win_pct', 50.0) / 100.0)
    blank = {'ml_home': ml_home, 'ml_away': ml_away, 'spread_home': '—',
             'spread_away': '—', 'total_line': '—', 'total_over': '', 'total_under': ''}

    dist = game.get('score_distribution') or {}
    home_counts, away_counts = dist.get('home') or [], dist.get('away') or []
    hsum, asum = sum(home_counts), sum(away_counts)
    if not hsum or not asum:
        return blank

    hp = [c / hsum for c in home_counts]   # P(home runs == i)
    ap = [c / asum for c in away_counts]    # P(away runs == j)
    margin, total = defaultdict(float), defaultdict(float)
    for h, ph in enumerate(hp):
        if not ph:
            continue
        for a, pa in enumerate(ap):
            if pa:
                margin[h - a] += ph * pa
                total[h + a] += ph * pa

    # ---- Total: nearest half-run line, fair over/under odds ----
    exp_total = sum(t * p for t, p in total.items())
    line = round(exp_total * 2) / 2                      # ends in .0 or .5
    p_over = sum(p for t, p in total.items() if t > line)
    p_under = sum(p for t, p in total.items() if t < line)
    denom = (p_over + p_under) or 1.0                    # drop pushes on a .0 line
    total_line = f"{line:.1f}"
    total_over, total_under = _american(p_over / denom), _american(p_under / denom)

    # ---- Run line: fixed 1.5, favorite by win probability ----
    fav_home = game.get('home_win_pct', 50.0) >= game.get('away_win_pct', 50.0)
    if fav_home:
        p_cover = sum(p for d, p in margin.items() if d >= 2)   # home wins by ≥2
        spread_home = f"-1.5 {_american(p_cover)}"
        spread_away = f"+1.5 {_american(1 - p_cover)}"
    else:
        p_cover = sum(p for d, p in margin.items() if d <= -2)  # away wins by ≥2
        spread_away = f"-1.5 {_american(p_cover)}"
        spread_home = f"+1.5 {_american(1 - p_cover)}"

    return {'ml_home': ml_home, 'ml_away': ml_away,
            'spread_home': spread_home, 'spread_away': spread_away,
            'total_line': total_line, 'total_over': total_over, 'total_under': total_under}


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
        'lines': _market_lines({**game, **core}),
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


def _market_compare(game: dict, mk: dict) -> dict:
    """Format averaged market odds for display and flag where the model disagrees."""
    def odds(v):
        return f"{int(round(v)):+d}" if v is not None else '—'

    lines = game.get('lines') or {}
    out = {
        'total': f"{mk['total']:.1f}" if mk.get('total') is not None else '—',
        'over_odds': odds(mk.get('over_odds')), 'under_odds': odds(mk.get('under_odds')),
        'ml_home': odds(mk.get('ml_home')), 'ml_away': odds(mk.get('ml_away')),
        'n_books': mk.get('n_books', 0), 'total_edge': None, 'ml_edge': None,
    }
    # Total edge: model's total line vs the market number.
    try:
        diff = float(lines.get('total_line')) - float(mk['total'])
        if abs(diff) >= 0.5:
            out['total_edge'] = f"model {'OVER' if diff > 0 else 'UNDER'} {abs(diff):.1f}"
    except (TypeError, ValueError):
        pass
    # Moneyline edge: model and market favor different sides.
    mh, ma = mk.get('ml_home'), mk.get('ml_away')
    if mh is not None and ma is not None:
        model_fav_home = game.get('home_win_pct', 50) >= game.get('away_win_pct', 50)
        if (mh < ma) != model_fav_home:
            out['ml_edge'] = f"model likes {game.get('home_abbr') if model_fav_home else game.get('away_abbr')}"
    return out


def _edge_metrics(game: dict, mk: dict) -> dict | None:
    """Numeric model-vs-market comparison + an edge score, for ranking 'Picks'."""
    lines = game.get('lines') or {}
    try:
        model_total = float(lines.get('total_line'))
    except (TypeError, ValueError):
        model_total = None
    mkt_total = mk.get('total')
    if model_total is None or mkt_total is None:
        return None

    def implied(ml):
        if ml is None:
            return None
        return (-ml) / ((-ml) + 100) if ml < 0 else 100 / (ml + 100)

    ih, ia = implied(mk.get('ml_home')), implied(mk.get('ml_away'))
    mkt_home_win = ih / (ih + ia) if (ih and ia) else None        # de-vigged
    model_home_win = game.get('home_win_pct', 50) / 100.0
    model_fav_home = model_home_win >= 0.5
    mkt_fav_home = mkt_home_win >= 0.5 if mkt_home_win is not None else model_fav_home
    ml_prob_edge = abs(model_home_win - mkt_home_win) if mkt_home_win is not None else 0.0
    total_diff = model_total - mkt_total
    return {
        'away_abbr': game.get('away_abbr'), 'home_abbr': game.get('home_abbr'),
        'model_total': model_total, 'mkt_total': mkt_total, 'total_diff': total_diff,
        'model_home_win': model_home_win * 100,
        'model_fav': game.get('home_abbr') if model_fav_home else game.get('away_abbr'),
        'mkt_fav': game.get('home_abbr') if mkt_fav_home else game.get('away_abbr'),
        'fav_disagree': model_fav_home != mkt_fav_home,
        'score': abs(total_diff) + 3.0 * ml_prob_edge + (1.0 if model_fav_home != mkt_fav_home else 0.0),
    }


def _picks_payload(games: list, top: int = 4) -> list:
    """The day's strongest model-vs-market edges, ranked, for the Picks summary."""
    edges = [g['edge'] for g in games if g.get('edge')]
    edges.sort(key=lambda e: e['score'], reverse=True)
    return edges[:top]


def run_daily_simulation(sim_date: str = None) -> list[dict]:
    if sim_date is None:
        sim_date = date.today().strftime('%Y-%m-%d')

    cores = {c['game_id']: c for c in get_prediction(sim_date)}
    market = get_market_odds(sim_date)
    results = []
    for game in get_schedule(sim_date):
        try:
            g = _enrich_game(game, cores.get(game['game_id'], {}))
            mk = market.get(market_key(g.get('away_name', ''), g.get('home_name', '')))
            g['market'] = _market_compare(g, mk) if mk else None
            g['edge'] = _edge_metrics(g, mk) if mk else None
            results.append(g)
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
    cached = _actuals_cache.get(result_date)
    if cached is not None:
        return cached
    year = int(result_date[:4])
    all_games = get_season_schedule(year)
    games = [g for g in all_games if g.get('game_date') == result_date]
    if not games or not all(_is_final_game(g) for g in games):
        if refresh_schedule_date(year, result_date) > 0:
            all_games = get_season_schedule(year)
            games = [g for g in all_games if g.get('game_date') == result_date]
        if not games:
            games = get_schedule(result_date)  # fallback for today/future
    actuals = {g['game_id']: g for g in games if _is_final_game(g)}
    # Memoize settled (older than the refresh window) dates: their finals never
    # change, so the live refresh + join runs once per process instead of once
    # per archived date per model version (e.g. /archive grades every version).
    if result_date < (date.today() - timedelta(days=_REFRESH_WINDOW_DAYS)).strftime('%Y-%m-%d'):
        _actuals_cache[result_date] = actuals
    return actuals


def compare_date(result_date: str, version: str = None) -> dict:
    """Join the frozen prediction for result_date with actual finals — no sim.

    For the current version, generates-on-miss; for an archived version, reads
    only what is stored (frozen). Accuracy is computed from the stored core's
    win% and median scores, so it never changes unless the model version does.
    """
    if version is None:
        version = _current_version()
    # Read-only: grading never simulates. Today's live prediction is generated by
    # run_daily_simulation/get_prediction; past dates are filled by the backfill.
    # (Avoids a full simulation running inside a request on a cold start.)
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

    # Memoize the rollup per (version, today, window, ready-set). Invalidates
    # when a new date finalizes (ready grows) or the day rolls over.
    cache_key = (str(_DATA_DIR), version, dates[0] if dates else '', n_days, len(ready))
    if cache_key in _aggregate_cache:
        return _aggregate_cache[cache_key]

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

    result = {
        'total_games': total_games,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'avg_score_err': avg_err,
        'daily': daily,
    }
    _aggregate_cache[cache_key] = result
    return result


_summary_cache: dict = {}  # immutable archived-version rollups (current version excluded)
_aggregate_cache: dict = {}  # _aggregate_days rollups keyed by (version, today, window, ready-count)


def _version_summary(version: str) -> dict:
    """Aggregate accuracy across every date stored for a model version (all of
    its history, not just the last 90 days). Used by the archive pages.

    Archived (non-current) versions are immutable, so their rollup is cached
    keyed by the stored-date set; the current version always recomputes.
    """
    dates = _pred.list_dates(_DATA_DIR, version)
    is_current = (version == _current_version())
    key = (str(_DATA_DIR), version, dates[-1] if dates else '', len(dates))
    if not is_current and key in _summary_cache:
        return _summary_cache[key]
    daily = []
    for d in dates:
        try:
            r = compare_date(d, version)
            if r.get('n_completed', 0) > 0:
                daily.append(r)
        except Exception:
            pass
    daily.sort(key=lambda x: x['result_date'], reverse=True)
    total = sum(r['n_completed'] for r in daily)
    correct = sum(round(r['winner_accuracy'] / 100 * r['n_completed']) for r in daily)
    accuracy = round(correct / total * 100, 1) if total else 0
    scored = [r for r in daily if r.get('avg_score_err') is not None]
    avg_err = round(sum(r['avg_score_err'] for r in scored) / len(scored), 2) if scored else None
    result = {
        'total_games': total,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'avg_score_err': avg_err,
        'daily': daily,
    }
    if not is_current:
        _summary_cache[key] = result
    return result


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


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['TESTING'] = testing
    if not testing:
        bootstrap_data_cache()
        bootstrap_model_cache()
        bootstrap_predictions_cache()
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
            edges = _picks_payload(_simulation_cache)
            if edges:
                threading.Thread(target=_pregenerate_picks, args=(edges, today), daemon=True).start()

        if not _results_cache or _last_results_date != yesterday:
            try:
                _results_cache = _get_results_for_date(yesterday)
                _last_results_date = yesterday
            except Exception:
                _results_cache = {}

        last_7 = _aggregate_days(7)
        last_90 = _aggregate_days(90)
        meta = _read_model_meta()
        cur = _current_version()
        model_name = next((e.get('name') for e in _pred.read_versions(_DATA_DIR)
                           if e.get('version') == cur), None) or f"v{meta.get('feature_version', '?')}"

        return render_template('index.html',
                               results=_simulation_cache, sim_date=today,
                               yesterday=_results_cache, yesterday_date=yesterday,
                               last_7=last_7, last_90=last_90,
                               model_name=model_name,
                               has_picks=bool(_picks_payload(_simulation_cache)),
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
        global _simulation_cache, _last_simulated_date, _results_cache
        _reset_model_caches()
        _get_models(force_retrain=True)
        _get_inning_model(force_retrain=True)
        # New pkls -> new version. Register it (archives the prior version's
        # prediction folder by leaving it intact) and re-sim today + 90 days
        # under the new version in the background.
        _current_version()
        _simulation_cache = []
        _last_simulated_date = ""
        _results_cache = {}
        threading.Thread(target=_backfill_results_cache, args=(90,), daemon=True).start()
        return redirect(url_for('index'))

    @app.route('/archive')
    def archive():
        current = _current_version()
        rows = []
        for v in _pred.read_versions(_DATA_DIR):
            s = _version_summary(v['version'])
            rows.append({
                **v,
                'total_games': s['total_games'],
                'days_with_games': s['days_with_games'],
                'winner_accuracy': s['winner_accuracy'],
                'avg_score_err': s['avg_score_err'],
                'is_current': v['version'] == current,
            })
        # Newest first by training/registration time (registry order is not relied on).
        rows.sort(key=lambda r: r.get('created_at', ''), reverse=True)
        return render_template('archive.html', versions=rows)

    @app.route('/archive/<version>')
    def archive_version(version: str):
        # Only serve known versions — `version` is used to build a filesystem
        # path (predictions/<version>/...), so never trust it from the URL.
        meta = next((v for v in _pred.read_versions(_DATA_DIR)
                     if v['version'] == version), None)
        if meta is None:
            abort(404)
        summary = _version_summary(version)
        return render_template('archive_version.html', version=version, meta=meta,
                               summary=summary, is_current=(version == _current_version()))

    @app.route('/picks')
    def picks():
        today = date.today().strftime('%Y-%m-%d')
        edges = _picks_payload(_simulation_cache)
        headers = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        if not edges:
            def _none():
                yield f"data: {json.dumps({'text': 'No standout edges vs the market today.'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_none()), mimetype='text/event-stream', headers=headers)
        return Response(stream_with_context(_stream_picks(edges, today)),
                        mimetype='text/event-stream', headers=headers)

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
