import json
import threading
import requests as _requests
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, Response, stream_with_context, abort, jsonify
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.fetcher import (
    get_schedule, get_pitching_stats, get_team_batting_stats,
    get_bullpen_stats, get_season_schedule, get_weather_for_game, get_game_linescore,
    get_game_lineup, bootstrap_data_cache, bootstrap_model_cache,
    bootstrap_predictions_cache, refresh_schedule_date,
    get_market_odds, market_key, devig_home_prob,
)
from src.features import build_game_features, build_inning_feature_row, _NEUTRAL_WEATHER, FEATURE_VERSION, FEATURE_COLUMNS
from src.model import (
    train_models, load_models, models_exist, predict_game, build_training_data,
    train_inning_model, load_inning_model, inning_model_exists, predict_inning_probs,
    _combine_inning_dist as _model_combine_inning,
)
from src.simulator import simulate_game
from src.stadiums import get_stadium
from src.teams import get_team_meta
from src import predictions as _pred
from src import calibration as _cal
from src import blend as _blend
from src.colors import hex_to_rgb_str as _hex_to_rgb_str, bar_color as _bar_color
from src.explanations import (
    _explanation_cache, _load_disk_explanations, _build_explain_prompt,
    _stream_ollama, _pregenerate_explanations, _stream_edge_summary,
    _edge_summary_cache,
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
_tomorrow_simulation_cache: list[dict] = []
_last_tomorrow_date: str = ""
_results_cache: dict = {}
_last_results_date: str = ""
_actuals_cache: dict = {}  # settled date -> {game_id: final game}; avoids redundant live refreshes
_calibrator_cache: dict = {}  # data-dir -> PlattCalibrator|None; win% calibration loaded lazily
_blender_cache: dict = {}  # data-dir -> MarketBlender|None; market blend loaded lazily


def _get_calibrator(filename=_cal.CALIBRATOR_FILE):
    """Load (and cache) a calibrator for the active data dir.

    Keyed on (_DATA_DIR, filename) so tests that patch it to a tmp_path (with no
    calibrator -> None -> identity) stay isolated from the real one.
    """
    key = (str(_DATA_DIR), filename)
    if key not in _calibrator_cache:
        _calibrator_cache[key] = _cal.load(_DATA_DIR, filename)
    return _calibrator_cache[key]


def _get_inning_dist_calibrator():
    """Load (and cache) the 3-class inning calibrator for the active data dir."""
    key = (str(_DATA_DIR), _cal.INNING_DIST_CALIBRATOR_FILE)
    if key not in _calibrator_cache:
        _calibrator_cache[key] = _cal.load_multiclass(_DATA_DIR)
    return _calibrator_cache[key]


def _calibrate_core(core: dict) -> dict:
    """Return a copy of a prediction core with its win% and per-inning run
    distribution calibrated for display.

    The raw simulated win% — and the inning classifier's bucket probabilities —
    are overconfident out-of-sample; the calibrators pull them back toward the
    true rate (see src/calibration.py). The win% map is monotonic through 50%, so
    the favored side — and therefore the graded pick — never changes. The stored
    core on disk is untouched; this only affects what is shown and the edge math.
    No-op for whichever calibrators are absent (identity)."""
    out = dict(core)
    win_cal = _get_calibrator()
    hwp = core.get('home_win_pct')
    if win_cal is not None and hwp is not None:
        hp = _cal.calibrate_pct(win_cal, hwp)
        out.update(home_win_pct=hp, away_win_pct=round(100.0 - hp, 1), raw_home_win_pct=hwp)
    # 3-class inning distribution (new cores). Recompute the combined row from the
    # calibrated per-team dists so the three stay consistent.
    dist_cal = _get_inning_dist_calibrator()
    if dist_cal is not None and core.get('home_innings_dist') and core.get('away_innings_dist'):
        hd = [_cal.calibrate_dist(dist_cal, c) for c in core['home_innings_dist']]
        ad = [_cal.calibrate_dist(dist_cal, c) for c in core['away_innings_dist']]
        out['home_innings_dist'] = hd
        out['away_innings_dist'] = ad
        out['combined_innings_dist'] = [_model_combine_inning(h, a) for h, a in zip(hd, ad)]
    # Legacy cores (binary P(score>=1)).
    inn_cal = _get_calibrator(_cal.INNING_CALIBRATOR_FILE)
    if inn_cal is not None:
        for key in ('home_innings_scoring_pct', 'away_innings_scoring_pct'):
            vals = core.get(key)
            if vals:
                out[key] = [_cal.calibrate_pct(inn_cal, v) for v in vals]
    return out


def _get_blender():
    """Load (and cache) the market blender for the active data dir."""
    key = str(_DATA_DIR)
    if key not in _blender_cache:
        _blender_cache[key] = _blend.load(_DATA_DIR)
    return _blender_cache[key]


def _blend_core(core: dict, market_home_prob: "float | None") -> dict:
    """Return a copy whose home/away win% is the market-blended ('Consensus')
    line, with the calibrated model-only value preserved under model_*.

    The blender consumes the RAW model prob (it was fit on raw logits). Falls back
    to the (calibrated) model line when there is no blender or no market line."""
    out = dict(core)
    model_home = core.get('home_win_pct')        # already calibrated by _calibrate_core
    out['model_home_win_pct'] = model_home
    out['model_away_win_pct'] = core.get('away_win_pct')
    raw_home = core.get('raw_home_win_pct', model_home)
    # market_home_prob is a 0-1 fraction (devig_home_prob); blend_pct wants 0-100.
    market_home_pct = market_home_prob * 100.0 if market_home_prob is not None else None
    blended = _blend.blend_pct(_get_blender(), raw_home, market_home_pct)
    if blended is not None:
        out['home_win_pct'] = blended
        out['away_win_pct'] = round(100.0 - blended, 1)
    return out


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
                    sim['home_innings_dist'] = inning_probs['home']
                    sim['away_innings_dist'] = inning_probs['away']
                    sim['combined_innings_dist'] = inning_probs['combined']
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
    """Derive the model-implied moneyline from a prediction core.

    The app supports the moneyline market only: fair (no-vig) American odds from
    the win probability. Run-line and total markets were removed — their edges
    were computed from the raw, uncalibrated score distribution (and the run line
    against a fabricated 50% baseline), which manufactured implausible edges.
    """
    return {
        'ml_home': _american(game.get('home_win_pct', 50.0) / 100.0),
        'ml_away': _american(game.get('away_win_pct', 50.0) / 100.0),
    }


def _enrich_game(game: dict, core: dict) -> dict:
    """Combine a schedule game shell with its frozen prediction core and the
    presentation/context fields (weather, lineup, features, logos, colors) that
    are re-derived at render time rather than persisted in the artifact."""
    core = _calibrate_core(core)
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


def _implied_prob(odds, default=-110):
    """De-vig-free implied probability of a single American price (default -110)."""
    o = odds if odds is not None else default
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def _fmt_american(v, default=None):
    v = v if v is not None else default
    return f"{int(round(v)):+d}" if v is not None else '—'


def _market_block(game: dict, mk: dict) -> dict:
    """Format the averaged ESPN moneyline and compute the model's moneyline edge %
    — the model's (calibrated) win probability minus the market's de-vigged implied
    probability, shown on the side the model favors.

    Moneyline only: the run-line and total markets were removed."""
    model_home_win = game.get('home_win_pct', 50.0) / 100.0
    ha, aa = game.get('home_abbr'), game.get('away_abbr')

    block = {
        'ml_home': _fmt_american(mk.get('ml_home')),
        'ml_away': _fmt_american(mk.get('ml_away')),
        'n_books': mk.get('n_books', 0),
    }

    # Moneyline edge — model win% vs the de-vigged market.
    ih, ia = _implied_prob(mk.get('ml_home')), _implied_prob(mk.get('ml_away'))
    mkt_home = ih / (ih + ia) if (ih + ia) else 0.5
    ml_edge = model_home_win - mkt_home
    pct = round(abs(ml_edge) * 100)
    block['edge_ml_side'] = ha if ml_edge >= 0 else aa
    # Only surface an edge once it clears the noise floor. Below it, the model-vs-
    # market gap doesn't reliably beat a sharp close (walk-forward CLV: bets <5%
    # edge ~breakeven at +1.1%, ≥5% edge +2.2% vs real 2019/21 closes), so showing
    # those as actionable "edges" over-promises. Sub-threshold -> no edge.
    block['edge_ml_pct'] = pct if pct >= _EDGE_MIN_PCT else 0
    block['edge_ml_raw_pct'] = pct          # pre-threshold gap, kept for detail/debug
    return block


# Minimum model-vs-market gap (percentage points) to count as an actionable edge.
# Below this, the gap is within model noise and doesn't beat a sharp close. Tunable.
_EDGE_MIN_PCT = 5

_EDGE_MARKETS = (('ML', 'edge_ml_side', 'edge_ml_pct'),)


def _best_edge(market: dict) -> dict | None:
    """The single biggest model-vs-market edge for a game: {market, side, pct}.

    None when there is no market block or no priced edge."""
    if not market:
        return None
    cands = [(label, market.get(sk), market.get(pk)) for label, sk, pk in _EDGE_MARKETS]
    cands = [(label, s, p) for label, s, p in cands if s and p]
    if not cands:
        return None
    label, side, pct = max(cands, key=lambda c: c[2])
    return {'market': label, 'side': side, 'pct': pct}


def _top_edges(games: list, n: int = 6) -> list[dict]:
    """Flatten every game's moneyline edge, rank by %, return the top n.

    Each entry: {game_id, matchup, market, side, pct}. Games without a market
    block (no ESPN odds) are skipped."""
    out = []
    for g in games:
        mk = g.get('market')
        if not mk:
            continue
        matchup = f"{g.get('away_abbr', '?')} @ {g.get('home_abbr', '?')}"
        for label, sk, pk in _EDGE_MARKETS:
            side, pct = mk.get(sk), mk.get(pk)
            if side and pct:
                out.append({'game_id': g.get('game_id'), 'matchup': matchup,
                            'market': label, 'side': side, 'pct': pct})
    out.sort(key=lambda e: e['pct'], reverse=True)
    return out[:n]


def run_daily_simulation(sim_date: str = None) -> list[dict]:
    if sim_date is None:
        sim_date = date.today().strftime('%Y-%m-%d')

    cores = {c['game_id']: c for c in get_prediction(sim_date)}
    market = get_market_odds(sim_date)
    schedule = get_schedule(sim_date)

    # Snapshot today's moneylines write-once — a load-bearing input to the blended
    # ("Consensus") line, so the headline is reproducible when this date is graded.
    snap = {}
    for game in schedule:
        mk = market.get(market_key(game.get('away_name', ''), game.get('home_name', '')))
        if mk and mk.get('ml_home') is not None and mk.get('ml_away') is not None:
            snap[game['game_id']] = {'ml_home': mk['ml_home'], 'ml_away': mk['ml_away']}
    try:
        _pred.save_market_odds(_DATA_DIR, sim_date, snap)
    except Exception:
        pass

    results = []
    for game in schedule:
        try:
            g = _enrich_game(game, cores.get(game['game_id'], {}))
            mk = market.get(market_key(g.get('away_name', ''), g.get('home_name', '')))
            market_home_prob = devig_home_prob(mk.get('ml_home'), mk.get('ml_away')) if mk else None
            g.update(_blend_core(g, market_home_prob))   # Consensus headline; model_* preserved
            g['market'] = _market_block(g, mk) if mk else None   # edge = Consensus vs market
            g['best_edge'] = _best_edge(g['market'])
            results.append(g)
        except Exception as e:
            results.append({**game, 'error': str(e)})

    # Chronological by first pitch (the default the UI shows); missing times last.
    results.sort(key=lambda g: g.get('game_datetime') or '~')

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
    market_odds = _pred.load_market_odds(_DATA_DIR, result_date)
    results = []
    correct = 0          # blended / Consensus (the headline pick of record)
    model_correct = 0    # model-only (kept for comparison)
    for core in cores:
        game = actuals.get(core.get('game_id'))
        if not game:
            continue  # not final yet — no comparison row
        core = _calibrate_core(core)  # model-only calibrated line (+ raw_home_win_pct)
        o = market_odds.get(str(core.get('game_id')))
        market_home_prob = devig_home_prob(o['ml_home'], o['ml_away']) if o else None
        core = _blend_core(core, market_home_prob)   # blended home_win_pct + model_* preserved
        actual_home = int(game['home_score'])
        actual_away = int(game['away_score'])
        actual_home_won = actual_home > actual_away
        predicted_home_won = core.get('home_win_pct', 50.0) > 50.0          # blended (Consensus) pick
        model_home_won = core.get('model_home_win_pct', 50.0) > 50.0        # model-only pick
        winner_correct = actual_home_won == predicted_home_won
        model_winner_correct = actual_home_won == model_home_won
        if winner_correct:
            correct += 1
        if model_winner_correct:
            model_correct += 1
        home_err = abs(core.get('median_home_score', 0) - actual_home)
        away_err = abs(core.get('median_away_score', 0) - actual_away)
        home_meta = get_team_meta(game['home_id'])
        away_meta = get_team_meta(game['away_id'])
        ml_pick = home_meta.get('abbr', 'HOME') if predicted_home_won else away_meta.get('abbr', 'AWAY')
        results.append({
            **game,
            **core,
            'actual_home_score': actual_home,
            'actual_away_score': actual_away,
            'actual_home_won': actual_home_won,
            'predicted_home_won': predicted_home_won,
            'home_score_err': round(home_err, 1),
            'away_score_err': round(away_err, 1),
            'winner_correct': winner_correct,
            'ml_correct': winner_correct,                 # headline = Consensus
            'model_winner_correct': model_winner_correct,  # model-only, for comparison
            'ml_pick': ml_pick,
            'home_logo': home_meta['logo_url'],
            'away_logo': away_meta['logo_url'],
            'home_color': home_meta['primary'],
            'away_color': away_meta['primary'],
            'home_color2': home_meta['secondary'],
            'away_color2': away_meta['secondary'],
        })

    n = len(results)
    accuracy = round(correct / n * 100, 1) if n else 0
    model_acc = round(model_correct / n * 100, 1) if n else 0
    scored = [r for r in results if 'home_score_err' in r]
    avg_err = round(
        sum(r['home_score_err'] + r['away_score_err'] for r in scored) / (2 * len(scored)), 2
    ) if scored else None

    return {
        'result_date': result_date,
        'games': results,
        'n_completed': n,
        'winner_accuracy': accuracy,
        'ml_accuracy': accuracy,
        'consensus_accuracy': accuracy,
        'model_accuracy': model_acc,
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
    model_total = sum(
        round((r.get('model_accuracy') or 0) / 100 * r['n_completed']) for r in daily
    )
    model_accuracy = round(model_total / total_games * 100, 1) if total_games else 0
    scored = [r for r in daily if r.get('avg_score_err') is not None]
    avg_err = round(
        sum(r['avg_score_err'] for r in scored) / len(scored), 2
    ) if scored else None

    result = {
        'total_games': total_games,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'ml_accuracy': accuracy,
        'consensus_accuracy': accuracy,
        'model_accuracy': model_accuracy,
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
    model_total = sum(round((r.get('model_accuracy') or 0) / 100 * r['n_completed']) for r in daily)
    model_accuracy = round(model_total / total * 100, 1) if total else 0
    scored = [r for r in daily if r.get('avg_score_err') is not None]
    avg_err = round(sum(r['avg_score_err'] for r in scored) / len(scored), 2) if scored else None
    result = {
        'total_games': total,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'ml_accuracy': accuracy,
        'consensus_accuracy': accuracy,
        'model_accuracy': model_accuracy,
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


def _fmt_short_date(d: str) -> str:
    """'2026-06-27' -> 'Jun 27' (platform-independent, no zero-pad)."""
    try:
        dt = datetime.strptime(d, '%Y-%m-%d')
        return f"{dt.strftime('%b')} {dt.day}"
    except Exception:
        return d


def _weather_str(g: dict) -> str:
    """One-line weather summary for the game-detail card."""
    w = g.get('weather') or {}
    if w.get('is_dome'):
        t = w.get('temperature_f')
        return f"Dome · {round(t)}°F" if t is not None else "Dome"
    t = w.get('temperature_f')
    if t is None:
        return "—"
    parts = [f"{round(t)}°F"]
    ws = w.get('wind_speed_mph')
    if ws is not None:
        parts.append(f"{round(ws)} mph")
    h = w.get('humidity_pct')
    if h is not None:
        parts.append(f"{round(h)}% RH")
    return " · ".join(parts)


def _to_int_score(v) -> "int | None":
    """Return v as int or None if absent/NaN."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _game_ui(g: dict) -> dict | None:
    """Project an enriched sim game into the JSON the Edge UI consumes.

    Moneyline only. All numbers are real model output; nothing is mocked."""
    if g.get('error') or 'home_win_pct' not in g:
        return None
    away_pct = round(g.get('away_win_pct', 50.0))
    home_pct = round(g.get('home_win_pct', 50.0))
    fav_home = home_pct >= away_pct
    pick = g.get('home_abbr') if fav_home else g.get('away_abbr')
    pick_pct = home_pct if fav_home else away_pct
    mk = g.get('market') or {}
    lines = g.get('lines') or {}
    has_edge = bool(mk) and (mk.get('edge_ml_pct') or 0) > 0
    model_ml = lines.get('ml_home') if fav_home else lines.get('ml_away')
    market_ml = (mk.get('ml_home') if fav_home else mk.get('ml_away')) if mk else None
    sd = g.get('score_distribution') or {}
    lineup = g.get('lineup') or {}
    home_color = g.get('home_bar_color') or g.get('home_color') or '#0064c8'
    away_color = g.get('away_bar_color') or g.get('away_color') or '#c83232'
    ma, mh = g.get('modal_away_score'), g.get('modal_home_score')
    if ma is None or mh is None:  # fall back to medians (real cores always have modal)
        ma, mh = round(g.get('median_away_score', 0)), round(g.get('median_home_score', 0))
    status = g.get('status', '')
    is_active = status in ('Final', 'In Progress', 'Game Over')
    home_actual = _to_int_score(g.get('home_score')) if is_active else None
    away_actual = _to_int_score(g.get('away_score')) if is_active else None
    return {
        'id': g.get('game_id'),
        'away': g.get('away_abbr'), 'home': g.get('home_abbr'),
        'awayName': g.get('away_name'), 'homeName': g.get('home_name'),
        'awayLogo': g.get('away_logo'), 'homeLogo': g.get('home_logo'),
        'awayColor': away_color, 'homeColor': home_color,
        'awayTint': away_color + '24', 'homeTint': home_color + '24',
        'awayPct': away_pct, 'homePct': home_pct,
        'pick': pick, 'pickPct': pick_pct, 'hasEdge': has_edge,
        'edgePct': mk.get('edge_ml_pct', 0) if mk else 0,
        'edgeSide': mk.get('edge_ml_side') if mk else None,
        'score': f"{ma}–{mh}",
        'utc': g.get('game_datetime') or '',
        'modelML': model_ml, 'marketML': market_ml,
        'scoreDist': {'labels': sd.get('labels', []),
                      'away': sd.get('away', []), 'home': sd.get('home', [])},
        'nSims': g.get('n_simulations', N_SIMULATIONS),
        'innings': {'away': g.get('away_innings_dist') or [],
                    'home': g.get('home_innings_dist') or [],
                    'both': g.get('combined_innings_dist') or []},
        'awayP': g.get('away_pitcher', 'TBD'), 'homeP': g.get('home_pitcher', 'TBD'),
        'venue': g.get('venue_name', ''), 'weather': _weather_str(g),
        'awayLineup': lineup.get('away', []), 'homeLineup': lineup.get('home', []),
        'status': status,
        'homeActual': home_actual,
        'awayActual': away_actual,
        'projHome': round(g.get('median_home_score') or 0),
        'projAway': round(g.get('median_away_score') or 0),
    }


def _track_ui(last_7: dict, last_30: dict, last_90: dict,
              yesterday: dict, yesterday_date: str) -> dict:
    """Track Record tab payload: 7/30/90 summary, yesterday's graded games,
    and the last-7-days daily accuracy bars."""
    def _summary(label, agg):
        # Headline = Consensus (market-blended) accuracy — the graded pick of
        # record; model-only accuracy is shown alongside for comparison (PR #15).
        acc = agg.get('consensus_accuracy', agg.get('winner_accuracy')) or 0
        return {'label': label, 'acc': acc,
                'model': agg.get('model_accuracy'),
                'games': agg.get('total_games', 0),
                'err': agg.get('avg_score_err'),
                'good': acc >= 53}

    yrows = []
    for r in (yesterday.get('games') or []):
        aw = r.get('away_abbr') or get_team_meta(r['away_id'])['abbr']
        hw = r.get('home_abbr') or get_team_meta(r['home_id'])['abbr']
        # HIT/MISS grades the Consensus moneyline pick (the pick of record),
        # which can differ from the model's projected score — so show the graded
        # pick, not the predicted score, or the row looks contradictory.
        pick = hw if r.get('predicted_home_won') else aw
        yrows.append({
            'away': aw, 'home': hw,
            'awayLogo': r.get('away_logo'), 'homeLogo': r.get('home_logo'),
            'actual': f"{r.get('actual_away_score')}–{r.get('actual_home_score')}",
            'pick': pick,
            'win': bool(r.get('winner_correct')),
        })

    daily = [{'date': _fmt_short_date(d['result_date']),
              'n': d.get('n_completed', 0),
              'acc': round(d.get('consensus_accuracy', d.get('winner_accuracy')) or 0)}
             for d in (last_7.get('daily') or [])]

    return {
        'summary': [_summary('7 days', last_7), _summary('30 days', last_30),
                    _summary('90 days', last_90)],
        'yesterday': yrows, 'yDate': _fmt_short_date(yesterday_date),
        'daily': daily,
    }


def _self_heal_closing_odds(days: int = 7) -> None:
    """Backfill the previous days' real closing lines not yet captured (launch-time).

    'Previous days only' — a true close exists only after a game ends, and the free
    /odds feed drops finished games, so a past close needs the paid historical
    endpoint (gated on ODDS_API_KEY; no-op without it). The resumable manifest makes
    steady state ~yesterday (one cheap day, ~40 credits); a per-launch credit cap
    bounds the cost if there's a gap.
    """
    from datetime import timedelta
    from src import closing_backfill as _cb
    end = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (date.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        _cb.backfill_range(_DATA_DIR, start, end, max_credits=2000)
    except Exception:
        pass


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['TESTING'] = testing
    if not testing:
        bootstrap_data_cache()
        bootstrap_model_cache()
        bootstrap_predictions_cache()
        threading.Thread(target=_backfill_results_cache, daemon=True).start()
        threading.Thread(target=_self_heal_closing_odds, daemon=True).start()

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
        last_30 = _aggregate_days(30)
        last_90 = _aggregate_days(90)
        meta = _read_model_meta()
        cur = _current_version()
        model_name = next((e.get('name') for e in _pred.read_versions(_DATA_DIR)
                           if e.get('version') == cur), None) or f"v{meta.get('feature_version', '?')}"

        games_ui = [u for g in _simulation_cache if (u := _game_ui(g)) is not None]
        payload = {
            'dateLabel': date.today().strftime('%a, %b ') + str(date.today().day),
            'accAccuracy': round(last_7.get('consensus_accuracy', last_7.get('winner_accuracy')) or 0),
            'games': games_ui,
            'track': _track_ui(last_7, last_30, last_90, _results_cache, yesterday),
            'model': {
                'name': model_name,
                'nGames': meta.get('n_games', '?'),
                'years': meta.get('training_years', []),
            },
        }

        return render_template('index.html', payload=payload)

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
                'consensus_accuracy': s.get('consensus_accuracy', s['winner_accuracy']),
                'model_accuracy': s.get('model_accuracy'),
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

    @app.route('/history')
    def history():
        """Every past game with a stored prediction: final score, the model's
        (Consensus) pick graded against the result, and the real closing line."""
        version = _current_version()
        days, totals = [], {'n': 0, 'correct': 0, 'with_close': 0}
        for d in sorted(_pred.list_dates(_DATA_DIR, version), reverse=True):
            day = compare_date(d, version)
            games = day.get('games') or []
            if not games:
                continue
            closing = _pred.load_closing_odds(_DATA_DIR, d)
            for g in games:
                c = closing.get(str(g.get('game_id')))
                g['close_home'] = _fmt_american(c.get('ml_home')) if c else None
                g['close_away'] = _fmt_american(c.get('ml_away')) if c else None
                totals['n'] += 1
                totals['correct'] += 1 if g.get('winner_correct') else 0
                totals['with_close'] += 1 if c else 0
            days.append(day)
        totals['accuracy'] = round(100 * totals['correct'] / totals['n'], 1) if totals['n'] else 0
        return render_template('history.html', days=days, totals=totals, version=version)

    @app.route('/api/games/tomorrow')
    def api_tomorrow():
        global _tomorrow_simulation_cache, _last_tomorrow_date
        tomorrow = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        if not _tomorrow_simulation_cache or _last_tomorrow_date != tomorrow:
            _tomorrow_simulation_cache = run_daily_simulation(tomorrow)
            _last_tomorrow_date = tomorrow
        games_ui = [u for g in _tomorrow_simulation_cache if (u := _game_ui(g)) is not None]
        dt = date.today() + timedelta(days=1)
        return jsonify({
            'games': games_ui,
            'dateLabel': dt.strftime('%a, %b ') + str(dt.day),
        })

    @app.route('/edge/<int:game_id>')
    def edge(game_id: int):
        """SSE: a one-sentence (<=10 word) AI edge angle for 'The bet', streamed
        as it generates. On an unknown game or Ollama failure the stream yields
        no text, so the client keeps its deterministic fallback line."""
        headers = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        game = next((g for g in _simulation_cache if g.get('game_id') == game_id), None)
        if not game:
            game = next((g for g in _tomorrow_simulation_cache if g.get('game_id') == game_id), None)
        if not game:
            return Response("data: [DONE]\n\n", mimetype='text/event-stream', headers=headers)
        if game_id in _edge_summary_cache:
            cached = _edge_summary_cache[game_id]
            def _from_cache():
                yield f"data: {json.dumps({'text': cached})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_from_cache()),
                            mimetype='text/event-stream', headers=headers)
        return Response(stream_with_context(_stream_edge_summary(game, game_id=game_id)),
                        mimetype='text/event-stream', headers=headers)

    @app.route('/explain/<int:game_id>')
    def explain(game_id: int):
        game = next((g for g in _simulation_cache if g.get('game_id') == game_id), None)
        if not game:
            game = next((g for g in _tomorrow_simulation_cache if g.get('game_id') == game_id), None)
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
