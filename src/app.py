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
    _combine_inning_dist as _model_combine_inning,
)
from src.simulator import simulate_game
from src.stadiums import get_stadium
from src.teams import get_team_meta
from src import predictions as _pred
from src import calibration as _cal
from src import grading as _grading
from src.colors import hex_to_rgb_str as _hex_to_rgb_str, bar_color as _bar_color
from src.explanations import (
    _explanation_cache, _load_disk_explanations, _build_explain_prompt,
    _stream_ollama, _pregenerate_explanations,
)
from src.chat import stream_chat
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
_calibrator_cache: dict = {}  # data-dir -> PlattCalibrator|None; win% calibration loaded lazily


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
    """Format the averaged ESPN line (with -110 defaults where a price is missing)
    and compute the model's edge % per market — the model's probability minus the
    market's de-vigged implied probability, shown on the side the model favors."""
    from collections import defaultdict

    # Model run-margin + total distributions, by convolving the per-team score
    # histograms (the sim draws each team's runs independently).
    dist = game.get('score_distribution') or {}
    hc, ac = dist.get('home') or [], dist.get('away') or []
    hsum, asum = sum(hc), sum(ac)
    margin, total = defaultdict(float), defaultdict(float)
    if hsum and asum:
        hp, ap = [c / hsum for c in hc], [c / asum for c in ac]
        for h, ph in enumerate(hp):
            if ph:
                for a, pa in enumerate(ap):
                    if pa:
                        margin[h - a] += ph * pa
                        total[h + a] += ph * pa
    model_home_win = game.get('home_win_pct', 50.0) / 100.0
    fav_home = model_home_win >= 0.5
    ha, aa = game.get('home_abbr'), game.get('away_abbr')

    block = {
        'ml_home': _fmt_american(mk.get('ml_home')),
        'ml_away': _fmt_american(mk.get('ml_away')),
        'total': f"{mk['total']:.1f}" if mk.get('total') is not None else '—',
        'total_over': _fmt_american(mk.get('over_odds'), default=-110),
        'total_under': _fmt_american(mk.get('under_odds'), default=-110),
        'runline_odds': '-110',   # ESPN gives the ±1.5 line but no price
        'n_books': mk.get('n_books', 0),
    }

    # Moneyline edge — model win% vs the de-vigged market.
    ih, ia = _implied_prob(mk.get('ml_home')), _implied_prob(mk.get('ml_away'))
    mkt_home = ih / (ih + ia) if (ih + ia) else 0.5
    ml_edge = model_home_win - mkt_home
    block['edge_ml_side'] = ha if ml_edge >= 0 else aa
    block['edge_ml_pct'] = round(abs(ml_edge) * 100)

    # Total edge — model P(over) at the MARKET line vs the de-vigged market over.
    line = mk.get('total')
    if line is not None and total:
        po = sum(p for t, p in total.items() if t > line)
        pu = sum(p for t, p in total.items() if t < line)
        model_over = po / (po + pu) if (po + pu) else 0.5
        io, iu = _implied_prob(mk.get('over_odds')), _implied_prob(mk.get('under_odds'))
        mkt_over = io / (io + iu) if (io + iu) else 0.5
        t_edge = model_over - mkt_over
        block['edge_total_side'] = 'Over' if t_edge >= 0 else 'Under'
        block['edge_total_pct'] = round(abs(t_edge) * 100)
    else:
        block['edge_total_side'], block['edge_total_pct'] = '', None

    # Run-line edge — model P(favorite wins by >=2) vs the market (-110 -> 50%).
    if margin:
        p_cover = sum(p for d, p in margin.items() if (d >= 2 if fav_home else d <= -2))
        rl_edge = p_cover - 0.5
        fav, dog = (ha, aa) if fav_home else (aa, ha)
        block['edge_rl_side'] = f"{fav} -1.5" if rl_edge >= 0 else f"{dog} +1.5"
        block['edge_rl_pct'] = round(abs(rl_edge) * 100)
    else:
        block['edge_rl_side'], block['edge_rl_pct'] = '', None
    return block


_EDGE_MARKETS = (('ML', 'edge_ml_side', 'edge_ml_pct'),
                 ('Total', 'edge_total_side', 'edge_total_pct'),
                 ('Run line', 'edge_rl_side', 'edge_rl_pct'))


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
    """Flatten every game's ML/total/run-line edges, rank by %, return the top n.

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

    # Snapshot the market total line each Total pick is graded against. Write-once,
    # so the first (closing-ish) line captured for a date's games is preserved.
    try:
        _pred.save_market_lines(_DATA_DIR, sim_date, _market_total_lines(schedule, market))
    except Exception:
        pass

    results = []
    for game in schedule:
        try:
            g = _enrich_game(game, cores.get(game['game_id'], {}))
            mk = market.get(market_key(g.get('away_name', ''), g.get('home_name', '')))
            g['market'] = _market_block(g, mk) if mk else None
            g['best_edge'] = _best_edge(g['market'])
            results.append(g)
        except Exception as e:
            results.append({**game, 'error': str(e)})

    # Chronological by first pitch (the default the UI shows); missing times last.
    results.sort(key=lambda g: g.get('game_datetime') or '~')

    return results


def _market_total_lines(games: list, market: dict) -> dict:
    """Map game_id -> market total line, for games with a priced market total."""
    out = {}
    for g in games:
        mk = market.get(market_key(g.get('away_name', ''), g.get('home_name', '')))
        if mk and mk.get('n_books', 0) > 0 and mk.get('total') is not None:
            out[g['game_id']] = mk['total']
    return out


def _inning_scoring_lines(g: dict) -> str:
    """Compact per-inning run-bucket distribution (0 / 1 / 2+ runs) for away,
    home, and the combined (any team) line."""
    ad, hd, cd = (g.get('away_innings_dist'), g.get('home_innings_dist'),
                  g.get('combined_innings_dist'))
    if ad and hd:
        aw, hw = g.get('away_abbr', 'AWAY'), g.get('home_abbr', 'HOME')
        fmt = lambda dist: " ".join(
            f"i{j+1} {round(c[0])}/{round(c[1])}/{round(c[2])}" for j, c in enumerate(dist))
        out = (f"Per-inning run distribution P(0/1/2+ runs) by inning 1-9 — "
               f"{aw}: {fmt(ad)}; {hw}: {fmt(hd)}")
        if cd:
            out += f"; both teams combined: {fmt(cd)}"
        return out + " (percent)."
    # Legacy cores (binary P(score>=1)).
    ap, hp = g.get('away_innings_scoring_pct'), g.get('home_innings_scoring_pct')
    if not ap or not hp:
        return ""
    aw, hw = g.get('away_abbr', 'AWAY'), g.get('home_abbr', 'HOME')
    either = [round((1 - (1 - a / 100) * (1 - h / 100)) * 100) for a, h in zip(ap, hp)]
    fmt = lambda xs: "/".join(f"{round(x)}" for x in xs)
    return (f"P(team scores >=1 run) by inning 1-9 — {aw}: {fmt(ap)}; "
            f"{hw}: {fmt(hp)}; either team: {fmt(either)} (percent).")


def _game_chat_line(g: dict) -> str:
    """A factual block about a game for the chatbot context: predictions, the
    full inning-by-inning scoring breakdown, venue/weather, lines and edges."""
    m = g.get('lines') or {}
    mk = g.get('market') or {}
    w = g.get('weather') or {}
    aw, hw = g.get('away_abbr', '?'), g.get('home_abbr', '?')
    parts = [
        f"{aw} @ {hw} ({g.get('away_name')} at {g.get('home_name')}):",
        f"model win% {aw} {g.get('away_win_pct')}% / {hw} {g.get('home_win_pct')}%,",
        f"predicted runs {aw} {g.get('predicted_away_runs')} / {hw} {g.get('predicted_home_runs')} "
        f"(most-likely score {g.get('modal_away_score')}-{g.get('modal_home_score')}, "
        f"median {g.get('median_away_score')}-{g.get('median_home_score')});",
        f"SP {g.get('away_pitcher')} vs {g.get('home_pitcher')};",
    ]
    venue = g.get('venue_name')
    if venue:
        cond = "indoor dome" if w.get('is_dome') else (
            f"{w.get('temperature_f')}F, wind {w.get('wind_speed_mph')}mph" if w.get('temperature_f') is not None else "")
        parts.append(f"venue {venue} ({g.get('elevation_ft', '?')} ft{', ' + cond if cond else ''});")
    inn = _inning_scoring_lines(g)
    if inn:
        parts.append(inn)
    if m:
        parts.append(f"model fair lines: total {m.get('total_line')}, run line {m.get('spread_home')} (home)/"
                     f"{m.get('spread_away')} (away), ML {aw} {m.get('ml_away')}/{hw} {m.get('ml_home')};")
    if mk:
        parts.append(
            f"ESPN avg line ({mk.get('n_books', 0)} books): total {mk.get('total')}, "
            f"ML {aw} {mk.get('ml_away')}/{hw} {mk.get('ml_home')}; "
            f"model edges vs market: ML {mk.get('edge_ml_side')} +{mk.get('edge_ml_pct')}%, "
            f"total {mk.get('edge_total_side')} +{mk.get('edge_total_pct')}%, "
            f"run line {mk.get('edge_rl_side')} +{mk.get('edge_rl_pct')}%."
        )
    return " ".join(p for p in parts if p)


def _build_chat_context(focus_game_id: int = None) -> str:
    today = date.today().strftime('%Y-%m-%d')
    meta = _read_model_meta()
    cur = _current_version()
    model_name = next((e.get('name') for e in _pred.read_versions(_DATA_DIR)
                       if e.get('version') == cur), None) or f"v{meta.get('feature_version', '?')}"
    out = [
        f"DATE: {today}.",
        f"MODEL: {model_name} — three XGBoost models (home-win classifier + two run "
        f"regressors) on a {len(FEATURE_COLUMNS)}-feature vector (pitcher ERA/FIP/WHIP, "
        f"team wOBA/OPS, bullpen, park factors, elevation, weather, handedness, rest, "
        f"recent form), trained on {meta.get('n_games', '?')} completed 2024–2026 games. "
        f"Win prob + run totals come from a 1000-run Monte Carlo (negative-binomial) per game; "
        f"the inning breakdown is a separate classifier. Betting lines are the model's fair "
        f"(no-vig) implied lines; 'edges' compare them to the average ESPN sportsbook line.",
    ]
    games = [g for g in _simulation_cache if not g.get('error')]
    if games:
        out.append("\nTODAY'S GAMES:")
        out.extend(f"- {_game_chat_line(g)}" for g in games)
    else:
        out.append("\nNo games scheduled today.")

    last7, last90 = _aggregate_days(7), _aggregate_days(90)
    out.append(
        f"\nHISTORICAL ACCURACY (in-sample backtest): last 7 days "
        f"{last7.get('winner_accuracy')}% winners over {last7.get('total_games')} games "
        f"(±{last7.get('avg_score_err')} avg run error); last 90 days "
        f"{last90.get('winner_accuracy')}% over {last90.get('total_games')} games."
    )
    if _results_cache and _results_cache.get('games'):
        y = _results_cache
        out.append(f"YESTERDAY ({_last_results_date}): {y.get('winner_accuracy')}% winners, "
                   f"{y.get('n_completed')} games graded.")

    if focus_game_id:
        g = next((x for x in _simulation_cache if x.get('game_id') == focus_game_id), None)
        if g:
            expl = _explanation_cache.get(focus_game_id)
            out.append(f"\nFOCUS GAME the user wants to discuss: {g.get('away_name')} @ "
                       f"{g.get('home_name')}. Full model analysis: "
                       f"{expl or '(analysis still generating)'}")
    return "\n".join(out)


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
    market_lines = _pred.load_market_lines(_DATA_DIR, result_date)
    results = []
    correct = 0
    spread_correct = spread_n = total_correct = total_n = 0
    for core in cores:
        game = actuals.get(core.get('game_id'))
        if not game:
            continue  # not final yet — no comparison row
        core = _calibrate_core(core)  # display win% calibrated; pick (>50) unchanged
        actual_home = int(game['home_score'])
        actual_away = int(game['away_score'])
        actual_home_won = actual_home > actual_away
        predicted_home_won = core.get('home_win_pct', 50.0) > 50.0
        home_err = abs(core.get('median_home_score', 0) - actual_home)
        away_err = abs(core.get('median_away_score', 0) - actual_away)
        if actual_home_won == predicted_home_won:
            correct += 1
        # Market-by-market grade: ML (== winner_correct), Spread (run-line +/-1.5),
        # Total (vs the captured market line; N/A when no line was snapshotted).
        total_line = market_lines.get(str(core.get('game_id')), {}).get('total_line')
        marks = _grading.grade_markets(core, actual_home, actual_away, total_line)
        if marks['spread'] is not None:
            spread_n += 1
            spread_correct += 1 if marks['spread'] else 0
        if marks['total'] not in (None, 'push'):
            total_n += 1
            total_correct += 1 if marks['total'] else 0
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
            'ml_correct': marks['ml'],
            'spread_correct': marks['spread'],
            'total_correct': marks['total'],
            'total_line': total_line,
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
    spread_accuracy = round(spread_correct / spread_n * 100, 1) if spread_n else None
    total_accuracy = round(total_correct / total_n * 100, 1) if total_n else None

    return {
        'result_date': result_date,
        'games': results,
        'n_completed': n,
        'winner_accuracy': accuracy,
        'ml_accuracy': accuracy,
        'spread_accuracy': spread_accuracy,
        'spread_graded': spread_n,
        'total_accuracy': total_accuracy,
        'total_graded': total_n,
        'avg_score_err': avg_err,
    }


def _get_results_for_date(date_str: str) -> dict:
    """Results for a past date by joining the frozen prediction with actuals."""
    return compare_date(date_str)


def _weighted_market_accuracy(daily: list, acc_key: str, n_key: str):
    """Pool a per-day market accuracy by its graded count -> (accuracy, total_n).

    Returns (None, 0) when nothing was graded for the market across the window.
    """
    n = sum(r.get(n_key, 0) for r in daily)
    if not n:
        return None, 0
    correct = sum(round((r.get(acc_key) or 0) / 100 * r.get(n_key, 0)) for r in daily)
    return round(correct / n * 100, 1), n


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
    spread_acc, spread_n = _weighted_market_accuracy(daily, 'spread_accuracy', 'spread_graded')
    total_acc, total_n = _weighted_market_accuracy(daily, 'total_accuracy', 'total_graded')

    result = {
        'total_games': total_games,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'ml_accuracy': accuracy,
        'spread_accuracy': spread_acc,
        'spread_graded': spread_n,
        'total_accuracy': total_acc,
        'total_graded': total_n,
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
    spread_acc, spread_n = _weighted_market_accuracy(daily, 'spread_accuracy', 'spread_graded')
    total_acc, total_n = _weighted_market_accuracy(daily, 'total_accuracy', 'total_graded')
    result = {
        'total_games': total,
        'days_with_games': len(daily),
        'winner_accuracy': accuracy,
        'ml_accuracy': accuracy,
        'spread_accuracy': spread_acc,
        'spread_graded': spread_n,
        'total_accuracy': total_acc,
        'total_graded': total_n,
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
                               top_edges=_top_edges(_simulation_cache),
                               yesterday=_results_cache, yesterday_date=yesterday,
                               last_7=last_7, last_90=last_90,
                               model_name=model_name,
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
                'spread_accuracy': s.get('spread_accuracy'),
                'total_accuracy': s.get('total_accuracy'),
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

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json(silent=True) or {}
        # cap history so the context window isn't blown; keep only role/content
        messages = [
            {'role': m.get('role'), 'content': str(m.get('content', ''))[:2000]}
            for m in (data.get('messages') or [])[-12:]
            if m.get('role') in ('user', 'assistant') and m.get('content')
        ]
        if not messages:
            abort(400)
        game_id = data.get('game_id')
        context = _build_chat_context(int(game_id) if game_id else None)
        return Response(stream_with_context(stream_chat(messages, context)),
                        mimetype='text/plain; charset=utf-8',
                        headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

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
