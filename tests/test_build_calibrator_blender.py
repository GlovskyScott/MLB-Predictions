from scripts.build_calibrator import build_blender
from src import predictions as P


def _fair_one(q):
    """Fair American price for a single-side probability q."""
    return -100 * q / (1 - q) if q >= 0.5 else 100 * (1 - q) / q


def _fair_american(p):
    """Fair (no-vig) American odds for home prob p, as (home_price, away_price)."""
    return _fair_one(p), _fair_one(1 - p)


def test_build_blender_joins_market_and_fits(tmp_path):
    # Market strongly predicts outcome; model is noise. Blender should weight market.
    import numpy as np
    rng = np.random.default_rng(1)
    pairs = []
    for d in range(1, 11):                       # 10 days x 80 = 800 games (> min 500)
        date = f'2026-04-{d:02d}'
        odds = {}
        for gid in range(1, 81):
            p = float(rng.uniform(0.25, 0.75))
            mlh, mla = _fair_american(p)
            odds[gid] = {'ml_home': mlh, 'ml_away': mla}
            pairs.append({'game_id': gid, 'date': date, 'model_home_prob': 0.5,
                          'home_win': int(rng.uniform() < p)})
        P.save_market_odds(tmp_path, date, odds)
    bl = build_blender(pairs, tmp_path)
    assert bl is not None and bl.b > 0.4         # leans on the market signal
    assert bl(0.5, 0.75) > bl(0.5, 0.25)


def test_build_blender_none_without_market(tmp_path):
    pairs = [{'game_id': 1, 'date': '2026-04-01', 'model_home_prob': 0.6, 'home_win': 1}]
    assert build_blender(pairs, tmp_path) is None   # no market_odds on disk
