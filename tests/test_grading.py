"""Market-pick grading derived from the frozen core's score distribution."""
from src import grading as G


def _core(home_hist, away_hist, home_win_pct):
    n = max(len(home_hist), len(away_hist))
    home = home_hist + [0] * (n - len(home_hist))
    away = away_hist + [0] * (n - len(away_hist))
    return {
        'home_win_pct': home_win_pct, 'away_win_pct': 100 - home_win_pct,
        'score_distribution': {'labels': list(range(n)), 'home': home, 'away': away},
    }


def _point(value):
    return [0] * value + [1]


# ---- ML ---------------------------------------------------------------------

def test_ml_correct_when_favorite_wins():
    core = _core(_point(5), _point(3), home_win_pct=60.0)
    assert G.grade_markets(core, 5, 3)['ml'] is True


def test_ml_wrong_when_favorite_loses():
    core = _core(_point(5), _point(3), home_win_pct=60.0)
    assert G.grade_markets(core, 3, 5)['ml'] is False


# ---- Spread (run-line +/-1.5) ----------------------------------------------

def test_spread_favorite_minus_1_5_covers():
    # home wins by 2 in the dist -> P(home cover) = 1 -> pick home -1.5
    core = _core(_point(5), _point(3), home_win_pct=99.0)
    assert G.grade_markets(core, 5, 3)['spread'] is True     # won by 2 -> covers
    assert G.grade_markets(core, 5, 4)['spread'] is False    # won by 1 -> no cover


def test_spread_underdog_plus_1_5_covers():
    # favorite only ever wins by 1 -> P(home cover) = 0 -> pick dog +1.5
    core = _core(_point(5), _point(4), home_win_pct=99.0)
    assert G.grade_markets(core, 5, 4)['spread'] is True     # fav by 1 -> +1.5 covers
    assert G.grade_markets(core, 6, 3)['spread'] is False    # fav by 3 -> +1.5 loses


# ---- Total (O/U at the market line) ----------------------------------------

def test_total_over_pick_hits():
    core = _core(_point(5), _point(3), home_win_pct=55.0)    # total always 8
    assert G.grade_markets(core, 5, 3, total_line=7.5)['total'] is True   # over 7.5


def test_total_under_pick_hits():
    core = _core(_point(5), _point(3), home_win_pct=55.0)    # total always 8
    assert G.grade_markets(core, 5, 3, total_line=8.5)['total'] is True   # under 8.5


def test_total_push_on_integer_line():
    core = _core(_point(5), _point(3), home_win_pct=55.0)
    assert G.grade_markets(core, 5, 3, total_line=8)['total'] == 'push'


def test_total_none_without_line():
    core = _core(_point(5), _point(3), home_win_pct=55.0)
    assert G.grade_markets(core, 5, 3, total_line=None)['total'] is None


def test_spread_and_total_none_without_distribution():
    # A minimal core (no score_distribution) can still grade ML, not Spread/Total.
    core = {'home_win_pct': 60.0}
    r = G.grade_markets(core, 5, 3, total_line=7.5)
    assert r['ml'] is True
    assert r['spread'] is None
    assert r['total'] is None
