"""Grade a frozen prediction core on the three betting markets — ML, Spread
(run-line +/-1.5), and Total (O/U) — without re-simulating.

The simulator draws each team's run total independently, so the core's two
marginal score histograms (``score_distribution``) carry the full joint under
independence. Every market pick is therefore recoverable by convolving those two
marginals at grade time. Picks are determined by the model's stored
probabilities; correctness by the actual final score.

(Extra-innings tie-breaking couples the two totals slightly in ~9% of games; for
Spread, margin>=2 games are never tied, so it is irrelevant, and for Total it is
negligible.)
"""


def _norm(hist: list) -> list:
    total = float(sum(hist))
    if total <= 0:
        return [0.0] * len(hist)
    return [h / total for h in hist]


def _marginals(core: dict):
    dist = core.get('score_distribution') or {}
    labels = dist.get('labels') or list(range(len(dist.get('home', []))))
    return labels, _norm(dist.get('home', [])), _norm(dist.get('away', []))


def home_cover_prob(core: dict) -> float:
    """P(home wins by >= 2 runs) under independence of the two marginals."""
    labels, ph, pa = _marginals(core)
    p = 0.0
    for i, hv in enumerate(labels):
        if ph[i] == 0:
            continue
        for j, av in enumerate(labels):
            if hv - av >= 2:
                p += ph[i] * pa[j]
    return p


def away_cover_prob(core: dict) -> float:
    """P(away wins by >= 2 runs)."""
    labels, ph, pa = _marginals(core)
    p = 0.0
    for i, hv in enumerate(labels):
        if ph[i] == 0:
            continue
        for j, av in enumerate(labels):
            if av - hv >= 2:
                p += ph[i] * pa[j]
    return p


def total_over_prob(core: dict, line: float) -> float:
    """P(home + away > line)."""
    labels, ph, pa = _marginals(core)
    p = 0.0
    for i, hv in enumerate(labels):
        if ph[i] == 0:
            continue
        for j, av in enumerate(labels):
            if hv + av > line:
                p += ph[i] * pa[j]
    return p


def grade_markets(core: dict, actual_home: int, actual_away: int,
                  total_line: float = None) -> dict:
    """Grade ML / Spread / Total for one game.

    Returns {'ml': bool, 'spread': bool, 'total': bool | 'push' | None}. ``total``
    is None when no market line was captured for the game.
    """
    actual_home = int(actual_home)
    actual_away = int(actual_away)
    home_won = actual_home > actual_away

    # ML — pick the side with win% > 50 (matches compare_date's winner_correct).
    pick_home = core.get('home_win_pct', 50.0) > 50.0
    ml = (pick_home == home_won)

    # Spread / Total both need the score distribution; without it they are N/A.
    dist = core.get('score_distribution') or {}
    has_dist = sum(dist.get('home', [])) > 0 and sum(dist.get('away', [])) > 0
    spread = None
    total = None
    if has_dist:
        # Spread — favorite is the ML side; fav -1.5 if it covers >50%, else dog +1.5.
        if pick_home:
            fav_margin = actual_home - actual_away
            fav_cover_p = home_cover_prob(core)
        else:
            fav_margin = actual_away - actual_home
            fav_cover_p = away_cover_prob(core)
        spread = (fav_margin >= 2) if fav_cover_p > 0.5 else (fav_margin <= 1)

        # Total — pick over/under at the captured market line.
        if total_line is not None:
            actual_total = actual_home + actual_away
            if actual_total == total_line:
                total = 'push'
            else:
                over = total_over_prob(core, total_line) > 0.5
                total = (actual_total > total_line) if over else (actual_total < total_line)

    return {'ml': ml, 'spread': spread, 'total': total}
