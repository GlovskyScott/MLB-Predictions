import numpy as np
from collections import Counter

# Historical MLB run distribution by inning (9 innings, sums to 1.0)
INNING_WEIGHTS = [0.120, 0.105, 0.095, 0.105, 0.100, 0.100, 0.110, 0.120, 0.145]


def _distribute_runs_to_innings(total_runs: int, rng: np.random.RandomState) -> list[int]:
    """Distribute total_runs across 9 innings using historical weights."""
    if total_runs == 0:
        return [0] * 9
    counts = rng.multinomial(total_runs, INNING_WEIGHTS)
    return counts.tolist()


def _runs_draw(rng: np.random.RandomState, mu: float, overdispersion: float) -> int:
    """Sample a team's run total. MLB run-scoring is overdispersed (variance ≈ 2×
    mean), which a plain Poisson (variance == mean) underestimates — it misses the
    blowout tail. A negative binomial with variance = overdispersion·mu captures
    that; overdispersion == 1 collapses back to Poisson.
    """
    if overdispersion <= 1.0 or mu <= 0:
        return int(rng.poisson(max(0.0, mu)))
    r = mu / (overdispersion - 1.0)   # NB dispersion: var = mu + mu^2/r = overdispersion*mu
    p = r / (r + mu)
    return int(rng.negative_binomial(r, p))


def simulate_game(prediction: dict, n_simulations: int = 1000, seed: int = None,
                  overdispersion: float = 2.0) -> dict:
    """
    Run Monte Carlo simulation for a single game.

    prediction: output from model.predict_game()
    overdispersion: variance/mean ratio of the per-team run distribution. ~2.0
        matches MLB (negative binomial); 1.0 reverts to Poisson.
    Returns aggregated statistics across all simulations.
    """
    rng = np.random.RandomState(seed)

    home_lambda = max(0.1, prediction['predicted_home_runs'])
    away_lambda = max(0.1, prediction['predicted_away_runs'])

    home_wins = 0
    away_wins = 0
    home_scores_all = []
    away_scores_all = []
    home_innings_matrix = np.zeros((n_simulations, 9), dtype=int)
    away_innings_matrix = np.zeros((n_simulations, 9), dtype=int)

    for i in range(n_simulations):
        home_runs = _runs_draw(rng, home_lambda, overdispersion)
        away_runs = _runs_draw(rng, away_lambda, overdispersion)

        home_by_inning = _distribute_runs_to_innings(home_runs, rng)
        away_by_inning = _distribute_runs_to_innings(away_runs, rng)

        # Extra innings if tied after 9
        if home_runs == away_runs:
            for _ in range(3):
                extra_away = int(rng.poisson(away_lambda / 9))
                extra_home = int(rng.poisson(home_lambda / 9))
                away_runs += extra_away
                home_runs += extra_home
                if home_runs != away_runs:
                    break
            if home_runs == away_runs:
                if rng.random() > 0.5:
                    home_runs += 1
                else:
                    away_runs += 1

        if home_runs > away_runs:
            home_wins += 1
        else:
            away_wins += 1

        home_scores_all.append(home_runs)
        away_scores_all.append(away_runs)
        home_innings_matrix[i] = home_by_inning
        away_innings_matrix[i] = away_by_inning

    # Mean per inning so values sum to predicted total; medians of Poisson(λ<0.5) are always 0
    home_innings_mean = [round(float(np.mean(home_innings_matrix[:, j])), 2) for j in range(9)]
    away_innings_mean = [round(float(np.mean(away_innings_matrix[:, j])), 2) for j in range(9)]
    home_innings_scoring_pct = [round(float(np.mean(home_innings_matrix[:, j] >= 1)) * 100, 1) for j in range(9)]
    away_innings_scoring_pct = [round(float(np.mean(away_innings_matrix[:, j] >= 1)) * 100, 1) for j in range(9)]

    median_home = float(np.median(home_scores_all))
    median_away = float(np.median(away_scores_all))

    # Most common actual game outcome — never ties since extra innings resolves them
    score_pairs = Counter(zip(away_scores_all, home_scores_all))
    modal_away, modal_home = score_pairs.most_common(1)[0][0]

    home_dist = Counter(home_scores_all)
    away_dist = Counter(away_scores_all)
    max_runs = max(max(home_scores_all), max(away_scores_all), 15)
    home_hist = [home_dist.get(r, 0) for r in range(max_runs + 1)]
    away_hist = [away_dist.get(r, 0) for r in range(max_runs + 1)]

    return {
        'home_win_pct': round((home_wins / n_simulations) * 100, 1),
        'away_win_pct': round((away_wins / n_simulations) * 100, 1),
        'median_home_score': round(median_home, 1),
        'median_away_score': round(median_away, 1),
        'modal_home_score': modal_home,
        'modal_away_score': modal_away,
        'home_innings': home_innings_mean,
        'away_innings': away_innings_mean,
        'home_innings_scoring_pct': home_innings_scoring_pct,
        'away_innings_scoring_pct': away_innings_scoring_pct,
        'score_distribution': {
            'home': home_hist,
            'away': away_hist,
            'labels': list(range(max_runs + 1)),
        },
        # Use the modal (most-common) outcome — it's the documented "most likely
        # score" and is tie-free (extra innings resolve ties), unlike the median
        # which can land on an impossible tie like 4-4.
        'predicted_score': f"{modal_home}-{modal_away}",
        'n_simulations': n_simulations,
    }
