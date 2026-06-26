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


def simulate_game(prediction: dict, n_simulations: int = 1000, seed: int = None) -> dict:
    """
    Run Monte Carlo simulation for a single game.

    prediction: output from model.predict_game()
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
        home_runs = int(rng.poisson(home_lambda))
        away_runs = int(rng.poisson(away_lambda))

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

    home_innings_median = [float(np.median(home_innings_matrix[:, j])) for j in range(9)]
    away_innings_median = [float(np.median(away_innings_matrix[:, j])) for j in range(9)]

    median_home = float(np.median(home_scores_all))
    median_away = float(np.median(away_scores_all))

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
        'home_innings': home_innings_median,
        'away_innings': away_innings_median,
        'score_distribution': {
            'home': home_hist,
            'away': away_hist,
            'labels': list(range(max_runs + 1)),
        },
        'predicted_score': f"{round(median_home)}-{round(median_away)}",
        'n_simulations': n_simulations,
    }
