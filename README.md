# MLB Predictions

A machine-learning web app that predicts MLB game outcomes for the current day using XGBoost models trained on 2024–2026 historical data, Monte Carlo simulation, real-time weather, and live MLB Stats API data.

---

## What it does

For every game on today's schedule the app:

1. Builds a 44-feature vector per game (pitcher stats, team batting, bullpen, weather, park factors, handedness matchups, recent form)
2. Runs the feature vector through three XGBoost models to get win probability and expected runs for each team
3. Simulates the game 500 times via Poisson Monte Carlo to produce a score distribution, inning-by-inning scoring percentages, and a most-likely final score
4. Overrides the inning percentages with a separate ML classifier trained on 100k+ individual inning observations
5. Displays everything in a dark-themed dashboard with historical accuracy tracked over the past 90 days

---

## Tech stack

| Layer | Library |
|---|---|
| Web framework | Flask 3 |
| ML models | XGBoost 2 (classifier + 2 regressors + inning classifier) |
| Feature engineering | mlb-statsapi, pybaseball |
| Simulation | NumPy Poisson Monte Carlo |
| Weather | Open-Meteo (free, no API key) |
| Charts | Chart.js |
| Data pipeline | pandas |

---

## Models

### Game model (FEATURE_VERSION 2 — 44 features)

Three XGBoost models are trained together from the same feature matrix:

| Model | Target | File |
|---|---|---|
| Win classifier | Home team win (0/1) | `data/model_win.pkl` |
| Home runs regressor | Home team final runs scored | `data/model_runs_home.pkl` |
| Away runs regressor | Away team final runs scored | `data/model_runs_away.pkl` |

**Features (44 total):**

- Away/home starting pitcher — ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9, IP
- Away/home team batting — wOBA, OPS
- Away/home bullpen — ERA, WHIP
- Park factors — runs factor, HR factor
- Weather — temperature (°F), wind speed, wind direction (out/in/cross), precipitation flag, dome flag
- Pitcher rest — days since last start (both starters)
- Recent team offense — runs per game over last 15 games (both teams)
- Bullpen stress — late-inning runs allowed over last 3 games (proxy for fatigue)
- Handedness matchup — starter is LHP flag + team OPS vs that handedness (both sides)

Training data: 5,921 completed games from 2024–2026 seasons.

### Inning model (10 features)

A separate XGBoost classifier predicts P(team scores ≥ 1 run) for each of the 9 innings independently, trained on 106,578 individual inning observations.

**Features:** inning number, is_home, batting wOBA, batting OPS, batting runs/game (L15), pitcher ERA, pitcher WHIP, starter days rest, bullpen stress L3, park runs factor.

---

## Project structure

```
MLB-Predictions/
├── src/
│   ├── app.py          # Flask app — routes, simulation orchestration, background backfill
│   ├── features.py     # Feature engineering (build_game_features, FEATURE_COLUMNS)
│   ├── fetcher.py      # All external data: MLB Stats API, pybaseball, Open-Meteo
│   ├── model.py        # Train, load, and predict with XGBoost models
│   ├── simulator.py    # Poisson Monte Carlo game simulation
│   ├── stadiums.py     # Stadium lat/lon, park factors, roof type
│   └── teams.py        # Team colors, logos, abbreviations
├── templates/
│   └── index.html      # Full dashboard UI
├── static/
│   ├── chart.umd.min.js
│   └── charts.js       # Win probability bar + score distribution chart renderers
├── scripts/
│   ├── prefetch_historical.py   # Pre-fetch pitcher handedness + batting splits to disk
│   └── upload_data_release.sh  # Bundle and upload data cache to GitHub release
├── tests/
│   ├── test_app.py
│   ├── test_features.py
│   ├── test_fetcher.py
│   ├── test_model.py
│   ├── test_simulator.py
│   └── test_stadiums.py
├── data/               # Git-ignored — downloaded at startup from GitHub releases
│   ├── model_win.pkl
│   ├── model_runs_home.pkl
│   ├── model_runs_away.pkl
│   ├── model_inning.pkl
│   ├── model_meta.json
│   ├── schedules/      # Season schedule CSVs (2024–2026)
│   ├── linescores/     # Per-game linescore cache
│   ├── pitcher_hand.json
│   ├── team_batting_splits_*.json
│   └── results_cache/  # 90-day daily result comparison snapshots
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- Git

### Install

```bash
git clone https://github.com/jackleh/MLB-Predictions.git
cd MLB-Predictions
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
flask --app src.app:create_app run
```

On first boot the app automatically:
1. Downloads the pre-built model pkl files from the `latest` GitHub release into `data/`
2. Downloads the historical data cache (schedules, linescores, splits) from the `data-cache` release
3. Starts a background thread that backfills 90 days of result comparisons (takes ~25 min the first time, then instant on subsequent restarts)

Open `http://localhost:5000` in your browser.

---

## Retraining

Click **⚙ Retrain** in the UI, or POST to `/retrain`. This:

1. Fetches every completed game from 2024–2026 via the MLB Stats API (using cached schedules and linescores — typically completes in ~33 seconds)
2. Builds the 44-feature matrix for each game using `for_training=True` (skips slow per-game pitcher splits and lineup calls)
3. Trains three XGBoost models and the inning classifier
4. Saves pkl files to `data/` and updates `data/model_meta.json`

### Pre-fetching handedness data (recommended before first retrain)

Pitcher handedness and team batting-vs-hand splits are looked up from disk during training. Run this once to populate the cache before retraining:

```bash
python scripts/prefetch_historical.py
```

This fetches handedness for every starter in 2024–2026 (~500 unique pitchers) and team batting splits for all 30 teams × 3 years, saving to `data/pitcher_hand.json` and `data/team_batting_splits_{year}.json`.

---

## Data releases

Model pkl files and the data cache are stored as GitHub release assets (excluded from git via `.gitignore`).

| Release tag | Contents |
|---|---|
| `latest` | `model_win.pkl`, `model_runs_home.pkl`, `model_runs_away.pkl`, `model_inning.pkl` |
| `data-cache` | `data_cache.tar.gz` — all data/ files except pkls |

To upload a new data cache after updating linescores or splits:

```bash
bash scripts/upload_data_release.sh
```

---

## Key implementation details

### `for_training=True` flag

`build_game_features` accepts a `for_training` flag. When set:
- Skips `get_pitcher_splits()` (requires a live game pk, not available for historical games)
- Skips `get_game_lineup()` (not available pre-game)
- Uses `cache_only=True` on linescore fetches — returns `None` instead of hitting the API for missing games, keeping retraining fast

### Schedule memory cache

`get_season_schedule()` checks an in-process `_season_schedule_memory` dict before reading from CSV. Prevents repeated disk reads during training across 3 years × many games.

### Background results backfill

On app startup a daemon thread calls `_backfill_results_cache(n_days=90)` using 6 parallel workers. Each worker calls `run_results_comparison()` for one historical date, caches the result to `data/results_cache/{date}.json`, and skips dates that are already cached. This means cold starts are ~25 min the first time, then instant thereafter.

### Inning probability model

The inning classifier overrides the simulation-derived scoring percentages (which use historical INNING_WEIGHTS). The ML model captures context — park, starter quality, bullpen — rather than relying on league-average base rates.

---

## API routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Main dashboard — today's predictions + 90-day historical results |
| `POST` | `/refresh` | Force re-run today's simulations |
| `POST` | `/retrain` | Retrain all models from historical data |

---

## Tests

```bash
pytest
```

The test suite mocks all external API calls (MLB Stats API, pybaseball, Open-Meteo). Tests cover feature building, simulation correctness, model train/predict, and app routes.

---

## Branch protection

The `master` branch requires all changes to come through a pull request. `@jackleh` is a required reviewer on all PRs (enforced via `.github/CODEOWNERS`).
