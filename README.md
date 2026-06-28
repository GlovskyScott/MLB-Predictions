# MLB Predictions

A machine-learning web app that predicts MLB game outcomes for the current day using XGBoost models trained on 2024–2026 historical data, Monte Carlo simulation, real-time weather, and live MLB Stats API data.

---

## What it does

For every game on today's schedule the app:

1. Builds a 44-feature vector per game (pitcher stats, team batting, bullpen, weather, park factors, elevation, handedness matchups, recent form)
2. Runs the feature vector through three XGBoost models to get win probability and expected runs for each team
3. Simulates the game 500 times via Poisson Monte Carlo to produce a score distribution, inning-by-inning scoring percentages, and a most-likely final score
4. Overrides the inning percentages with a separate ML classifier trained on 100k+ individual inning observations
5. Auto-generates an analyst-style AI explanation for each game using a local Ollama model (llama3.1:8b)
6. Displays everything in a dark-themed dashboard with historical accuracy tracked over the past 90 days

---

## Tech stack

| Layer | Library / Tool |
|---|---|
| Web framework | Flask 3 |
| ML models | XGBoost 2 (classifier + 2 regressors + inning classifier) |
| Feature engineering | mlb-statsapi, pybaseball |
| Simulation | NumPy Poisson Monte Carlo |
| Weather | Open-Meteo (free, no API key) |
| AI explanations | Ollama — llama3.1:8b (local) |
| Charts | Chart.js |
| Data pipeline | pandas |

---

## Models

### Game model (FEATURE_VERSION 4 — 44 features)

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
- Stadium elevation — `elevation_ft` (higher altitude = thinner air = more carry; Coors Field = 5,200 ft)
- Weather — temperature (°F), wind speed, wind direction (out/in/cross), precipitation flag, humidity (% RH), dome flag
- Pitcher rest — days since last start (both starters)
- Recent team offense — runs per game over last 15 games (both teams)
- Bullpen stress — late-inning runs allowed over last 3 games (proxy for fatigue)
- Handedness matchup — starter is LHP flag + team OPS vs that handedness (both sides)

Training data: 5,921 completed games from 2024–2026 seasons, with real historical weather (temperature, wind, precipitation, humidity) fetched per game per stadium.

### Inning model (11 features)

A separate XGBoost classifier predicts P(team scores ≥ 1 run) for each of the 9 innings independently, trained on 100k+ individual inning observations.

**Features:** inning number, is_home, batting wOBA, batting OPS, batting runs/game (L15), pitcher ERA, pitcher WHIP, starter days rest, bullpen stress L3, park runs factor, elevation_ft.

---

## Project structure

```
MLB-Predictions/
├── src/
│   ├── app.py          # Flask app — routes, simulation, background backfill, AI explanations
│   ├── features.py     # Feature engineering (build_game_features, FEATURE_COLUMNS)
│   ├── fetcher.py      # All external data: MLB Stats API, pybaseball, Open-Meteo
│   ├── model.py        # Train, load, and predict with XGBoost models
│   ├── simulator.py    # Poisson Monte Carlo game simulation
│   ├── stadiums.py     # Stadium lat/lon, elevation, park factors, roof type
│   └── teams.py        # Team colors, logos, abbreviations
├── templates/
│   └── index.html      # Full dashboard UI
├── static/
│   ├── chart.umd.min.js
│   └── charts.js       # Win probability bar + score distribution chart renderers
├── scripts/
│   ├── prefetch_historical.py   # Pre-fetch pitcher handedness + batting splits to disk
│   ├── prefetch_weather.py      # Bulk-fetch historical weather (12 months × all stadiums × all years)
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
│   ├── stadiums.json           # Stadium metadata including elevation_ft
│   ├── weather_cache.csv       # Historical weather per (date, lat, lon)
│   ├── schedules/              # Season schedule CSVs (2024–2026)
│   ├── linescores/             # Per-game linescore cache
│   ├── pitcher_hand.json
│   ├── team_batting_splits_*.json
│   ├── explanations/           # AI game explanations — {date}/{game_id}.txt
│   └── results_cache/          # 90-day daily result comparison snapshots
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- Git
- [Ollama](https://ollama.com) (for AI game explanations)

### Install

```bash
git clone https://github.com/jackleh/MLB-Predictions.git
cd MLB-Predictions
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Install Ollama and pull the model

```bash
brew install ollama
brew services start ollama
ollama pull llama3.1:8b
```

### Run

```bash
flask --app src.app:create_app run
```

On first boot the app automatically:
1. Downloads the pre-built model pkl files from the `latest` GitHub release into `data/`
2. Downloads the historical data cache (schedules, linescores, weather, splits) from the `data-cache` release
3. Starts a background thread that backfills 90 days of result comparisons (takes ~25 min the first time, then instant on subsequent restarts)

Open `http://localhost:5000` in your browser.

---

## UI features

Each game card shows:
- **Win probability bar** — home vs away, derived from Monte Carlo simulation
- **Predicted score** — most common final score across 500 simulations
- **Game start time** — local time converted from UTC in the browser
- **Starting pitchers** — both starters with handedness
- **Weather strip** — temperature, wind speed, humidity (% RH), venue, and elevation (e.g. `72°F · 5 mph · 58% RH · Coors Field · 5200 ft`; domes show `Dome · Rogers Centre · 276 ft`)
- **Inning breakdown** — scoring % and average runs per inning from the ML inning classifier
- **Score distribution chart** — run total probabilities across simulations
- **AI analysis** — auto-generated analyst-style explanation streamed from a local llama3.1:8b model, covering the starter matchup, offensive edges, bullpen state, handedness advantage, park/weather factors, and elevation
- **Lineup** — batting order (1–9) for both teams, when posted

---

## Retraining

Click **⚙ Retrain** in the UI, or POST to `/retrain`. This:

1. Fetches every completed game from 2024–2026 via the MLB Stats API (using cached schedules and linescores — typically completes in ~50 seconds)
2. Builds the 44-feature matrix for each game using `for_training=True` (skips slow per-game pitcher splits and lineup calls)
3. Trains three XGBoost models and the inning classifier
4. Saves pkl files to `data/` and updates `data/model_meta.json`

A version mismatch between `FEATURE_VERSION` in code and `model_meta.json` triggers an automatic retrain on the next page load.

### Pre-fetching scripts (run once before first retrain)

**Pitcher handedness + batting splits:**
```bash
python scripts/prefetch_historical.py
```
Fetches handedness for ~500 unique starters and team batting splits (vs LHP/RHP) for all 30 teams × 3 years.

**Historical weather:**
```bash
python scripts/prefetch_weather.py
```
Fetches real temperature, wind, precipitation, and humidity for every completed game via Open-Meteo — 12 monthly API calls per outdoor stadium per year (~1,008 calls total, ~10 min). Populates `data/weather_cache.csv`. Run this before retraining so all historical games use real weather values instead of defaults.

---

## Model versioning & the archive

Predictions are **immutable artifacts keyed by model version**. A version id is a
hash of the four model `.pkl` files, so any retrain produces a new version.

- Within a version, a date's prediction is generated once, frozen to
  `data/predictions/<version>/<date>.json`, and served verbatim forever after —
  the dashboard never re-simulates a stored date.
- A retrain forks a **new** version (new current model) and leaves the previous
  version's predictions intact as an **archive**. The new version's dates are
  re-simulated in the background.
- Accuracy is computed by joining a frozen prediction with the actual final
  score, so a model's historical accuracy never changes unless its version does.

Browse versions at **`/archive`** (click ▤ Archive in the header): each model
version with its rollup winner-accuracy and average score error, drilling into
per-date predictions vs actuals. `data/predictions/versions.json` is the registry
of known versions.

### Legacy models (pre-v4 releases)

The `v1.0.0` / `v1.1.0` release models were trained on a 32-feature pipeline that
the current 44-feature code can't reproduce, so they're graded **faithfully** by
replaying each model's own era's code. To reconstruct one:

```bash
# 1) generate its predictions using its own pipeline, in a worktree at its commit
git worktree add --detach /tmp/mlb-legacy <commit-with-32-feature-pipeline>
ln -s "$(pwd)/data" /tmp/mlb-legacy/data
cd /tmp/mlb-legacy
OLD_PKL=/path/to/release/pkls OUT_DIR=/tmp/legacy_staging \
    python scripts/import_legacy_predictions.py

# 2) register it into the archive (from the current checkout)
python scripts/register_legacy_version.py \
    --pkl-dir /path/to/release/pkls --staging /tmp/legacy_staging \
    --release v1.0.0 --created-at 2026-06-26T07:10:49Z --features 32 --n-games 5921
```

Legacy versions carry a `release` tag and `features` count in the registry (no
`feature_version`, since they predate that numbering) and grade on the same
simulation settings as current models, isolating the model as the variable.

## Data releases

Model pkl files and the data cache are stored as GitHub release assets (excluded from git via `.gitignore`).

| Release tag | Contents |
|---|---|
| `latest` | `model_win.pkl`, `model_runs_home.pkl`, `model_runs_away.pkl`, `model_inning.pkl` |
| `data-cache` | `data_cache.tar.gz` — all `data/` files except pkls and results_cache |
| `prediction-archive` | `predictions.tar.gz` — the versioned prediction store (`data/predictions/`): frozen per-game predictions for every trained model version, plus `versions.json` |

To upload a new data cache after updating linescores, splits, or weather:

```bash
bash scripts/upload_data_release.sh
```

To publish the prediction archive after a retrain (which forks a new model
version and archives the previous one):

```bash
bash scripts/upload_predictions_release.sh
```

On first boot the app restores this archive via `bootstrap_predictions_cache()`,
so the **Archive** page (per–model-version historical accuracy) works without
re-simulating. See [Model versioning & the archive](#model-versioning--the-archive).

---

## Key implementation details

### `for_training=True` flag

`build_game_features` accepts a `for_training` flag. When set:
- Skips `get_pitcher_splits()` (requires a live game pk, not available for historical games)
- Skips `get_game_lineup()` (not available pre-game)
- Uses `cache_only=True` on linescore fetches — returns `None` instead of hitting the API for missing games, keeping retraining fast

### Weather cache

Historical weather is cached in `data/weather_cache.csv` keyed by `(date, lat, lon)`. Cache entries missing `humidity_pct` (written before it was added) are skipped on load and re-fetched lazily with the full schema. Run `scripts/prefetch_weather.py` to bulk-populate the cache before retraining.

### Stadium elevation

Elevation data for all 30 stadiums is stored in `data/stadiums.json` (fetched once from the Open-Meteo elevation API). It is used as a feature in both the game and inning models and displayed in the UI. Higher elevation means thinner air and more ball carry — Coors Field (5,200 ft) is the extreme example.

### AI explanations

After each simulation run, a background thread calls Ollama (`llama3.1:8b`) sequentially for each game and stores the result in `data/explanations/{date}/{game_id}.txt`. On page load, the browser receives each explanation via Server-Sent Events, streaming text into the card as it generates. Subsequent loads for the same day serve from disk instantly. The prompt includes both starters' ERA/FIP/WHIP/handedness, team wOBA/OPS/recent form, bullpen ERA and fatigue, handedness OPS edges, park factor, elevation, and weather.

### Schedule memory cache

`get_season_schedule()` checks an in-process `_season_schedule_memory` dict before reading from CSV. Prevents repeated disk reads during training across 3 years × many games.

### Background results backfill

On app startup a daemon thread calls `_backfill_results_cache(n_days=90)` using 6 parallel workers. Each worker calls `run_results_comparison()` for one historical date, caches the result to `data/results_cache/{date}.json`, and skips dates that are already cached. Cold starts are ~25 min the first time, then instant thereafter.

### Inning probability model

The inning classifier overrides the simulation-derived scoring percentages (which use historical `INNING_WEIGHTS`). The ML model captures context — park, elevation, starter quality, bullpen fatigue — rather than relying on league-average base rates.

---

## API routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Main dashboard — today's predictions + 90-day historical results |
| `POST` | `/refresh` | Force re-run today's simulations |
| `POST` | `/retrain` | Retrain all models from historical data |
| `GET` | `/explain/<game_id>` | Stream AI explanation for one game via SSE |

---

## Tests

```bash
pytest
```

The test suite mocks all external API calls (MLB Stats API, pybaseball, Open-Meteo). Tests cover feature building, simulation correctness, model train/predict, and app routes.

---

## Branch protection

The `master` branch requires all changes to come through a pull request. `@jackleh` is a required reviewer on all PRs (enforced via `.github/CODEOWNERS`).
