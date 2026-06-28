# MLB Predictions

A machine-learning web app that predicts MLB game outcomes for the current day using XGBoost models trained on 2024–2026 historical data, Poisson Monte Carlo simulation, real weather, and live MLB Stats API data. Every prediction is frozen as an immutable, **model-versioned** artifact, so historical accuracy is graded against exactly what each model predicted — and you can compare model versions head-to-head on a dedicated archive page.

---

## What it does

For every game on today's schedule the app:

1. Builds a **44-feature** vector per game (pitcher stats, team batting, bullpen, weather, park factors, elevation, handedness matchups, recent form).
2. Runs the vector through three XGBoost models to get **win probability** and **expected runs** for each team.
3. Simulates the game **1,000 times** via Poisson Monte Carlo (fixed per-game seed) to produce a score distribution, inning-by-inning scoring percentages, and a most-likely final score.
4. Overrides the simulation's inning percentages with a separate **inning classifier** trained on 100k+ individual inning observations.
5. **Freezes** the prediction to disk under the current model version (`data/predictions/<version>/<date>.json`) — it is served verbatim forever after and never re-simulated.
6. Auto-generates an analyst-style **AI explanation** per game using a local Ollama model (`llama3.1:8b`), streamed into the card.
7. Displays everything in a dark dashboard with **historical accuracy** over the last 90 days, plus an **archive** comparing every model version's accuracy.

---

## Tech stack

| Layer | Library / Tool |
|---|---|
| Web framework | Flask 3 |
| ML models | XGBoost (win classifier + 2 run regressors + inning classifier) |
| Feature engineering | mlb-statsapi, pybaseball |
| Simulation | NumPy Poisson Monte Carlo |
| Weather | Open-Meteo (free, no API key) |
| AI explanations | Ollama — `llama3.1:8b` (local) |
| Charts | Chart.js |
| Data pipeline | pandas |
| Tests | pytest, pytest-mock |

---

## Models

### Game models (`FEATURE_VERSION = 4` — 44 features)

Three XGBoost models trained together from the same feature matrix:

| Model | Target | File |
|---|---|---|
| Win classifier | Home team win (0/1) | `data/model_win.pkl` |
| Home runs regressor | Home team final runs | `data/model_runs_home.pkl` |
| Away runs regressor | Away team final runs | `data/model_runs_away.pkl` |

**Features (44):** away/home starting-pitcher ERA/FIP/xFIP/WHIP/K9/BB9/HR9/IP; away/home team wOBA & OPS; away/home bullpen ERA & WHIP; park runs & HR factors; stadium `elevation_ft`; weather (temp °F, wind speed, wind direction out/in/cross, precipitation flag, humidity % RH, dome flag); pitcher days-rest (both); recent runs/game over last 15 (both); bullpen stress (late-inning runs allowed L3); handedness matchup (starter-is-LHP flag + team OPS vs that hand, both sides).

Training data: ~5,900 completed games from 2024–2026, with real historical weather fetched per game per stadium.

### Inning model (11 features)

A separate XGBoost classifier predicts P(team scores ≥ 1 run) for each of the 9 innings independently, trained on 100k+ inning observations. **Features:** inning number, is_home, batting wOBA, batting OPS, batting runs/game (L15), pitcher ERA, pitcher WHIP, starter days rest, bullpen stress L3, park runs factor, elevation_ft.

### How accurate is it, really?

The dashboard's 90-day accuracy is **in-sample** (the models train on the same completed games they're then graded on), so it overstates real skill. `scripts/backtest.py` runs proper temporal validation: in-sample ~74%, but **true out-of-sample ~58–60%** (vs a 53% home-team baseline) — genuinely good for MLB, just not 74%. See **[docs/MODELING.md](docs/MODELING.md)** for the full ladder, the negative-binomial run model, real FIP/xFIP, and the roadmap.

```bash
python scripts/backtest.py
```

### Calibrated win probabilities

The same overconfidence shows up in the *probabilities*: walk-forward, a simulated "72%" actually wins ~61%. Left uncorrected, comparing those inflated numbers to an efficient sportsbook line manufactures fake "edges" of 7–14% — which is what a real edge never looks like. So the displayed win% is run through a **temperature-scaling calibrator** (`src/calibration.py`, fit on out-of-sample predictions) before it's shown or used for the moneyline edge. It only rescales confidence — it passes exactly through 50%, so the winner pick and every graded accuracy number are unchanged; the fake moneyline edges collapse to a realistic 0–3%. The per-inning scoring probabilities get the same treatment, though there the model turns out to already be well-calibrated (walk-forward slope ≈ 0.97), so that map is essentially the identity. Rebuild both after a retrain with `python -m scripts.build_calibrator`.

---

## Project structure

```
MLB-Predictions/
├── src/
│   ├── app.py          # Flask app: routes, get_prediction/compare_date, backfill
│   ├── predictions.py  # Versioned prediction store: model_version(), registry, load/save, next_build()
│   ├── training.py     # Model training orchestration + model caches (get_models, needs_retrain, …)
│   ├── explanations.py # Ollama AI explanations (prompt, SSE stream, pre-generator)
│   ├── colors.py       # Team-color helpers for the dark UI
│   ├── features.py     # Feature engineering: build_game_features(), FEATURE_COLUMNS, FEATURE_VERSION
│   ├── fetcher.py      # All external data + caches: MLB Stats API, pybaseball, Open-Meteo, bootstrap_*()
│   ├── model.py        # Train / load / predict for the XGBoost models
│   ├── simulator.py    # Poisson Monte Carlo game simulation (simulate_game)
│   ├── calibration.py  # Serve-time win% calibration (temperature scaling, fit out-of-sample)
│   ├── stadiums.py     # Stadium lat/lon, elevation, park factors, roof type, wind classification
│   └── teams.py        # Team colors, logos, abbreviations
├── templates/
│   ├── index.html             # Main dashboard
│   ├── archive.html           # Model-version list with rollup accuracy
│   ├── archive_version.html   # One version's per-date predictions vs actuals
│   └── game.html              # Single-game detail
├── static/
│   ├── chart.umd.min.js
│   └── charts.js              # Win-prob bar + score-distribution chart renderers
├── scripts/
│   ├── prefetch_historical.py        # Pre-fetch pitcher handedness + batting splits
│   ├── prefetch_weather.py           # Bulk-fetch historical weather into weather_cache.csv
│   ├── upload_data_release.sh        # Publish the data cache to the data-cache release
│   ├── upload_predictions_release.sh # Publish data/predictions/ to the prediction-archive release
│   ├── import_legacy_predictions.py  # Regenerate a legacy model's predictions (run in a legacy worktree)
│   └── register_legacy_version.py    # Register a reconstructed legacy version into the archive
├── tests/                    # pytest; all external APIs mocked
│   ├── test_app.py  test_predictions.py  test_features.py
│   ├── test_fetcher.py  test_model.py  test_simulator.py  test_stadiums.py
├── docs/superpowers/         # Design spec + implementation plan for the versioned store
├── data/                     # Git-ignored — restored at startup from GitHub releases
│   ├── model_*.pkl, model_meta.json          # current models + metadata
│   ├── schedule_{2024,2025,2026}.csv         # full season schedules (status + final scores)
│   ├── linescore_cache.csv                    # per-game inning linescores
│   ├── pitching_*.csv, batting_*.csv, team_batting_*.csv
│   ├── pitcher_hand.json, pitcher_splits_*.json, team_batting_splits_*.json
│   ├── stadiums.json, park_factors.csv, weather_cache.csv
│   ├── explanations/<date>/<game_id>.txt     # cached AI explanations
│   └── predictions/                          # the versioned prediction store (see below)
│       ├── versions.json                     # registry of every model version
│       └── <version>/<date>.json             # frozen per-game prediction cores
└── requirements.txt
```

> `data/results_cache/` may exist from older builds; it is **deprecated** and ignored. Accuracy is now computed by joining the versioned store with live finals.

---

## Setup

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Git**
- **[Ollama](https://ollama.com)** with `llama3.1:8b` — only needed to *generate new* AI explanations; cached ones display without it
- **OpenMP runtime** for XGBoost — on macOS this is `libomp` (`brew install libomp`). Without it `import xgboost` fails with a `libxgboost.dylib could not be loaded` error.

### Install

```bash
git clone https://github.com/jackleh/MLB-Predictions.git
cd MLB-Predictions
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> No Homebrew? [`uv`](https://docs.astral.sh/uv/) is a drop-in: `uv venv --python 3.12 .venv && uv pip install -r requirements.txt`. If XGBoost can't find `libomp`, drop a `libomp.dylib` (e.g. from a conda-forge `llvm-openmp` package) onto the dynamic-loader path.

### Install Ollama and pull the model

```bash
brew install ollama && brew services start ollama
ollama pull llama3.1:8b            # ~4.9 GB
```

### Run

```bash
flask --app src.app:create_app run            # default http://localhost:5000
```

On macOS, AirPlay Receiver squats on port 5000 — use `--port 5001` if needed.

**On first boot** the app (`create_app`, when not testing):
1. Downloads pre-built model pkls from the **`latest`** GitHub release into `data/` (`bootstrap_model_cache`).
2. Downloads the historical data cache (schedules, linescores, weather, splits) from the **`data-cache`** release (`bootstrap_data_cache`).
3. Restores the versioned prediction archive from the **`prediction-archive`** release (`bootstrap_predictions_cache`).
4. Starts a daemon thread (`_backfill_results_cache`) that generates-and-freezes predictions for the trailing 90 days under the current model version, using 6 workers.

The first page load runs today's live simulation (~10 s); subsequent loads serve the frozen prediction instantly.

### Deployment

`flask run` uses the Werkzeug **development server** — single-threaded, so overlapping requests queue (a slow `/archive` will block other requests). It's fine for local use but **not for production**. For a real deployment use a WSGI server, e.g.:

```bash
pip install waitress
waitress-serve --port 8000 --call src.app:create_app
```

Note: in-process state (today's predictions, rollup caches, model objects) is **per-process**, so run a **single worker** — multiple gunicorn/waitress workers would each hold separate caches, and `/retrain` would only refresh the worker that handled it. (Background prediction generation is shared via the on-disk store, so a single worker is sufficient.)

---

## Dashboard (`/`)

Each game card shows:
- **Win-probability bar** (home vs away, from the simulation)
- **Betting board** — the model's implied **moneyline** (fair, no-vig; from the calibrated win probability), compared against the **average live sportsbook line** pulled from ESPN's public API (no key). A green badge flags where the model disagrees with the market (e.g. *model likes HOU*). (The run-line and total markets were removed — their edges came from the raw, uncalibrated run distribution and produced implausibly large numbers.)
- **Predicted score** (median final score across simulations)
- **Game start time** (local, converted from UTC in the browser)
- **Starting pitchers** with handedness
- **Weather strip** — e.g. `72°F · 5 mph · 58% RH · Coors Field · 5200 ft`; domes show `Dome · Rogers Centre · 276 ft`
- **Inning breakdown** — scoring % and average runs per inning (from the inning classifier)
- **Score-distribution chart** — run-total probabilities
- **AI analysis** — streamed from `llama3.1:8b` via SSE, covering starter matchup, offensive edges, bullpen state, handedness, park/weather/elevation
- **Lineup** — batting order 1–9 for both teams, when posted

Below the cards: yesterday's graded results, plus 7-day and 90-day accuracy rollups. The footer shows the current model name (e.g. `Model v4.1`).

**Dashboard Assistant** — a chat widget (bottom-right) backed by the local Ollama model. It's given the day's predictions, betting lines + edges, recent accuracy, and model facts as context, so you can ask things like *"which game has the biggest moneyline edge?"*, *"how did the model do yesterday?"*, or *"how are win probabilities computed?"*. Each game card has a **Discuss** button that hands that game (and its AI analysis) to the chat for follow-up questions.

---

## Model versioning & the archive

Predictions are **immutable artifacts keyed by model version**. A **version id** is the first 12 hex of a SHA-256 over the four model pkls, so any retrain produces a new version id.

### Naming rule — `v<major>.<build>`

- **major** = the feature generation the model trains on (the `FEATURE_VERSION` constant — currently `4`, the 44-feature set). Legacy 32-feature releases are generation `1`.
- **build** = increments from `0` for each new model trained at that generation.

A retrain at the same feature set bumps the build (`v4.0 → v4.1`); changing the feature set bumps the major and resets the build (`→ v5.0`). Names are assigned **once** at registration and never change. The pkl-hash id stays the storage key; `name` and `major` live alongside it in `versions.json`.

| Name | Hash | Origin | Features |
|---|---|---|---|
| `v1.0` | `947baaba…` | release v1.0.0 | 32 |
| `v1.1` | `6c70da42…` | release v1.1.0 | 32 |
| `v4.0` | `672d9a30…` | released `latest` | 44 |
| `v4.1` | `15dac968…` | retrain | 44 |

### How it behaves

- A date's prediction is generated once, frozen, and served verbatim — the dashboard never re-simulates a stored date (`get_prediction`).
- A **retrain forks a new version**: the previous version's predictions stay intact as an archive, and the new version's 90-day window is re-simulated in the background.
- **Accuracy** = join a frozen prediction with the actual final score (`compare_date`). It never changes unless the model version does. No simulation runs on the read/grading path.
- All versions are graded on the **same** simulation settings (`N_SIMULATIONS = 1000`, seed = `game_id % 100000`), isolating the model as the only variable.

### Archive (`/archive`)

Click **Archive** in the header. The list shows every model version with rollup winner-accuracy and average score error; click a version (`/archive/<version>`) to drill into its per-date predictions vs actuals. `data/predictions/versions.json` is the registry.

### Legacy models (pre-v4 releases)

The `v1.0.0`/`v1.1.0` release models use a 32-feature pipeline the current 44-feature code can't reproduce, so they're graded **faithfully** by replaying each model's own era's code in a git worktree:

```bash
# 1) generate predictions with the model's own pipeline, in a worktree at its commit
git worktree add --detach /tmp/mlb-legacy <commit-with-32-feature-pipeline>
ln -s "$(pwd)/data" /tmp/mlb-legacy/data
cd /tmp/mlb-legacy
OLD_PKL=/path/to/release/pkls OUT_DIR=/tmp/legacy_staging \
    python scripts/import_legacy_predictions.py

# 2) register it into the archive (from the current checkout)
python scripts/register_legacy_version.py \
    --pkl-dir /path/to/release/pkls --staging /tmp/legacy_staging \
    --release v1.0.0 --created-at 2026-06-26T07:10:49Z --features 32 --n-games 5921 --major 1
```

---

## Retraining

Click **Retrain** in the UI, or `POST /retrain`. This:

1. Rebuilds the 44-feature matrix for every completed 2024–2026 game (`for_training=True` skips slow per-game splits/lineups; ~50 s using the cached schedules/linescores).
2. Trains the three game models + the inning classifier and writes new pkls to `data/`.
3. Forks a **new model version** (new pkl hash → new `v<major>.<build>` name), leaving prior versions archived.
4. Re-simulates today + the trailing 90 days under the new version in a background thread.

A mismatch between `FEATURE_VERSION` / `training_years` in code and `data/model_meta.json` triggers an automatic retrain on next page load (`_needs_retrain`).

Training is deterministic (`random_state=42`), so retraining on identical data produces identical pkls (same version). A retrain only forks a new version when the training data or feature set actually changes.

### Pre-fetch scripts (run once before a from-scratch retrain)

```bash
python scripts/prefetch_historical.py   # pitcher handedness + team batting splits
python scripts/prefetch_weather.py      # real historical weather → data/weather_cache.csv (~10 min)
```

---

## Data releases

`data/` is git-ignored; bulky artifacts live as GitHub release assets.

| Release tag | Contents | Restored by |
|---|---|---|
| `latest` | the four `model_*.pkl` | `bootstrap_model_cache` |
| `data-cache` | `data_cache.tar.gz` — all `data/` files except pkls | `bootstrap_data_cache` |
| `prediction-archive` | `predictions.tar.gz` — the full `data/predictions/` store + `versions.json` | `bootstrap_predictions_cache` |

```bash
bash scripts/upload_data_release.sh          # after updating linescores/splits/weather
bash scripts/upload_predictions_release.sh   # after a retrain forks a new version
```

---

## API routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Dashboard — today's predictions + 7/90-day accuracy |
| `POST` | `/refresh` | Force re-run today's predictions |
| `POST` | `/retrain` | Retrain all models (forks a new version) |
| `GET` | `/archive` | Model-version list with rollup accuracy |
| `GET` | `/archive/<version>` | One version's per-date predictions vs actuals |
| `GET` | `/explain/<game_id>` | Stream a game's AI explanation via SSE |

---

## Key implementation details

- **`for_training=True`** — `build_game_features` skips `get_pitcher_splits()` and `get_game_lineup()` (unavailable for historical/pre-game) and reads linescores `cache_only`, keeping retraining fast.
- **Schedule cache + live backfill** — `get_season_schedule(year)` reads `schedule_<year>.csv` (memoized in `_season_schedule_memory`). The bundled cache can be snapshotted before a day's games finish; `refresh_schedule_date(year, date)` re-fetches a date's finals from the live API and persists them back to the CSV (thread-safe). `_actuals_for_date` memoizes settled dates so grading doesn't make redundant live calls.
- **Weather cache** — `data/weather_cache.csv` keyed by `(date, lat, lon)`; entries missing `humidity_pct` are re-fetched with the full schema.
- **AI explanations** — a background thread calls Ollama per game and writes `data/explanations/<date>/<game_id>.txt`; the browser streams them via SSE and serves cached text instantly on later loads.
- **Inning model override** — the inning classifier replaces the simulation's league-average inning weights with context-aware (park/elevation/starter/bullpen) probabilities.

---

## Tests

```bash
source .venv/bin/activate
pytest                      # all external APIs are mocked
```

Covers feature building, simulation, model train/predict, the versioned store (`test_predictions.py`), and app routes (`test_app.py`).

---

## Branch protection

`master` requires changes via pull request; `@jackleh` is a required reviewer (enforced via `.github/CODEOWNERS`).
