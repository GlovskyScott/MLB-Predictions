# CLAUDE.md

Guidance for AI agents (and humans) working in this repository. Read this before making changes. The user-facing project overview is in [README.md](README.md); this file is about *how the code works and how to change it safely*.

---

## TL;DR

- Flask app that predicts MLB games with XGBoost + Monte Carlo, then **freezes** each prediction as an immutable, **model-versioned** JSON artifact. Historical accuracy is graded by joining frozen predictions with real final scores — never by re-simulating.
- Source lives in `src/`. The hot subsystem is the **versioned prediction store** (`src/predictions.py` + `get_prediction`/`compare_date` in `src/app.py`). Understand it before touching prediction or archive code.
- Everything under `data/` is **git-ignored** and restored from GitHub releases at startup. Never commit `data/`.
- Tests mock all network I/O. Run `pytest` and keep it green (currently 63 tests).

---

## Environment & running (this machine has no Homebrew)

This box does **not** have Homebrew, pyenv, or conda, and system Python is 3.9 (too old). The working setup is:

- **Python**: [`uv`](https://docs.astral.sh/uv/) at `~/.local/bin/uv` manages CPython **3.12**. The venv is `.venv/`.
- **XGBoost needs `libomp`**: there is no `brew install libomp` here. A `libomp.dylib` (from a conda-forge `llvm-openmp` package) was placed on the dynamic-loader search path so `import xgboost` works. If you see `libxgboost.dylib could not be loaded`, that's the missing OpenMP runtime.
- **Ollama** is off-PATH: app at `~/Applications/Ollama.app`, CLI symlinked to `~/.local/bin/ollama`, model `llama3.1:8b` pulled. Start it with `ollama serve` (listens on `:11434`).

### Commands

```bash
source .venv/bin/activate                      # always, before python/pytest
pytest                                          # full suite (~16 s; hits live weather APIs in a few tests)
flask --app src.app:create_app run --port 5001  # run the app (5000 is taken by macOS AirPlay)
```

Run the Flask dev server **detached** so it survives the shell call, e.g. `nohup flask … &` — a backgrounded `flask … &` inside a wrapper that then exits gets reaped.

---

## Architecture

Request → prediction → render flow:

```
GET /  ──► run_daily_simulation(today)
             ├─ get_prediction(today)            # read frozen core, or simulate-once-and-freeze
             │    └─ _simulate_core_for_date()    # build_game_features → predict_game → simulate_game
             └─ _enrich_game()                    # add weather, lineup, logos, colors (NOT persisted)
        ──► compare_date(yesterday)               # join frozen prediction + live finals (no sim)
        ──► _aggregate_days(7|90)                 # rollup accuracy over stored dates

POST /retrain ──► train models → new pkl hash → new version → background re-sim (_backfill_results_cache)

GET /archive ──► _version_summary(v) for each registered version  (join store + actuals)
```

### Module map (`src/`)

| File | Responsibility | Key functions |
|---|---|---|
| `app.py` | Flask app, routes, prediction/grade orchestration, caches | `create_app`, `get_prediction`, `_simulate_core_for_date`, `run_daily_simulation`, `_enrich_game`, `compare_date`, `_actuals_for_date`, `_version_summary`, `_current_version`, `_backfill_results_cache` |
| `predictions.py` | The versioned store (pure, filesystem-only) | `model_version` (memoized on pkl signature), `read_versions`/`append_version` (locked), `next_build`, `load_prediction`/`save_prediction`, `list_dates`, `extract_core`, `PREDICTION_FIELDS` |
| `training.py` | Model training orchestration + model caches | `get_models`/`get_inning_model`, `build_training_df`/`build_inning_training_df`, `read_model_meta`/`write_model_meta`, `needs_retrain`, `reset_model_caches` (retrain calls this — don't poke the caches directly) |
| `explanations.py` | Ollama AI explanations (prompt/stream/pregenerate + cache) | `_stream_ollama`, `_build_explain_prompt`, `_pregenerate_explanations`, `_load_disk_explanations`, `_explanation_cache` |
| `colors.py` | Team-color helpers (pure) | `hex_to_rgb_str`, `bar_color` |
| `features.py` | Feature engineering | `build_game_features`, `build_inning_feature_row`, `FEATURE_COLUMNS`, `FEATURE_VERSION` |
| `fetcher.py` | All external data + disk caches | `bootstrap_{model,data,predictions}_cache`, `get_schedule`, `get_season_schedule`, `refresh_schedule_date`, `get_*_stats`, `get_weather_*`, splits/handedness/rest/recent-runs/bullpen helpers |
| `model.py` | XGBoost train/load/predict | `train_models`, `load_models`, `predict_game`, `train_inning_model`, `predict_inning_probs`, `models_exist` |
| `simulator.py` | Poisson Monte Carlo | `simulate_game(prediction, n_simulations, seed)` |
| `stadiums.py` | Park/elevation/roof, wind | `get_stadium`, `classify_wind` |
| `teams.py` | Colors/logos/abbrs | `get_team_meta` |

---

## The versioned prediction store (read this before touching predictions/archive)

**Storage layout** (under `data/`, git-ignored):

```
predictions/
  versions.json                 # registry: [{version, name, major, created_at, features, n_games, feature_version, release?}]
  <version>/<date>.json         # list of per-game prediction "cores" (PREDICTION_FIELDS only)
```

**Invariants — do not break these:**

1. **Immutability.** Once `predictions/<version>/<date>.json` exists it is served verbatim. `get_prediction` returns the stored file if present and only simulates on a miss. Grading (`compare_date`) **never** calls `simulate_game`.
2. **Version id** = first 12 hex of SHA-256 over the four pkls (`predictions.model_version`). It's the storage key. Any retrain that changes pkl bytes → new id.
3. **The frozen core is sim-only.** `PREDICTION_FIELDS` holds win%, median/modal scores, score distribution, inning pcts, n_simulations, and game identity. **No actual scores** (joined on read) and **no presentation fields** (weather/logos/colors/lineup are re-derived at render in `_enrich_game`).
4. **Determinism.** Generation always uses `N_SIMULATIONS = 1000` and `seed = game_id % 100000`, in both the live and backfill paths, so what's shown "today" equals what later appears under "yesterday." Don't introduce an unseeded or differently-sized sim on either path.
5. **Naming rule** `v<major>.<build>`: `major` = `FEATURE_VERSION` for trained models (legacy = explicit `--major`); `build` = `predictions.next_build(data_dir, major)` (count of existing versions with that major). Names are assigned once and never recomputed.

**Accuracy/grading:** `compare_date(date, version=None)` loads the frozen core (current version generates-on-miss; archived versions are read-only), gets finals via `_actuals_for_date`, and computes `winner_correct` from `home_win_pct > 50` and score error from the median scores. `_version_summary` aggregates across `list_dates`.

**Retrain → fork:** `/retrain` trains, then `_current_version()` registers the new pkl hash (new `name`), leaving old version folders intact, then a daemon thread re-sims today + 90 days into the new version's folder.

---

## Conventions & gotchas

- **Never commit `data/`.** It's git-ignored and lives in releases. Prediction data goes in the `prediction-archive` release via `scripts/upload_predictions_release.sh`.
- **Schedule cache can be incomplete.** The bundled `schedule_<year>.csv` may be snapshotted before games finish. `refresh_schedule_date` fills finals from the live API **and persists them back to the CSV** (guarded by `_schedule_cache_lock`; the backfill runs 6 workers). `_actuals_for_date` memoizes settled dates (older than `_REFRESH_WINDOW_DAYS`) so `/archive` doesn't make a live call per date per version.
- **`/archive` is O(versions × dates).** It re-grades every version's stored dates; first load after a restart can take ~20 s (fills the actuals memo + persists missing finals), then ~4 s. If you make it slower, memoize — don't drop the refresh (that under-counts graded games).
- **Flask dev server caches templates** (no debug mode). After editing a template, **restart the server** — a reload of the page is not enough.
- **The dev server is single-threaded.** Overlapping curl/browser probes queue up and look like hangs. Probe once, with a generous timeout.
- **Browsers cache `/archive` aggressively.** When verifying a template change in a browser, append a unique query string (`?v=<n>`) to bust the cache.
- **Training is deterministic** (`random_state=42`, fixed data → identical pkls → same version id). To force a genuinely new version you must change the training data or feature set. The released `latest` pkls differ from a local retrain because they were trained in a different environment — that's why a first local retrain forks a new version.
- **`_DATA_DIR` is patched in tests.** Anything touching the store reads `app._DATA_DIR` / `fetcher._DATA_DIR`; tests `mocker.patch.object(app, '_DATA_DIR', tmp_path)`. Keep new store I/O going through these module globals so it stays test-isolatable.
- **Don't let `/retrain` or the backfill run unmocked in tests.** They spawn real training + a 90-day live backfill thread. `test_retrain_redirects` mocks `_get_models`, `_get_inning_model`, `_current_version`, and `_backfill_results_cache` for this reason.

---

## Common tasks

- **Add a game feature:** edit `FEATURE_COLUMNS` and `build_game_features` in `features.py`, **bump `FEATURE_VERSION`**, retrain. The version mismatch triggers an auto-retrain; the new model registers as `v<new>.0`.
- **Retrain:** `POST /retrain` (or click ⚙ Retrain). After it forks a version, publish with `bash scripts/upload_predictions_release.sh`.
- **Reconstruct a legacy model into the archive:** see the README "Legacy models" section — generate in a git worktree at the model's commit with `scripts/import_legacy_predictions.py`, then `scripts/register_legacy_version.py --major <N>`.
- **Publish releases:** `scripts/upload_data_release.sh` (data cache), `scripts/upload_predictions_release.sh` (prediction archive). Both use `gh`.

---

## Testing

- `pytest` from the activated venv. External APIs (statsapi, pybaseball, Open-Meteo) are mocked; a few weather tests do hit the network briefly.
- New store/versioning logic → add to `tests/test_predictions.py`; new route/orchestration logic → `tests/test_app.py`. Patch `_DATA_DIR` to `tmp_path` and seed the store with `predictions.save_prediction` / `append_version`.
- When adding a path that could call `simulate_game` on read, add a test asserting `simulate_game.assert_not_called()` — re-simulation on the grading path is a regression.

---

## Git / PRs

- `master` is protected: PRs only, `@jackleh` required reviewer (`.github/CODEOWNERS`). Branch for changes; don't push to `master`.
- End commit messages with the `Co-Authored-By: Claude` trailer.
- Design docs for the versioned store live in `docs/superpowers/` (spec + plan).
