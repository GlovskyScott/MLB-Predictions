# Versioned Prediction Store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make predictions immutable artifacts keyed by model version — served verbatim within a version, re-simulated only when the model changes, with old versions archived on a separate page.

**Architecture:** A new `src/predictions.py` owns the on-disk versioned store (hashing, registry, load/save). `src/app.py` gains `get_prediction(date)` (read-or-generate), a deterministic core-simulation helper shared by the live and backfill paths, a join-based `compare_date()` that replaces the re-simulating `run_results_comparison()`, and `/archive` routes. Retrain forks a new version folder and re-sims in the background.

**Tech Stack:** Flask 3, XGBoost, NumPy Monte Carlo, pandas, pytest/pytest-mock. Python 3.12 venv at `.venv`.

## Global Constraints

- Run tests with `.venv` active: `source .venv/bin/activate && python -m pytest`.
- `data/` is gitignored — never commit prediction files, only code/tests/docs.
- Determinism: prediction generation uses seed `game_id % 100000` and `N_SIMULATIONS = 1000` everywhere (live + backfill).
- Version id = first 12 hex of `sha256` over the concatenated bytes of `model_win.pkl, model_runs_home.pkl, model_runs_away.pkl, model_inning.pkl` in that fixed order.
- Frozen artifact stores the simulation core only (no actuals, no presentation fields). Actuals joined on read from the schedule cache.

---

### Task 1: `src/predictions.py` — versioned store primitives

**Files:**
- Create: `src/predictions.py`
- Test: `tests/test_predictions.py`

**Interfaces:**
- Produces:
  - `PREDICTION_FIELDS: tuple[str, ...]` — core fields persisted per game.
  - `model_version(data_dir: Path) -> str | None` — 12-hex hash of the 4 pkls; `None` if any missing.
  - `read_versions(data_dir: Path) -> list[dict]` — registry list (possibly empty).
  - `append_version(data_dir: Path, entry: dict) -> None` — add entry if `entry['version']` not already present (idempotent).
  - `load_prediction(data_dir, version, date) -> list[dict] | None`
  - `save_prediction(data_dir, version, date, games: list[dict]) -> None` — atomic write to `predictions/<version>/<date>.json`.
  - `extract_core(game: dict) -> dict` — slim a full sim result to `PREDICTION_FIELDS`.

- [ ] **Step 1: Write failing tests** in `tests/test_predictions.py`:

```python
import json
from pathlib import Path
import pytest
from src import predictions as P

def _write_pkls(d: Path, payloads):
    d.mkdir(parents=True, exist_ok=True)
    for name, b in payloads.items():
        (d / name).write_bytes(b)

PKLS = ['model_win.pkl', 'model_runs_home.pkl', 'model_runs_away.pkl', 'model_inning.pkl']

def test_model_version_stable_and_changes(tmp_path):
    _write_pkls(tmp_path, {n: b'a' for n in PKLS})
    v1 = P.model_version(tmp_path)
    assert isinstance(v1, str) and len(v1) == 12
    assert P.model_version(tmp_path) == v1            # stable
    (tmp_path / 'model_win.pkl').write_bytes(b'different')
    assert P.model_version(tmp_path) != v1            # changes on pkl change

def test_model_version_none_when_missing(tmp_path):
    assert P.model_version(tmp_path) is None

def test_append_version_idempotent(tmp_path):
    P.append_version(tmp_path, {'version': 'abc', 'n_games': 1})
    P.append_version(tmp_path, {'version': 'abc', 'n_games': 1})
    P.append_version(tmp_path, {'version': 'def', 'n_games': 2})
    vs = P.read_versions(tmp_path)
    assert [v['version'] for v in vs] == ['abc', 'def']

def test_save_load_roundtrip(tmp_path):
    games = [{'game_id': 1, 'home_win_pct': 60.0, 'extra': 'dropped'}]
    P.save_prediction(tmp_path, 'v1', '2026-06-26', games)
    loaded = P.load_prediction(tmp_path, 'v1', '2026-06-26')
    assert loaded[0]['game_id'] == 1 and loaded[0]['home_win_pct'] == 60.0

def test_load_missing_returns_none(tmp_path):
    assert P.load_prediction(tmp_path, 'v1', '2026-06-26') is None

def test_extract_core_keeps_only_fields():
    core = P.extract_core({'game_id': 1, 'home_win_pct': 60.0, 'home_logo': 'x', 'weather': {}})
    assert 'home_logo' not in core and 'weather' not in core
    assert core['game_id'] == 1
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_predictions.py -q` → ImportError/fail.

- [ ] **Step 3: Implement `src/predictions.py`:**

```python
import hashlib
import json
from pathlib import Path

_PKL_NAMES = ('model_win.pkl', 'model_runs_home.pkl', 'model_runs_away.pkl', 'model_inning.pkl')

PREDICTION_FIELDS = (
    'game_id', 'game_date', 'home_id', 'away_id', 'home_name', 'away_name',
    'venue_id', 'venue_name', 'home_win_pct', 'away_win_pct',
    'median_home_score', 'median_away_score', 'modal_home_score', 'modal_away_score',
    'predicted_score', 'score_distribution',
    'home_innings_scoring_pct', 'away_innings_scoring_pct',
    'home_innings', 'away_innings', 'n_simulations',
)

def _pred_dir(data_dir: Path) -> Path:
    return Path(data_dir) / 'predictions'

def model_version(data_dir: Path):
    h = hashlib.sha256()
    for name in _PKL_NAMES:
        p = Path(data_dir) / name
        if not p.exists():
            return None
        h.update(p.read_bytes())
    return h.hexdigest()[:12]

def _versions_file(data_dir: Path) -> Path:
    return _pred_dir(data_dir) / 'versions.json'

def read_versions(data_dir: Path) -> list:
    f = _versions_file(data_dir)
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception:
        return []

def append_version(data_dir: Path, entry: dict) -> None:
    versions = read_versions(data_dir)
    if any(v.get('version') == entry.get('version') for v in versions):
        return
    versions.append(entry)
    f = _versions_file(data_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(versions, indent=2))

def load_prediction(data_dir: Path, version: str, date: str):
    f = _pred_dir(data_dir) / version / f'{date}.json'
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None

def save_prediction(data_dir: Path, version: str, date: str, games: list) -> None:
    d = _pred_dir(data_dir) / version
    d.mkdir(parents=True, exist_ok=True)
    core = [extract_core(g) for g in games]
    tmp = d / f'{date}.json.tmp'
    tmp.write_text(json.dumps(core, default=str))
    tmp.replace(d / f'{date}.json')

def extract_core(game: dict) -> dict:
    return {k: game[k] for k in PREDICTION_FIELDS if k in game}
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_predictions.py -q`.
- [ ] **Step 5: Commit** — `git add src/predictions.py tests/test_predictions.py && git commit -m "feat: versioned prediction store primitives"`.

---

### Task 2: `get_prediction()` + deterministic core sim in `app.py`

**Files:**
- Modify: `src/app.py` (add helpers near `run_daily_simulation`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes Task 1: `predictions.model_version/append_version/load_prediction/save_prediction`.
- Produces:
  - `N_SIMULATIONS = 1000` constant.
  - `_current_version() -> str | None` — hash of live pkls; appends registry entry on first sight.
  - `_simulate_core_for_date(date, models, inning_model) -> list[dict]` — builds features/weather, predicts, simulates with `seed=game_id % 100000, n_simulations=N_SIMULATIONS`, returns core dicts (via `extract_core`).
  - `get_prediction(date) -> list[dict]` — returns stored core for current version, else generates+persists then returns. Never simulates when a stored file exists.

- [ ] **Step 1: Failing test** — `get_prediction` does NOT call `simulate_game` when a stored file exists:

```python
def test_get_prediction_uses_store_no_resim(tmp_path, mocker):
    import src.app as app
    import src.predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch('src.predictions.Path', Path)  # no-op safety
    # seed a stored prediction for the current version
    mocker.patch.object(app, '_current_version', return_value='vTEST')
    P.save_prediction(tmp_path, 'vTEST', '2026-06-26', [{'game_id': 1, 'home_win_pct': 55.0}])
    sim = mocker.patch('src.app.simulate_game')
    out = app.get_prediction('2026-06-26')
    assert out[0]['game_id'] == 1
    sim.assert_not_called()
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** in `src/app.py`:
  - Add `from src import predictions as _pred` and `N_SIMULATIONS = 1000`.
  - `_current_version()`: `v = _pred.model_version(_DATA_DIR)`; if `v` and not in registry, `append_version` with `{version, created_at: utcnow iso, **_read_model_meta()}`; return `v`.
  - `_simulate_core_for_date(date, models, inning_model)`: same body as the per-game loop in `run_daily_simulation` (features, weather, predict, `simulate_game(prediction, n_simulations=N_SIMULATIONS, seed=game['game_id'] % 100000)`, inning override) but append `_pred.extract_core({**game, **sim})`. Reuse for both paths.
  - `get_prediction(date)`: `v = _current_version()`; `stored = _pred.load_prediction(_DATA_DIR, v, date)`; if stored is not None return it; else `core = _simulate_core_for_date(...)`, `_pred.save_prediction(_DATA_DIR, v, date, core)`, return core.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — "feat: get_prediction read-or-generate with deterministic core".

---

### Task 3: Refactor `run_daily_simulation` to enrich stored core

**Files:** Modify `src/app.py:193-261`; Test `tests/test_app.py`.

**Interfaces:**
- Consumes Task 2: `get_prediction`.
- Produces: `run_daily_simulation(sim_date)` returns the same enriched dicts as before (logos/colors/weather/lineup/features) but the prediction core now comes from `get_prediction` (so today == future "yesterday").

- [ ] **Step 1: Failing test** — calling the index route twice for the same day reads from the store the second time (assert `simulate_game` called only on first). Mock `get_schedule`, `_get_models`, `_get_inning_model`, `simulate_game` returning `MOCK` core, and team/stadium helpers.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — rewrite `run_daily_simulation` to: `cores = get_prediction(sim_date)`; build a `{game_id: core}` map; fetch `games = get_schedule(sim_date)`; for each game enrich `{**game, **core, weather, lineup, features, logos/colors/...}`. Keep weather/features/lineup computation (needed for cards + explanations). Games without a core (edge) fall back to on-the-fly enrich with empty core.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — "refactor: daily view renders from versioned store".

---

### Task 4: Join-based `compare_date()` replaces re-simulation

**Files:** Modify `src/app.py` (`run_results_comparison` → `compare_date`; `_get_results_for_date`, `_backfill_one`, `_aggregate_days`); Test `tests/test_app.py`.

**Interfaces:**
- Consumes Task 2 (`get_prediction`/store) + `refresh_schedule_date`.
- Produces:
  - `compare_date(date, version=None) -> dict` with keys `{result_date, games, n_completed, winner_accuracy, avg_score_err}`. `version=None` → current version. Loads stored core (or, for current version, `get_prediction` to generate-on-miss), joins actual finals from the schedule cache (calling `refresh_schedule_date` when the date isn't all-final), computes winner_correct + score errs. **Never calls `simulate_game`.**
  - `_get_results_for_date(date)` → `compare_date(date)` (drop results_cache read/write).
  - `_backfill_one(date)` → ensure `get_prediction(date)` exists (generate-on-miss), no-op if already stored.
  - `_aggregate_days(n)` → iterate dates whose stored prediction exists for current version, call `compare_date`, aggregate.

- [ ] **Step 1: Failing test** — `compare_date` joins a stored prediction with an actual final and reports `winner_correct`, without calling `simulate_game`:

```python
def test_compare_date_joins_without_resim(tmp_path, mocker):
    import src.app as app
    from src import predictions as P
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    P.save_prediction(tmp_path, 'v1', '2026-06-26',
        [{'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
          'home_name': 'NYY', 'away_name': 'BOS', 'home_win_pct': 60.0, 'away_win_pct': 40.0,
          'median_home_score': 5.0, 'median_away_score': 3.0, 'predicted_score': '5-3'}])
    mocker.patch('src.app.get_season_schedule', return_value=[
        {'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
         'home_score': 6, 'away_score': 2, 'home_id': 147, 'away_id': 111}])
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    sim = mocker.patch('src.app.simulate_game')
    mocker.patch('src.app.get_team_meta', return_value={'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})
    data = app.compare_date('2026-06-26')
    assert data['n_completed'] == 1
    assert data['games'][0]['winner_correct'] is True   # predicted home win, home won
    sim.assert_not_called()
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `compare_date` (join logic from old `run_results_comparison` lines 274-337 but sourcing `sim`-like fields from the stored core instead of simulating; `predicted_home_won = core['home_win_pct'] > 50.0`; score errs from `median_*`). Rewire `_get_results_for_date/_backfill_one/_aggregate_days`. Delete `run_results_comparison`, `_RESULTS_CACHE_DIR` usage.
- [ ] **Step 4: Run, verify pass** — full `python -m pytest -q` green.
- [ ] **Step 5: Commit** — "feat: accuracy via join on versioned store, retire results_cache".

---

### Task 5: Retrain forks a new version + background re-sim

**Files:** Modify `src/app.py` (`/retrain` route, `_backfill_results_cache`); Test `tests/test_app.py`.

**Interfaces:**
- Consumes Task 2/4.
- Produces: `/retrain` retrains, computes the new `_current_version()` (registry appended), leaves old version folders intact, and starts a daemon thread running `_backfill_results_cache(90)` which now generates predictions under the new version (today + trailing 90 days).

- [ ] **Step 1: Failing test** — after retrain, `read_versions` contains ≥1 entry and the old version's prediction folder still exists. Mock training to write new pkl bytes.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — in `retrain()`: after `_get_models(force_retrain=True)` + inning, call `_current_version()` (appends new), clear `_simulation_cache/_results_cache`, start `threading.Thread(target=_backfill_results_cache, args=(90,), daemon=True)`. `_backfill_results_cache` already calls `_backfill_one` → `get_prediction` (generate-on-miss) under current version.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — "feat: retrain forks new prediction version + re-sim".

---

### Task 6: Archive routes + template

**Files:** Create `templates/archive.html`; Modify `src/app.py` (routes), `templates/index.html` (nav link); Test `tests/test_app.py`.

**Interfaces:**
- Consumes Task 1/4: `read_versions`, `compare_date(date, version)`.
- Produces:
  - `GET /archive` — renders a table of `read_versions()` each with rollup winner-accuracy + avg err (aggregate `compare_date(d, version)` across that version's stored dates).
  - `GET /archive/<version>` — per-date predictions-vs-actuals for `version`, reusing card/table partials.

- [ ] **Step 1: Failing test** — `GET /archive` returns 200 and lists a known version; `GET /archive/<v>` returns 200. Seed a version + one stored date in `tmp_path`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** routes + minimal `archive.html` (extends the dark theme; table of versions linking to `/archive/<version>`; per-version page lists dates with actual vs predicted). Add a header link `Archive` in `index.html`.
- [ ] **Step 4: Run, verify pass** — full suite green; manual `curl /archive`.
- [ ] **Step 5: Commit** — "feat: model-version archive pages".

---

## Self-Review notes

- Spec coverage: storage (T1), never-resim read/generate (T2), live==yesterday determinism (T2/T3), join accuracy + retire results_cache (T4), retrain archive+resim (T5), archive pages (T6). All covered.
- The slim artifact omits `home_pitcher/away_pitcher` strings — those are presentation; cards read them from the live `game` dict at render, comparison table doesn't need them. OK.
- `compare_date` for a non-current version must read that version's store directly (no generate-on-miss — old versions are frozen); only the current version generates on miss.
