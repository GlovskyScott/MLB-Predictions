# Model-Versioned Prediction Store — Design

**Date:** 2026-06-27
**Status:** Approved (design); implementation pending

## Problem

Predictions for past dates are not durable artifacts. Today's predictions live
only in the in-memory `_simulation_cache` and are never written as predictions.
For past dates, `run_results_comparison()` **re-simulates from scratch** using
the *current* models and a *different* simulation config than the live view
(live: 1000 sims, unseeded; comparison: 200 sims, seeded). Consequences:

1. The "prediction" shown for a past day is a hindsight re-simulation, not what
   was served that day.
2. Retraining silently changes every not-yet-frozen historical date's prediction.
3. The frozen `results_cache/<date>.json` captures the *first re-simulation*
   (possibly days late, possibly a newer model), not the original prediction.
4. The past-date predicted score/win% never matches the live "today" view for the
   same game, because the two paths use different sim counts and seeding.

## Goal

Predictions generated for a date X are **the** prediction for X and are reused
forever after — never re-simulated — **except** when the model changes. A model
change (any retrain) archives the previous model's predictions on a separate page
and re-simulates all dates once under the new model for the main page.

## Behavior rules

- **Never re-simulate within a model version.** Once a date's prediction exists
  under the current version, it is served verbatim.
- **Generate-on-miss.** If a date has no stored prediction under the current
  version, simulate it once with the current models, persist it, then serve it.
  (Occurs the day a date is new, and during the one-time backfill of the 90-day
  window after a new version appears.)
- **Any retrain is a new version.** The ⚙ Retrain button and the auto-retrain on
  `feature_version` mismatch both produce a new version. Old version predictions
  are retained (archived), not overwritten.
- **Determinism.** Generation uses a fixed per-`game_id` seed and a single
  `n_simulations` constant for both the live "today" path and the stored artifact,
  so what is shown live is byte-for-byte what later appears under "yesterday".

## Data model & storage

```
data/predictions/
  versions.json               # registry (ordered, newest last)
  <version>/<date>.json       # frozen per-game prediction core for that date
```

- `version` = first 12 hex chars of `sha256` over the concatenated bytes of the
  model `.pkl` files (`model_win.pkl`, `model_runs_home.pkl`,
  `model_runs_away.pkl`, `model_inning.pkl`), in fixed order. Deterministic; no
  clock dependence; any retrain changes it.
- `versions.json` entries:
  `{version, created_at (ISO8601 UTC), feature_version, n_games, training_years}`.
  Appended when a version is first observed. The **current** version is whichever
  matches the live pkl hash (it may already be present from a prior run).
- `<date>.json` = list of per-game prediction-core dicts. Stored fields:
  `game_id, game_date, home_id, away_id, home_name, away_name, venue_id,
  venue_name, home_win_pct, away_win_pct, median_home_score, median_away_score,
  modal_home_score, modal_away_score, predicted_score, score_distribution,
  home_innings_scoring_pct, away_innings_scoring_pct, n_simulations`.
  **No actual scores** are stored — actuals are joined on read.

## Read & generation flow

- On startup, compute `current_version` from the live pkl hash; ensure it is in
  `versions.json`.
- `get_prediction(date)`:
  - Read `predictions/<current_version>/<date>.json`. If present → return it.
  - If missing → simulate each game (current models, fixed seed + constant
    `n_simulations`), write the file atomically, return it.
- **Accuracy/comparison** (`yesterday`, 90-day table) = join stored predictions
  with actual finals from the schedule cache (using `refresh_schedule_date()` for
  recent dates that aren't yet final), then tally winner-correct and score error.
  No simulation occurs on this path. This replaces `run_results_comparison()`'s
  re-simulation.
- The today/live view (`run_daily_simulation`) calls `get_prediction(today)` for
  the prediction core (which generates-and-persists on first load of the day), and
  then enriches it with presentation/context fields needed for the dashboard —
  logos, colors, abbreviations, lineup, weather strip, and the feature vector used
  by the AI-explanation prompt. These enrichment fields are **re-derived at render
  time and are NOT part of the frozen artifact**; the frozen artifact is purely the
  simulation core. Because the core is read from the store (not re-simulated), the
  win%/score shown today are identical to what later appears under "yesterday",
  while weather/features are always recomputed fresh for display.
- Note on weather/features during generation: producing the prediction core
  requires the feature vector and weather as inputs to the models/simulator, so a
  cache miss computes them as part of generation. The point is only that they are
  not *persisted* in the artifact — they are recomputed on read.

## Retrain & archive

- **Retrain** (`/retrain` or auto on feature_version mismatch):
  1. Retrain models → new pkls → new `current_version` hash.
  2. Append the new version to `versions.json`.
  3. Leave all existing `predictions/<old_version>/` folders intact (the archive).
  4. Background job re-simulates today + the trailing 90-day window into
     `predictions/<new_version>/` (the existing backfill mechanism, now writing to
     the versioned store).
  5. Main page switches to the new version; frozen again until the next retrain.
- **`/archive`**: table of past versions (fingerprint, created_at, n_games,
  feature_version) each with rollup winner-accuracy + avg score error, computed by
  joining that version's stored predictions with actual finals.
- **`/archive/<version>`**: per-date predictions-vs-actuals for that version,
  reusing the existing card/table rendering.

## Replaced / removed

- `results_cache/` re-simulation is **retired**. Accuracy is computed on read from
  the versioned store. Existing `results_cache/*.json` files are ignored and may be
  deleted; they are derived data, not source of truth.

## Out of scope (YAGNI)

- No pruning/retention policy for old versions (kept indefinitely; cheap JSON).
- No diffing UI between two specific versions beyond the per-version drill-down.
- No change to model training, feature engineering, or the simulator math.

## Testing

- Version hashing: stable for unchanged pkls; changes when pkl bytes change.
- Registry: appends a new version once; idempotent on repeat startup.
- `get_prediction`: returns stored file when present; generates + persists exactly
  once when missing; never simulates when present.
- Determinism: same game_id + same models → identical stored prediction across
  runs; live "today" artifact equals the one read later as "yesterday".
- Accuracy join: stored prediction + actual final → correct winner/err tally;
  no simulation invoked (assert simulate_game not called on the read path).
- Retrain: new version folder created, old folder untouched, registry appended.
- Archive routes: list versions; drill-down renders a version's dates.
