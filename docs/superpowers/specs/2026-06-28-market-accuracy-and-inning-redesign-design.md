# Market-by-market accuracy + 3-class inning breakdown — Design

**Date:** 2026-06-28
**Status:** Approved (brainstorm)

## Summary

Two related improvements to how the app judges and presents the model, delivered
as one spec in two phases:

1. **Phase 1 — Market-by-market accuracy.** Grade the model on three betting
   markets (Moneyline, Total, Spread/run-line) instead of only "winner correct",
   and surface all three hit-rates in the yesterday summary, the game page, and
   the `/archive` rollup.
2. **Phase 2 — Inning breakdown redesign.** Replace the per-inning *average runs*
   grid with a 3-class distribution — P(0) / P(1) / P(2+) runs per inning per
   team, plus a combined "P(any team scores)" row — driven by retargeting the
   inning classifier from binary to 3-class.

Phase 1 is fully self-contained (no model change, no version fork). Phase 2
retargets `model_inning.pkl`, which **is** one of the four version-hash pkls, so
it forks a new model version and triggers the normal re-sim — it is sequenced
after Phase 1.

---

## Phase 1 — ML / Total / Spread accuracy

### Background

`compare_date` currently grades only `winner_correct` (`home_win_pct > 50` vs
actual winner) plus median-score MAE. The app already derives both model-implied
and de-vigged **market** lines for ML/Total/Run-line and computes edges
(`src/app.py` `_market_lines`, `_market_block`, fed by
`fetcher.get_market_odds(date)`), but those market lines are **not persisted**.

### Key insight: grade from the stored distribution

The simulator draws home and away run totals *independently* (`simulator.py`,
separate `_runs_draw` calls), and the frozen core already stores both marginal
score histograms in `score_distribution`. Therefore every market pick can be
recovered at grade time by convolving those two marginals — **no change to
`PREDICTION_FIELDS`, no re-simulation, immutability preserved.**

Minor approximation: extra-innings run-adding slightly couples the two totals in
~9% of games (the tied ones). For Spread (margin ≥ 2, never tied) this is
irrelevant; for Total it is negligible. Documented, accepted.

### Pick definitions (graded against finals)

- **ML.** Pick the side with model win% > 50; correct if it wins. (This is
  today's `winner_correct`, relabeled "ML". No push.)
- **Spread (run-line ±1.5).** From the marginal histograms compute
  `P(home wins by ≥2)` and `P(away wins by ≥2)`. Pick the favorite at −1.5 if its
  cover probability > 50%, else the underdog at +1.5. Correct if the pick covers
  the actual margin. Self-contained; no push on ±1.5.
- **Total (O/U).** Pick Over if `P(home+away > line) > 50%` evaluated at the
  **captured market total line**, else Under. Correct/push vs the actual total.
  Requires the stored line; when absent the game's Total is graded as N/A.

### New infrastructure: market-line snapshot store

`data/market_lines/<date>.json` → `{ "<game_id>": {"total_line": float,
"captured_at": iso8601} }`.

- **Version-independent.** The book line is identical across model versions, so
  it lives outside `data/predictions/<version>/`.
- **Write-once.** Captured the first time a date's predictions are generated
  live (in the `run_daily_simulation` / generation path) via `get_market_odds`,
  and never overwritten — it preserves the line the pick was made against.
- **"From now on".** Backfilled/historical dates without a captured line grade
  Total as N/A; ML and Spread are always gradeable from the frozen core.
- New module-level `_DATA_DIR`-relative helpers (test-isolatable like the rest of
  the store): `save_market_lines`, `load_market_lines`.

### Grading + aggregation changes

- A small pure grading module (e.g. `src/grading.py`) holds the convolution and
  pick logic: `cover_probs(core)`, `total_over_prob(core, line)`, and
  `grade_markets(core, actual_home, actual_away, total_line) -> {ml, spread,
  total}` where each value is `True` / `False` / `None` (N/A) / `"push"`.
- `compare_date` attaches `ml_correct` / `spread_correct` / `total_correct` per
  game and returns aggregate `ml_accuracy` / `spread_accuracy` / `total_accuracy`
  (each over its graded subset; pushes excluded from the denominator).
  `winner_accuracy` stays as an alias of `ml_accuracy` for back-compat.
- `_version_summary` aggregates the three across all stored dates.

### UI

The yesterday summary, game page, and `/archive` show three labelled hit-rates
(**ML / Total / Spread**) with their graded counts, replacing the single winner
%. Total shows "N/A" (with a small "no line" note) for ungraded dates.

---

## Phase 2 — Inning breakdown redesign (3-class model)

### Model retarget

`build_inning_training_df` already reads exact `inn_runs`; relabel into three
classes — `0` (0 runs), `1` (exactly 1), `2` (2+). `train_inning_model` switches
to multiclass (`objective='multi:softprob'`, `num_class=3`), keeping the existing
`INNING_FEATURE_COLUMNS` and `random_state=42`. `predict_inning_probs` returns
P(0)/P(1)/P(2+) per inning for each team.

Because `model_inning.pkl` is hashed into the version id, this **forks a new
model version** and triggers the standard re-sim; the inning calibrator is
rebuilt.

### Calibration

`build_calibrator`'s inning branch becomes 3-class: per-class temperature scaling
fit on walk-forward out-of-sample pairs, renormalized to sum to 1. The binary map
today is near-identity (slope ≈ 0.97); the 3-class map is expected to be similar
and is applied for consistency/robustness. The win% calibrator is unchanged.

### Core fields (`PREDICTION_FIELDS`)

Remove `home_innings_scoring_pct`, `away_innings_scoring_pct`, `home_innings`,
`away_innings`. Add:

- `home_innings_dist`, `away_innings_dist`: 9×3 arrays of [P0, P1, P2+] percents.
- `combined_innings_dist`: 9×3, **P(any team scores)** per inning bucketed by
  *total* runs that inning — 0 / exactly 1 / 2+ — derived by convolving the two
  teams' 3-class distributions under independence:
  - `P(0) = Ph0·Pa0`
  - `P(1) = Ph0·Pa1 + Ph1·Pa0`
  - `P(2+) = 1 − P(0) − P(1)`

Old archived cores keep their old fields and render the old grid; the template
branches on field presence so both render correctly.

### UI

Replace the avg-runs grid with three rows — Away / Home / Combined ("Both") —
each inning cell a compact 3-segment stacked bar (0 / 1 / 2+) with the three
percentages on hover/label:

```
Inning            1     2     3   ...   9
Away   0 / 1 / 2+ [▇▃▁] [▇▃▂]  ...
Home   0 / 1 / 2+ [▇▂▂] [▇▃▁]  ...
Both   0 / 1 / 2+ [▆▃▃] [▅▃▄]  ...   ← P(any team scores)
```

## Out of scope (YAGNI)

- No inning-level *accuracy* grading (Phase 2 is a display redesign, not a graded
  market).
- No ROI / bankroll / edge-tracking beyond the three Phase-1 hit-rates.
- No change to the win% calibrator or the run regressors.

## Testing

- Phase 1 grading helpers: unit-test pick logic against hand-built cores
  (known histograms) — Over/Under/push, favorite covers / fails, ML. Assert
  `simulate_game` is never called on the grading path. Market-line store:
  save/load round-trip + write-once.
- Phase 2: relabel correctness (0/1/2+), multiclass `predict_inning_probs` shape
  and per-cell sum-to-1, combined convolution math, template renders both old and
  new cores.
- Patch `_DATA_DIR`/`fetcher._DATA_DIR` to `tmp_path` throughout; keep `pytest`
  green.
