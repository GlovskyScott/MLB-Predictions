# Modeling notes — accuracy, honesty, and roadmap

This documents the predictive methodology: how well it really works, the
improvements made on `jack/prediction-methodology`, and the concrete next steps.

## How good is it, honestly?

The dashboard's "90-day accuracy" is **in-sample** — the models train on every
completed 2024–2026 game and are then graded on those same games. That inflates
the number. `scripts/backtest.py` runs proper temporal validation:

| Split | Winner accuracy | What it measures |
|---|---|---|
| In-sample (the dashboard) | **74.2%** | inflated — graded on training games |
| Temporal holdout (test 2026) | **60.0%** | model never saw 2026 outcomes |
| Walk-forward (monthly) | **58.3%** | train only on games before each month |
| Walk-forward, point-in-time team features | **48.5%** | zero season-aggregate stats |
| Naive baseline (home team always) | 53.0% | — |

**Reading:** the model's *real* out-of-sample skill is **~58–60%** — genuinely
good for MLB (the best models / closing lines sit ~57–60%; this clears the 53%
baseline by 5–7 points). The 74% was never real; it's training-on-test-data.

The point-in-time row (48.5%, below baseline) is the most instructive: with
**only** leak-free team records (runs scored/allowed/win% to date), the model
can't beat "always pick home." So the genuine edge lives in the
**season-aggregate pitcher and offense features** — which carry a within-season
look-ahead. That tension defines the roadmap below.

Run it yourself:

```bash
python scripts/backtest.py            # uses the cached feature matrix
python scripts/backtest.py --rebuild  # rebuild features first
```

## What changed on this branch

- **Honest backtest** (`scripts/backtest.py`) — temporal holdout, walk-forward,
  and a fully leak-free point-in-time floor, side by side with the in-sample
  number. This is the headline fix: you can now measure real skill.
- **Negative-binomial run model** (`simulator.py`) — per-team runs are drawn
  from a negative binomial (variance ≈ 2× mean, matching MLB's overdispersion)
  instead of a Poisson (variance == mean), which had too thin a tail and
  understated blowouts. `overdispersion=1` reverts to Poisson.
- **Real FIP/xFIP** (`fetcher.py`) — were set equal to ERA (pure collinearity).
  Now computed from the component rates: `FIP = (13·HR9 + 3·BB9 − 2·K9)/9 + c`,
  with xFIP regressing the HR rate to league average. Adopt with a pitching-cache
  refresh + retrain.

## Roadmap (in priority order)

### 1. Point-in-time *pitcher* features (the real leakage fix)

The remaining look-ahead is on the feature side: `get_pitching_stats(year)` and
`get_team_batting_stats(year)` return **full-season** aggregates, so an April
game is featurized with a pitcher's whole-season ERA. The point-in-time backtest
shows team records alone aren't enough — we need **as-of-date pitcher** ERA/FIP/
K-BB and **as-of-date team** wOBA/OPS.

Approach: cache per-game player logs (MLB Stats API `playerStats` / boxscores,
or pybaseball game logs) once, then compute cumulative/rolling stats up to each
game date in a single forward pass (same pattern as `point_in_time_matrix`).
Then the production feature builder takes an `as_of_date` and the walk-forward
becomes fully leak-free. Expect the honest number to settle slightly below 58%,
but it will finally be a number you can trust and optimize against.

### 2. Negative-binomial dispersion fit + recalibration

`overdispersion=2.0` is a reasonable constant; fit it (and check win-prob
calibration with a reliability curve) against historical score distributions per
run-environment. Adopting NB in production changes the prediction-of-record, so
regenerate predictions (or bump the model version) so a version stays internally
consistent.

### 3. Batter-vs-pitcher / matchup features (deferred — data-heavy)

The biggest untapped signal in baseball is matchup-level, but it needs data the
app doesn't currently cache:

- **Lineup-weighted handedness/quality** — the current handedness feature is a
  team-level OPS-vs-hand; weight it by the *actual posted lineup* (already
  fetched for display in `get_game_lineup`) and each batter's platoon split.
- **Batter-vs-pitcher history** — career BvP via the MLB Stats API; small samples,
  so regress heavily toward each batter's overall line.
- **Statcast expected stats** — xwOBA / barrel% / whiff% (pybaseball
  `statcast`) are the strongest public signals, but the pull is multi-GB and
  rate-limited, so it needs its own cached ETL, not an inline fetch.

This is deferred deliberately: it's the lowest-priority item and demands a
dedicated data pipeline. Do it **after** the point-in-time fix (#1) — there's no
point adding signal until the evaluation can measure whether it helps.

### 4. Smaller items

- Replace the crude extra-innings model (3 Poisson innings then a coin flip)
  with a proper inning-by-inning continuation.
- Model home/away run correlation (currently independent draws) — a pitcher's
  duel suppresses both sides.
- Reconsider redundant/low-signal features once the honest backtest exists
  (feature importance under walk-forward, not in-sample).
