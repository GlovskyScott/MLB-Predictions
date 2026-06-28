#!/usr/bin/env python3
"""Faithfully regenerate a legacy model's predictions in the current store format.

Older model releases (e.g. v1.0.0 / v1.1.0) were trained on a 32-feature
pipeline that the current 44-feature code can't reproduce. To grade them in the
archive on equal footing, run this from a **git worktree checked out at that
model's commit** so it uses that era's `src.features` / `src.model` /
`src.simulator`, then import the output into `data/predictions/<version>/`.

Usage (from a legacy worktree, with this repo's .venv active):

    git worktree add --detach /tmp/mlb-legacy <commit-with-matching-features>
    ln -s "$(pwd)/data" /tmp/mlb-legacy/data        # share the cached data
    cd /tmp/mlb-legacy
    OLD_PKL=/path/to/release/pkls OUT_DIR=/tmp/legacy_out \
        DATES=2026-06-25,2026-06-26 python scripts/import_legacy_predictions.py

`OLD_PKL` holds the release's model_win/model_runs_home/model_runs_away pkls.
`DATES` is optional (defaults to every completed date in the cached schedule).
The new-format cores written to OUT_DIR are then registered with
`scripts/register_legacy_version.py` from the current checkout.
"""
import os
import sys
import json
import joblib
from pathlib import Path

OLD_PKL = Path(os.environ["OLD_PKL"])
OUT_DIR = Path(os.environ["OUT_DIR"])
DATES = os.environ["DATES"].split(",") if os.environ.get("DATES") else None
N_SIM = int(os.environ.get("N_SIM", "1000"))

sys.path.insert(0, os.getcwd())
from src.features import build_game_features            # noqa: E402  (legacy src)
from src.model import predict_game                      # noqa: E402
from src.simulator import simulate_game                 # noqa: E402
from src.fetcher import get_season_schedule             # noqa: E402

# Mirror src.predictions.PREDICTION_FIELDS (kept inline so this runs under old src).
PREDICTION_FIELDS = (
    "game_id", "game_date", "home_id", "away_id", "home_name", "away_name",
    "venue_id", "venue_name", "home_win_pct", "away_win_pct",
    "median_home_score", "median_away_score", "modal_home_score", "modal_away_score",
    "predicted_score", "score_distribution",
    "home_innings_scoring_pct", "away_innings_scoring_pct",
    "home_innings", "away_innings", "n_simulations",
)

models = {
    "win": joblib.load(OLD_PKL / "model_win.pkl"),
    "home_runs": joblib.load(OLD_PKL / "model_runs_home.pkl"),
    "away_runs": joblib.load(OLD_PKL / "model_runs_away.pkl"),
}

by_date = {}
for g in get_season_schedule(2026):
    if g.get("status") == "Final" and g.get("home_score") is not None:
        by_date.setdefault(str(g["game_date"])[:10], []).append(g)

OUT_DIR.mkdir(parents=True, exist_ok=True)
written = 0
for d in (DATES if DATES else sorted(by_date)):
    games = by_date.get(d, [])
    if not games:
        continue
    cores = []
    for game in games:
        try:
            feats = build_game_features(game, year=2026)
            pred = predict_game(feats, models)
            sim = simulate_game(pred, n_simulations=N_SIM, seed=game["game_id"] % 100000)
            merged = {**game, **sim}
            cores.append({k: merged[k] for k in PREDICTION_FIELDS if k in merged})
        except Exception as e:  # one bad game shouldn't drop the date
            print(f"  {d} game {game.get('game_id')}: {type(e).__name__}: {e}")
    (OUT_DIR / f"{d}.json").write_text(json.dumps(cores, default=str))
    written += 1
print(f"DONE: {written} dates -> {OUT_DIR}")
