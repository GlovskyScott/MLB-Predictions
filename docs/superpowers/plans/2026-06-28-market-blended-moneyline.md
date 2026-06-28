# Market-Blended Moneyline ("Consensus" line) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the displayed model moneyline with a learned blend of the model's win% and the de-vigged market line ("Consensus"), fit on walk-forward data, and grade that blended line for historical accuracy.

**Architecture:** A new serve-time artifact `model_market_blender.pkl` holds `blended = sigmoid(a·logit(model_p) + b·logit(market_p) + c)`, fit by logistic regression on out-of-sample `(raw model prob, de-vigged market prob, outcome)` triples — the same walk-forward methodology as `scripts/build_calibrator.py`. The blend is applied at serve time and grade time (like calibration), but unlike the pick-preserving calibrator it can move the favored side toward the market, so the **blended line becomes the graded prediction of record**. That requires the market moneyline as a load-bearing input, stored write-once per date in `data/market_odds/<date>.json` (version-independent) and restored/backfilled from ESPN's public API (which returns closing moneylines for past games).

**Tech Stack:** Python 3.12, scikit-learn (LogisticRegression), joblib, Flask/Jinja, pytest. No new dependencies.

## Global Constraints

- **No new third-party dependencies.** scikit-learn, joblib, numpy, requests are already used.
- **`data/` is git-ignored.** New artifacts (`model_market_blender.pkl`, `data/market_odds/`) ship via GitHub releases, never committed. The blender pkl rides in the `latest` model release (`_CALIBRATOR_ASSETS`); `market_odds/` rides in `data-cache`.
- **Serve-time only for the blend math.** The frozen prediction core on disk is never rewritten; the stored core keeps the raw model `home_win_pct`. No version fork from this feature.
- **Determinism preserved.** The blender pkl and the write-once market-odds snapshot make the blended line reproducible. Refit the blender only on retrain (it is fit per model version, like the calibrator).
- **Graceful degradation.** No blender file OR no market line for a game → fall back to the calibrated model line (existing `_calibrate_core`). Tests that patch `_DATA_DIR` to `tmp_path` get the fallback automatically.
- **Money probabilities are de-vigged.** `market_p` is always `ih/(ih+ia)` from the two American prices, never a raw single-side implied prob.
- Run `pytest` from the activated venv (`source .venv/bin/activate`); keep it green (currently 111 tests).

---

### Task 1: `MarketBlender` model (new module `src/blend.py`)

**Files:**
- Create: `src/blend.py`
- Test: `tests/test_blend.py`

**Interfaces:**
- Consumes: nothing (leaf module, mirrors `src/calibration.py` style).
- Produces:
  - `BLENDER_FILE = "model_market_blender.pkl"`
  - `class MarketBlender(a: float, b: float, c: float = 0.0)`, callable `(model_p: float, market_p: float) -> float` (probabilities in [0,1]).
  - `fit(model_probs, market_probs, outcomes) -> MarketBlender`
  - `save(blender, data_dir, filename=BLENDER_FILE) -> Path`
  - `load(data_dir, filename=BLENDER_FILE) -> MarketBlender | None`
  - `blend_pct(blender: MarketBlender | None, model_home_pct: float, market_home_pct: float | None) -> float | None` — blends two home win **percentages** (0–100); returns `None` when `blender is None` or `market_home_pct is None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_blend.py
import math
import pytest
from src import blend as B


def test_blender_identity_on_model_when_market_weight_zero():
    # b=0, a=1, c=0 -> blended == model prob (market ignored)
    bl = B.MarketBlender(a=1.0, b=0.0, c=0.0)
    assert bl(0.70, 0.55) == pytest.approx(0.70, abs=1e-6)


def test_blender_pulls_toward_market():
    # Equal weight on both logits averages them in logit space.
    bl = B.MarketBlender(a=0.5, b=0.5, c=0.0)
    model_p, market_p = 0.80, 0.50
    expected = 1 / (1 + math.exp(-(0.5 * math.log(model_p / (1 - model_p))
                                   + 0.5 * math.log(market_p / (1 - market_p)))))
    assert bl(model_p, market_p) == pytest.approx(expected, abs=1e-9)
    assert 0.50 < bl(model_p, market_p) < 0.80   # between the two


def test_fit_recovers_market_when_model_is_noise():
    # If outcomes track the market and the model is constant 0.5 (logit 0),
    # the fit should lean on the market term (b > 0) to separate the classes.
    import numpy as np
    rng = np.random.default_rng(0)
    market = rng.uniform(0.2, 0.8, size=2000)
    model = [0.5] * 2000                     # pure noise (logit 0)
    outcomes = (rng.uniform(size=2000) < market).astype(int)
    bl = B.fit(model, market, outcomes)
    assert bl.b > 0.5            # market carries the signal
    # blended prob tracks the market direction
    assert bl(0.5, 0.75) > bl(0.5, 0.25)


def test_blend_pct_none_without_market_or_blender():
    bl = B.MarketBlender(a=1.0, b=0.0)
    assert B.blend_pct(bl, 60.0, None) is None         # no market line
    assert B.blend_pct(None, 60.0, 55.0) is None       # no blender


def test_blend_pct_returns_percentage():
    bl = B.MarketBlender(a=0.5, b=0.5, c=0.0)
    out = B.blend_pct(bl, 80.0, 50.0)
    assert 50.0 < out < 80.0 and round(out, 1) == out


def test_save_load_round_trip(tmp_path):
    bl = B.MarketBlender(a=0.6, b=0.7, c=-0.05)
    B.save(bl, tmp_path)
    loaded = B.load(tmp_path)
    assert (loaded.a, loaded.b, loaded.c) == pytest.approx((0.6, 0.7, -0.05))


def test_load_missing_returns_none(tmp_path):
    assert B.load(tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_blend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.blend'`.

- [ ] **Step 3: Implement `src/blend.py`**

```python
"""Market-blended win probability.

The displayed moneyline is a learned blend of the model's raw simulated win% and
the de-vigged market line:

    blended = sigmoid(a*logit(model_p) + b*logit(market_p) + c)

fit by logistic regression on **walk-forward** (out-of-sample) triples of
(raw model prob, de-vigged market prob, actual outcome) — see
scripts/build_calibrator.py for the methodology. Because the market is sharp and
the raw model is overconfident, the fit typically lands a < 1 (shrinks the model)
and b > 0 (leans on the market). Unlike the win% calibrator this is NOT
pick-preserving: it can move the favored side toward the market. That is the
point — so the blended line is the prediction of record and is graded as such.

If no blender file is present, or a game has no market line, callers fall back to
the calibrated model line (load() returns None, blend_pct() returns None).
"""
import math
from pathlib import Path

import joblib

BLENDER_FILE = "model_market_blender.pkl"
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class MarketBlender:
    """Blend two probabilities in logit space: sigmoid(a*logit(model)+b*logit(mkt)+c)."""

    def __init__(self, a: float, b: float, c: float = 0.0):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def __call__(self, model_p: float, market_p: float) -> float:
        return _sigmoid(self.a * _logit(model_p) + self.b * _logit(market_p) + self.c)


def fit(model_probs, market_probs, outcomes) -> MarketBlender:
    """Fit the blend by logistic regression on the two logit features.

    Intercept is allowed (the blend is not pick-preserving, so a small learned
    home-field term is fine and usually helps). High C ~= unregularized.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([
        [_logit(float(p)) for p in model_probs],
        [_logit(float(p)) for p in market_probs],
    ])
    y = np.asarray(outcomes, dtype=int)
    lr = LogisticRegression(C=1e6, solver="lbfgs", fit_intercept=True)
    lr.fit(X, y)
    return MarketBlender(a=float(lr.coef_[0][0]), b=float(lr.coef_[0][1]),
                         c=float(lr.intercept_[0]))


def save(blender: MarketBlender, data_dir, filename: str = BLENDER_FILE) -> Path:
    path = Path(data_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"a": blender.a, "b": blender.b, "c": blender.c}, path)
    return path


def load(data_dir, filename: str = BLENDER_FILE) -> "MarketBlender | None":
    """Load a blender for data_dir, or None if none has been built.

    Safety: written by build_calibrator.py in this codebase, never from user
    input or network. Joblib is acceptable here.
    """
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    d = joblib.load(path)
    return MarketBlender(a=d["a"], b=d["b"], c=d.get("c", 0.0))


def blend_pct(blender: "MarketBlender | None", model_home_pct: float,
              market_home_pct: "float | None") -> "float | None":
    """Blend two home win *percentages* (0-100). None if blender or market absent."""
    if blender is None or model_home_pct is None or market_home_pct is None:
        return None
    return round(blender(model_home_pct / 100.0, market_home_pct / 100.0) * 100.0, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_blend.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/blend.py tests/test_blend.py
git commit -m "feat(blend): MarketBlender — logit-space model+market blend"
```

---

### Task 2: De-vig helper in `src/fetcher.py`

**Files:**
- Modify: `src/fetcher.py` (add near `market_key`, ~line 950)
- Test: `tests/test_fetcher.py` (create if absent)

**Interfaces:**
- Consumes: nothing.
- Produces: `devig_home_prob(ml_home, ml_away) -> float | None` — de-vigged home win probability from two American moneyline prices; `None` if either price is missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetcher.py
import pytest
from src.fetcher import devig_home_prob


def test_devig_home_prob_pick_em():
    # -110 / -110 -> ~0.5 after removing vig
    assert devig_home_prob(-110, -110) == pytest.approx(0.5, abs=1e-6)


def test_devig_home_prob_favorite():
    # home -200 (0.667 raw) vs away +170 (0.370 raw) -> de-vig ~0.643
    p = devig_home_prob(-200, 170)
    assert 0.60 < p < 0.68


def test_devig_home_prob_none_when_missing():
    assert devig_home_prob(None, 150) is None
    assert devig_home_prob(-150, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher.py -q`
Expected: FAIL — `ImportError: cannot import name 'devig_home_prob'`.

- [ ] **Step 3: Implement the helper**

Add to `src/fetcher.py` (after `market_key`):

```python
def _implied(odds) -> "float | None":
    if odds is None:
        return None
    o = float(odds)
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def devig_home_prob(ml_home, ml_away) -> "float | None":
    """De-vigged home win probability from two American prices, or None if either
    price is missing. Removes the bookmaker hold by normalizing the two raw
    implied probabilities to sum to 1."""
    ih, ia = _implied(ml_home), _implied(ml_away)
    if ih is None or ia is None or (ih + ia) == 0:
        return None
    return ih / (ih + ia)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetcher.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fetcher.py tests/test_fetcher.py
git commit -m "feat(fetcher): devig_home_prob helper"
```

---

### Task 3: Write-once market-odds store in `src/predictions.py`

**Files:**
- Modify: `src/predictions.py` (add a store section; `threading` and `json` are already imported)
- Test: `tests/test_predictions.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_market_odds(data_dir, date: str) -> dict` — `{game_id(str): {"ml_home": float, "ml_away": float, "captured_at": str}}`, or `{}`.
  - `save_market_odds(data_dir, date: str, odds: dict) -> None` — `odds` maps `game_id -> {"ml_home","ml_away"}`; **write-once per game** (first capture preserved), games missing either price are skipped.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_predictions.py  (append)
def test_market_odds_round_trip(tmp_path):
    P.save_market_odds(tmp_path, '2026-06-28',
                       {1: {'ml_home': -130, 'ml_away': 110},
                        2: {'ml_home': 105, 'ml_away': -125}})
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert got['1']['ml_home'] == -130 and got['1']['ml_away'] == 110
    assert 'captured_at' in got['1']


def test_market_odds_missing_returns_empty(tmp_path):
    assert P.load_market_odds(tmp_path, '2026-01-01') == {}


def test_market_odds_write_once(tmp_path):
    P.save_market_odds(tmp_path, '2026-06-28', {1: {'ml_home': -130, 'ml_away': 110}})
    ts = P.load_market_odds(tmp_path, '2026-06-28')['1']['captured_at']
    # later capture must NOT move an existing game, but may add new ones
    P.save_market_odds(tmp_path, '2026-06-28',
                       {1: {'ml_home': -200, 'ml_away': 170},
                        3: {'ml_home': 100, 'ml_away': -120}})
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert got['1']['ml_home'] == -130 and got['1']['captured_at'] == ts  # preserved
    assert got['3']['ml_home'] == 100                                      # added


def test_market_odds_skips_incomplete(tmp_path):
    P.save_market_odds(tmp_path, '2026-06-28',
                       {1: {'ml_home': None, 'ml_away': 110},
                        2: {'ml_home': -120, 'ml_away': 100}})
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert '1' not in got and got['2']['ml_home'] == -120
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predictions.py -k market_odds -q`
Expected: FAIL — `AttributeError: module 'src.predictions' has no attribute 'save_market_odds'`.

- [ ] **Step 3: Implement the store**

Add to `src/predictions.py` (after the imports / lock section):

```python
# ---- market-odds snapshot store --------------------------------------------
# De-vigged inputs for the market-blended ("Consensus") moneyline. Version-
# independent (the book line is the same regardless of model) and write-once: the
# first line captured for a game is preserved (a closing-ish line); later captures
# only add games not yet seen. Absent line -> the game falls back to the model.

_market_odds_lock = threading.Lock()


def _market_odds_dir(data_dir) -> Path:
    return Path(data_dir) / 'market_odds'


def load_market_odds(data_dir, date: str) -> dict:
    """Return {game_id(str): {ml_home, ml_away, captured_at}} for date, or {}."""
    f = _market_odds_dir(data_dir) / f'{date}.json'
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def save_market_odds(data_dir, date: str, odds: dict) -> None:
    """Merge moneyline snapshots for date, write-once per game.

    ``odds`` maps game_id -> {'ml_home': float|None, 'ml_away': float|None}.
    Games missing either price are skipped; games already on disk keep their
    first-captured line and timestamp.
    """
    import datetime
    with _market_odds_lock:
        existing = load_market_odds(data_dir, date)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        changed = False
        for gid, line in odds.items():
            key = str(gid)
            mlh, mla = (line or {}).get('ml_home'), (line or {}).get('ml_away')
            if mlh is None or mla is None or key in existing:
                continue
            existing[key] = {'ml_home': float(mlh), 'ml_away': float(mla),
                             'captured_at': now}
            changed = True
        if not changed and (_market_odds_dir(data_dir) / f'{date}.json').exists():
            return
        d = _market_odds_dir(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f'{date}.json.tmp'
        tmp.write_text(json.dumps(existing, default=str))
        tmp.replace(d / f'{date}.json')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predictions.py -k market_odds -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/predictions.py tests/test_predictions.py
git commit -m "feat(predictions): write-once market-odds (moneyline) store"
```

---

### Task 4: Backfill script `scripts/backfill_market_odds.py`

**Files:**
- Create: `scripts/backfill_market_odds.py`
- Test: `tests/test_backfill_market_odds.py`

**Interfaces:**
- Consumes: `fetcher.get_market_odds(date)` (already returns `{market_key: {ml_home, ml_away, total, ...}}`), `fetcher.get_schedule(date)` / `get_season_schedule`, `predictions.save_market_odds`.
- Produces: `collect_market_odds(date: str, data_dir) -> int` — fetches the date's lines, maps them to game_ids, persists via `save_market_odds`, returns the number of games written. A `main()` CLI loops a date range.

**Note:** `get_market_odds` keys by `market_key(away_name, home_name)`; the schedule gives `game_id` + team names. Join on `market_key` to attach `game_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill_market_odds.py
from scripts import backfill_market_odds as BF
from src import predictions as P
from src.fetcher import market_key


def test_collect_market_odds_writes_by_game_id(tmp_path, mocker):
    mocker.patch('scripts.backfill_market_odds.get_schedule', return_value=[
        {'game_id': 11, 'away_name': 'Boston Red Sox', 'home_name': 'New York Yankees'},
        {'game_id': 12, 'away_name': 'Chicago Cubs', 'home_name': 'St. Louis Cardinals'},
    ])
    mocker.patch('scripts.backfill_market_odds.get_market_odds', return_value={
        market_key('Boston Red Sox', 'New York Yankees'): {'ml_home': -130, 'ml_away': 110},
        # game 12 has no odds -> skipped
    })
    n = BF.collect_market_odds('2026-06-28', tmp_path)
    assert n == 1
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert got['11']['ml_home'] == -130 and '12' not in got
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backfill_market_odds.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.backfill_market_odds'`.

- [ ] **Step 3: Implement the script**

```python
"""Backfill the write-once market-odds (moneyline) store from ESPN's public API.

ESPN's scoreboard/summary returns closing moneylines for completed games, so we
can snapshot history once and grade the blended line reproducibly. Run after a
retrain (before scripts/build_calibrator.py, which needs these lines to fit the
blender) and as a daily top-up.

Usage:
  python -m scripts.backfill_market_odds [--days N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse
import datetime as dt
from pathlib import Path

from src.fetcher import get_schedule, get_market_odds, market_key
from src import predictions as P

_DATA = Path(__file__).resolve().parent.parent / "data"


def collect_market_odds(date: str, data_dir=_DATA) -> int:
    """Fetch and persist (write-once) the moneylines for one date. Returns games written."""
    odds = get_market_odds(date)
    if not odds:
        return 0
    by_game = {}
    for g in get_schedule(date):
        mk = odds.get(market_key(g.get('away_name', ''), g.get('home_name', '')))
        if mk and mk.get('ml_home') is not None and mk.get('ml_away') is not None:
            by_game[g['game_id']] = {'ml_home': mk['ml_home'], 'ml_away': mk['ml_away']}
    before = len(P.load_market_odds(data_dir, date))
    P.save_market_odds(data_dir, date, by_game)
    return len(P.load_market_odds(data_dir, date)) - before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    if args.start and args.end:
        d0 = dt.date.fromisoformat(args.start)
        d1 = dt.date.fromisoformat(args.end)
    else:
        d1 = dt.date.today()
        d0 = d1 - dt.timedelta(days=args.days)
    total = 0
    d = d0
    while d <= d1:
        ds = d.isoformat()
        try:
            n = collect_market_odds(ds)
            total += n
            print(f"{ds}: +{n} games")
        except Exception as e:
            print(f"{ds}: ERROR {e}")
        d += dt.timedelta(days=1)
    print(f"Done. {total} game lines captured.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backfill_market_odds.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_market_odds.py tests/test_backfill_market_odds.py
git commit -m "feat(scripts): backfill_market_odds — snapshot ESPN moneylines"
```

---

### Task 5: Fit the blender in `scripts/build_calibrator.py`

**Files:**
- Modify: `scripts/build_calibrator.py` (add a blender stage after the win-cal stage; it already accumulates walk-forward `(game, raw_prob, outcome)` data and knows each game's `game_id`/`date`)
- Test: `tests/test_build_calibrator_blender.py`

**Interfaces:**
- Consumes: the walk-forward pairs already produced for the win calibrator, `predictions.load_market_odds`, `fetcher.devig_home_prob`, `blend.fit`, `blend.save`.
- Produces: writes `data/model_market_blender.pkl`. Factor the join+fit into a testable helper: `build_blender(pairs, data_dir) -> blend.MarketBlender | None`, where `pairs` is an iterable of `{'game_id', 'date', 'model_home_prob', 'home_win'}`.

**Note:** Read `build_calibrator.py` first to match how it stores each walk-forward row (it must expose `game_id` + `game_date` + raw home prob + outcome). If it currently keeps only `(prob, outcome)`, extend the accumulation to also keep `game_id` and `date` — that is part of this task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_calibrator_blender.py
from scripts.build_calibrator import build_blender
from src import predictions as P


def test_build_blender_joins_market_and_fits(tmp_path):
    # Market strongly predicts outcome; model is noise. Blender should weight market.
    import numpy as np
    rng = np.random.default_rng(1)
    pairs = []
    for d in range(1, 6):
        date = f'2026-04-0{d}'
        odds = {}
        for gid in range(1, 81):
            p = float(rng.uniform(0.25, 0.75))
            # encode a market prob via a -vig-free pair: home -EV from p
            mlh = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
            mla = 100 * (1 - p) / p if p >= 0.5 else -100 * p / (1 - p)
            odds[gid] = {'ml_home': mlh, 'ml_away': mla}
            pairs.append({'game_id': gid, 'date': date, 'model_home_prob': 0.5,
                          'home_win': int(rng.uniform() < p)})
        P.save_market_odds(tmp_path, date, odds)
    bl = build_blender(pairs, tmp_path)
    assert bl is not None and bl.b > 0.4      # leans on the market signal
    assert bl(0.5, 0.75) > bl(0.5, 0.25)


def test_build_blender_none_without_market(tmp_path):
    pairs = [{'game_id': 1, 'date': '2026-04-01', 'model_home_prob': 0.6, 'home_win': 1}]
    assert build_blender(pairs, tmp_path) is None   # no market_odds on disk
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_calibrator_blender.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_blender'`.

- [ ] **Step 3: Implement `build_blender` + wire it into `main()`**

Add to `scripts/build_calibrator.py`:

```python
from src import blend as _blend
from src.fetcher import devig_home_prob


def build_blender(pairs, data_dir):
    """Fit the market blender from walk-forward pairs joined to the market-odds store.

    pairs: iterable of {'game_id', 'date', 'model_home_prob' (0-1), 'home_win' (0/1)}.
    Returns a MarketBlender, or None if too few games have a captured market line.
    """
    from src import predictions as P
    model_ps, market_ps, ys = [], [], []
    odds_cache = {}
    for r in pairs:
        date = r['date']
        if date not in odds_cache:
            odds_cache[date] = P.load_market_odds(data_dir, date)
        o = odds_cache[date].get(str(r['game_id']))
        if not o:
            continue
        mp = devig_home_prob(o['ml_home'], o['ml_away'])
        if mp is None:
            continue
        model_ps.append(r['model_home_prob'])
        market_ps.append(mp)
        ys.append(int(r['home_win']))
    if len(ys) < 500:
        return None
    return _blend.fit(model_ps, market_ps, ys)
```

In `main()`, after the win calibrator is saved and you have the accumulated
walk-forward rows (each with `game_id`, `game_date`, raw home prob, outcome),
add:

```python
    blender = build_blender(
        [{'game_id': gid, 'date': gd, 'model_home_prob': mp, 'home_win': y}
         for gid, gd, mp, y in walkforward_rows],   # adapt to the actual accumulator
        _DATA,
    )
    if blender is not None:
        path = _blend.save(blender, _DATA)
        print(f"Saved market blender (a={blender.a:.3f} model, "
              f"b={blender.b:.3f} market, c={blender.c:+.3f}) -> {path}")
    else:
        print("Market blender NOT built — too few games with captured market odds. "
              "Run scripts/backfill_market_odds.py first.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_calibrator_blender.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_calibrator.py tests/test_build_calibrator_blender.py
git commit -m "feat(calibrator): fit market blender on walk-forward + market odds"
```

---

### Task 6: Serve-time blend + daily snapshot in `src/app.py`

**Files:**
- Modify: `src/app.py` — add `_get_blender`, `_blend_core`; call them in `run_daily_simulation`; snapshot today's moneylines write-once.
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `blend.load`/`blend.blend_pct`, `fetcher.devig_home_prob`, `predictions.save_market_odds`/`load_market_odds`, existing `_calibrate_core`, `get_market_odds`, `market_key`.
- Produces:
  - `_get_blender() -> blend.MarketBlender | None` (cached per `_DATA_DIR`, like `_get_calibrator`).
  - `_blend_core(core: dict, market_home_prob: float | None) -> dict` — returns a copy with `home_win_pct`/`away_win_pct` set to the **blended** value and `model_home_win_pct`/`model_away_win_pct` preserving the calibrated model-only value; if no blender or no market prob, returns the calibrated core unchanged (model-only) with `model_home_win_pct == home_win_pct`.

**Design:** `_blend_core` is always called *after* `_calibrate_core`. It blends the **raw** model prob (`core['raw_home_win_pct']` when present, else the calibrated `home_win_pct`) with the market prob, because the blender was fit on raw model logits. The calibrated model value is kept under `model_*` for fallback and (internal) reference.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py  (append)
def test_blend_core_replaces_winpct_with_blend(tmp_path, mocker):
    import src.app as app
    from src import blend
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._blender_cache.clear()
    blend.save(blend.MarketBlender(a=0.5, b=0.5, c=0.0), tmp_path)
    core = {'home_win_pct': 80.0, 'away_win_pct': 20.0, 'raw_home_win_pct': 80.0}
    out = app._blend_core(core, market_home_prob=0.50)
    assert 50.0 < out['home_win_pct'] < 80.0          # pulled toward market
    assert out['away_win_pct'] == round(100 - out['home_win_pct'], 1)
    assert out['model_home_win_pct'] == 80.0          # model-only preserved


def test_blend_core_falls_back_without_market(tmp_path, mocker):
    import src.app as app
    from src import blend
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._blender_cache.clear()
    blend.save(blend.MarketBlender(a=0.5, b=0.5, c=0.0), tmp_path)
    core = {'home_win_pct': 64.0, 'away_win_pct': 36.0, 'raw_home_win_pct': 64.0}
    out = app._blend_core(core, market_home_prob=None)
    assert out['home_win_pct'] == 64.0                # unchanged
    assert out['model_home_win_pct'] == 64.0


def test_blend_core_identity_without_blender(tmp_path, mocker):
    import src.app as app
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    app._blender_cache.clear()                         # no blender file
    core = {'home_win_pct': 64.0, 'away_win_pct': 36.0, 'raw_home_win_pct': 64.0}
    out = app._blend_core(core, market_home_prob=0.50)
    assert out['home_win_pct'] == 64.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -k blend_core -q`
Expected: FAIL — `AttributeError: module 'src.app' has no attribute '_blend_core'`.

- [ ] **Step 3: Implement in `src/app.py`**

Add the import and cache near the other calibrator plumbing:

```python
from src import blend as _blend
from src.fetcher import devig_home_prob

_blender_cache: dict = {}  # data-dir -> MarketBlender|None
```

Add the loader + blend function:

```python
def _get_blender():
    """Load (and cache) the market blender for the active data dir."""
    key = str(_DATA_DIR)
    if key not in _blender_cache:
        _blender_cache[key] = _blend.load(_DATA_DIR)
    return _blender_cache[key]


def _blend_core(core: dict, market_home_prob: "float | None") -> dict:
    """Return a copy whose home/away win% is the market-blended ('Consensus')
    line, with the calibrated model-only value preserved under model_*.

    The blender consumes the RAW model prob (it was fit on raw logits). Falls back
    to the (calibrated) model line when there is no blender or no market line."""
    out = dict(core)
    model_home = core.get('home_win_pct')        # already calibrated by _calibrate_core
    out['model_home_win_pct'] = model_home
    out['model_away_win_pct'] = core.get('away_win_pct')
    blender = _get_blender()
    raw_home = core.get('raw_home_win_pct', model_home)
    blended = _blend.blend_pct(blender, raw_home, market_home_prob)
    if blended is not None:
        out['home_win_pct'] = blended
        out['away_win_pct'] = round(100.0 - blended, 1)
    return out
```

In `run_daily_simulation`, after `market = get_market_odds(sim_date)` and the
schedule loop builds each `g`, snapshot the moneylines and apply the blend.
Locate the loop that sets `g['market'] = _market_block(g, mk) if mk else None`
and extend it:

```python
    # Snapshot today's moneylines write-once (load-bearing input to the blend).
    snap = {}
    for game in schedule:
        mk = market.get(market_key(game.get('away_name', ''), game.get('home_name', '')))
        if mk and mk.get('ml_home') is not None and mk.get('ml_away') is not None:
            snap[game['game_id']] = {'ml_home': mk['ml_home'], 'ml_away': mk['ml_away']}
    try:
        _pred.save_market_odds(_DATA_DIR, sim_date, snap)
    except Exception:
        pass
```

And where each enriched game `g` is finalized (after `g['market'] = ...`):

```python
        mk = g.get('market')
        market_home_prob = None
        if mk:
            market_home_prob = devig_home_prob(market.get(
                market_key(g.get('away_name', ''), g.get('home_name', '')), {}).get('ml_home'),
                market.get(market_key(g.get('away_name', ''), g.get('home_name', '')), {}).get('ml_away'))
        g.update(_blend_core(g, market_home_prob))
```

(Refactor the repeated `market.get(market_key(...))` into a local `mk_raw`
variable for readability; the test only depends on `_blend_core`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -k blend_core -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite (no regressions) and commit**

```bash
pytest -q
git add src/app.py tests/test_app.py
git commit -m "feat(app): serve-time market blend + daily moneyline snapshot"
```

---

### Task 7: Grade the blended line in `compare_date`

**Files:**
- Modify: `src/app.py` — `compare_date` blends each core with the date's stored market line before computing `winner_correct`; also compute model-only correctness for comparison; aggregators expose both.
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `predictions.load_market_odds`, `devig_home_prob`, `_blend_core`.
- Produces: each result row gains `model_winner_correct` (model-only pick vs outcome) alongside `winner_correct` (now the **blended** pick). `compare_date`/`_aggregate_days`/`_version_summary` return `consensus_accuracy` (== `winner_accuracy`, the blended/headline number) and `model_accuracy` (model-only), plus `model_correct`/graded counts for weighted pooling.

**Design:** `winner_correct` stays the headline-graded field (now blended) so existing callers/templates keep working; `ml_accuracy`/`winner_accuracy` therefore already reflect the Consensus line. Add `model_accuracy` as the secondary comparison number.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py  (append)
def test_compare_date_grades_blended_pick(tmp_path, mocker):
    import src.app as app
    from src import predictions as P, blend
    mocker.patch.object(app, '_DATA_DIR', tmp_path)
    mocker.patch.object(app, '_current_version', return_value='v1')
    app._blender_cache.clear(); app._calibrator_cache.clear()
    # Blender that leans hard on the market (a=0, b=3): the market decides the pick.
    blend.save(blend.MarketBlender(a=0.0, b=3.0, c=0.0), tmp_path)
    # Model favors HOME (70%), market favors AWAY (home -110/+ -> away favorite).
    P.save_prediction(tmp_path, 'v1', '2026-06-26', [{
        'game_id': 1, 'game_date': '2026-06-26', 'home_id': 147, 'away_id': 111,
        'home_win_pct': 70.0, 'away_win_pct': 30.0, 'raw_home_win_pct': 70.0,
        'median_home_score': 4.0, 'median_away_score': 5.0}])
    P.save_market_odds(tmp_path, '2026-06-26', {1: {'ml_home': 200, 'ml_away': -240}})  # away favored
    mocker.patch('src.app.get_season_schedule', return_value=[{
        'game_id': 1, 'game_date': '2026-06-26', 'status': 'Final',
        'home_score': 3, 'away_score': 6, 'home_id': 147, 'away_id': 111}])  # away won
    mocker.patch('src.app.refresh_schedule_date', return_value=0)
    sim = mocker.patch('src.app.simulate_game')
    mocker.patch('src.app.get_team_meta', return_value={
        'logo_url': '', 'primary': '#111', 'secondary': '#222', 'abbr': 'X'})

    data = app.compare_date('2026-06-26')
    g = data['games'][0]
    assert g['winner_correct'] is True        # blended pick followed market -> AWAY -> correct
    assert g['model_winner_correct'] is False  # model picked HOME -> wrong
    assert data['consensus_accuracy'] == 100.0
    assert data['model_accuracy'] == 0.0
    sim.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -k grades_blended_pick -q`
Expected: FAIL — `KeyError: 'model_winner_correct'` / `consensus_accuracy`.

- [ ] **Step 3: Implement in `compare_date`**

Read the current `compare_date` body. Replace the per-core grading block so it
loads the date's market odds once, blends, and grades the blended pick:

```python
    market_odds = _pred.load_market_odds(_DATA_DIR, result_date)
    results = []
    correct = 0          # blended (headline)
    model_correct = 0    # model-only (comparison)
    for core in cores:
        game = actuals.get(core.get('game_id'))
        if not game:
            continue
        core = _calibrate_core(core)
        o = market_odds.get(str(core.get('game_id')))
        market_home_prob = devig_home_prob(o['ml_home'], o['ml_away']) if o else None
        core = _blend_core(core, market_home_prob)   # sets blended home_win_pct + model_*
        actual_home = int(game['home_score'])
        actual_away = int(game['away_score'])
        actual_home_won = actual_home > actual_away
        predicted_home_won = core.get('home_win_pct', 50.0) > 50.0          # blended pick
        model_home_won = core.get('model_home_win_pct', 50.0) > 50.0        # model-only pick
        winner_correct = actual_home_won == predicted_home_won
        model_winner_correct = actual_home_won == model_home_won
        if winner_correct:
            correct += 1
        if model_winner_correct:
            model_correct += 1
        home_err = abs(core.get('median_home_score', 0) - actual_home)
        away_err = abs(core.get('median_away_score', 0) - actual_away)
        home_meta = get_team_meta(game['home_id'])
        away_meta = get_team_meta(game['away_id'])
        ml_pick = home_meta.get('abbr', 'HOME') if predicted_home_won else away_meta.get('abbr', 'AWAY')
        results.append({
            **game, **core,
            'actual_home_score': actual_home, 'actual_away_score': actual_away,
            'actual_home_won': actual_home_won, 'predicted_home_won': predicted_home_won,
            'home_score_err': round(home_err, 1), 'away_score_err': round(away_err, 1),
            'winner_correct': winner_correct, 'ml_correct': winner_correct,
            'model_winner_correct': model_winner_correct, 'ml_pick': ml_pick,
            'home_logo': home_meta['logo_url'], 'away_logo': away_meta['logo_url'],
            'home_color': home_meta['primary'], 'away_color': away_meta['primary'],
            'home_color2': home_meta['secondary'], 'away_color2': away_meta['secondary'],
        })

    n = len(results)
    accuracy = round(correct / n * 100, 1) if n else 0
    model_acc = round(model_correct / n * 100, 1) if n else 0
    scored = [r for r in results if 'home_score_err' in r]
    avg_err = round(sum(r['home_score_err'] + r['away_score_err'] for r in scored)
                    / (2 * len(scored)), 2) if scored else None
    return {
        'result_date': result_date, 'games': results, 'n_completed': n,
        'winner_accuracy': accuracy, 'ml_accuracy': accuracy,
        'consensus_accuracy': accuracy, 'model_accuracy': model_acc,
        'model_correct': model_correct, 'avg_score_err': avg_err,
    }
```

Then in `_aggregate_days` and `_version_summary`, pool `model_accuracy` the same
way `winner_accuracy` is pooled (weight by `n_completed`) and add
`model_accuracy` + `consensus_accuracy` (== existing `winner_accuracy`) to both
return dicts:

```python
    model_correct_total = sum(round((r.get('model_accuracy') or 0) / 100 * r['n_completed'])
                              for r in daily)
    model_accuracy = round(model_correct_total / total_games * 100, 1) if total_games else 0
    # add to result dict:
    #   'consensus_accuracy': accuracy, 'model_accuracy': model_accuracy,
```

- [ ] **Step 4: Run test + full suite**

Run: `pytest tests/test_app.py -k 'grades_blended_pick or compare_date or aggregate' -q && pytest -q`
Expected: PASS; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/app.py tests/test_app.py
git commit -m "feat(app): grade the blended Consensus pick; expose model vs consensus accuracy"
```

---

### Task 8: UI — headline = Consensus, hide model-only, archive shows both

**Files:**
- Modify: `templates/index.html` (game card win-prob/odds area, results "Pick" column, `market_stats` macro), `templates/archive.html`, `templates/archive_version.html`
- Test: `tests/test_app.py` (render assertions)

**Interfaces:**
- Consumes: `g.home_win_pct` (now blended), `g.model_home_win_pct` (hidden), `summary.consensus_accuracy`, `summary.model_accuracy`.
- Produces: rendered pages where the headline win% is the blend, labeled "Consensus", with no separate model-only number surfaced; archive shows both Consensus and Model accuracy columns.

- [ ] **Step 1: Write the failing render test**

```python
# tests/test_app.py  (append)
def test_index_headline_is_consensus(client, mocker):
    sim = dict(MOCK_SIM_RESULT)
    sim['home_win_pct'] = 58.0; sim['away_win_pct'] = 42.0
    sim['model_home_win_pct'] = 64.0   # model-only, must NOT be shown
    mocker.patch('src.app.run_daily_simulation', return_value=[sim])
    mocker.patch('src.app._get_results_for_date', return_value=MOCK_RESULTS_DATA)
    mocker.patch('src.app._aggregate_days', return_value={
        'winner_accuracy': 58.0, 'consensus_accuracy': 58.0, 'model_accuracy': 55.0,
        'total_games': 100, 'avg_score_err': 2.1, 'daily': []})
    html = client.get('/').data.decode()
    assert 'Consensus' in html
    assert '58' in html and '>64<' not in html   # model-only number not surfaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -k headline_is_consensus -q`
Expected: FAIL — `assert 'Consensus' in html` fails.

- [ ] **Step 3: Update templates**

In `templates/index.html`:
- Where the win-probability bar / headline label renders, label the number **"Consensus win %"** (it already reads `r.home_win_pct`, which is now blended — no data change needed).
- In the odds table, keep the **ESPN** row and the model-vs-market **Edge** row (the edge is now Consensus vs market — typically small, which is the honest result). Do **not** add a model-only row.
- `market_stats(s)` macro: change the `ML` tile title to "Consensus moneyline accuracy" and add a second tile reading `s.get('model_accuracy')` titled "Model-only accuracy (for comparison)":

```html
{% set ml = s.get('consensus_accuracy', s.get('ml_accuracy', s.get('winner_accuracy'))) %}
{% set mo = s.get('model_accuracy') %}
<div class="summary-stat"><div class="s-label" title="Consensus (market-blended) moneyline: % of headline picks correct">Consensus</div><div class="s-value {% if ml is not none and ml >= 55 %}good{% else %}neutral{% endif %}">{% if ml is not none %}{{ ml }}%{% else %}—{% endif %}</div></div>
<div class="summary-stat"><div class="s-label" title="Model-only moneyline accuracy, for comparison">Model</div><div class="s-value neutral">{% if mo is not none %}{{ mo }}%{% else %}—{% endif %}</div></div>
```

In `templates/archive.html` (header + row): replace the single `ML` column with
two — `Consensus` (`v.consensus_accuracy`) and `Model` (`v.model_accuracy`):

```html
<!-- header -->
<th>Days</th><th>Graded games</th><th>Consensus</th><th>Model</th><th>Avg err</th>
<!-- row -->
<td class="acc">{% if v.total_games %}{{ v.consensus_accuracy }}%{% else %}—{% endif %}</td>
<td class="acc">{% if v.model_accuracy is not none %}{{ v.model_accuracy }}%{% else %}—{% endif %}</td>
```

In `templates/archive_version.html` summary stats: replace the single ML stat
with Consensus + Model:

```html
<div class="stat"><div class="v acc">{% if summary.total_games %}{{ summary.consensus_accuracy }}%{% else %}—{% endif %}</div><div class="l">Consensus</div></div>
<div class="stat"><div class="v acc">{% if summary.model_accuracy is not none %}{{ summary.model_accuracy }}%{% else %}—{% endif %}</div><div class="l">Model</div></div>
```

Update the `/archive` route in `src/app.py` to pass `consensus_accuracy` +
`model_accuracy` through the per-version row dict (mirror how `winner_accuracy`
is passed today).

- [ ] **Step 4: Run test + full suite**

Run: `pytest tests/test_app.py -k 'headline_is_consensus or archive' -q && pytest -q`
Expected: PASS; full suite green.

- [ ] **Step 5: Commit**

```bash
git add templates/ src/app.py tests/test_app.py
git commit -m "feat(ui): headline Consensus line; archive shows Consensus vs Model"
```

---

### Task 9: Ship the blender in the `latest` release restore

**Files:**
- Modify: `src/fetcher.py` — add `model_market_blender.pkl` to `_CALIBRATOR_ASSETS` so `bootstrap_model_cache` restores it best-effort.
- Test: `tests/test_fetcher.py`

**Interfaces:**
- Consumes: existing `_CALIBRATOR_ASSETS` list + `bootstrap_model_cache`.
- Produces: `model_market_blender.pkl` listed among the optional model artifacts downloaded from the `latest` release.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetcher.py  (append)
def test_blender_in_calibrator_assets():
    from src import fetcher
    assert 'model_market_blender.pkl' in fetcher._CALIBRATOR_ASSETS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher.py -k blender_in_calibrator_assets -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `src/fetcher.py`, add to the `_CALIBRATOR_ASSETS` list:

```python
_CALIBRATOR_ASSETS = ["model_calibrator.pkl", "model_inning_calibrator.pkl",
                      "model_inning_dist_calibrator.pkl", "model_market_blender.pkl"]
```

- [ ] **Step 4: Run test + full suite**

Run: `pytest tests/test_fetcher.py -q && pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fetcher.py tests/test_fetcher.py
git commit -m "feat(fetcher): restore market blender from the latest release"
```

---

### Task 10: Docs — CLAUDE.md + README

**Files:**
- Modify: `CLAUDE.md` (calibration section), `README.md` (betting board bullet)

- [ ] **Step 1: Update CLAUDE.md**

In the "Win-probability calibration" section, add a bullet:

```markdown
- **Market-blended "Consensus" line (headline).** The displayed/ graded moneyline is `sigmoid(a·logit(model_raw) + b·logit(market_devig) + c)` from `src/blend.py` (`model_market_blender.pkl`), fit walk-forward by `scripts/build_calibrator.py` on `(raw model prob, de-vigged market prob, outcome)` triples. Unlike the win% calibrator it is **not pick-preserving** — it can move the favored side toward the market — so the blended line is the **graded prediction of record** (`winner_correct`/`ml_accuracy`/`consensus_accuracy`), with model-only accuracy kept alongside as `model_accuracy`. Inputs: the write-once moneyline store `data/market_odds/<date>.json` (`predictions.save/load_market_odds`, captured live in `run_daily_simulation`, backfilled by `scripts/backfill_market_odds.py` from ESPN closing lines). Serve-time only (no fork); falls back to the calibrated model line when a game has no market line or the blender is absent. Rebuild order after a retrain: `backfill_market_odds.py` → `build_calibrator.py` (fits both calibrators + the blender).
```

- [ ] **Step 2: Update README.md** betting-board bullet to describe the Consensus line.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: market-blended Consensus moneyline"
```

---

### Task 11: Deploy (operational — run once, after merge)

**Not a code change — a runbook.** Run from the merged v5 code against the live data dir.

- [ ] **Step 1:** Backfill historical moneylines:
  `python -m scripts.backfill_market_odds --days 120`
  Expected: per-date `+N games`, hundreds–thousands total. Verify `ls data/market_odds | wc -l`.
- [ ] **Step 2:** Fit calibrators + blender:
  `python -m scripts.build_calibrator`
  Expected: existing calibrator output **plus** `Saved market blender (a=… model, b=… market, c=…)`. Sanity-check: `b > 0` (market carries signal) and `a < 1` (model shrunk). If it prints "NOT built", Step 1 didn't capture enough lines.
- [ ] **Step 3:** Smoke-test locally: launch the app, confirm the headline reads "Consensus", `/archive` shows Consensus vs Model columns, and a game with no captured line still renders (falls back to model).
- [ ] **Step 4:** Publish: upload `data/model_market_blender.pkl` to the `latest` release (`gh release upload latest data/model_market_blender.pkl --clobber -R jackleh/MLB-Predictions`); refresh `data-cache` (`bash scripts/upload_data_release.sh`) so `data/market_odds/` ships.
- [ ] **Step 5:** Commit nothing (artifacts only); confirm `pytest -q` green on the branch before opening the PR.

---

## Self-Review

**Spec coverage:**
- Learned logit blend → Tasks 1, 5. ✓
- De-vigged market prob → Task 2. ✓
- Historical market lines as load-bearing input (store + backfill) → Tasks 3, 4. ✓
- Fit on walk-forward → Task 5 (reuses `build_calibrator`). ✓
- Blend replaces model line as headline; model hidden → Tasks 6, 8. ✓
- Grade the blended line; show Consensus vs Model → Tasks 7, 8. ✓
- Graceful fallback (no blender / no line) → Tasks 1, 6 (tests). ✓
- Ship via releases → Tasks 9, 11. ✓
- Docs → Task 10. ✓

**Placeholder scan:** Task 5 Step 3 and Task 7 Step 3 say "adapt to the actual accumulator" / "read the current body" — these are unavoidable because they modify existing code whose exact local variable names must be read first; the surrounding code to add is given in full. No TBD/TODO logic gaps.

**Type consistency:** `MarketBlender(a,b,c)` / `blend_pct(blender, model_pct, market_pct)` / `devig_home_prob(ml_home, ml_away)` / `load_market_odds(data_dir, date) -> {gid: {ml_home, ml_away, captured_at}}` / `_blend_core(core, market_home_prob)` are used consistently across Tasks 1–8. `consensus_accuracy`/`model_accuracy` introduced in Task 7 and consumed in Task 8 match.

**Open risk to verify during execution:** ESPN historical moneyline coverage for 2024–2025 may be sparse; if `build_blender` returns None or fits on too few games, the blender is fit on the 2026 window only (still thousands of games) — acceptable, and the app degrades to the calibrated model line where no line exists.
