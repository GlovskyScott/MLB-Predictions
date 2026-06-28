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
