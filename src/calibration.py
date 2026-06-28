"""Win-probability calibration.

The dashboard's win% is the Monte-Carlo output of the run regressors (the share
of simulated games the home team wins). On unseen games those raw probabilities
are systematically *overconfident* — when the simulation says "72%", teams in
that bucket actually win closer to ~62%. Compared against an efficient
sportsbook line that gap manufactures large, fake "edges".

This module corrects the headline probability with a Platt (logistic-on-logit)
calibrator fit on **walk-forward** predictions — honest, out-of-sample pairs of
(raw sim win%, actual outcome) produced by `scripts/build_calibrator.py`. The
map is monotonic and passes through 0.5, so it never flips which side is
favored (the winner pick and grading are unchanged); it only pulls overconfident
probabilities back toward reality, which collapses the inflated edges.

If no calibrator file is present the functions degrade to the identity, so the
app and tests run unchanged without one.
"""
import math
from pathlib import Path

import joblib

CALIBRATOR_FILE = "model_calibrator.pkl"              # win% calibrator
INNING_CALIBRATOR_FILE = "model_inning_calibrator.pkl"  # legacy per-inning P(score) calibrator
INNING_DIST_CALIBRATOR_FILE = "model_inning_dist_calibrator.pkl"  # 3-class P(0/1/2+) calibrator
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class PlattCalibrator:
    """Monotonic prob->prob map: calibrated = sigmoid(a*logit(p) + b).

    a < 1 shrinks confidence toward 0.5 (the overconfidence fix). a == 1, b == 0
    is the identity. Always passes a probability of 0.5 through unchanged when
    b == 0, and is monotonically increasing for a > 0, so it never changes which
    side is favored.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def __call__(self, p: float) -> float:
        return _sigmoid(self.a * _logit(p) + self.b)


def fit(raw_probs, outcomes) -> PlattCalibrator:
    """Fit a temperature-scaling calibrator on (raw win prob, 0/1 outcome) pairs.

    Logistic regression on the single feature logit(raw_prob) with **no
    intercept** (b == 0), so calibrated = sigmoid(a*logit(p)). Forcing b == 0
    pins the curve through 0.5: it can only rescale confidence (a < 1 shrinks the
    overconfidence), never shift which side is favored. That keeps the winner
    pick — and every graded historical result — identical to the raw model.
    (A fitted intercept picks up a small home-field bias but would flip
    near-coinflip picks, which we deliberately avoid.)
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x = np.array([[_logit(float(p))] for p in raw_probs])
    y = np.asarray(outcomes, dtype=int)
    lr = LogisticRegression(C=1e6, solver="lbfgs", fit_intercept=False)
    lr.fit(x, y)
    return PlattCalibrator(a=float(lr.coef_[0][0]), b=0.0)


def save(cal: PlattCalibrator, data_dir, filename: str = CALIBRATOR_FILE) -> Path:
    path = Path(data_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"a": cal.a, "b": cal.b}, path)
    return path


def load(data_dir, filename: str = CALIBRATOR_FILE) -> PlattCalibrator | None:
    """Load a calibrator for `data_dir`, or None if none has been built.

    Safety: this pkl is written by build_calibrator.py in this codebase, never
    sourced from user input or network. Joblib is acceptable here.
    """
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    d = joblib.load(path)
    return PlattCalibrator(a=d["a"], b=d["b"])


def calibrate_pct(cal: PlattCalibrator | None, home_win_pct: float) -> float:
    """Calibrate a home win *percentage* (0-100). Identity when cal is None."""
    if cal is None or home_win_pct is None:
        return home_win_pct
    return round(cal(home_win_pct / 100.0) * 100.0, 1)


class MulticlassInningCalibrator:
    """Calibrate a per-inning 3-class run distribution [P0, P1, P2+].

    Holds one one-vs-rest Platt map per class; at apply time each class
    probability is mapped independently then the three are renormalized to sum to
    1. Near-identity in practice (the inning classifier is already well
    calibrated), applied for consistency/robustness like the win% map.
    """

    def __init__(self, maps):
        self.maps = list(maps)  # 3 PlattCalibrators


def fit_multiclass(prob_rows, labels) -> MulticlassInningCalibrator:
    """Fit a one-vs-rest calibrator per class from walk-forward pairs.

    prob_rows: iterable of [p0, p1, p2] predicted probabilities (fractions).
    labels:    iterable of the actual bucket (0/1/2).
    """
    import numpy as np
    P = np.asarray(prob_rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    maps = [fit(P[:, c], (y == c).astype(int)) for c in range(3)]
    return MulticlassInningCalibrator(maps)


def calibrate_dist(mc: "MulticlassInningCalibrator | None", dist_pct):
    """Calibrate a [P0, P1, P2+] distribution in percent. Identity when mc None."""
    if mc is None:
        return [round(float(p), 1) for p in dist_pct]
    raw = [mc.maps[c](dist_pct[c] / 100.0) for c in range(3)]
    s = sum(raw) or 1.0
    return [round(p / s * 100.0, 1) for p in raw]


def save_multiclass(mc: MulticlassInningCalibrator, data_dir,
                    filename: str = INNING_DIST_CALIBRATOR_FILE) -> Path:
    path = Path(data_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"maps": [{"a": m.a, "b": m.b} for m in mc.maps]}, path)
    return path


def load_multiclass(data_dir,
                    filename: str = INNING_DIST_CALIBRATOR_FILE) -> "MulticlassInningCalibrator | None":
    """Load the 3-class inning calibrator, or None if not built.

    Safety: written by build_calibrator.py in this codebase, never from user
    input or network. Joblib is acceptable here.
    """
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    d = joblib.load(path)
    return MulticlassInningCalibrator([PlattCalibrator(a=m["a"], b=m["b"]) for m in d["maps"]])
