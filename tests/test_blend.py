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
