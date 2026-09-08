"""Two-stage calibration: a well-posed per-regime smile fit, and a switching-rate fit against a
forward-variance target. Both are checked by synthetic round-trip recovery.
"""

import numpy as np
import pytest

from tarf_rslv import (
    build_generator,
    calibrate_regime_smiles,
    calibrate_switching_rate,
    model_forward_variance,
    regime_probabilities,
)
from tarf_rslv.calibration import _mixture_vol


def test_regime_smile_fit_reproduces_market_smile():
    weights = np.array([0.25, 0.5, 0.25])
    k = np.linspace(-0.3, 0.3, 9)
    market = _mixture_vol(k, atm=0.185, skew=-0.05, curv=0.30, spread=0.05, weights=weights)

    result = calibrate_regime_smiles(k, market, weights, regime_spread=0.05)

    assert result["success"]
    assert result["fit_error"] < 1e-4
    assert result["base"]["atm_vol"] == pytest.approx(0.185, abs=1e-3)
    assert result["base"]["skew"] == pytest.approx(-0.05, abs=1e-3)
    assert len(result["regimes"]) == 3
    lows = [r["local_vol"] for r in result["regimes"]]
    assert lows[0] < lows[1] < lows[2]  # low / mid / high vol regimes
    for regime in result["regimes"]:
        assert regime["local_vol"] > 0.0


def test_regime_smile_fit_is_stable_under_small_market_perturbations():
    weights = np.array([0.3, 0.4, 0.3])
    k = np.linspace(-0.25, 0.25, 11)
    base = _mixture_vol(k, atm=0.20, skew=-0.03, curv=0.15, spread=0.04, weights=weights)

    rng = np.random.default_rng(0)
    atm_levels = []
    for _ in range(8):
        bumped = base + rng.normal(scale=2e-4, size=base.shape)
        atm_levels.append(calibrate_regime_smiles(k, bumped, weights, regime_spread=0.04)["base"]["atm_vol"])
    # a 2bp market wiggle must not move the fitted ATM level by more than a few bp
    assert np.std(atm_levels) < 5e-4


def test_switching_rate_round_trip_recovery():
    pi0 = np.array([0.05, 0.90, 0.05])          # currently in the mid-vol regime
    stationary = np.array([0.25, 0.50, 0.25])   # long-run mix
    regime_vols = np.array([0.12, 0.20, 0.32])

    for true_rate in (0.75, 2.5, 6.0):
        target = model_forward_variance(pi0, build_generator(stationary, true_rate), regime_vols, 0.5, 1.5)
        calibrated = calibrate_switching_rate(pi0, stationary, regime_vols, 0.5, 1.5, target)
        assert calibrated["switch_rate"] == pytest.approx(true_rate, rel=1e-4)
        np.testing.assert_allclose(calibrated["q"].sum(axis=1), 0.0, atol=1e-12)


def test_switching_rate_requires_non_stationary_start():
    pi = np.array([0.25, 0.5, 0.25])
    with pytest.raises(ValueError):
        calibrate_switching_rate(pi, pi, [0.1, 0.2, 0.3], 0.5, 1.5, 0.01)


def test_generator_has_prescribed_stationary_distribution():
    stationary = np.array([0.2, 0.5, 0.3])
    q = build_generator(stationary, switch_rate=3.0)
    # after a long time any start converges to the stationary distribution
    far = regime_probabilities([1.0, 0.0, 0.0], q, 50.0)
    np.testing.assert_allclose(far, stationary, atol=1e-6)
