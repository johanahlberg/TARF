"""Barrier / touch pricing and the regime calibration from one-touch quotes."""

import dataclasses

import numpy as np
import pytest
from scipy.stats import norm

from tarf_rslv import (
    FXVolSurface,
    OneTouchQuote,
    RegimeBarrierPricer,
    SingleRegimeLocalVolModel,
    SmileQuotes,
    TARFAccumulator,
    calibrate_regime_model,
)
from tarf_rslv.calibration_surface import _generator


class _FlatModel:
    """Single flat-vol regime, wrapped for RegimeBarrierPricer."""

    def __init__(self, spot, rate, div, vol):
        self.spot = spot
        self.rate = rate
        self.dividend_yield = div
        self.q = np.zeros((1, 1))
        self.regime_weights = np.array([1.0])
        self.time_varying = False
        self.regimes = [SingleRegimeLocalVolModel(spot=spot, rate=rate, dividend_yield=div, local_vol=vol, strike=spot)]


def _rr_one_touch(spot, barrier, r, carry, vol, t, eta):
    """Rubinstein-Reiner cash-at-hit one-touch, rebate 1. eta = +1 down, -1 up."""
    mu = (carry - 0.5 * vol * vol) / (vol * vol)
    lam = np.sqrt(mu * mu + 2.0 * r / (vol * vol))
    z = np.log(barrier / spot) / (vol * np.sqrt(t)) + lam * vol * np.sqrt(t)
    return (barrier / spot) ** (mu + lam) * norm.cdf(eta * z) \
        + (barrier / spot) ** (mu - lam) * norm.cdf(eta * (z - 2.0 * lam * vol * np.sqrt(t)))


# --------------------------------------------------------------------------------------------------
# pricer
# --------------------------------------------------------------------------------------------------
def test_one_touch_matches_closed_form():
    m = _FlatModel(1.0, 0.025, 0.01, 0.10)
    p = RegimeBarrierPricer(m)
    carry = m.rate - m.dividend_yield
    for x in (0.03, 0.05, 0.10):
        fd_dn = p.one_touch(1.0 - x, 1.0, direction="down", payment="hit")
        cf_dn = _rr_one_touch(1.0, 1.0 - x, m.rate, carry, 0.10, 1.0, 1.0)
        assert abs(fd_dn - cf_dn) < 3e-3           # < 30 bp of payout
        fd_up = p.one_touch(1.0 + x, 1.0, direction="up", payment="hit")
        cf_up = _rr_one_touch(1.0, 1.0 + x, m.rate, carry, 0.10, 1.0, -1.0)
        assert abs(fd_up - cf_up) < 3e-3


def test_touch_parity_identities():
    m = _FlatModel(1.0, 0.02, 0.005, 0.09)
    p = RegimeBarrierPricer(m)
    df = float(np.exp(-m.rate * 1.0))
    for b in (0.90, 0.85):
        ot = p.one_touch(b, 1.0, direction="down", payment="expiry")
        nt = p.no_touch(b, 1.0, direction="down")
        assert abs(ot + nt - df) < 1e-4            # exactly one of touched / not-touched pays at T

    ko = p.knock_out(1.0, 0.85, 1.0, option_type="call", direction="down")
    ki = p.knock_in(1.0, 0.85, 1.0, option_type="call", direction="down")
    assert abs(ko + ki - p._vanilla(1.0, 1.0, "call")) < 1e-6


def test_one_touch_is_monotone_in_barrier():
    m = _FlatModel(1.0, 0.02, 0.0, 0.10)
    p = RegimeBarrierPricer(m)
    prices = [p.one_touch(b, 1.0, direction="down", payment="hit") for b in (0.80, 0.85, 0.90, 0.95)]
    assert all(np.diff(prices) > 0) and 0.0 < prices[0] and prices[-1] < 1.0


# --------------------------------------------------------------------------------------------------
# calibration from one-touch quotes
# --------------------------------------------------------------------------------------------------
def _surface():
    quotes = [
        SmileQuotes(1 / 12, 0.070, -0.004, 0.0015, -0.008, 0.005),
        SmileQuotes(0.25, 0.072, -0.006, 0.0018, -0.011, 0.006),
        SmileQuotes(0.50, 0.074, -0.007, 0.0020, -0.013, 0.007),
        SmileQuotes(1.00, 0.076, -0.008, 0.0022, -0.015, 0.008),
    ]
    return FXVolSurface(spot=0.81, smiles=quotes, domestic_zero=lambda t: 0.011, foreign_zero=lambda t: 0.046)


def test_barrier_calibration_recovers_the_regime_structure():
    surface = _surface()
    true, _ = calibrate_regime_model(
        surface, n_regimes=3, level_spread=0.22, skew_spread=0.12, calibrate_q=False,
        num_x=401, steps_per_year=300, dupire_nt=21, dupire_nx=101,
    )
    true = dataclasses.replace(true, q=_generator(4.0, np.array([0.25, 0.5, 0.25])))
    pr = RegimeBarrierPricer(true)
    specs = [(0.75, 0.5, "down"), (0.72, 1.0, "down"), (0.70, 1.0, "down"),
             (0.88, 0.5, "up"), (0.90, 1.0, "up"), (0.93, 1.0, "up")]
    quotes = [
        OneTouchQuote(barrier=b, maturity=T, direction=d,
                      market_price=pr.one_touch(b, T, direction=d, payment="hit"))
        for b, T, d in specs
    ]

    model, report = calibrate_regime_model(
        surface, n_regimes=3, barrier_quotes=quotes, n_barrier_rounds=2,
        num_x=401, steps_per_year=300, dupire_nt=21, dupire_nx=101,
    )
    assert report.n_barrier_quotes == 6
    assert report.barrier_rms_price_error < 3e-2                 # < 3% of payout
    # the overall vol dispersion is well identified by touches
    assert abs(report.level_spread - 0.22) < 0.06
    assert model.n_regimes == 3
    tarf = TARFAccumulator(
        target_level=0.10, strike=0.82, fixing_dates=tuple((k + 1) / 12 for k in range(12)),
        is_call_option=False, notional1=1.0, notional2=2.0, target_adjustment=0,
    )
    assert np.isfinite(model.price_tarf(tarf, 1.0, num_spot=101, num_target=50, n_steps=60))
