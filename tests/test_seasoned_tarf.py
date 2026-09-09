"""Seasoned-deal handling: past fixings dropped, a valuation-date fixing kept, and the
finite-difference grid resolution exposed through the Front Arena wrapper."""

import numpy as np

from tarf_rslv import (
    RegimeSwitchingLocalVolModel,
    SingleRegimeLocalVolModel,
    TARFAccumulator,
    ThreeRegimePricer,
    build_default_regime_matrix,
    price_tarf,
)
from tarf_rslv.solver import _tarf_fixing_schedule


def _model():
    regimes = [
        SingleRegimeLocalVolModel(spot=0.81, rate=0.011, dividend_yield=0.046, local_vol=v, strike=0.82)
        for v in (0.077, 0.072, 0.071)
    ]
    return RegimeSwitchingLocalVolModel(
        spot=0.81, rate=0.011, dividend_yield=0.046, regimes=regimes, q=build_default_regime_matrix()
    )


def _tarf(fixing_dates, accumulated_value):
    return TARFAccumulator(
        target_level=0.10,
        strike=0.82,
        fixing_dates=tuple(sorted(fixing_dates)),
        is_call_option=False,
        notional1=1.0,
        notional2=2.0,
        target_adjustment=0,
        accumulated_value=accumulated_value,
    )


def _price(fixing_dates, accumulated_value, maturity, num_target=160):
    pricer = ThreeRegimePricer(
        _model(), [0.25, 0.5, 0.25], num_spot=161, num_target=num_target, n_steps=160
    )
    return pricer.price_tarf(_tarf(fixing_dates, accumulated_value), maturity)


# --------------------------------------------------------------------------------------------------
# _tarf_fixing_schedule
# --------------------------------------------------------------------------------------------------
def test_schedule_drops_past_fixings_and_keeps_valuation_date_fixing():
    tarf = _tarf([-0.25, -0.08, 0.0, 0.25, 0.5], accumulated_value=0.05)
    schedule = _tarf_fixing_schedule(tarf, maturity=0.5)

    # past fixings gone; the t=0 fixing kept and snapped to exactly 0.0; indices preserved
    assert schedule == [(2, 0.0), (3, 0.25), (4, 0.5)]


def test_schedule_snaps_tiny_negative_valuation_date_fixing_to_zero():
    tarf = _tarf([-1e-12, 0.25], accumulated_value=0.0)
    schedule = _tarf_fixing_schedule(tarf, maturity=0.25)

    assert schedule[0] == (0, 0.0)


def test_schedule_drops_fixings_beyond_maturity():
    tarf = _tarf([0.25, 0.5, 0.9], accumulated_value=0.0)
    assert _tarf_fixing_schedule(tarf, maturity=0.5) == [(0, 0.25), (1, 0.5)]


# --------------------------------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------------------------------
def test_past_fixings_do_not_change_the_price():
    future = [k / 12 for k in range(1, 8)]
    maturity = future[-1]

    without_history = _price(future, accumulated_value=0.062, maturity=maturity)
    with_history = _price([-0.3, -0.2, -0.1] + future, accumulated_value=0.062, maturity=maturity)

    assert np.isclose(without_history, with_history, rtol=0, atol=1e-9)


def test_valuation_date_fixing_is_priced_in():
    future = [k / 12 for k in range(1, 8)]
    maturity = future[-1]

    with_today = _price([0.0] + future, accumulated_value=0.062, maturity=maturity)
    without_today = _price([-1e-4] + future, accumulated_value=0.062, maturity=maturity)

    # spot (0.81) is below the strike (0.82): today's fixing is ITM, so keeping it is worth money
    assert with_today > without_today + 1e-3


def test_valuation_date_fixing_matches_treating_it_as_already_observed():
    future = [k / 12 for k in range(1, 8)]
    maturity = future[-1]
    today_intrinsic = 0.82 - 0.81  # ITM cashflow realised now, notional1 = 1

    kept = _price([0.0] + future, accumulated_value=0.062, maturity=maturity, num_target=320)
    observed = (
        _price([-1e-4] + future, accumulated_value=0.062 + today_intrinsic, maturity=maturity, num_target=320)
        + today_intrinsic
    )

    assert np.isclose(kept, observed, rtol=0, atol=2e-4)


# --------------------------------------------------------------------------------------------------
# grid resolution through the Front Arena wrapper
# --------------------------------------------------------------------------------------------------
def _fa_price(**overrides):
    params = dict(
        spot=0.81,
        strike=0.82,
        target_level=0.10,
        fixing_times=[0.0] + [k / 12 for k in range(1, 8)],
        maturity=7 / 12,
        domestic_rate=0.011,
        foreign_rate=0.046,
        regime_volatilities=[0.077, 0.072, 0.071],
        regime_weights=[0.25, 0.5, 0.25],
        q_matrix=build_default_regime_matrix().tolist(),
        is_call_option=False,
        notional1=1.0,
        notional2=2.0,
        target_adjustment=0,
        accumulated_value=0.062,
    )
    params.update(overrides)
    return price_tarf(**params)


def test_front_arena_wrapper_accepts_grid_resolution_and_converges():
    coarse = _fa_price(num_spot=81, num_target=60, n_steps=60)
    fine = _fa_price(num_spot=201, num_target=320, n_steps=200)

    assert np.isfinite(coarse) and np.isfinite(fine)
    assert abs(coarse - fine) < 5e-3  # same deal, just resolution

    # default call still works unchanged
    assert np.isfinite(_fa_price())
