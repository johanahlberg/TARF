"""The design brief's headline goal: TARF delta/gamma read off the PDE grid must be far smoother under
a spot sweep than bump-and-revalue, because each revalue re-lands the spot on a different grid.
"""

import numpy as np

from tarf_rslv import SingleRegimeLocalVolModel, SingleRegimePricer, TARFAccumulator

_FIXINGS = [(i + 1) * 30.0 / 365.0 for i in range(12)]


def _tarf() -> TARFAccumulator:
    return TARFAccumulator(
        target_level=0.30,
        strike=1.0,
        fixing_dates=_FIXINGS,
        notional1=1.0,
        notional2=1.5,
        barrier=0.9,
        target_adjustment=2,
    )


def _pricer(spot: float) -> SingleRegimePricer:
    model = SingleRegimeLocalVolModel(spot=spot, rate=0.0, dividend_yield=0.0, local_vol=0.12, strike=1.0)
    return SingleRegimePricer(model, num_spot=301, num_target=70, n_steps=300)


def test_on_grid_delta_is_much_smoother_than_bump_and_revalue():
    tarf = _tarf()
    spots = np.linspace(1.00, 1.10, 15)

    on_grid = np.array([_pricer(float(s)).tarf_greeks(tarf, _FIXINGS[-1])["delta"] for s in spots])

    bump = []
    for s in spots:
        h = 1e-3 * s
        up = _pricer(float(s + h)).price_tarf(tarf, _FIXINGS[-1])
        dn = _pricer(float(s - h)).price_tarf(tarf, _FIXINGS[-1])
        bump.append((up - dn) / (2.0 * h))
    bump = np.array(bump)

    # roughness = std of the discrete third difference (curvature of the delta curve)
    on_grid_roughness = np.std(np.diff(on_grid, 2))
    bump_roughness = np.std(np.diff(bump, 2))
    assert on_grid_roughness < 0.1 * bump_roughness


def test_on_grid_greeks_are_finite_and_signed_correctly():
    tarf = _tarf()
    g = _pricer(1.05).tarf_greeks(tarf, _FIXINGS[-1])
    assert np.isfinite(g["delta"]) and np.isfinite(g["gamma"]) and np.isfinite(g["vega"])
    # A call-style accumulator is long the underlying.
    assert g["delta"] > 0.0
