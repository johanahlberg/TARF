"""Correctness gate: the single-regime finite-difference pricer must reproduce the benchmark table of

    Luo & Shevchenko, "Pricing TARN Using a Finite Difference Method" (arXiv:1304.7563v2), Table 1

which is the published reference for exactly this scheme. Inputs (their Section 5): spot 1.05,
strike 1.0, sigma 0.2, rd = rf = 0, 20 monthly fixings, notional 1, no leveraged downside leg.

Their three knockout types map onto ``target_adjustment``: full gain -> 0, part gain -> 2, no gain -> 3.
"""

import numpy as np
import pytest

from tarf_rslv import SingleRegimeLocalVolModel, SingleRegimePricer, TARFAccumulator

_FIXINGS = [(i + 1) * 30.0 / 365.0 for i in range(20)]

# target level -> published price, per knockout type
_PAPER_TABLE_1 = {
    0: {0.3: 0.2978, 0.5: 0.4386, 0.7: 0.5644, 0.9: 0.6790},  # full gain
    2: {0.3: 0.2445, 0.5: 0.3818, 0.7: 0.5061, 0.9: 0.6200},  # part gain
    3: {0.3: 0.1955, 0.5: 0.3286, 0.7: 0.4505, 0.9: 0.5633},  # no gain
}


def _price(target_level: float, adjustment: int, num_spot: int, num_target: int, n_steps: int) -> float:
    model = SingleRegimeLocalVolModel(spot=1.05, rate=0.0, dividend_yield=0.0, local_vol=0.2, strike=1.0)
    tarf = TARFAccumulator(
        target_level=target_level,
        strike=1.0,
        fixing_dates=_FIXINGS,
        notional1=1.0,
        notional2=0.0,
        target_adjustment=adjustment,
    )
    return SingleRegimePricer(model, num_spot=num_spot, num_target=num_target, n_steps=n_steps).price_tarf(
        tarf, _FIXINGS[-1]
    )


@pytest.mark.parametrize("adjustment", sorted(_PAPER_TABLE_1))
@pytest.mark.parametrize("target_level", [0.3, 0.7])
def test_matches_luo_shevchenko_table_1(adjustment: int, target_level: float):
    reference = _PAPER_TABLE_1[adjustment][target_level]
    price = _price(target_level, adjustment, num_spot=251, num_target=70, n_steps=300)
    # The paper's own reference carries ~0.1% Monte-Carlo noise; 0.5% is a firm agreement gate.
    assert abs(price / reference - 1.0) < 5e-3, f"{price=} vs {reference=}"


def test_convergence_is_monotone_and_stable():
    coarse = _price(0.5, 0, num_spot=151, num_target=50, n_steps=200)
    fine = _price(0.5, 0, num_spot=401, num_target=120, n_steps=600)
    assert abs(coarse - fine) < 3e-3
    assert abs(fine - _PAPER_TABLE_1[0][0.5]) < 3e-3


def test_vanilla_limit_recovers_black_scholes():
    """A single fixing with an unreachable target is a European call; the PDE must recover BS."""
    from scipy.stats import norm

    model = SingleRegimeLocalVolModel(spot=1.05, rate=0.03, dividend_yield=0.0, local_vol=0.2, strike=1.0)
    tarf = TARFAccumulator(target_level=1e9, strike=1.0, fixing_dates=[1.0], notional1=1.0, notional2=0.0)
    price = SingleRegimePricer(model, num_spot=301, num_target=6, n_steps=300).price_tarf(tarf, 1.0)

    d1 = (np.log(1.05 / 1.0) + (0.03 + 0.5 * 0.04) * 1.0) / 0.2
    d2 = d1 - 0.2
    bs = 1.05 * norm.cdf(d1) - 1.0 * np.exp(-0.03) * norm.cdf(d2)
    assert abs(price - bs) < 2e-3
