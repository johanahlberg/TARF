import numpy as np

from tarf_rslv import (
    SingleRegimeLocalVolModel,
    TARFAccumulator,
    SingleRegimePricer,
    build_default_regime_matrix,
)


def test_single_regime_matches_trivial_regime_matrix():
    model = SingleRegimeLocalVolModel(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        local_vol=0.2,
        strike=100.0,
    )
    pricer = SingleRegimePricer(model=model, num_spot=80, num_target=60)

    tarf = TARFAccumulator(
        target_level=100.0,
        coupon=0.05,
        fixing_dates=[0.5, 1.0],
        trigger_mode="accumulation",
    )

    price = pricer.price_tarf(tarf=tarf, maturity=1.0)

    assert np.isfinite(price)
    assert price > 0.0
    assert price < model.spot * 1.5


def test_european_baseline_is_positive_and_reasonable():
    model = SingleRegimeLocalVolModel(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        local_vol=0.2,
        strike=100.0,
    )
    pricer = SingleRegimePricer(model=model, num_spot=120, num_target=60)

    price = pricer.price_european(strike=100.0, maturity=1.0)

    assert np.isfinite(price)
    assert 0.0 < price < 100.0


def test_default_regime_matrix_is_stochastic_and_valid():
    q = build_default_regime_matrix()

    assert q.shape == (3, 3)
    np.testing.assert_allclose(q.sum(axis=1), 0.0)
    assert np.all(np.diag(q) <= 0.0)
    assert np.all(q[np.triu_indices(3, k=1)] >= 0.0)
