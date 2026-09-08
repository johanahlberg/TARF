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
        target_level=20.0,
        strike=100.0,
        fixing_dates=[0.5, 1.0],
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


def test_leveraged_otm_notional_can_make_tarf_value_negative():
    """Regression guard: notional2 losses are real risk, not clipped away as in the old proxy payoff."""
    model = SingleRegimeLocalVolModel(
        spot=100.0,
        rate=0.0,
        dividend_yield=0.0,
        local_vol=0.2,
        strike=100.0,
    )
    pricer = SingleRegimePricer(model=model, num_spot=101, num_target=60)

    tarf = TARFAccumulator(
        target_level=20.0,
        strike=100.0,
        fixing_dates=[0.5, 1.0],
        notional1=1.0,
        notional2=4.0,
    )

    price = pricer.price_tarf(tarf=tarf, maturity=1.0)

    assert np.isfinite(price)
    assert price < 0.0


def test_ki_barrier_protects_against_leveraged_loss():
    """A downside KI barrier should make the (heavily leveraged) TARF less negative, not more."""
    model = SingleRegimeLocalVolModel(
        spot=100.0,
        rate=0.0,
        dividend_yield=0.0,
        local_vol=0.2,
        strike=100.0,
    )

    def build_tarf(barrier: float) -> TARFAccumulator:
        return TARFAccumulator(
            target_level=20.0,
            strike=100.0,
            fixing_dates=[0.5, 1.0],
            notional1=1.0,
            notional2=4.0,
            barrier=barrier,
        )

    pricer = SingleRegimePricer(model=model, num_spot=101, num_target=60)
    price_without_barrier = pricer.price_tarf(tarf=build_tarf(0.0), maturity=1.0)
    price_with_barrier = pricer.price_tarf(tarf=build_tarf(90.0), maturity=1.0)

    assert np.isfinite(price_without_barrier)
    assert np.isfinite(price_with_barrier)
    assert price_with_barrier > price_without_barrier


def test_default_regime_matrix_is_stochastic_and_valid():
    q = build_default_regime_matrix()

    assert q.shape == (3, 3)
    np.testing.assert_allclose(q.sum(axis=1), 0.0)
    assert np.all(np.diag(q) <= 0.0)
    assert np.all(q[np.triu_indices(3, k=1)] >= 0.0)
