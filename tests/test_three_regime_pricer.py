import numpy as np

from tarf_rslv import (
    SingleRegimeLocalVolModel,
    RegimeSwitchingLocalVolModel,
    TARFAccumulator,
    ThreeRegimePricer,
    SingleRegimePricer,
)


def test_three_regime_price_is_finite_and_weighted():
    model = RegimeSwitchingLocalVolModel(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
    )
    pricer = ThreeRegimePricer(model=model, regime_weights=np.array([0.2, 0.5, 0.3]))

    tarf = TARFAccumulator(
        target_level=20.0,
        strike=100.0,
        fixing_dates=[0.5, 1.0],
    )

    price = pricer.price_tarf(tarf=tarf, maturity=1.0)
    assert np.isfinite(price)
    assert price > 0.0


def test_three_regime_european_price_matches_weighted_average():
    model = RegimeSwitchingLocalVolModel(spot=100.0, rate=0.03, dividend_yield=0.0)
    weights = np.array([0.3, 0.4, 0.3])
    pricer = ThreeRegimePricer(model=model, regime_weights=weights)

    european = pricer.price_european(strike=100.0, maturity=1.0)

    assert np.isfinite(european)
    assert european > 0.0
    assert european < 100.0


def test_three_regime_coupled_tarf_price_is_finite():
    model = RegimeSwitchingLocalVolModel(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
    )
    pricer = ThreeRegimePricer(model=model, regime_weights=np.array([0.2, 0.5, 0.3]))
    tarf = TARFAccumulator(
        target_level=20.0,
        strike=100.0,
        fixing_dates=[0.5, 1.0],
    )

    price = pricer.price_tarf_coupled(tarf=tarf, maturity=1.0)

    assert np.isfinite(price)
    assert price > 0.0


def test_three_regime_coupled_tarf_matches_single_regime_when_regimes_are_identical():
    model = RegimeSwitchingLocalVolModel(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        regimes=[
            SingleRegimeLocalVolModel(spot=100.0, rate=0.03, dividend_yield=0.0, local_vol=0.2, strike=100.0),
            SingleRegimeLocalVolModel(spot=100.0, rate=0.03, dividend_yield=0.0, local_vol=0.2, strike=100.0),
            SingleRegimeLocalVolModel(spot=100.0, rate=0.03, dividend_yield=0.0, local_vol=0.2, strike=100.0),
        ],
        q=np.array(
            [
                [-0.6, 0.3, 0.3],
                [0.2, -0.4, 0.2],
                [0.1, 0.1, -0.2],
            ],
            dtype=float,
        ),
    )
    pricer = ThreeRegimePricer(
        model=model,
        regime_weights=np.array([0.2, 0.5, 0.3]),
        num_spot=121,
        num_target=80,
        n_steps=80,
    )

    tarf = TARFAccumulator(
        target_level=20.0,
        strike=100.0,
        fixing_dates=[0.5, 1.0],
    )

    expected = SingleRegimePricer(model=model.regimes[0]).price_tarf(tarf=tarf, maturity=1.0)
    coupled = pricer.price_tarf_coupled(tarf=tarf, maturity=1.0)

    assert np.isclose(coupled, expected, rtol=1e-8, atol=1e-8)
