import numpy as np

from tarf_rslv import (
    SingleRegimeLocalVolModel,
    SingleRegimePricer,
    calibrate_regime_smile,
    fit_switching_matrix,
)


def test_regime_smile_calibration_returns_sensible_params():
    strikes = np.array([90.0, 100.0, 110.0])
    vols = np.array([0.18, 0.2, 0.24])

    params = calibrate_regime_smile(strikes=strikes, implied_vols=vols)

    assert set(params.keys()) >= {"atm_vol", "skew", "curvature", "fit_error"}
    assert params["atm_vol"] > 0.0
    assert np.isfinite(params["fit_error"])


def test_switching_matrix_fit_is_conservative():
    weights = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    q = fit_switching_matrix(weights)

    assert q.shape == (3, 3)
    np.testing.assert_allclose(q.sum(axis=1), 0.0, atol=1e-10)
    assert np.all(np.diag(q) <= 0.0)


def test_single_regime_greeks_are_finite():
    model = SingleRegimeLocalVolModel(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        local_vol=0.2,
        strike=100.0,
    )
    pricer = SingleRegimePricer(model=model)

    greeks = pricer.compute_greeks(strike=100.0, maturity=1.0)

    assert set(greeks.keys()) >= {"delta", "gamma", "vega"}
    assert np.isfinite(greeks["delta"])
    assert np.isfinite(greeks["gamma"])
    assert np.isfinite(greeks["vega"])
