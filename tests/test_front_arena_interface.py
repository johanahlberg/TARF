from datetime import date

import numpy as np

from tarf_rslv import (
    DEFAULT_REGIME_DELTAS,
    FallbackDenominatedValue,
    market_data_from_front_arena,
    price_tarf,
    tarf_model,
    tarf_model_from_market_data,
)


class FakeDenominatedValue:
    def __init__(self, number, unit):
        self._number = number
        self._unit = unit

    def Number(self):
        return self._number

    def Unit(self):
        return self._unit


class FakeIrCurveInformation:
    """Stand-in for FIrCurveInformation; Rate() mimics the continuous Act/365 spot rate."""

    def __init__(self, flat_rate):
        self._flat_rate = flat_rate

    def Rate(self, start_day, end_day):
        assert end_day > start_day
        return self._flat_rate


class FakeMalzParametricVolatilityInformation:
    """Stand-in for FMalzParametricVolatilityInformation with a linear skew in delta."""

    def __init__(self, atm_vol, skew=0.0):
        self._atm_vol = atm_vol
        self._skew = skew

    def Value(self, expiry_date, delta, foreign_rate, domestic_rate):
        return self._atm_vol + self._skew * (delta - 0.5)


def _market_parameters():
    return {
        "spot": 100.0,
        "strike": 100.0,
        "target_level": 20.0,
        "fixing_times": [0.5, 1.0],
        "maturity": 1.0,
        "domestic_rate": 0.03,
        "foreign_rate": 0.0,
        "regime_volatilities": [0.15, 0.20, 0.30],
        "regime_weights": [0.2, 0.5, 0.3],
        "q_matrix": [[-0.5, 0.25, 0.25], [0.25, -0.5, 0.25], [0.25, 0.25, -0.5]],
    }


def test_front_arena_wrapper_returns_mandatory_result_key():
    parameters = _market_parameters()
    parameters["spot_value"] = FakeDenominatedValue(parameters.pop("spot"), "EUR")
    parameters["strike_value"] = FakeDenominatedValue(parameters.pop("strike"), "EUR")
    result = tarf_model(valuation_date=date(2026, 9, 8), **parameters)

    assert set(result) == {"result"}
    assert isinstance(result["result"], FallbackDenominatedValue)
    assert result["result"].unit == "EUR"
    assert result["result"].value_date == date(2026, 9, 8)
    assert np.isfinite(result["result"].number)


def test_front_arena_pure_function_is_serializable_and_deterministic():
    parameters = _market_parameters()
    first = price_tarf(**parameters)
    second = price_tarf(**parameters)

    assert np.isfinite(first)
    assert first == second


def test_market_data_from_front_arena_extracts_rates_and_regime_vols():
    domestic_curve = FakeIrCurveInformation(0.03)
    foreign_curve = FakeIrCurveInformation(0.01)
    vol_surface = FakeMalzParametricVolatilityInformation(atm_vol=0.20, skew=0.10)

    market = market_data_from_front_arena(
        valuation_date=date(2026, 9, 8),
        maturity_date=date(2027, 9, 8),
        domestic_curve=domestic_curve,
        foreign_curve=foreign_curve,
        vol_surface=vol_surface,
    )

    assert market["domestic_rate"] == 0.03
    assert market["foreign_rate"] == 0.01
    put_delta, atm_delta, call_delta = DEFAULT_REGIME_DELTAS
    expected = [0.20 + 0.10 * (delta - 0.5) for delta in (put_delta, atm_delta, call_delta)]
    assert np.allclose(market["regime_volatilities"], expected)


def test_market_data_from_front_arena_rejects_non_positive_tenor():
    domestic_curve = FakeIrCurveInformation(0.03)
    foreign_curve = FakeIrCurveInformation(0.01)
    vol_surface = FakeMalzParametricVolatilityInformation(atm_vol=0.20)

    try:
        market_data_from_front_arena(
            valuation_date=date(2026, 9, 8),
            maturity_date=date(2026, 9, 8),
            domestic_curve=domestic_curve,
            foreign_curve=foreign_curve,
            vol_surface=vol_surface,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-positive maturity tenor")


def test_tarf_model_from_market_data_returns_mandatory_result_key():
    domestic_curve = FakeIrCurveInformation(0.03)
    foreign_curve = FakeIrCurveInformation(0.0)
    vol_surface = FakeMalzParametricVolatilityInformation(atm_vol=0.20, skew=0.08)

    result = tarf_model_from_market_data(
        valuation_date=date(2026, 9, 8),
        maturity_date=date(2027, 9, 8),
        spot_value=FakeDenominatedValue(100.0, "EUR"),
        strike_value=FakeDenominatedValue(100.0, "EUR"),
        target_level=20.0,
        fixing_times=[0.5, 1.0],
        domestic_curve=domestic_curve,
        foreign_curve=foreign_curve,
        vol_surface=vol_surface,
        regime_weights=[0.2, 0.5, 0.3],
        q_matrix=[[-0.5, 0.25, 0.25], [0.25, -0.5, 0.25], [0.25, 0.25, -0.5]],
    )

    assert set(result) == {"result"}
    assert isinstance(result["result"], FallbackDenominatedValue)
    assert result["result"].unit == "EUR"
    assert result["result"].value_date == date(2026, 9, 8)
    assert np.isfinite(result["result"].number)