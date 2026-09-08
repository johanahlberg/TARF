from datetime import date

import numpy as np

from tarf_rslv import FallbackDenominatedValue, price_tarf, tarf_model


class FakeDenominatedValue:
    def __init__(self, number, unit):
        self._number = number
        self._unit = unit

    def Number(self):
        return self._number

    def Unit(self):
        return self._unit


def _market_parameters():
    return {
        "spot": 100.0,
        "strike": 100.0,
        "target_level": 100.0,
        "coupon": 0.05,
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
    assert result["result"].number > 0.0


def test_front_arena_pure_function_is_serializable_and_deterministic():
    parameters = _market_parameters()
    first = price_tarf(**parameters)
    second = price_tarf(**parameters)

    assert np.isfinite(first)
    assert first == second