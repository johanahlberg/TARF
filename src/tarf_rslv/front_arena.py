"""Front Arena AEF boundary for the TARF valuation engine.

The functions in this module deliberately accept only scalar values, lists, and dates. That keeps the
valuation state-free and compatible with AEF distributed-calculation serialization. Front Arena-specific
ACM objects are created only at the final result boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from .model import RegimeSwitchingLocalVolModel, SingleRegimeLocalVolModel
from .product import TARFAccumulator
from .solver import ThreeRegimePricer


@dataclass(frozen=True)
class FallbackDenominatedValue:
    """Local substitute used by tests when the Front Arena ``acm`` module is unavailable."""

    number: float
    unit: str
    value_date: date


def _make_denominated_value(value: float, currency: str, value_date: date) -> object:
    """Create the ACM result object when running inside PRIME, with a test fallback outside PRIME."""
    try:
        import acm  # type: ignore
    except ImportError:
        return FallbackDenominatedValue(float(value), currency, value_date)
    return acm.DenominatedValue(float(value), currency, value_date)


def _denominated_number(value: object, name: str) -> float:
    number = getattr(value, "Number", None)
    if not callable(number):
        raise TypeError(f"{name} must provide a Number() method")
    return float(number())


def _denominated_unit(value: object, name: str) -> str:
    unit = getattr(value, "Unit", None)
    if not callable(unit):
        raise TypeError(f"{name} must provide a Unit() method")
    return unit()


def _validate_vector(values: Sequence[float], name: str, size: int = 3) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _build_model(
    spot: float,
    strike: float,
    domestic_rate: float,
    foreign_rate: float,
    regime_volatilities: Sequence[float],
    q_matrix: Sequence[Sequence[float]],
) -> RegimeSwitchingLocalVolModel:
    volatilities = _validate_vector(regime_volatilities, "regime_volatilities")
    if np.any(volatilities <= 0.0):
        raise ValueError("regime_volatilities must be positive")

    q = np.asarray(q_matrix, dtype=float)
    if q.shape != (3, 3):
        raise ValueError("q_matrix must have shape 3x3")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_matrix must contain only finite values")
    if not np.allclose(q.sum(axis=1), 0.0, atol=1e-10):
        raise ValueError("q_matrix rows must sum to zero")
    if np.any(q - np.diag(np.diag(q)) < 0.0):
        raise ValueError("q_matrix off-diagonal transition rates must be non-negative")

    regimes = [
        SingleRegimeLocalVolModel(
            spot=float(spot),
            rate=float(domestic_rate),
            dividend_yield=float(foreign_rate),
            local_vol=float(volatility),
            strike=float(strike),
        )
        for volatility in volatilities
    ]
    return RegimeSwitchingLocalVolModel(
        spot=float(spot),
        rate=float(domestic_rate),
        dividend_yield=float(foreign_rate),
        regimes=regimes,
        q=q,
    )


def price_tarf(
    spot: float,
    strike: float,
    target_level: float,
    coupon: float,
    fixing_times: Sequence[float],
    maturity: float,
    domestic_rate: float,
    foreign_rate: float,
    regime_volatilities: Sequence[float],
    regime_weights: Sequence[float],
    q_matrix: Sequence[Sequence[float]],
) -> float:
    """Pure, state-free TARF valuation function used by the AEF wrapper."""
    weights = _validate_vector(regime_weights, "regime_weights")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("regime_weights must be non-negative and sum to 1")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")

    model = _build_model(
        spot=spot,
        strike=strike,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        regime_volatilities=regime_volatilities,
        q_matrix=q_matrix,
    )
    tarf = TARFAccumulator(
        target_level=float(target_level),
        coupon=float(coupon),
        fixing_dates=tuple(float(time) for time in fixing_times),
    )
    return ThreeRegimePricer(model=model, regime_weights=weights).price_tarf(tarf, float(maturity))


def tarf_model(
    valuation_date: date,
    spot_value: object,
    strike_value: object,
    target_level: float,
    coupon: float,
    fixing_times: Sequence[float],
    maturity: float,
    domestic_rate: float,
    foreign_rate: float,
    regime_volatilities: Sequence[float],
    regime_weights: Sequence[float],
    q_matrix: Sequence[Sequence[float]],
) -> dict[str, object]:
    """AEF-compatible wrapper; the mandatory return key is ``result``."""
    value = price_tarf(
        spot=_denominated_number(spot_value, "spot_value"),
        strike=_denominated_number(strike_value, "strike_value"),
        target_level=target_level,
        coupon=coupon,
        fixing_times=fixing_times,
        maturity=maturity,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        regime_volatilities=regime_volatilities,
        regime_weights=regime_weights,
        q_matrix=q_matrix,
    )
    return {"result": _make_denominated_value(value, _denominated_unit(strike_value, "strike_value"), valuation_date)}


ADFL_EXAMPLE = """\
[AEF PV&R]FObject:tarfModel =
    Definition=tarfModel(date valuationDate, denominatedvalue spotValue,
        denominatedvalue strikeValue,
        double targetLevel, double coupon, array(double) fixingTimes,
        double maturity, double domesticRate, double foreignRate,
        array(double) regimeVolatilities, array(double) regimeWeights,
        matrix(double) qMatrix): FDictionary
    Function=TarfValuation.tarf_model

[AEF PV&R]FInstrument:tarfValuationDescriptor = {
    theoreticalModelCall->"tarfTheoreticalModelCall"
}

[AEF PV&R]FInstrument:tarfTheoreticalModelCall = tarfModel(
    valuationDate, spotValue, strikeValue, targetLevel, coupon, fixingTimes,
    maturity, domesticRate, foreignRate, regimeVolatilities,
    regimeWeights, qMatrix);
"""
