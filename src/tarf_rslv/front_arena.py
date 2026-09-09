"""Front Arena AEF boundary for the TARF valuation engine.

The functions in this module deliberately accept only scalar values, lists, and dates. That keeps the
valuation state-free and compatible with AEF distributed-calculation serialization. Front Arena-specific
ACM objects are created only at the final result boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

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


# 25-delta put / ATM / 25-delta call, used as proxies for the low/mid/high vol regimes.
DEFAULT_REGIME_DELTAS: tuple[float, float, float] = (-0.25, 0.5, 0.25)


def _curve_rate(curve: object, start_day: date, end_day: date, name: str) -> float:
    """Read a continuous Act/365 zero rate off an FIrCurveInformation via Rate(startDay, endDay)."""
    rate = getattr(curve, "Rate", None)
    if not callable(rate):
        raise TypeError(f"{name} must provide a Rate(startDay, endDay) method")
    return float(rate(start_day, end_day))


def _vol_surface_value(
    vol_surface: object,
    expiry_date: date,
    delta: float,
    foreign_rate: float,
    domestic_rate: float,
    name: str,
) -> float:
    """Read a smile volatility off an FMalzParametricVolatilityInformation via Value(expiry, delta, ...)."""
    value = getattr(vol_surface, "Value", None)
    if not callable(value):
        raise TypeError(f"{name} must provide a Value(expiryDate, delta, foreignRate, domesticRate) method")
    return float(value(expiry_date, delta, foreign_rate, domestic_rate))


def market_data_from_front_arena(
    valuation_date: date,
    maturity_date: date,
    domestic_curve: object,
    foreign_curve: object,
    vol_surface: object,
    regime_deltas: Sequence[float] = DEFAULT_REGIME_DELTAS,
) -> dict[str, object]:
    """Extract scalar rates/vols from FA market-data objects.

    This is the only place FIrCurveInformation and FMalzParametricVolatilityInformation objects are
    touched; the extracted floats keep everything downstream (price_tarf, tarf_model) state-free.
    """
    if maturity_date <= valuation_date:
        raise ValueError("maturity_date must be after valuation_date")

    domestic_rate = _curve_rate(domestic_curve, valuation_date, maturity_date, "domestic_curve")
    foreign_rate = _curve_rate(foreign_curve, valuation_date, maturity_date, "foreign_curve")

    deltas = _validate_vector(regime_deltas, "regime_deltas")
    regime_volatilities = [
        _vol_surface_value(vol_surface, maturity_date, float(delta), foreign_rate, domestic_rate, "vol_surface")
        for delta in deltas
    ]

    return {
        "domestic_rate": domestic_rate,
        "foreign_rate": foreign_rate,
        "regime_volatilities": regime_volatilities,
    }


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
    fixing_times: Sequence[float],
    maturity: float,
    domestic_rate: float,
    foreign_rate: float,
    regime_volatilities: Sequence[float],
    regime_weights: Sequence[float],
    q_matrix: Sequence[Sequence[float]],
    is_call_option: bool = True,
    notional1: Sequence[float] | float = 1.0,
    notional2: Sequence[float] | float = 1.0,
    barrier: float = 0.0,
    target_adjustment: int = 0,
    inverted_target: bool = False,
    accumulated_value: float = 0.0,
    num_spot: int = 121,
    num_target: int = 80,
    n_steps: int = 80,
) -> float:
    """Pure, state-free TARF valuation function used by the AEF wrapper.

    The payoff mirrors the udmcTRFPayoff ESL script (see ``TARFAccumulator``): ``notional1`` pays the
    accumulation leg while ITM (toward ``target_level``), ``notional2`` pays the leveraged loss while
    OTM (gated by the local, no-memory KI ``barrier``), and ``target_adjustment`` controls how the
    triggering fixing is settled (0 = none, 1 = strike-adjusted, 2 = notional-adjusted).

    ``fixing_times`` are year fractions from the valuation date. Fixings already in the past are
    dropped; a fixing on the valuation date that has not yet been observed is kept (pass it as ``0.0``
    or a tiny positive number) and ``accumulated_value`` carries the amount realised by past fixings.
    ``num_spot`` / ``num_target`` / ``n_steps`` set the finite-difference grid resolution -- raise
    ``num_target`` for a heavily seasoned deal, where fewer accumulated-amount nodes remain live.
    """
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
        strike=float(strike),
        fixing_dates=tuple(float(time) for time in fixing_times),
        is_call_option=bool(is_call_option),
        notional1=notional1,
        notional2=notional2,
        barrier=float(barrier),
        target_adjustment=int(target_adjustment),
        inverted_target=bool(inverted_target),
        accumulated_value=float(accumulated_value),
    )
    return ThreeRegimePricer(
        model=model,
        regime_weights=weights,
        num_spot=int(num_spot),
        num_target=int(num_target),
        n_steps=int(n_steps),
    ).price_tarf(tarf, float(maturity))


def tarf_model(
    valuation_date: date,
    spot_value: object,
    strike_value: object,
    target_level: float,
    fixing_times: Sequence[float],
    maturity: float,
    domestic_rate: float,
    foreign_rate: float,
    regime_volatilities: Sequence[float],
    regime_weights: Sequence[float],
    q_matrix: Sequence[Sequence[float]],
    is_call_option: bool = True,
    notional1: Sequence[float] | float = 1.0,
    notional2: Sequence[float] | float = 1.0,
    barrier: float = 0.0,
    target_adjustment: int = 0,
    inverted_target: bool = False,
    accumulated_value: float = 0.0,
    num_spot: int = 121,
    num_target: int = 80,
    n_steps: int = 80,
) -> dict[str, object]:
    """AEF-compatible wrapper; the mandatory return key is ``result``.

    See ``price_tarf`` for the seasoning conventions (past fixings dropped, valuation-date fixing
    kept, ``accumulated_value`` carries realised amount) and the grid-resolution parameters.
    """
    value = price_tarf(
        spot=_denominated_number(spot_value, "spot_value"),
        strike=_denominated_number(strike_value, "strike_value"),
        target_level=target_level,
        fixing_times=fixing_times,
        maturity=maturity,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        regime_volatilities=regime_volatilities,
        regime_weights=regime_weights,
        q_matrix=q_matrix,
        is_call_option=is_call_option,
        notional1=notional1,
        notional2=notional2,
        barrier=barrier,
        target_adjustment=target_adjustment,
        inverted_target=inverted_target,
        accumulated_value=accumulated_value,
        num_spot=num_spot,
        num_target=num_target,
        n_steps=n_steps,
    )
    return {"result": _make_denominated_value(value, _denominated_unit(strike_value, "strike_value"), valuation_date)}


def tarf_model_from_market_data(
    valuation_date: date,
    maturity_date: date,
    spot_value: object,
    strike_value: object,
    target_level: float,
    fixing_times: Sequence[float],
    domestic_curve: object,
    foreign_curve: object,
    vol_surface: object,
    regime_weights: Sequence[float],
    q_matrix: Sequence[Sequence[float]],
    regime_deltas: Sequence[float] = DEFAULT_REGIME_DELTAS,
    is_call_option: bool = True,
    notional1: Sequence[float] | float = 1.0,
    notional2: Sequence[float] | float = 1.0,
    barrier: float = 0.0,
    target_adjustment: int = 0,
    inverted_target: bool = False,
    accumulated_value: float = 0.0,
    num_spot: int = 121,
    num_target: int = 80,
    n_steps: int = 80,
) -> dict[str, object]:
    """AEF-compatible wrapper taking real FA market data objects instead of pre-extracted scalars.

    domestic_curve/foreign_curve are FIrCurveInformation objects, vol_surface is an
    FMalzParametricVolatilityInformation object. The mandatory return key is ``result``. See
    ``price_tarf`` for the seasoning conventions and the grid-resolution parameters.
    """
    market = market_data_from_front_arena(
        valuation_date=valuation_date,
        maturity_date=maturity_date,
        domestic_curve=domestic_curve,
        foreign_curve=foreign_curve,
        vol_surface=vol_surface,
        regime_deltas=regime_deltas,
    )
    maturity = (maturity_date - valuation_date).days / 365.0

    value = price_tarf(
        spot=_denominated_number(spot_value, "spot_value"),
        strike=_denominated_number(strike_value, "strike_value"),
        target_level=target_level,
        fixing_times=fixing_times,
        maturity=maturity,
        domestic_rate=market["domestic_rate"],
        foreign_rate=market["foreign_rate"],
        regime_volatilities=market["regime_volatilities"],
        regime_weights=regime_weights,
        q_matrix=q_matrix,
        is_call_option=is_call_option,
        notional1=notional1,
        notional2=notional2,
        barrier=barrier,
        target_adjustment=target_adjustment,
        inverted_target=inverted_target,
        accumulated_value=accumulated_value,
        num_spot=num_spot,
        num_target=num_target,
        n_steps=n_steps,
    )
    return {"result": _make_denominated_value(value, _denominated_unit(strike_value, "strike_value"), valuation_date)}


ADFL_EXAMPLE = """\
[AEF PV&R]FObject:tarfModel =
    Definition=tarfModel(date valuationDate, denominatedvalue spotValue,
        denominatedvalue strikeValue,
        double targetLevel, array(double) fixingTimes,
        double maturity, double domesticRate, double foreignRate,
        array(double) regimeVolatilities, array(double) regimeWeights,
        matrix(double) qMatrix, int isCallOption, matrix(double) notional1,
        matrix(double) notional2, double barrier, int targetAdjustment,
        int invertedTarget, double accumulatedValue): FDictionary
    Function=TarfValuation.tarf_model

[AEF PV&R]FInstrument:tarfValuationDescriptor = {
    theoreticalModelCall->"tarfTheoreticalModelCall"
}

[AEF PV&R]FInstrument:tarfTheoreticalModelCall = tarfModel(
    valuationDate, spotValue, strikeValue, targetLevel, fixingTimes,
    maturity, domesticRate, foreignRate, regimeVolatilities, regimeWeights,
    qMatrix, isCallOption, notional1, notional2, barrier, targetAdjustment,
    invertedTarget, accumulatedValue);
"""

ADFL_EXAMPLE_MARKET_DATA = """\
[AEF PV&R]FObject:tarfModelFromMarketData =
    Definition=tarfModelFromMarketData(date valuationDate, date maturityDate,
        denominatedvalue spotValue, denominatedvalue strikeValue,
        double targetLevel, array(double) fixingTimes,
        FIrCurveInformation domesticCurve, FIrCurveInformation foreignCurve,
        FMalzParametricVolatilityInformation volSurface,
        array(double) regimeWeights, matrix(double) qMatrix,
        int isCallOption, matrix(double) notional1, matrix(double) notional2,
        double barrier, int targetAdjustment, int invertedTarget,
        double accumulatedValue): FDictionary
    Function=TarfValuation.tarf_model_from_market_data

[AEF PV&R]FInstrument:tarfValuationDescriptor = {
    theoreticalModelCall->"tarfTheoreticalModelCallFromMarketData"
}

[AEF PV&R]FInstrument:tarfTheoreticalModelCallFromMarketData = tarfModelFromMarketData(
    valuationDate, maturityDate, spotValue, strikeValue, targetLevel,
    fixingTimes, domesticCurve, foreignCurve, volSurface, regimeWeights,
    qMatrix, isCallOption, notional1, notional2, barrier, targetAdjustment,
    invertedTarget, accumulatedValue);
"""
