"""Front Arena AEF boundary for the TARF valuation engine.

The functions in this module deliberately accept only scalar values, lists, and dates. That keeps the
valuation state-free and compatible with AEF distributed-calculation serialization. Front Arena-specific
ACM objects are created only at the final result boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .calibration_surface import calibrate_regime_model
from .model import RegimeSwitchingLocalVolModel, SingleRegimeLocalVolModel
from .product import TARFAccumulator
from .solver import ThreeRegimePricer
from .vol_surface import DeltaConvention, FXVolSurface, SmileQuotes


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


def _zero_rate_fn(curve: object, valuation_date: date, name: str):
    """Wrap an FA ``FIrCurveInformation`` as ``t -> continuous Act/365 zero rate``."""
    rate = getattr(curve, "Rate", None)
    if not callable(rate):
        raise TypeError(f"{name} must provide a Rate(startDay, endDay) method")

    def zero(t: float) -> float:
        end = valuation_date + timedelta(days=max(int(round(float(t) * 365.0)), 1))
        return float(rate(valuation_date, end))

    return zero


def fx_surface_from_quotes(
    spot: float,
    valuation_date: date,
    tenor_years: Sequence[float],
    atm: Sequence[float],
    rr25: Sequence[float],
    bf25: Sequence[float],
    rr10: Sequence[float],
    bf10: Sequence[float],
    domestic_curve: object,
    foreign_curve: object,
    *,
    bf_convention: str = "market_strangle",
    delta_type: str = "spot",
    premium_adjusted: bool = False,
    atm_convention: str = "dns",
) -> FXVolSurface:
    """Build an :class:`FXVolSurface` from quote arrays (ATM / 25d & 10d RR & BF per tenor) and two
    FA ``FIrCurveInformation`` objects.

    ``bf_convention="market_strangle"`` for broker quotes (default), ``"smile"`` if the butterflies
    are already smile-vol butterflies. Set the delta / ATM conventions to match your quote source
    (for USDCHF the premium is CHF, so ``premium_adjusted=False``).
    """
    smiles = [
        SmileQuotes(float(t), float(a), float(r25), float(b25), float(r10), float(b10), bf_convention)
        for t, a, r25, b25, r10, b10 in zip(tenor_years, atm, rr25, bf25, rr10, bf10)
    ]
    return FXVolSurface(
        spot=float(spot),
        smiles=smiles,
        domestic_zero=_zero_rate_fn(domestic_curve, valuation_date, "domestic_curve"),
        foreign_zero=_zero_rate_fn(foreign_curve, valuation_date, "foreign_curve"),
        convention=DeltaConvention(delta_type, premium_adjusted, atm_convention),
    )


def market_surface_from_front_arena(
    spot: float,
    valuation_date: date,
    maturity_dates: Sequence[date],
    domestic_curve: object,
    foreign_curve: object,
    vol_surface: object,
    *,
    bf_convention: str = "smile",
    delta_type: str = "spot",
    premium_adjusted: bool = False,
    atm_convention: str = "dns",
) -> FXVolSurface:
    """Query a delta-parametrised FA vol object (``Value(expiry, delta, fRate, dRate)``) at
    +-10d / +-25d / ATM for each expiry and assemble an :class:`FXVolSurface`.

    A pure ``FMalzParametricVolatilityInformation`` carries only 25d information, so its 10d values
    are the parabola's own extrapolation -- pass genuine 10d quotes through ``fx_surface_from_quotes``
    when you have them.
    """
    value = getattr(vol_surface, "Value", None)
    if not callable(value):
        raise TypeError("vol_surface must provide a Value(expiryDate, delta, foreignRate, domesticRate) method")
    dom_zero = _zero_rate_fn(domestic_curve, valuation_date, "domestic_curve")
    for_zero = _zero_rate_fn(foreign_curve, valuation_date, "foreign_curve")

    tenors, atm, rr25, bf25, rr10, bf10 = [], [], [], [], [], []
    for md in maturity_dates:
        t = (md - valuation_date).days / 365.0
        if t <= 0.0:
            raise ValueError("every maturity_date must be after valuation_date")
        r_d, r_f = dom_zero(t), for_zero(t)
        v_atm = float(value(md, 0.50, r_f, r_d))
        v_25c = float(value(md, 0.25, r_f, r_d))
        v_25p = float(value(md, -0.25, r_f, r_d))
        v_10c = float(value(md, 0.10, r_f, r_d))
        v_10p = float(value(md, -0.10, r_f, r_d))
        tenors.append(t)
        atm.append(v_atm)
        rr25.append(v_25c - v_25p)
        bf25.append(0.5 * (v_25c + v_25p) - v_atm)
        rr10.append(v_10c - v_10p)
        bf10.append(0.5 * (v_10c + v_10p) - v_atm)

    return fx_surface_from_quotes(
        spot, valuation_date, tenors, atm, rr25, bf25, rr10, bf10, domestic_curve, foreign_curve,
        bf_convention=bf_convention, delta_type=delta_type,
        premium_adjusted=premium_adjusted, atm_convention=atm_convention,
    )


def calibrated_tarf_model_from_surface(
    valuation_date: date,
    spot_value: object,
    strike_value: object,
    target_level: float,
    fixing_times: Sequence[float],
    maturity: float,
    surface: FXVolSurface,
    *,
    regime_weights: Sequence[float] = (0.25, 0.5, 0.25),
    q_matrix: Sequence[Sequence[float]] | None = None,
    regime_spread: float = 0.03,
    is_call_option: bool = True,
    notional1: Sequence[float] | float = 1.0,
    notional2: Sequence[float] | float = 1.0,
    barrier: float = 0.0,
    target_adjustment: int = 0,
    inverted_target: bool = False,
    accumulated_value: float = 0.0,
    num_spot: int = 161,
    num_target: int = 100,
    n_steps: int = 120,
    calibrate_q: bool = True,
    calibration_kwargs: dict | None = None,
) -> dict[str, object]:
    """Calibrate the full three-regime model to ``surface`` (arbitrage-free SVI smile, analytic
    Dupire local vol, forward regime-switching PDE, plus the switch rate) and price the TARF with it.
    ``result`` holds the DenominatedValue; ``calibration`` the fit report.
    """
    weights = np.asarray(regime_weights, dtype=float)
    q = None if q_matrix is None else np.asarray(q_matrix, dtype=float)
    model, report = calibrate_regime_model(
        surface, weights, q=q, spread=regime_spread, calibrate_q=calibrate_q, **(calibration_kwargs or {})
    )

    tarf = TARFAccumulator(
        target_level=float(target_level),
        strike=_denominated_number(strike_value, "strike_value"),
        fixing_dates=tuple(float(time) for time in fixing_times),
        is_call_option=bool(is_call_option),
        notional1=notional1,
        notional2=notional2,
        barrier=float(barrier),
        target_adjustment=int(target_adjustment),
        inverted_target=bool(inverted_target),
        accumulated_value=float(accumulated_value),
    )
    price = ThreeRegimePricer(
        model, model.regime_weights, num_spot=num_spot, num_target=num_target, n_steps=n_steps
    ).price_tarf(tarf, float(maturity))

    unit = _denominated_unit(strike_value, "strike_value")
    return {
        "result": _make_denominated_value(price, unit, valuation_date),
        "calibration": {
            "rms_vol_error": report.rms_vol_error,
            "max_vol_error": report.max_vol_error,
            "svi_rms_vol_error": surface.svi_report.rms_vol_error if surface.svi_report else None,
            "max_butterfly_violation": report.max_butterfly_violation,
            "max_calendar_violation": report.max_calendar_violation,
            "switch_rate": report.switch_rate,
            "success": report.success,
        },
    }


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
