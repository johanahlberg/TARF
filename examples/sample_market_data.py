"""Shared sample market data for the pricing examples.

These classes are *stand-ins* for the Front Arena market-data objects the real
engine consumes:

    ZeroCurve       ~ FIrCurveInformation               (Rate(startDay, endDay))
    MalzVolSurface  ~ FMalzParametricVolatilityInformation  (Value(expiry, delta, ...))

A real integration passes the genuine FA objects straight into
``market_data_from_front_arena`` -- nothing here is part of the model.
"""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np


class ZeroCurve:
    """``Rate(startDay, endDay)`` returns the continuously-compounded Act/365 zero
    rate for the period, from ``(tenor_years, zero_rate)`` pillars with linear
    interpolation in zero-rate space and a forward-rate calc off-spot."""

    def __init__(self, valuation_date: date, pillars: list[tuple[float, float]]) -> None:
        self.valuation_date = valuation_date
        self._t = np.array([p[0] for p in pillars], dtype=float)
        self._z = np.array([p[1] for p in pillars], dtype=float)

    def _zero(self, t: float) -> float:
        return float(np.interp(max(t, 0.0), self._t, self._z))

    def Rate(self, start_day: date, end_day: date) -> float:
        t1 = (start_day - self.valuation_date).days / 365.0
        t2 = (end_day - self.valuation_date).days / 365.0
        if t2 <= max(t1, 0.0):
            return self._zero(max(t2, 0.0))
        if t1 <= 0.0:
            return self._zero(t2)
        z1, z2 = self._zero(t1), self._zero(t2)
        return (z2 * t2 - z1 * t1) / (t2 - t1)


class MalzVolSurface:
    """``Value(expiryDate, delta, foreignRate, domesticRate)`` returns a smile vol
    for a signed delta (-0.25 = 25d put, 0.5 = ATM, +0.25 = 25d call) from an ATM
    term structure plus a fixed 25-delta risk-reversal / butterfly, scaled
    linearly in |delta| away from the wings.

        sigma(25dC) = ATM + BF + RR / 2
        sigma(25dP) = ATM + BF - RR / 2     RR = sigma(25dC) - sigma(25dP)
    """

    def __init__(
        self,
        valuation_date: date,
        atm_pillars: list[tuple[float, float]],
        rr_25: float = -0.006,
        bf_25: float = 0.0020,
    ) -> None:
        self.valuation_date = valuation_date
        self._t = np.array([p[0] for p in atm_pillars], dtype=float)
        self._atm = np.array([p[1] for p in atm_pillars], dtype=float)
        self.rr_25 = rr_25
        self.bf_25 = bf_25

    def _atm_vol(self, t: float) -> float:
        return float(np.interp(max(t, 0.0), self._t, self._atm))

    def Value(self, expiry_date: date, delta: float, foreign_rate: float, domestic_rate: float) -> float:
        t = max((expiry_date - self.valuation_date).days / 365.0, 1e-6)
        atm = self._atm_vol(t)
        d = abs(float(delta))
        if d >= 0.49:  # treat as ATM
            return atm
        scale = (0.5 - d) / 0.25  # 1.0 at the 25-delta wings, 0 at the money
        wing = self.bf_25 * scale
        half_skew = 0.5 * self.rr_25 * scale
        return atm + wing + (half_skew if delta > 0 else -half_skew)


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    year = d.year + m // 12
    month = m % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


# --------------------------------------------------------------------------------------------------
# A single USDCHF snapshot used by every example (valuation 2026-09-09)
# --------------------------------------------------------------------------------------------------
VALUATION_DATE = date(2026, 9, 9)

# CHF (domestic / terms) zero curve -- low, gently upward sloping.
CHF_CURVE = ZeroCurve(
    VALUATION_DATE,
    [(0.08, 0.0090), (0.25, 0.0095), (0.50, 0.0100), (1.00, 0.0110), (2.00, 0.0125)],
)

# USD (foreign / base) zero curve -- around 4.5%.
USD_CURVE = ZeroCurve(
    VALUATION_DATE,
    [(0.08, 0.0440), (0.25, 0.0450), (0.50, 0.0455), (1.00, 0.0460), (2.00, 0.0465)],
)

# USDCHF ATM vol term structure; safe-haven skew (25d RR favours USD puts).
VOL_SURFACE = MalzVolSurface(
    VALUATION_DATE,
    [(0.08, 0.065), (0.25, 0.068), (0.50, 0.070), (1.00, 0.072), (2.00, 0.075)],
    rr_25=-0.006,
    bf_25=0.0020,
)
