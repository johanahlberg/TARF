"""Sample TARF pricing on USDCHF through the Front Arena market-data interface.

Deal -- seller TARF (client sells USD / buys CHF every fixing at the strike):

    pair            USDCHF   (quoted CHF per USD)
    spot            0.8100
    strike          0.8200
    fixings         monthly, 12 fixings over ~1 year
    leverage        2.0      (notional2 = 2 x notional1 on the OTM side)
    target          0.10     (10 big figures, in CHF pips)
    knockout        full participation on the fixing that reaches the target
                    (target_adjustment = 0)

Because USDCHF is quoted CHF-per-USD, the domestic / discounting (terms)
currency is CHF and the foreign (base) currency is USD, so:

    domestic_curve  -> CHF zero curve   -> model ``rate``
    foreign_curve   -> USD zero curve   -> model ``dividend_yield``

The client sells USD, i.e. gains when the USDCHF fixing is *below* the strike,
which is a put on USDCHF -> ``is_call_option = False``.

Run:  python examples/price_usdchf_tarf.py
(add ``src`` to PYTHONPATH, or ``pip install -e .`` first)
"""

from __future__ import annotations

import calendar
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tarf_rslv import (
    DEFAULT_REGIME_DELTAS,
    build_default_regime_matrix,
    market_data_from_front_arena,
    price_tarf,
)


def monte_carlo_price(
    spot: float,
    strike: float,
    target_level: float,
    fixing_times: list[float],
    domestic_rate: float,
    foreign_rate: float,
    sigma: float,
    notional1: float,
    notional2: float,
    n_paths: int = 200_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Flat-vol GBM cross-check of the seller (put) TARF, target_adjustment = 0, no barrier.

    Returns ``(price, standard_error)`` as value to the holder, in the same units
    as ``price_tarf`` (CHF per unit of notional1).
    """
    rng = np.random.default_rng(seed)
    s = np.full(n_paths, spot)
    accumulated = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)

    t_prev = 0.0
    drift = domestic_rate - foreign_rate - 0.5 * sigma * sigma
    for t_k in fixing_times:
        dt = t_k - t_prev
        t_prev = t_k
        z = rng.standard_normal(n_paths)
        s = s * np.exp(drift * dt + sigma * np.sqrt(dt) * z)

        intrinsic = strike - s  # put: client sells USD at the strike
        itm = intrinsic > 0.0
        new_accumulated = accumulated + np.where(itm, intrinsic, 0.0)
        terminated = itm & ((target_level - new_accumulated) < 1e-8)
        cashflow = np.where(itm, notional1 * intrinsic, notional2 * intrinsic)

        discount = np.exp(-domestic_rate * t_k)
        pv += np.where(alive, discount * cashflow, 0.0)
        accumulated = np.where(alive, new_accumulated, accumulated)
        alive &= ~terminated

    return float(pv.mean()), float(pv.std() / np.sqrt(n_paths))


# ---------------------------------------------------------------------------
# Minimal stand-ins for the Front Arena market-data objects
# ---------------------------------------------------------------------------
class ZeroCurve:
    """Stand-in for an FA ``FIrCurveInformation``.

    ``Rate(startDay, endDay)`` returns the continuously-compounded Act/365 zero
    rate for the period, built from a few ``(tenor_years, zero_rate)`` pillars
    with linear interpolation in zero-rate space and a forward-rate calc for
    non-spot-starting windows.
    """

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
    """Stand-in for an FA ``FMalzParametricVolatilityInformation``.

    ``Value(expiryDate, delta, foreignRate, domesticRate)`` returns a smile vol
    for a signed delta (-0.25 = 25d put, 0.5 = ATM, +0.25 = 25d call) from an
    ATM term structure plus a fixed 25-delta risk-reversal / butterfly, scaled
    linearly in |delta| away from the 25-delta wings.

        sigma(25dC) = ATM + BF + RR / 2
        sigma(25dP) = ATM + BF - RR / 2       RR = sigma(25dC) - sigma(25dP)
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


# ---------------------------------------------------------------------------
# Sample market data (valuation 2026-09-09)
# ---------------------------------------------------------------------------
VALUATION_DATE = date(2026, 9, 9)
MATURITY_DATE = date(2027, 9, 9)
SPOT = 0.8100
STRIKE = 0.8200

# CHF (domestic / terms) zero curve -- low rates, gently upward sloping.
chf_curve = ZeroCurve(
    VALUATION_DATE,
    [(0.08, 0.0090), (0.25, 0.0095), (0.50, 0.0100), (1.00, 0.0110), (2.00, 0.0125)],
)

# USD (foreign / base) zero curve -- around 4.5%.
usd_curve = ZeroCurve(
    VALUATION_DATE,
    [(0.08, 0.0440), (0.25, 0.0450), (0.50, 0.0455), (1.00, 0.0460), (2.00, 0.0465)],
)

# USDCHF ATM vol term structure; safe-haven skew (25d RR favours USD puts).
vol_surface = MalzVolSurface(
    VALUATION_DATE,
    [(0.08, 0.065), (0.25, 0.068), (0.50, 0.070), (1.00, 0.072), (2.00, 0.075)],
    rr_25=-0.006,
    bf_25=0.0020,
)

# 3-regime layer: 25d-put / ATM / 25d-call smile proxies, generator, and the
# prior weight on each regime at valuation.
REGIME_DELTAS = DEFAULT_REGIME_DELTAS            # (-0.25, 0.5, 0.25)
REGIME_WEIGHTS = [0.25, 0.50, 0.25]
Q_MATRIX = build_default_regime_matrix()


# ---------------------------------------------------------------------------
# Monthly fixing schedule
# ---------------------------------------------------------------------------
def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    year = d.year + m // 12
    month = m % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


fixing_dates = [add_months(VALUATION_DATE, k) for k in range(1, 13)]
fixing_times = [(d - VALUATION_DATE).days / 365.0 for d in fixing_dates]
maturity = (MATURITY_DATE - VALUATION_DATE).days / 365.0


def main() -> None:
    market = market_data_from_front_arena(
        valuation_date=VALUATION_DATE,
        maturity_date=MATURITY_DATE,
        domestic_curve=chf_curve,
        foreign_curve=usd_curve,
        vol_surface=vol_surface,
        regime_deltas=REGIME_DELTAS,
    )

    fwd = SPOT * np.exp((market["domestic_rate"] - market["foreign_rate"]) * maturity)

    print("=" * 72)
    print("USDCHF seller TARF  (client sells USD / buys CHF at the strike)")
    print("=" * 72)
    print(f"  valuation date        {VALUATION_DATE}")
    print(f"  maturity date         {MATURITY_DATE}   (T = {maturity:.4f}y)")
    print(f"  spot                  {SPOT:.4f}")
    print(f"  strike                {STRIKE:.4f}")
    print(f"  1y outright forward   {fwd:.4f}   (USD at a forward discount to CHF)")
    print(f"  fixings               {len(fixing_dates)} monthly")
    print("  leverage              2.0  (notional1 = 1, notional2 = 2)")
    print("  target                0.10  (10 big figures, CHF)")
    print("  knockout style        full participation  (target_adjustment = 0)")
    print()
    print("  Extracted market data")
    print(f"    CHF zero rate (dom)  {market['domestic_rate'] * 100:.3f}%")
    print(f"    USD zero rate (for)  {market['foreign_rate'] * 100:.3f}%")
    vols = market["regime_volatilities"]
    print(f"    regime vols          25dP {vols[0] * 100:.3f}%   "
          f"ATM {vols[1] * 100:.3f}%   25dC {vols[2] * 100:.3f}%")
    print()
    print("  Fixing schedule")
    for k, (d, tau) in enumerate(zip(fixing_dates, fixing_times), start=1):
        print(f"    #{k:<2d} {d}   t = {tau:.4f}")
    print()

    print("  Pricing (3-regime FD, default 121 x 80 grid, 80 steps) ...")
    value = price_tarf(
        spot=SPOT,
        strike=STRIKE,
        target_level=0.10,
        fixing_times=fixing_times,
        maturity=maturity,
        domestic_rate=market["domestic_rate"],
        foreign_rate=market["foreign_rate"],
        regime_volatilities=market["regime_volatilities"],
        regime_weights=REGIME_WEIGHTS,
        q_matrix=Q_MATRIX,
        is_call_option=False,     # selling USD -> put on USDCHF
        notional1=1.0,
        notional2=2.0,            # leverage 2 on the OTM leg
        barrier=0.0,              # no knock-in barrier
        target_adjustment=0,      # full gain on the knockout fixing
        inverted_target=False,    # target measured in CHF pips (natural quote)
        accumulated_value=0.0,
    )

    mc_value, mc_err = monte_carlo_price(
        spot=SPOT,
        strike=STRIKE,
        target_level=0.10,
        fixing_times=fixing_times,
        domestic_rate=market["domestic_rate"],
        foreign_rate=market["foreign_rate"],
        sigma=float(market["regime_volatilities"][1]),  # ATM regime as the flat-vol proxy
        notional1=1.0,
        notional2=2.0,
    )

    print()
    print("-" * 72)
    print("  TARF value to the holder (USD seller)")
    print(f"    finite-difference (3-regime)   {value:+.6f} CHF per USD of notional1")
    print(f"    Monte Carlo (flat ATM vol)     {mc_value:+.6f}  +/- {mc_err:.6f}")
    print(f"    = {value * 1e4:+.1f} CHF pips")
    print("-" * 72)
    print("  Sign: gains are capped by the 0.10 target (the deal knocks out), while")
    print("  the 2x leverage on fixings above the strike is uncapped, so under the")
    print("  risk-neutral measure this particular strike carries negative value to")
    print("  the holder -- i.e. the bank's margin / the cost of the enhanced rate.")


if __name__ == "__main__":
    main()
