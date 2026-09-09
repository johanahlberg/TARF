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
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sample_market_data import CHF_CURVE, USD_CURVE, VALUATION_DATE, VOL_SURFACE, add_months
from tarf_rslv import (
    DEFAULT_REGIME_DELTAS,
    build_default_regime_matrix,
    market_data_from_front_arena,
    price_tarf,
)

MATURITY_DATE = date(2027, 9, 9)
SPOT = 0.8100
STRIKE = 0.8200
TARGET = 0.10

REGIME_DELTAS = DEFAULT_REGIME_DELTAS         # (-0.25, 0.5, 0.25) -> 25dP / ATM / 25dC
REGIME_WEIGHTS = [0.25, 0.50, 0.25]
Q_MATRIX = build_default_regime_matrix()

fixing_dates = [add_months(VALUATION_DATE, k) for k in range(1, 13)]
fixing_times = [(d - VALUATION_DATE).days / 365.0 for d in fixing_dates]
maturity = (MATURITY_DATE - VALUATION_DATE).days / 365.0


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
    accumulated_value: float = 0.0,
    n_paths: int = 200_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Flat-vol GBM cross-check of the seller (put) TARF, target_adjustment = 0, no barrier.

    Only fixings with ``t > 0`` are simulated; ``accumulated_value`` seeds the target counter.
    Returns ``(price, standard_error)`` as value to the holder (CHF per unit of notional1).
    """
    live = [t for t in fixing_times if t > 0.0]
    rng = np.random.default_rng(seed)
    s = np.full(n_paths, spot)
    accumulated = np.full(n_paths, accumulated_value)
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)

    t_prev = 0.0
    drift = domestic_rate - foreign_rate - 0.5 * sigma * sigma
    for t_k in live:
        dt = t_k - t_prev
        t_prev = t_k
        s = s * np.exp(drift * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths))

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


def main() -> None:
    market = market_data_from_front_arena(
        valuation_date=VALUATION_DATE,
        maturity_date=MATURITY_DATE,
        domestic_curve=CHF_CURVE,
        foreign_curve=USD_CURVE,
        vol_surface=VOL_SURFACE,
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

    print("  Pricing (3-regime FD, default 121 x 80 grid, 80 steps) ...")
    value = price_tarf(
        spot=SPOT,
        strike=STRIKE,
        target_level=TARGET,
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
        target_level=TARGET,
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
