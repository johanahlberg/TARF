"""Pricing a seasoned USDCHF seller TARF -- a deal already part-way through its life.

Same trade as ``price_usdchf_tarf.py`` (sell USD / buy CHF monthly at 0.8200,
leverage 2, 0.10 target, full participation), a 1-year deal traded 2026-05-09,
now valued mid-life on 2026-09-09:

    * 3 monthly fixings have already been observed,
    * one fixing falls on the valuation date and has NOT been fixed yet,
    * 8 monthly fixings remain in the future,
    * 0.062 of the 0.10 target has already been accumulated by the past fixings.

Conventions the engine applies:

    fixing_times   year fractions from the valuation date. Past fixings (t < 0)
                   are dropped automatically; the t = 0 fixing is kept and
                   settled at the known current spot.
    accumulated_value   the amount of the target realised so far (CHF pips here).
    num_target     accumulated-amount grid nodes -- raised below because a
                   seasoned deal has fewer live nodes between accumulated_value
                   and the target.

Run:  python examples/price_seasoned_tarf.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sample_market_data import CHF_CURVE, USD_CURVE, VOL_SURFACE, add_months
from tarf_rslv import (
    DEFAULT_REGIME_DELTAS,
    build_default_regime_matrix,
    market_data_from_front_arena,
    price_tarf,
)

# --- the seasoned deal --------------------------------------------------------
TRADE_DATE = date(2026, 5, 9)          # original trade / first schedule anchor
VALUATION_DATE = date(2026, 9, 9)      # today: just after the 4th monthly fixing
SPOT = 0.8100
STRIKE = 0.8200
TARGET = 0.10
ACCUMULATED = 0.062                    # realised by fixings 1-4, target left = 0.038

REGIME_WEIGHTS = [0.25, 0.50, 0.25]
Q_MATRIX = build_default_regime_matrix()

# 12 monthly fixings from the trade date; #1-#4 are in the past, #5 is today.
all_fixings = [add_months(TRADE_DATE, k) for k in range(1, 13)]
maturity_date = all_fixings[-1]
fixing_times = [(d - VALUATION_DATE).days / 365.0 for d in all_fixings]
maturity = (maturity_date - VALUATION_DATE).days / 365.0

past = [t for t in fixing_times if t < 0.0]
today = [t for t in fixing_times if abs(t) < 1e-9]
future = [t for t in fixing_times if t > 1e-9]


def monte_carlo_price(sigma: float, domestic_rate: float, foreign_rate: float,
                      n_paths: int = 300_000, seed: int = 0) -> tuple[float, float]:
    """Flat-vol GBM cross-check. The t = 0 fixing settles immediately at SPOT; only
    t > 0 fixings are diffused. ``ACCUMULATED`` seeds the target counter."""
    live = today + future
    rng = np.random.default_rng(seed)
    s = np.full(n_paths, SPOT)
    accumulated = np.full(n_paths, ACCUMULATED)
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    drift = domestic_rate - foreign_rate - 0.5 * sigma * sigma

    t_prev = 0.0
    for t_k in live:
        dt = t_k - t_prev
        t_prev = t_k
        if dt > 0.0:
            s = s * np.exp(drift * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths))
        intrinsic = STRIKE - s
        itm = intrinsic > 0.0
        new_accumulated = accumulated + np.where(itm, intrinsic, 0.0)
        terminated = itm & ((TARGET - new_accumulated) < 1e-8)
        cashflow = np.where(itm, intrinsic, 2.0 * intrinsic)  # notional1 = 1, notional2 = 2
        pv += np.where(alive, np.exp(-domestic_rate * max(t_k, 0.0)) * cashflow, 0.0)
        accumulated = np.where(alive, new_accumulated, accumulated)
        alive &= ~terminated

    return float(pv.mean()), float(pv.std() / np.sqrt(n_paths))


def main() -> None:
    market = market_data_from_front_arena(
        valuation_date=VALUATION_DATE,
        maturity_date=maturity_date,
        domestic_curve=CHF_CURVE,
        foreign_curve=USD_CURVE,
        vol_surface=VOL_SURFACE,
        regime_deltas=DEFAULT_REGIME_DELTAS,
    )

    print("=" * 72)
    print("Seasoned USDCHF seller TARF")
    print("=" * 72)
    print(f"  valuation date        {VALUATION_DATE}")
    print(f"  final fixing / expiry {maturity_date}   (T = {maturity:.4f}y)")
    print(f"  spot / strike         {SPOT:.4f} / {STRIKE:.4f}")
    print(f"  target                {TARGET:.4f}   accumulated {ACCUMULATED:.4f}   "
          f"left {TARGET - ACCUMULATED:.4f}")
    print(f"  fixings               {len(past)} past (dropped), "
          f"{len(today)} today (kept, settles at spot), {len(future)} future")
    print(f"    CHF rate {market['domestic_rate'] * 100:.3f}%   "
          f"USD rate {market['foreign_rate'] * 100:.3f}%   "
          f"ATM vol {market['regime_volatilities'][1] * 100:.3f}%")
    print()

    common = dict(
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
        is_call_option=False,
        notional1=1.0,
        notional2=2.0,
        target_adjustment=0,
        accumulated_value=ACCUMULATED,
    )

    print("  Finite-difference value vs accumulated-amount grid resolution")
    for num_target in (80, 160, 320, 640):
        value = price_tarf(**common, num_spot=161, num_target=num_target, n_steps=160)
        print(f"    num_target = {num_target:>4}   {value:+.6f} CHF per USD   ({value * 1e4:+.1f} pips)")

    mc_value, mc_err = monte_carlo_price(
        sigma=float(market["regime_volatilities"][1]),
        domestic_rate=market["domestic_rate"],
        foreign_rate=market["foreign_rate"],
    )
    print()
    print("-" * 72)
    print(f"  Monte Carlo (flat ATM vol) cross-check   {mc_value:+.6f}  +/- {mc_err:.6f}")
    print("-" * 72)


if __name__ == "__main__":
    main()
