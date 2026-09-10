"""Calibrate the three-regime structural parameters from one-touch quotes.

A static vanilla surface does not identify regime dispersion or switching speed. Touch / barrier
options do (they are path-dependent and forward-smile-sensitive). This example:

  1. builds the USDCHF surface (arbitrage-free SVI + analytic Dupire),
  2. takes a set of one-touch quotes -- here generated from a 'true' regime model so the recovery
     is checkable; in production these come from the broker sheet,
  3. calibrates ``{level_spread, skew_spread, switch_rate}`` to them (alternating with the per-tenor
     SVI (a, b) refit to the vanilla surface),
  4. prices the seasoned TARF with the calibrated 3-regime model.

Run:  python examples/calibrate_from_barriers.py   (~30-60 s)
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sample_market_data import USDCHF_SMILE_QUOTES
from tarf_rslv import (
    FXVolSurface,
    OneTouchQuote,
    RegimeBarrierPricer,
    SmileQuotes,
    TARFAccumulator,
    calibrate_regime_model,
)
from tarf_rslv.calibration_surface import _generator

SPOT, STRIKE, TARGET = 0.8100, 0.8200, 0.10
R_CHF, R_USD = 0.011, 0.046

# use the shorter end of the sample surface to keep the example quick
QUOTES = [q for q in USDCHF_SMILE_QUOTES if q[0] <= 1.0]
TRUE_LEVEL_SPREAD, TRUE_SKEW_SPREAD, TRUE_SWITCH_RATE = 0.20, 0.14, 3.5


def main() -> None:
    surface = FXVolSurface(
        spot=SPOT, smiles=[SmileQuotes(*row) for row in QUOTES],
        domestic_zero=lambda t: R_CHF, foreign_zero=lambda t: R_USD,
    )
    print(f"USDCHF surface: {len(surface.smiles)} tenors, SVI RMS "
          f"{surface.svi_report.rms_vol_error * 1e4:.2f} bp\n")

    # --- one-touch "market" quotes (generated from a known regime structure) ---------------
    truth, _ = calibrate_regime_model(
        surface, n_regimes=3, level_spread=TRUE_LEVEL_SPREAD, skew_spread=TRUE_SKEW_SPREAD,
        calibrate_q=False,
    )
    truth = dataclasses.replace(truth, q=_generator(TRUE_SWITCH_RATE, np.array([0.25, 0.5, 0.25])))
    truth_pricer = RegimeBarrierPricer(truth)

    specs = [
        (0.78, 0.5, "down"), (0.76, 1.0, "down"), (0.74, 1.0, "down"), (0.80, 0.25, "down"),
        (0.85, 0.5, "up"), (0.87, 1.0, "up"), (0.89, 1.0, "up"), (0.84, 0.25, "up"),
    ]
    quotes = [
        OneTouchQuote(barrier=b, maturity=t, direction=d,
                      market_price=truth_pricer.one_touch(b, t, direction=d, payment="hit"))
        for b, t, d in specs
    ]
    print("  one-touch quotes (barrier / T / dir / price)")
    for q in quotes:
        print(f"    {q.barrier:.3f}  {q.maturity:.2f}y  {q.direction:<4}  {q.market_price:.4f}")

    # --- calibrate the regime structure to the touches ------------------------------------
    print(f"\n  Calibrating {{level_spread, skew_spread, switch_rate}} to {len(quotes)} one-touches ...")
    t0 = time.time()
    model, report = calibrate_regime_model(
        surface, n_regimes=3, barrier_quotes=quotes, n_barrier_rounds=2, verbose=True,
    )
    print(f"  -> {time.time() - t0:.0f} s")
    print(f"     level_spread {report.level_spread:.3f}  (true {TRUE_LEVEL_SPREAD})")
    print(f"     skew_spread  {report.skew_spread:.3f}  (true {TRUE_SKEW_SPREAD})   "
          f"-- partly trades off against level_spread with few / symmetric quotes")
    print(f"     switch_rate  {report.switch_rate:.2f}   (true {TRUE_SWITCH_RATE})")
    print(f"     vanilla surface RMS {report.rms_vol_error * 1e4:.2f} bp   "
          f"one-touch RMS {report.barrier_rms_price_error * 1e4:.1f} bp")

    # --- price the seasoned TARF with the barrier-calibrated model ------------------------
    fixings = [(k + 1) / 12 for k in range(12)]
    off = 4.5 / 12
    tarf = TARFAccumulator(
        target_level=TARGET, strike=STRIKE, fixing_dates=tuple(sorted(f - off for f in fixings)),
        is_call_option=False, notional1=1.0, notional2=2.0, target_adjustment=0, accumulated_value=0.062,
    )
    px = model.price_tarf(tarf, fixings[-1] - off, num_spot=161, num_target=120, n_steps=140)
    print(f"\n  Seasoned USDCHF seller TARF (barrier-calibrated 3-regime model): "
          f"{px:+.6f} CHF/USD  ({px * 1e4:+.1f} pips)")


if __name__ == "__main__":
    main()
