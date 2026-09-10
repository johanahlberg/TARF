"""Calibrate the local-vol model to a full USDCHF vol surface, then price the seasoned seller TARF.

Pipeline:

    broker quotes (ATM, 25d & 10d RR/BF; market strangles)   [sample_market_data]
        -> FXVolSurface        (5-point smile per tenor -> arbitrage-free SVI slice)
        -> calibrate_regime_model(n_regimes=3)   regimes differ in level AND skew; analytic Dupire;
                                                 forward regime-switching PDE; then the switch rate
        -> CalibratedRegimeModel.price_tarf      (dispatches to ThreeRegimePricer)

Then prices the same deal with ``n_regimes=1`` (a single Dupire local-vol model, ~3x faster) and the
crude 3-point proxy that ``front_arena.tarf_model_from_market_data`` uses, to show the regime effect.

Run:  python examples/calibrate_usdchf_surface.py   (~30 s: the calibration runs the forward PDE)
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sample_market_data import CHF_CURVE, USD_CURVE, USDCHF_SMILE_QUOTES, VALUATION_DATE
from tarf_rslv import (
    FXVolSurface,
    SmileQuotes,
    TARFAccumulator,
    build_default_regime_matrix,
    calibrate_regime_model,
    price_tarf,
)

SPOT = 0.8100
STRIKE = 0.8200
TARGET = 0.10
REGIME_WEIGHTS = [0.25, 0.50, 0.25]


def main() -> None:
    # Curves as t -> zero rate (the sample curves are near-flat, so a single point is fine here;
    # a real curve would be sampled at each t).
    r_chf = CHF_CURVE.Rate(VALUATION_DATE, date(VALUATION_DATE.year + 1, VALUATION_DATE.month, VALUATION_DATE.day))
    r_usd = USD_CURVE.Rate(VALUATION_DATE, date(VALUATION_DATE.year + 1, VALUATION_DATE.month, VALUATION_DATE.day))
    surface = FXVolSurface(
        spot=SPOT,
        smiles=[SmileQuotes(*row) for row in USDCHF_SMILE_QUOTES],
        domestic_zero=lambda t: r_chf,
        foreign_zero=lambda t: r_usd,
    )

    print("=" * 74)
    print("USDCHF vol surface  (broker quotes -> 5-point smile per tenor)")
    print("=" * 74)
    print(f"  spot {SPOT}   r_CHF {r_chf * 100:.2f}%   r_USD {r_usd * 100:.2f}%")
    print(f"  {len(surface.smiles)} tenors, {len(surface.market_nodes())} smile nodes\n")
    print("   tenor    ATM     25dRR    25dBF    10dRR    10dBF")
    for row in USDCHF_SMILE_QUOTES:
        t, atm, rr25, bf25, rr10, bf10 = row
        print(f"   {t:6.3f}y  {atm * 100:5.2f}%  {rr25 * 100:+6.2f}  {bf25 * 100:5.2f}  "
              f"{rr10 * 100:+6.2f}  {bf10 * 100:5.2f}")

    print(f"\n  Arbitrage-free SVI surface fit: RMS {surface.svi_report.rms_vol_error * 1e4:.2f} bp, "
          f"butterfly {surface.svi_report.max_butterfly_violation:.1e}, "
          f"calendar {surface.svi_report.max_calendar_violation:.1e}")
    print("\n  Calibrating full 3-regime model (SVI (a,b) per tenor -> analytic Dupire -> forward")
    print("  regime-switching PDE; regimes differ in level AND skew; then the switch rate) ...")
    t0 = time.time()
    model, report = calibrate_regime_model(surface, REGIME_WEIGHTS, verbose=True)
    print(f"  -> {time.time() - t0:.1f}s   RMS {report.rms_vol_error * 1e4:.2f} bp   "
          f"max {report.max_vol_error * 1e4:.2f} bp")
    print(f"     residual arbitrage: butterfly {report.max_butterfly_violation:.1e}, "
          f"calendar {report.max_calendar_violation:.1e}")
    print(f"     regime structure (prior): level_spread {report.level_spread}, "
          f"skew_spread {report.skew_spread}")
    print(f"     switch rate {report.switch_rate:.2f}/yr  "
          f"(identified from the RR/BF term structure: {report.switch_rate_identified})")

    # per-tenor fit error
    print("\n  Fit error by tenor (model implied vol - market, bp)")
    errs = report.node_errors
    for t in sorted({row[0] for row in errs}):
        row_errs = errs[np.isclose(errs[:, 0], t)][:, 2] * 1e4
        print(f"    {t:6.3f}y   " + "  ".join(f"{e:+5.1f}" for e in row_errs))

    # regime local-vol snapshot at 6m
    f6 = surface.forward(0.5)
    grid = f6 * np.exp(np.array([-0.05, 0.0, 0.05]))
    print("\n  Calibrated regime LOCAL vols at 6m  (spot -5% / ATM / +5%)")
    for i, r in enumerate(model.regimes):
        lv = r.local_volatility(grid, 0.5)
        print(f"    regime {i}:  {lv[0] * 100:5.2f}%   {lv[1] * 100:5.2f}%   {lv[2] * 100:5.2f}%")

    # ---- price the seasoned TARF two ways ------------------------------------------------
    fixings = [(k + 1) / 12 for k in range(12)]
    val_offset = 4.5 / 12
    fixing_times = [f - val_offset for f in fixings]
    maturity = fixings[-1] - val_offset
    accumulated = 0.062

    tarf = TARFAccumulator(
        target_level=TARGET, strike=STRIKE, fixing_dates=tuple(sorted(fixing_times)),
        is_call_option=False, notional1=1.0, notional2=2.0, target_adjustment=0,
        accumulated_value=accumulated,
    )
    px3 = model.price_tarf(tarf, maturity, num_spot=161, num_target=120, n_steps=140)

    # the same deal on the single Dupire local-vol model (n_regimes=1) -- ~3x faster to price
    t0 = time.time()
    single, single_rep = calibrate_regime_model(surface, n_regimes=1)
    px1 = single.price_tarf(tarf, maturity, num_spot=161, num_target=120, n_steps=140)
    print(f"\n  Single Dupire model (n_regimes=1): calibrated in {time.time() - t0:.1f}s, "
          f"surface RMS {single_rep.rms_vol_error * 1e4:.2f} bp")

    # crude 3-point proxy: 25dP / ATM / 25dC of the 6m smile as flat regime vols
    smile_6m = next(s for s in surface.smiles if abs(s.tenor - 0.5) < 1e-6)
    _, vols = smile_6m.knots(SPOT, f6, surface.foreign_df(0.5), surface.domestic_df(0.5), surface.convention)
    proxy_px = price_tarf(
        spot=SPOT, strike=STRIKE, target_level=TARGET, fixing_times=fixing_times, maturity=maturity,
        domestic_rate=r_chf, foreign_rate=r_usd, regime_volatilities=[float(vols[1]), float(vols[2]), float(vols[3])],
        regime_weights=REGIME_WEIGHTS, q_matrix=build_default_regime_matrix().tolist(),
        is_call_option=False, notional1=1.0, notional2=2.0, target_adjustment=0,
        accumulated_value=accumulated, num_target=120,
    )

    print("\n" + "-" * 74)
    print(f"  Seasoned USDCHF seller TARF  (T={maturity:.3f}y, accumulated {accumulated})")
    print(f"    3-regime calibrated model       {px3:+.6f} CHF/USD  ({px3 * 1e4:+.1f} pips)")
    print(f"    single Dupire model             {px1:+.6f} CHF/USD  ({px1 * 1e4:+.1f} pips)   "
          f"[regime effect {(px3 - px1) * 1e4:+.1f} pips]")
    print(f"    crude 3-point proxy             {proxy_px:+.6f} CHF/USD  ({proxy_px * 1e4:+.1f} pips)")
    print("-" * 74)


if __name__ == "__main__":
    main()
