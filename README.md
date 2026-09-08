# TARF RSLV Pricer

This repository is structured as a compact numerical-pricing prototype for a Target Accumulation Redemption Forward (TARF) using finite-difference local-volatility solvers and a coupled 3-regime extension.

## Layout

- `src/tarf_rslv/model.py`: model objects, including the single-regime local-vol setup and the default regime generator matrix.
- `src/tarf_rslv/product.py`: `TARFAccumulator`, a port of the `udmcTRFPayoff` ESL script's per-fixing payoff/accumulation logic.
- `src/tarf_rslv/solver.py`: single-regime and coupled three-regime PDE pricers built around that payoff.
- `src/tarf_rslv/calibration.py`: compact smile and switching-matrix calibration helpers.
- `src/tarf_rslv/front_arena.py`: AEF-compatible, state-free valuation wrapper and ADFL example.
- `tests/test_single_regime_pricer.py`: validation checks for the baseline architecture.
- `tests/test_three_regime_pricer.py`: regime coupling, finiteness, and degenerate-case checks.
- `tests/test_tarf_payoff.py`: unit coverage of `TARFAccumulator.settle` against the ESL script's per-fixing logic (accumulation, target adjustment, barrier, inverted quoting).
- `tests/test_front_arena_interface.py`: ACM boundary and serialization-oriented adapter checks.

## Current status

The repository currently contains a stable implementation of the main architecture described in the brief:

- single-regime local-vol model shell,
- a `TARFAccumulator` payoff ported directly from the `udmcTRFPayoff` ESL script: ITM fixings accrue `notional1 * intrinsic` toward `target_level` (or its inverted-quote equivalent), OTM fixings pay a leveraged `notional2 * intrinsic` loss gated by a local (no-memory) KI `barrier`, and the triggering fixing settles per `target_adjustment` (none / strike-adjusted / notional-adjusted),
- implicit finite-difference spot stepping on a 2D (spot, accumulated) grid, with the accumulated state jumping by a spot-dependent amount at each fixing date rather than drifting deterministically,
- operator-split 3-regime PDE coupling with exact short-step matrix exponentials,
- compact parametric smile and generator-matrix calibration helpers,
- regression coverage for the single-regime limit of the coupled solver, and for the leveraged-loss/barrier economics.

## Numerical design

Between fixing dates, each (spot, accumulated) slice diffuses independently via an implicit tridiagonal spot solve (Neumann/zero-gradient boundaries, since the payoff includes leveraged OTM losses with no simple closed-form edge value); regimes are then coupled via `expm(Q * dt)`. At each fixing date, `_apply_tarf_fixing` applies the payoff jump: on the ITM side the accumulated grid advances by a spot-dependent increment, which is not generally on-grid and is handled by interpolation along the accumulation axis; on the OTM side the accumulated state is unchanged and the leveraged/barrier-gated cashflow is added directly. Discounting is embedded in the implicit spot operator across the whole backward sweep, so the final price is read directly off the grid at `(spot, accumulated_value)` with no separate terminal discount factor.

## Front Arena AEF interface

`front_arena.tarf_model` follows the AEF proprietary-valuation contract. Its inputs are scalar values, arrays, a valuation date, and Front Arena `denominatedvalue` objects for spot and strike. It returns a dictionary with the mandatory `result` key containing an ACM `DenominatedValue`. The module falls back to a small local result object when tested outside PRIME.

The ADFL `FCustomFunction`, theoretical-model-call, and valuation-model-descriptor templates are available as `front_arena.ADFL_EXAMPLE`. Rates, regime volatilities, weights, and the `Q` matrix are passed in as plain scalars/arrays, which is also the shape required for distributed calculation serialization.

`front_arena.tarf_model_from_market_data` (ADFL template: `front_arena.ADFL_EXAMPLE_MARKET_DATA`) is the market-data-object variant: it takes `FIrCurveInformation` objects for the domestic/foreign curves and an `FMalzParametricVolatilityInformation` object for the vol surface directly, alongside the valuation and maturity dates. `front_arena.market_data_from_front_arena` is the boundary function that does the extraction — it is the only place these FA objects are touched:

- domestic/foreign rate: continuous Act/365 zero rate from `FIrCurveInformation.Rate(valuationDate, maturityDate)`.
- regime volatilities: `FMalzParametricVolatilityInformation.Value(maturityDate, delta, foreignRate, domesticRate)` sampled at `front_arena.DEFAULT_REGIME_DELTAS` (25-delta put / ATM / 25-delta call), used as proxies for the low/mid/high vol regimes.

Everything extracted is handed off as plain floats to the existing state-free `price_tarf`, so the PDE solver never depends on live FA objects.

## Run tests

```bash
cd "/Users/johanahlberg/Visual Studio Code/TARF"
/opt/homebrew/bin/python3 -m pip install -e '.[dev]'
/opt/homebrew/bin/python3 -m pytest -q
```
