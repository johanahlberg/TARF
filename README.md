# TARF RSLV Pricer

This repository is structured as a compact numerical-pricing prototype for a Target Accumulation Redemption Forward (TARF) using finite-difference local-volatility solvers and a coupled 3-regime extension.

## Layout

- `src/tarf_rslv/model.py`: model objects, including the single-regime local-vol setup and the default regime generator matrix.
- `src/tarf_rslv/product.py`: TARF accumulation and payoff logic.
- `src/tarf_rslv/solver.py`: single-regime and coupled three-regime PDE pricers.
- `src/tarf_rslv/calibration.py`: compact smile and switching-matrix calibration helpers.
- `src/tarf_rslv/front_arena.py`: AEF-compatible, state-free valuation wrapper and ADFL example.
- `tests/test_single_regime_pricer.py`: validation checks for the baseline architecture.
- `tests/test_three_regime_pricer.py`: regime coupling, finiteness, and degenerate-case checks.
- `tests/test_front_arena_interface.py`: ACM boundary and serialization-oriented adapter checks.

## Current status

The repository currently contains a stable implementation of the main architecture described in the brief:

- single-regime local-vol model shell,
- TARF accumulation logic,
- implicit finite-difference spot stepping on a 2D spot/accumulation grid,
- Rannacher-style startup around the initial kink and fixing dates,
- operator-split 3-regime PDE coupling with exact short-step matrix exponentials,
- compact parametric smile and generator-matrix calibration helpers,
- regression coverage for the single-regime limit of the coupled solver.

## Numerical design

The solver uses an implicit tridiagonal spot solve for each accumulation slice and regime. Accumulation advection is applied on the `A` grid, followed by `expm(Q * dt)` coupling across regimes. This operator split keeps the spatial linear algebra cheap while retaining stable generator propagation. The current implementation uses a compact `A` grid and geometric spot grid; local target-boundary refinement and full market calibration remain natural production extensions.

## Front Arena AEF interface

`front_arena.tarf_model` follows the AEF proprietary-valuation contract. Its inputs are scalar values, arrays, a valuation date, and Front Arena `denominatedvalue` objects for spot and strike. It returns a dictionary with the mandatory `result` key containing an ACM `DenominatedValue`. The module falls back to a small local result object when tested outside PRIME.

The ADFL `FCustomFunction`, theoretical-model-call, and valuation-model-descriptor templates are available as `front_arena.ADFL_EXAMPLE`. Keep market-data retrieval in ADFL: pass the resolved Malz inputs, rates, fixing times, regime parameters, weights, and `Q` matrix into the state-free Python function. This is also the shape required for distributed calculation serialization.

## Run tests

```bash
cd "/Users/johanahlberg/Visual Studio Code/TARF"
/opt/homebrew/bin/python3 -m pip install -e '.[dev]'
/opt/homebrew/bin/python3 -m pytest -q
```
