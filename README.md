# TARF RSLV Pricer

A finite-difference pricer for Target Accumulation Redemption Forwards (TARF) using a regime-switching
local-vol (RSLV) model, built for numerically stable Greeks. The single-regime core follows the
published scheme of **Luo & Shevchenko, *Pricing TARN Using a Finite Difference Method*
(arXiv:1304.7563v2)** and reproduces their benchmark table to within its Monte-Carlo noise; the
three-regime layer is an operator-split extension of the same scheme.

## Layout

- `src/tarf_rslv/model.py`: model objects. `SingleRegimeLocalVolModel` carries a compact parametric
  smile (`local_vol` ATM level + `skew` + `curvature` in log-moneyness); `RegimeSwitchingLocalVolModel`
  is three such layers plus a 3x3 generator matrix.
- `src/tarf_rslv/product.py`: `TARFAccumulator`, a port of the `udmcTRFPayoff` ESL per-fixing
  payoff/accumulation logic, with `target_adjustment` covering the paper's three knockout types
  (0 full gain / 2 part gain / 3 no gain) plus the ESL strike-adjusted variant (1).
- `src/tarf_rslv/solver.py`: `SingleRegimePricer` and `ThreeRegimePricer` -- the finite-difference
  engine (see *Numerical scheme* below) and on-grid Greeks.
- `src/tarf_rslv/calibration.py`: the two calibration stages (see *Calibration* below).
- `src/tarf_rslv/front_arena.py`: AEF-compatible, state-free valuation wrapper and ADFL example.

## Numerical scheme

Matches Luo & Shevchenko:

- **Log-space PDE.** Between fixings each accumulated-amount slice solves the 1-D pricing PDE in
  `x = ln S` on a **uniform** grid (their Section 4.4), with the spot and the strike pinned to grid
  nodes. A uniform log grid is what makes the central-difference stencil second-order consistent --
  an earlier version discretised on a geometric grid with a uniform-grid stencil, which was only
  first-order and biased the price by several percent with no convergence.
- **theta-scheme with Rannacher startup.** Crank-Nicolson (`theta = 1/2`) for the bulk of each
  inter-event segment, preceded by `RANNACHER_STEPS` (2) fully-implicit steps after every non-smooth
  event -- the terminal condition and each fixing jump -- to damp CN oscillation at the payoff kink.
- **Forward jump condition (their eq. 10), explicit in the on-grid accumulated amount:**
  `V(S, t_k^-, A_j) = V(S, t_k, A_j + C_k) + C_k`, with `A_j + C_k` allowed to exceed the target so
  the knockout falls out naturally. The final fixing's jump is applied first (`T -> T^-`) from the
  zero terminal condition.
- **Cubic spline** interpolation in the accumulated-amount coordinate at each jump (their Section
  4.2.2) -- linear/quadratic interpolation can converge to the wrong answer for discretely-sampled
  path dependence (Forsyth et al. 2002). Linear fallback below 4 accumulation nodes.
- **Boundary condition** `d^2V/dS^2 = 0` at both far ends (their Section 4.3), as linear-in-S
  extrapolation; the resulting linear system is pentadiagonal.
- **Discounting** is embedded in the spatial operator, so the price is read straight off the grid at
  `(spot, accumulated_value)`.

### Seasoned deals

`fixing_times` are year fractions **from the valuation date**. Fixings already in the past come
through negative and are dropped automatically; a fixing on the valuation date that has not yet been
observed is kept (pass `0.0`) and settled at the current spot, undiscounted. The realised amount from
past fixings is carried in `accumulated_value` (same units as `target_level` -- CHF-pip intrinsic
unless `inverted_target`), and the knockout uses `remaining = target_level - accumulated`. Pass
per-fixing `notional1` / `notional2` as full-length vectors (one per original fixing); indices stay
aligned with the untrimmed schedule.

The accumulated-amount grid spans `[0, target_level]` regardless of how seasoned the deal is, so a
deal with little target left has fewer live nodes -- raise `num_target` (the default 80 is converged
for a fresh deal; ~320 covers a heavily seasoned one). `num_spot` / `num_target` / `n_steps` are
constructor args on the pricers and trailing keyword args on the `front_arena` wrappers.

The three-regime solve reuses the same operator per regime and couples them with `exp(Q dt)` applied
after each diffusion sub-step (operator splitting). In the single-regime limit (identical regimes)
`exp(Q dt)` acts as the identity on the common value, so `ThreeRegimePricer` reduces exactly to
`SingleRegimePricer`.

## Greeks

`SingleRegimePricer.tarf_greeks` / `ThreeRegimePricer.tarf_greeks` read delta and gamma **directly
off the PDE grid** at the pinned spot node (central differences in `x`, converted to `S`), with no
bump-and-revalue -- each revalue would re-land the spot on a different grid and inject noise. Vega is
a single vol bump (aggregate, plus per-regime for the coupled pricer). `test_greek_stability.py`
shows the on-grid delta curve is >10x smoother than bump-and-revalue under a spot sweep.

## Calibration

Two independent stages (`calibration.py`), deliberately kept separate:

1. **Per-regime smile** -- `calibrate_regime_smiles(log_moneyness, market_vols, regime_weights, ...)`
   fits a shared base smile (`atm_vol`, `skew`, `curvature`) so the regime-weighted model reproduces
   the market vanilla smile. Well-posed, meant to run daily. The regime vol-dispersion
   (`regime_spread`) is held fixed here -- at the money it is degenerate with the ATM level, so it is
   treated as a switching-structure parameter. Returns per-regime dicts that drop straight into
   `SingleRegimeLocalVolModel(**d)`.
2. **Switching matrix** -- `calibrate_switching_rate(pi0, stationary, regime_vols, t1, t2,
   target_forward_variance)` fits the scalar switch speed of `Q = rate * (1 pi^T - I)` against a
   **forward-variance** target (the simplest quantity carrying forward-smile / switching-speed
   information). Requires `pi0 != stationary` for identification. Meant to run less frequently.

Both are verified by synthetic round-trip recovery in `test_calibration_stage.py`. The retained
helpers `calibrate_regime_smile` (single quadratic smile) and `fit_switching_matrix` (P -> Q = P - I)
are lightweight and used by the interface tests.

## Tests

- `test_paper_benchmark.py` -- **correctness gate**: reproduces Luo & Shevchenko Table 1 (all three
  knockout types) and the Black-Scholes vanilla limit.
- `test_greek_stability.py` -- on-grid vs bump-and-revalue Greek smoothness.
- `test_calibration_stage.py` -- round-trip recovery of both calibration stages.
- `test_single_regime_pricer.py`, `test_three_regime_pricer.py` -- architecture, regime-coupling
  reduction, leveraged-loss/barrier economics.
- `test_tarf_payoff.py` -- `TARFAccumulator.settle` against the ESL per-fixing logic.
- `test_front_arena_interface.py` -- ACM boundary and serialization adapter checks.
- `test_seasoned_tarf.py` -- past fixings dropped, valuation-date fixing priced in, grid-resolution
  passthrough on the `front_arena` wrappers.

## Front Arena AEF interface

`front_arena.tarf_model` follows the AEF proprietary-valuation contract: scalar/array inputs, a
valuation date, and `denominatedvalue` objects for spot and strike; returns a dict with the mandatory
`result` key holding an ACM `DenominatedValue` (with a local fallback outside PRIME).
`front_arena.tarf_model_from_market_data` is the variant taking `FIrCurveInformation` /
`FMalzParametricVolatilityInformation` objects directly; `front_arena.market_data_from_front_arena`
is the single boundary function that touches them. ADFL templates are in `front_arena.ADFL_EXAMPLE`
and `front_arena.ADFL_EXAMPLE_MARKET_DATA`.

Worked examples with mock FA market-data objects are in `examples/` -- `price_usdchf_tarf.py`
(a fresh USDCHF seller TARF) and `price_seasoned_tarf.py` (the same deal valued mid-life), each with
a flat-vol Monte-Carlo cross-check. Run e.g. `python examples/price_seasoned_tarf.py`.

## Run tests

```bash
cd "/Users/johanahlberg/Visual Studio Code/TARF"
/opt/homebrew/bin/python3 -m pip install -e '.[dev]'
/opt/homebrew/bin/python3 -m pytest -q
```
