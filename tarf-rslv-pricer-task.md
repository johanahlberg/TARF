# Task: TARF pricer with 3-regime local vol (finite difference)

## Objective
Build a finite-difference pricer for Target Accumulation Redemption Forwards (TARF) using a regime-switching local vol (RSLV) model, to replace an existing bump-and-revalue local vol pricer whose Greeks are unstable. Priorities: numerical stability of Greeks over exotic model sophistication, and a clean architecture I can extend later.

## Background / why this design
Two separate sources of Greek noise need separate fixes — don't conflate them:
1. **Local vol recalibration noise**: bumping market inputs and re-inverting a nonparametric Dupire surface is an ill-conditioned inverse problem — small bumps produce disproportionate jagged changes. Fix: calibrate each regime to a *compact parametric smile* (SABR-style — level, skew, curvature; 3-4 params) instead of a full nonparametric surface. A well-posed low-dimensional fit is far smoother under bumping.
2. **Payoff-kink noise**: TARFs have a discontinuous value function at the accumulated-target boundary. Finite-difference Greeks near that kink are noisy on any grid regardless of vol model. Fix: Rannacher startup (a few fully-implicit Euler steps before switching to Crank-Nicolson) plus local grid refinement near the target/knockout boundary.
3. **Prefer analytic/PDE Greeks over bump-and-revalue.** Since we're already solving on a grid, extract delta/gamma via finite differences *on the grid* (or via the adjoint/dual PDE) at the current spot node, rather than re-solving the whole PDE at bumped market levels. This alone removes most of the recalibration-noise problem, independent of the regime model.

## Model
Three-state regime-switching local vol: spot $S$, accumulated coupon/target level $A$, regime $i \in \{1,2,3\}$ (e.g. normal / low-vol / high-vol). Three coupled 2D PDEs over $(S, A)$ linked by a 3×3 generator matrix $Q$:

$$\frac{\partial V_i}{\partial t} + \mathcal{L}_i V_i + \sum_{j \neq i} q_{ij}(V_j - V_i) = 0, \quad i = 1,2,3$$

where $\mathcal{L}_i$ is the standard local-vol diffusion/drift operator parameterized by regime $i$'s smile.

## Numerics
- **Grid**: 2D grid in $(S, A)$ per regime (3 grids, same shape, coupled). Non-uniform spacing in $S$ concentrated near spot and near the barrier/target region; $A$ dimension can often be handled semi-analytically at fixing dates rather than gridded, if that's simpler — use judgement and flag the tradeoff.
- **Time-stepping**: Crank-Nicolson for the diffusion part, with **Rannacher startup** (2-4 fully implicit steps after each non-smooth event: initial payoff, and each fixing/accumulation date) to damp CN oscillation.
- **Operator splitting**: within each timestep, (a) solve the local-vol diffusion PDE independently per regime via ADI/Crank-Nicolson, (b) apply the regime-coupling term — either implicitly in the same solve, or via $\exp(Q \Delta t)$ applied as a separate step. Pick whichever is more stable/simpler to implement correctly; note the tradeoff in code comments.
- **Fixing/accumulation dates**: at each fixing, apply the TARF accumulation logic across all three regime layers identically — bump $A$ by the realized coupon, check against target, knock out if breached. This is a jump condition applied after solving up to that date, before continuing backward.
- **Grid refinement**: refine locally around the accumulated-target boundary in $A$ to control payoff-kink noise (see Background point 2).

## Calibration (two separate stages — don't conflate)
1. **Per-regime smile calibration**: for each regime, fit SABR-style (or similar compact parametric) smile parameters such that the probability-weighted mixture across regimes reproduces the market vanilla smile at each relevant tenor. This should run frequently (e.g. daily) — it's a well-posed weighted fit.
2. **Switching matrix ($Q$) calibration**: smile marginals alone under-determine switching speed (fast- and slow-switching models can both match static smiles with different implied forward-smile dynamics). Calibrate $Q$ against a target that carries dynamic information — e.g. risk-reversal/butterfly term structure convexity, forward-starting vanillas if available, or historical realized-vol regime clustering as a prior. Run this less frequently than the smile fit (e.g. weekly) so day-to-day Greeks aren't destabilized by re-fitting the switching structure under daily noise.

## Deliverables
1. Core PDE solver module: coupled 3-regime ADI/Crank-Nicolson solver with Rannacher startup, parameterized by regime smile params and $Q$.
2. TARF payoff/product module: accumulation logic, target/knockout handling, fixing schedule.
3. Greeks extraction: on-grid delta/gamma/vega (regime-level vega + aggregate), avoiding re-solve-from-scratch bump-and-revalue.
4. Calibration module: (a) per-regime smile fit to market vanilla smile, (b) switching matrix fit, kept as separate, independently-callable routines.
5. Validation/test suite:
   - Degenerate case: single active regime (others given ~zero weight/very slow switching) should reproduce a standard single-regime local-vol TARF price to a specified tolerance — validates the PDE/payoff mechanics independent of the regime-switching machinery.
   - Vanilla check: with no TARF payoff (plain European), the regime-weighted mixture price should match market vanilla prices used in calibration.
   - Greek stability check: compare delta/gamma smoothness (as spot is perturbed) against the existing bump-and-revalue pricer's output, to demonstrate the improvement quantitatively.
6. Brief README explaining the module layout, how to run calibration, how to add/adjust regimes.

## Tech stack
Python, numpy/scipy as the default numerical stack unless you (Claude, working in my repo) see an existing convention in the codebase that should be followed instead — check for that first before scaffolding.

## Suggested approach
Start with a single-regime local-vol TARF PDE pricer (skip the coupling entirely) and get it validated against known vanilla/TARF benchmarks first. Then extend to 3 coupled regimes. This gives a working baseline early and isolates bugs in the regime-coupling logic from bugs in the core PDE/payoff mechanics.
