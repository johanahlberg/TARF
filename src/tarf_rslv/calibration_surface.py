"""Full FX-surface calibration for the local-vol model, single- or three-regime.

``calibrate_regime_model(surface, n_regimes=1 or 3)`` returns a :class:`CalibratedRegimeModel` whose
``price_tarf`` / ``tarf_greeks`` dispatch to ``SingleRegimePricer`` (``n_regimes=1`` -- a single
Dupire local-vol model, ~3x faster, no forward-smile dynamics) or ``ThreeRegimePricer``
(``n_regimes=3``, default). It runs two stages, kept independent (as in :mod:`tarf_rslv.calibration`):

**Stage 1 -- static smile, to the whole surface.** Each tenor is fitted once with an arbitrage-free
**SVI** slice (``svi.fit_svi_surface``); the calibration then moves only that slice's level /
skew-scale ``(a, b)`` per tenor, with the shape ``(rho, m, sigma)`` frozen -- so a candidate slice is
linear in the free variables and needs no inner fit. Regime ``i`` is the base slice with total
variance scaled by ``(1 +- level_spread)^2`` **and** ``rho`` shifted by ``-+ skew_spread`` -- the
stressed regime is higher-vol *and* steeper-skew. Those two structural scalars are a regime
interpretation, not calibrated (see :class:`SharedSmileParams`). Each regime slice gives an
**analytic Dupire** local-vol surface, and the model's own implied-vol surface -- the forward
regime-switching PDE of :mod:`tarf_rslv.vanilla` -- is driven onto the market by least squares over
the ``(a, b)``.

**Stage 2 -- switching speed.** ``calibrate_switch_rate`` fits the scalar rate of
``Q = rate (1 pi^T - I)`` so the model's 25d **RR and BF** term structures (from the forward PDE)
match the market's decay with maturity -- the smile-flattening the switch rate controls. It is only
weakly identified by vanillas (~a few bp of response); the fit is a coarse grid and falls back to a
persistence prior when the term structure does not respond.

The result is a :class:`CalibratedRegimeModel` -- three time-dependent Dupire regimes plus the
generator -- which plugs straight into ``ThreeRegimePricer``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, least_squares

from .svi import _ARB_GRID, SVISlice, svi_local_vol, svi_total_variance_and_derivs
from .vanilla import RegimeForwardPDE
from .vol_surface import FXVolSurface, bs_forward_price, implied_vol_from_forward_price

REGIME_MULTIPLIERS = np.array([-1.0, 0.0, 1.0])
PDE_MIN_TENOR = 1.0 / 26.0                     # ~2 weeks; below this the forward PDE uses the mixture
_SVI_LO = np.array([-2.0, 1e-4, -0.999, -1.5, 1e-3])
_SVI_HI = np.array([2.0, 3.0, 0.999, 1.5, 1.5])
_PRIOR_SWITCH_RATE = 2.0                       # regimes persist ~6 months when the rate is unidentified
# Default regime interpretation (structural priors -- not identified by a static surface):
#   vol multipliers ~ (1 - 0.18, 1, 1 + 0.18) = calm / mid / stressed
#   the stressed regime also carries rho shifted by -0.15 (steeper put skew)
_DEFAULT_LEVEL_SPREAD = 0.18
_DEFAULT_SKEW_SPREAD = 0.15


# --------------------------------------------------------------------------------------------------
# Stage-1 parameters
# --------------------------------------------------------------------------------------------------
@dataclass
class SharedSmileParams:
    """Per-tenor SVI with the shape ``(rho, m, sigma)`` frozen from the market fit and the level /
    wing-scale ``(a, b)`` free -- a candidate slice is analytic in those, no inner fit. (Freeing
    ``sigma`` as well is poorly conditioned and thrashes the optimiser, so the residual ~2 bp of
    wing-shape mismatch under strong regime dispersion is accepted, not chased.)

    Two global structural scalars set how the three regimes differ from the base slice:

        regime i:  a, b   ->  a, b  * (1 + m_i * level_spread)^2      (level / vol dispersion)
                   rho    ->  rho - m_i * skew_spread                 (skew dispersion)

    with ``m = (-1, 0, +1)`` -- so the high-vol regime is also the steeper-skew regime. These are
    **not** identified by a static vanilla surface (free them in the fit and they collapse to zero),
    so they are set from a regime interpretation / realised-vol prior, not calibrated.
    """

    tenors: np.ndarray
    frozen: np.ndarray               # (N, 3): rho, m, sigma   (fixed)
    free: np.ndarray                 # (N, 2): a, b            (free)
    level_spread: float              # structural prior
    skew_spread: float               # structural prior
    free0: np.ndarray                # the market-fit (a, b), for the bounding box

    @property
    def n_buckets(self) -> int:
        return len(self.tenors)

    def slices(self) -> list[SVISlice]:
        return [
            SVISlice(float(self.tenors[b]), float(self.free[b, 0]), float(self.free[b, 1]),
                     float(self.frozen[b, 0]), float(self.frozen[b, 1]), float(self.frozen[b, 2]))
            for b in range(self.n_buckets)
        ]

    def free_vector(self) -> np.ndarray:
        return self.free.ravel()

    def with_free_vector(self, vector: np.ndarray) -> "SharedSmileParams":
        return replace(self, free=np.asarray(vector, dtype=float).reshape(self.n_buckets, 2))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        a0, b0 = self.free0[:, 0], self.free0[:, 1]
        lo = np.stack([a0 - 0.02, np.maximum(b0 * 0.4, 1e-4)], axis=1).ravel()
        hi = np.stack([a0 + 0.02, b0 * 2.2 + 0.02], axis=1).ravel()
        return lo, hi

    def regime_slices(self, regime: int) -> list[SVISlice]:
        m = REGIME_MULTIPLIERS[regime]
        var_mult = (1.0 + m * self.level_spread) ** 2
        rho_shift = m * self.skew_spread
        return [
            SVISlice(s.tenor, s.a * var_mult, s.b * var_mult,
                     min(max(s.rho - rho_shift, -0.999), 0.999), s.m, s.sigma)
            for s in self.slices()
        ]

    @classmethod
    def from_surface(cls, surface: FXVolSurface, level_spread: float, skew_spread: float) -> "SharedSmileParams":
        frozen = np.array([[s.rho, s.m, s.sigma] for s in surface.svi_slices], dtype=float)
        free = np.array([[s.a, s.b] for s in surface.svi_slices], dtype=float)
        return cls(np.asarray(surface._tenors, dtype=float), frozen, free.copy(),
                   float(level_spread), float(skew_spread), free.copy())


# --------------------------------------------------------------------------------------------------
# Calibrated model
# --------------------------------------------------------------------------------------------------
class _LocalVolRegime:
    """One regime for ``ThreeRegimePricer``: a bilinear-interpolated Dupire local-vol grid."""

    def __init__(self, x_grid, t_grid, sigma_grid, rate, dividend_yield, atm_level, vol_floor):
        self._x = x_grid
        self._t = t_grid
        self._sigma = sigma_grid
        self.rate = float(rate)
        self.dividend_yield = float(dividend_yield)
        self.local_vol = float(atm_level)
        self.vol_floor = float(vol_floor)
        self.skew = 0.0
        self.curvature = 0.0
        self.smile_ref = float(np.exp(x_grid[len(x_grid) // 2]))
        self.sigma_ref = float(atm_level)
        self.spot = self.smile_ref

    def local_volatility(self, spot, t: float = 0.0) -> np.ndarray:
        x = np.log(np.maximum(np.asarray(spot, dtype=float), 1e-300))
        it = int(np.clip(np.searchsorted(self._t, t) - 1, 0, len(self._t) - 2))
        t0, t1 = self._t[it], self._t[it + 1]
        wt = 0.0 if t1 == t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        row = (1.0 - wt) * self._sigma[it] + wt * self._sigma[it + 1]
        return np.interp(x, self._x, row)

    def shifted(self, dv: float) -> "_LocalVolRegime":
        out = _LocalVolRegime.__new__(_LocalVolRegime)
        out.__dict__.update(self.__dict__)
        out._sigma = np.maximum(self._sigma + dv, self.vol_floor)
        out.local_vol = self.local_vol + dv
        return out


@dataclass
class CalibratedRegimeModel:
    """Calibrated model. Holds one *or* three time-dependent Dupire regimes; ``price_tarf`` /
    ``tarf_greeks`` dispatch to ``SingleRegimePricer`` (n = 1) or ``ThreeRegimePricer`` (n = 3)."""

    spot: float
    rate: float
    dividend_yield: float
    regimes: list[_LocalVolRegime]
    q: np.ndarray
    regime_weights: np.ndarray
    params: SharedSmileParams
    surface: FXVolSurface
    time_varying: bool = True

    @property
    def n_regimes(self) -> int:
        return len(self.regimes)

    def bumped_vol(self, dv: float) -> "CalibratedRegimeModel":
        return replace(self, regimes=[r.shifted(dv) for r in self.regimes])

    def regime_implied_vol(self, i: int, t: float, y: np.ndarray) -> np.ndarray:
        w, _, _, _ = svi_total_variance_and_derivs(self.params.regime_slices(i), max(float(t), 1e-8), y)
        return np.sqrt(np.maximum(w, 1e-12) / max(float(t), 1e-8))

    # -- SingleRegimeLocalVolModel interface (used when n_regimes == 1) ---------------------
    @property
    def sigma_ref(self) -> float:
        return self.regimes[0].sigma_ref

    @property
    def local_vol(self) -> float:
        return self.regimes[0].local_vol

    @property
    def strike(self) -> float:
        return self.regimes[0].smile_ref

    skew = 0.0
    curvature = 0.0

    @property
    def smile_ref(self) -> float:
        return self.regimes[0].smile_ref

    @property
    def vol_floor(self) -> float:
        return self.regimes[0].vol_floor

    def local_volatility(self, spot, t: float = 0.0) -> np.ndarray:
        return self.regimes[0].local_volatility(spot, t)

    # -- pricing dispatch -----------------------------------------------------------------
    def _pricer(self, num_spot: int, num_target: int, n_steps: int):
        from .solver import SingleRegimePricer, ThreeRegimePricer

        if self.n_regimes == 1:
            return SingleRegimePricer(self, num_spot, num_target, n_steps)
        return ThreeRegimePricer(self, self.regime_weights, num_spot, num_target, n_steps)

    def price_tarf(self, tarf, maturity: float, *, num_spot: int = 161, num_target: int = 100,
                   n_steps: int = 120) -> float:
        return self._pricer(num_spot, num_target, n_steps).price_tarf(tarf, float(maturity))

    def tarf_greeks(self, tarf, maturity: float, *, num_spot: int = 161, num_target: int = 100,
                    n_steps: int = 120, vega_bump: float = 1e-4) -> dict:
        return self._pricer(num_spot, num_target, n_steps).tarf_greeks(tarf, float(maturity), vega_bump)


def _regime_local_vol_grids(params, forward_of_t, x_grid, t_grid, vol_floor, vol_cap, n_regimes):
    grids = []
    for i in range(n_regimes):
        regime_slices = params.regime_slices(i)
        rows = np.empty((len(t_grid), len(x_grid)))
        for kk, t in enumerate(t_grid):
            k = x_grid - np.log(forward_of_t(float(t)))
            rows[kk] = svi_local_vol(regime_slices, float(max(t, 1e-4)), k, vol_floor=vol_floor, vol_cap=vol_cap)
        grids.append(rows)
    return grids


def _build_model(surface, params, q, weights, x_grid, t_grid, max_t, n_regimes=3):
    grids = _regime_local_vol_grids(
        params, surface.forward, x_grid, t_grid, surface.vol_floor, surface.vol_cap, n_regimes
    )
    front = params.slices()[0]
    front_atm = float(np.sqrt(max(front.total_variance(0.0), 1e-12) / front.tenor))
    m = REGIME_MULTIPLIERS if n_regimes == 3 else np.zeros(n_regimes)
    regimes = [
        _LocalVolRegime(
            x_grid, t_grid, grids[i], surface.domestic_zero(max_t), surface.foreign_zero(max_t),
            front_atm * (1.0 + m[i] * params.level_spread), surface.vol_floor,
        )
        for i in range(n_regimes)
    ]
    return CalibratedRegimeModel(
        spot=surface.spot, rate=surface.domestic_zero(max_t), dividend_yield=surface.foreign_zero(max_t),
        regimes=regimes, q=np.asarray(q, dtype=float), regime_weights=np.asarray(weights, dtype=float),
        params=params, surface=surface,
    )


# --------------------------------------------------------------------------------------------------
# Model-implied vanilla surface
# --------------------------------------------------------------------------------------------------
def _time_averaged_regime_probs(q: np.ndarray, weights: np.ndarray, t: float) -> np.ndarray:
    grid = np.linspace(0.0, t, 17)
    probs = np.array([expm(q.T * u) @ weights for u in grid])
    return np.sum(0.5 * (probs[1:] + probs[:-1]) * np.diff(grid)[:, None], axis=0) / t


def _mixture_implied_vol(model: CalibratedRegimeModel, t: float, y: np.ndarray, pi_bar: np.ndarray) -> np.ndarray:
    forward = model.spot * np.exp((model.rate - model.dividend_yield) * t)
    out = np.empty(np.size(y))
    for j, yy in enumerate(np.atleast_1d(y)):
        strike = forward * np.exp(yy)
        price = sum(
            pi_bar[i] * bs_forward_price(forward, strike, float(model.regime_implied_vol(i, t, np.array([yy]))[0]), t, True)
            for i in range(model.n_regimes)
        )
        out[j] = implied_vol_from_forward_price(price, forward, strike, t, True)
    return out


def _mixture_implied_vol_nodes(model: CalibratedRegimeModel, nodes) -> np.ndarray:
    out = np.empty(len(nodes))
    prob_cache: dict[float, np.ndarray] = {}
    for n, (t, y, _, _) in enumerate(nodes):
        if t not in prob_cache:
            prob_cache[t] = _time_averaged_regime_probs(model.q, model.regime_weights, t)
        out[n] = _mixture_implied_vol(model, t, np.array([y]), prob_cache[t])[0]
    return out


def _pde_implied_vol_nodes(model, nodes, *, num_x, steps_per_year):
    tenors = sorted({t for t, _, _, _ in nodes})
    pde_tenors = [t for t in tenors if t >= PDE_MIN_TENOR]
    result: dict[tuple[float, float], float] = {}

    if pde_tenors:
        max_t = max(pde_tenors)
        sig_ref = float(max(model.regime_implied_vol(model.n_regimes // 2, max_t, np.array([0.0]))[0], 0.05))
        local_vol_fns = [(lambda x, tt, r=r: r.local_volatility(np.exp(x), tt)) for r in model.regimes]
        pde = RegimeForwardPDE(
            model.spot, model.rate, model.dividend_yield, local_vol_fns,
            model.q, model.regime_weights, max_t=max_t, sigma_ref=sig_ref,
            num_x=num_x, steps_per_year=steps_per_year,
        )
        y_by_tenor = {t: np.array(sorted({yy for tt, yy, _, _ in nodes if tt == t})) for t in pde_tenors}
        strikes_by_tenor = {
            t: model.spot * np.exp((model.rate - model.dividend_yield) * t) * np.exp(y_by_tenor[t])
            for t in pde_tenors
        }
        pde_iv = pde.implied_vols(pde_tenors, strikes_by_tenor)
        for t in pde_tenors:
            for y, iv in zip(y_by_tenor[t], pde_iv[t]):
                result[(t, float(y))] = float(iv)

    short_nodes = [nd for nd in nodes if nd[0] < PDE_MIN_TENOR]
    if short_nodes:
        for nd, iv in zip(short_nodes, _mixture_implied_vol_nodes(model, short_nodes)):
            result[(nd[0], float(nd[1]))] = float(iv)

    return np.array([result[(t, float(y))] for t, y, _, _ in nodes])


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
@dataclass
class SurfaceCalibrationReport:
    success: bool
    rms_vol_error: float
    max_vol_error: float
    node_errors: np.ndarray
    svi_rms_vol_error: float
    max_butterfly_violation: float
    max_calendar_violation: float
    level_spread: float
    skew_spread: float
    switch_rate: float
    switch_rate_identified: bool
    n_pde_passes: int
    n_residual_evals: int
    barrier_rms_price_error: float = float("nan")   # set when barrier_quotes are given
    n_barrier_quotes: int = 0


def _fit_barrier_structural(surface, params_at_x, weights, quotes, x_grid, t_grid, max_t, x0, verbose):
    """Fit (level_spread, skew_spread, switch_rate) to the barrier/touch quotes, given the
    already-fitted per-tenor SVI (a, b)."""
    from .barriers import RegimeBarrierPricer

    market = np.array([qt.market_price for qt in quotes])
    qweight = np.array([qt.weight for qt in quotes])

    def resid(theta: np.ndarray) -> np.ndarray:
        ls, ss, rate = float(theta[0]), float(theta[1]), float(theta[2])
        params = replace(params_at_x, level_spread=ls, skew_spread=ss)
        model = _build_model(surface, params, _generator(rate, weights), weights, x_grid, t_grid, max_t, 3)
        pricer = RegimeBarrierPricer(model, num_spot=221, steps_per_year=400, min_steps=100)
        model_prices = np.array([qt.model_price(pricer) for qt in quotes])
        # weak Tikhonov on skew_spread only: with few / symmetric touches it trades off against
        # level_spread; the switch rate and overall dispersion are identified, the split is not
        skew_pull = 0.04 * (ss - _DEFAULT_SKEW_SPREAD * ls / max(_DEFAULT_LEVEL_SPREAD, 1e-6))
        return np.concatenate([(model_prices - market) * qweight, [skew_pull]])

    sol = least_squares(
        resid, np.asarray(x0, dtype=float),
        bounds=([0.0, 0.0, 0.2], [0.45, 0.55, 12.0]),
        x_scale=[0.05, 0.05, 2.0], diff_step=[0.03, 0.03, 0.15], max_nfev=45, ftol=1e-6, xtol=1e-6,
    )
    rms = float(np.sqrt(np.mean(sol.fun[:len(quotes)] ** 2)))
    if verbose:
        print(f"  [barriers] level_spread={sol.x[0]:.3f} skew_spread={sol.x[1]:.3f} "
              f"switch_rate={sol.x[2]:.2f}  rms price err {rms * 1e4:.1f} bp")
    return tuple(float(v) for v in sol.x), rms


def _arbitrage_diagnostics(params: SharedSmileParams, n_regimes: int = 3) -> tuple[float, float]:
    """Worst butterfly violation over the regimes, and worst calendar crossing of the base."""
    bf = 0.0
    for i in range(n_regimes):
        for slc in params.regime_slices(i):
            bf = max(bf, float(np.max(np.maximum(-slc.durrleman_g(_ARB_GRID), 0.0))))
    cal = 0.0
    base = params.slices()
    for a, b in zip(base[:-1], base[1:]):
        cal = max(cal, float(np.max(np.maximum(a.total_variance(_ARB_GRID) - b.total_variance(_ARB_GRID), 0.0))))
    return bf, cal


# --------------------------------------------------------------------------------------------------
# stage 2: switch rate from the 25d risk-reversal + butterfly term structure
# --------------------------------------------------------------------------------------------------
def _generator(rate: float, stationary: np.ndarray) -> np.ndarray:
    n = len(np.asarray(stationary))
    return rate * (np.ones((n, 1)) @ np.asarray(stationary)[None, :] - np.eye(n))


def _rr_bf_term_structure_from_iv(iv_by_tenor: dict[float, np.ndarray], tenors) -> tuple[np.ndarray, np.ndarray]:
    """``iv_by_tenor[t]`` holds ``[25dP, ATM, 25dC]``; returns ``(RR(T), BF(T))``."""
    rr = np.array([iv_by_tenor[t][2] - iv_by_tenor[t][0] for t in tenors])
    bf = np.array([0.5 * (iv_by_tenor[t][2] + iv_by_tenor[t][0]) - iv_by_tenor[t][1] for t in tenors])
    return rr, bf


def calibrate_switch_rate(
    model: CalibratedRegimeModel,
    *,
    num_x: int = 501,
    steps_per_year: int = 400,
    rate_bounds: tuple[float, float] = (0.2, 12.0),
) -> tuple[float, bool]:
    """Fit the switch rate so the model's 25d **RR and BF term structures** (from the forward PDE --
    the mixture surrogate is switch-rate-blind when the regime distribution starts stationary) match
    the market's decay with maturity. Returns ``(rate, identified)``; ``identified`` is ``False``
    (rate = the persistence prior) when neither term structure responds to the rate."""
    surface = model.surface
    tenors = [float(s.tenor) for s in surface.smiles if s.tenor >= PDE_MIN_TENOR]
    if len(tenors) < 3:
        return _PRIOR_SWITCH_RATE, False
    knot_k = {float(s.tenor): k for s, k in zip(surface.smiles, surface._knot_y)}

    market: dict[float, np.ndarray] = {}
    for s, k in zip(surface.smiles, surface._knot_y):
        if float(s.tenor) in tenors:
            market[float(s.tenor)] = np.asarray(
                next(sl for sl in surface.svi_slices if sl.tenor == s.tenor).implied_vol(
                    np.array([k[1], 0.0, k[-2]])), dtype=float)
    m_rr, m_bf = _rr_bf_term_structure_from_iv(market, tenors)
    weight = np.array(tenors)

    bf_nodes = [(t, float(k), 0.0, 0.0) for t in tenors for k in (knot_k[t][1], 0.0, knot_k[t][-2])]

    def model_rr_bf(rate: float) -> tuple[np.ndarray, np.ndarray]:
        m = replace(model, q=_generator(rate, model.regime_weights))
        iv = _pde_implied_vol_nodes(m, bf_nodes, num_x=num_x, steps_per_year=steps_per_year)
        by_t = {t: iv[3 * i:3 * i + 3] for i, t in enumerate(tenors)}
        return _rr_bf_term_structure_from_iv(by_t, tenors)

    # coarse grid -- the RR/BF-vs-rate response is only a few bp over the whole range, so a fine
    # optimiser would chase noise
    grid = np.array([0.3, 0.7, 1.5, 3.0, 6.0, 10.0])
    curves = {r: model_rr_bf(float(r)) for r in grid}
    obj = np.array([
        float(np.sum(((curves[r][0] - m_rr) ** 2 + (curves[r][1] - m_bf) ** 2) * weight)) for r in grid
    ])
    rr_swing = max(np.max(np.abs(c[0] - curves[grid[0]][0])) for c in curves.values())
    bf_swing = max(np.max(np.abs(c[1] - curves[grid[0]][1])) for c in curves.values())
    if max(rr_swing, bf_swing) < 3e-4 or (obj.max() - obj.min()) < 0.05 * obj.max():
        return _PRIOR_SWITCH_RATE, False
    return float(grid[int(np.argmin(obj))]), True


# kept for backward compatibility / direct use
def calibrate_switch_rate_to_term_structure(surface, regime_atm_levels, pi0, stationary=None, *, max_rate=40.0):
    """Fit ``Q = rate (1 stationary^T - I)`` to the ATM forward-variance term structure."""
    regime_atm = np.asarray(regime_atm_levels, dtype=float)
    pi0 = np.asarray(pi0, dtype=float)
    stationary = pi0 if stationary is None else np.asarray(stationary, dtype=float)
    if regime_atm.shape != (3,) or pi0.shape != (3,) or stationary.shape != (3,):
        raise ValueError("regime_atm_levels, pi0, stationary must all be length 3")
    if np.allclose(pi0, stationary, atol=1e-9):
        raise ValueError("pi0 must differ from stationary for the switch rate to be identifiable")
    tenors = np.array([s.tenor for s in surface.smiles], dtype=float)
    market = np.array([s.atm for s in surface.smiles], dtype=float) ** 2 * tenors

    def fwd_var(rate):
        q = _generator(rate, stationary)
        out = np.empty(len(tenors))
        for k, t in enumerate(tenors):
            g = np.linspace(0.0, float(t), 48)
            v = np.array([(expm(q.T * u) @ pi0) @ regime_atm ** 2 for u in g])
            out[k] = float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(g)))
        return out

    def gap(rate):
        return float(np.sum((fwd_var(rate) - market) * tenors))

    lo, hi = gap(1e-8), gap(max_rate)
    rate = float(brentq(gap, 1e-8, max_rate, xtol=1e-8)) if np.sign(lo) != np.sign(hi) else (
        0.0 if abs(lo) < abs(hi) else max_rate)
    return {"switch_rate": rate, "q": _generator(rate, stationary),
            "model_forward_variance": fwd_var(rate), "market_forward_variance": market, "tenors": tenors}


# --------------------------------------------------------------------------------------------------
# the calibration
# --------------------------------------------------------------------------------------------------
def calibrate_regime_model(
    surface: FXVolSurface,
    regime_weights: np.ndarray | list[float] = (0.25, 0.5, 0.25),
    *,
    n_regimes: int = 1,
    level_spread: float = _DEFAULT_LEVEL_SPREAD,
    skew_spread: float = _DEFAULT_SKEW_SPREAD,
    calibrate_q: bool = True,
    barrier_quotes: list | None = None,
    n_barrier_rounds: int = 2,
    q: np.ndarray | None = None,
    target: str = "hybrid",
    n_pde_passes: int = 3,
    q_refit_passes: int = 2,
    num_x: int = 701,
    steps_per_year: int = 600,
    dupire_nt: int = 41,
    dupire_nx: int = 161,
    max_nfev_inner: int = 140,
    verbose: bool = False,
) -> tuple[CalibratedRegimeModel, SurfaceCalibrationReport]:
    """Calibrate the model to ``surface`` and return a ready-to-price :class:`CalibratedRegimeModel`.

    ``n_regimes=1`` (default) -- a **single Dupire local-vol** model on the arbitrage-free SVI
    surface (priced by ``SingleRegimePricer``, ~3x faster, no forward-smile dynamics).
    ``n_regimes=3`` -- the coupled regime-switching model: regimes differ in level and skew by
    ``level_spread`` / ``skew_spread``.

    A static vanilla surface does **not** identify the regime structure. There are three ways to set
    it, in increasing order of soundness:

    * defaults / a realised-vol regime prior (``level_spread`` / ``skew_spread`` as given; the switch
      rate from ``calibrate_q`` against the RR/BF decay, which is only weakly identified);
    * ``barrier_quotes`` -- a list of :class:`~tarf_rslv.barriers.OneTouchQuote`: the calibration then
      **alternates** ``n_barrier_rounds`` times between refitting the per-tenor SVI ``(a, b)`` to the
      vanilla surface and fitting ``{level_spread, skew_spread, switch_rate}`` to the touch/barrier
      prices (which *do* carry the forward-smile / path-dependence information). ``calibrate_q`` is
      then ignored.

    Stage 1 always fits per-tenor SVI ``(a, b)`` so the model reprices the whole surface.
    ``target``: ``"hybrid"`` (default) fits a smooth mixture surrogate re-anchored ``n_pde_passes``
    times by the forward-PDE correction; ``"pde"`` / ``"mixture"`` are also available.
    """
    if n_regimes not in (1, 3):
        raise ValueError("n_regimes must be 1 or 3")
    if target not in {"hybrid", "pde", "mixture"}:
        raise ValueError("target must be 'hybrid', 'pde' or 'mixture'")
    if barrier_quotes and n_regimes != 3:
        raise ValueError("barrier calibration needs n_regimes=3 (there is nothing to fit otherwise)")

    if n_regimes == 1:
        weights = np.array([1.0])
        level_spread = skew_spread = 0.0
        calibrate_q = False
    else:
        weights = np.asarray(regime_weights, dtype=float)
        if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("regime_weights must be a length-3 vector summing to 1")
    q = (_generator(_PRIOR_SWITCH_RATE, weights) if q is None else np.asarray(q, dtype=float))

    nodes = surface.market_nodes()
    market_iv = np.array([iv for _, _, iv, _ in nodes])
    vega_w = np.array([w for _, _, _, w in nodes])
    vega_w = vega_w / vega_w.mean()

    template = SharedSmileParams.from_surface(surface, level_spread, skew_spread)
    tenors = template.tenors
    n = template.n_buckets
    max_t = float(tenors[-1])

    node_bucket = np.array([int(np.argmin(np.abs(tenors - t))) for t, _, _, _ in nodes])
    jac_sparsity = np.zeros((len(nodes), 2 * n), dtype=bool)
    for row, b in enumerate(node_bucket):
        for coef in range(2):
            for bb in (b - 1, b, b + 1):
                if 0 <= bb < n:
                    jac_sparsity[row, bb * 2 + coef] = True

    max_atm = float(max(surface.svi_slices[i].implied_vol(np.array([0.0]))[0] for i in range(n)))
    half = max(8.0 * max_atm * np.sqrt(max_t), 0.35)
    x_grid = np.log(surface.spot) + np.linspace(-half, half, int(dupire_nx) | 1)
    t_grid = np.concatenate([[1e-4], np.linspace(max_t / dupire_nt, max_t, dupire_nt)])

    lo, hi = template.bounds()
    evals = {"n": 0}

    def run_stage1(x0, q_, passes, label=""):
        x = np.clip(x0, lo, hi)
        correction = np.zeros(len(nodes))
        ok = False
        for outer in range(passes):
            ref = market_iv - correction

            def residual(vec, _ref=ref):
                evals["n"] += 1
                model = _build_model(surface, template.with_free_vector(vec), q_, weights, x_grid, t_grid, max_t, n_regimes)
                if target == "pde":
                    return (_pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                            - market_iv) * vega_w
                return (_mixture_implied_vol_nodes(model, nodes) - _ref) * vega_w

            sol = least_squares(residual, x, bounds=(lo, hi), method="trf", jac_sparsity=jac_sparsity,
                                x_scale="jac", max_nfev=max_nfev_inner, ftol=1e-8, xtol=1e-10, gtol=1e-10)
            x, ok = sol.x, bool(sol.success)

            if target == "hybrid" and outer < passes - 1:
                model = _build_model(surface, template.with_free_vector(x), q_, weights, x_grid, t_grid, max_t, n_regimes)
                new_corr = (_pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                            - _mixture_implied_vol_nodes(model, nodes))
                correction = new_corr if outer == 0 else 0.6 * new_corr + 0.4 * correction  # damped
            if verbose:
                model = _build_model(surface, template.with_free_vector(x), q_, weights, x_grid, t_grid, max_t, n_regimes)
                chk = _pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                print(f"  {label}pass {outer + 1}/{passes}  PDE rms="
                      f"{np.sqrt(np.nanmean((chk - market_iv) ** 2)) * 1e4:6.2f} bp  (evals {evals['n']})")
        return x, ok

    passes = 1 if target != "hybrid" else max(1, n_pde_passes)
    x, ok = run_stage1(template.free_vector(), q, passes, label="[stage1] ")

    switch_rate, identified = _PRIOR_SWITCH_RATE, False
    barrier_rms = float("nan")
    n_bq = len(barrier_quotes) if barrier_quotes else 0

    if barrier_quotes:
        ls, ss, rate = level_spread, skew_spread, _PRIOR_SWITCH_RATE
        for rnd in range(max(1, n_barrier_rounds)):
            template = SharedSmileParams.from_surface(surface, ls, ss)
            x, ok = run_stage1(x, _generator(rate, weights), max(2, passes - 1), label=f"[bar{rnd + 1} s1] ")
            (ls, ss, rate), barrier_rms = _fit_barrier_structural(
                surface, template.with_free_vector(x), weights, barrier_quotes,
                x_grid, t_grid, max_t, (ls, ss, rate), verbose,
            )
        template = SharedSmileParams.from_surface(surface, ls, ss)
        q = _generator(rate, weights)
        switch_rate, identified = rate, True
    elif calibrate_q:
        model = _build_model(surface, template.with_free_vector(x), q, weights, x_grid, t_grid, max_t, n_regimes)
        switch_rate, identified = calibrate_switch_rate(model)
        q = _generator(switch_rate, weights)
        if verbose:
            print(f"  [stage2] switch rate = {switch_rate:.3f} / year  (identified={identified})")
        if identified:
            x, ok = run_stage1(x, q, max(1, q_refit_passes), label="[refit]  ")

    params = template.with_free_vector(x)
    bf_viol, cal_viol = _arbitrage_diagnostics(params, n_regimes)
    if bf_viol > 5e-4 or cal_viol > 1e-3:                  # nudge the wing scale down to clear it
        params.free[:, 1] = np.minimum(params.free[:, 1], template.free0[:, 1] * 1.4)
        bf_viol, cal_viol = _arbitrage_diagnostics(params, n_regimes)

    model = _build_model(surface, params, q, weights, x_grid, t_grid, max_t, n_regimes)
    final_iv = _pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
    errors = final_iv - market_iv

    report = SurfaceCalibrationReport(
        success=ok,
        rms_vol_error=float(np.sqrt(np.nanmean(errors ** 2))),
        max_vol_error=float(np.nanmax(np.abs(errors))),
        node_errors=np.array([(t, y, e) for (t, y, _, _), e in zip(nodes, errors)]),
        svi_rms_vol_error=float(surface.svi_report.rms_vol_error) if surface.svi_report else float("nan"),
        max_butterfly_violation=bf_viol,
        max_calendar_violation=cal_viol,
        level_spread=float(params.level_spread),
        skew_spread=float(params.skew_spread),
        switch_rate=switch_rate,
        switch_rate_identified=identified,
        n_pde_passes=passes,
        n_residual_evals=evals["n"],
        barrier_rms_price_error=barrier_rms,
        n_barrier_quotes=n_bq,
    )
    return model, report


def calibrate_regime_surface(surface, regime_weights=(0.25, 0.5, 0.25), q=None, **kwargs):
    """Backward-compatible alias: the three-regime model, stage 1 only unless ``calibrate_q=True``.
    Accepts the old ``spread=`` keyword as ``level_spread``."""
    if "spread" in kwargs:
        kwargs["level_spread"] = kwargs.pop("spread")
    kwargs.setdefault("n_regimes", 3)
    kwargs.setdefault("calibrate_q", False)
    return calibrate_regime_model(surface, regime_weights, q=q, **kwargs)
