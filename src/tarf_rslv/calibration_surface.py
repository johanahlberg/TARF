"""Full FX-surface calibration for the three-regime local-vol model.

``calibrate_regime_model`` runs two stages, kept independent (as in :mod:`tarf_rslv.calibration`):

**Stage 1 -- static smile, to the whole surface.** Each tenor is fitted once with an arbitrage-free
**SVI** slice (``svi.fit_svi_surface``); the calibration then moves only that slice's level /
skew-scale ``(a, b)`` per tenor, with the shape ``(rho, m, sigma)`` frozen -- so a candidate slice is
linear in the free variables and needs no inner fit. Regime ``i`` is the base slice with total
variance scaled by ``(1 +- spread)^2``; each regime slice gives an **analytic Dupire** local-vol
surface; and the model's own implied-vol surface -- the forward regime-switching PDE of
:mod:`tarf_rslv.vanilla` -- is driven onto the market surface by least squares over the ``(a, b)``.

**Stage 2 -- switching speed.** ``calibrate_switch_rate`` fits the scalar rate of
``Q = rate (1 pi^T - I)`` so the model's 25d **butterfly** term structure matches the market's --
the smile-flattening-with-maturity that the switch rate controls in a level-dispersion regime model.
It is only weakly identified by vanillas; the fit is bounded and falls back to a persistence prior.

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
_PRIOR_SWITCH_RATE = 2.0                       # regimes persist ~6 months when stage 2 is unidentified


def _regime_variance_multiplier(spread: float, regime: int) -> float:
    return float((1.0 + REGIME_MULTIPLIERS[regime] * spread) ** 2)


# --------------------------------------------------------------------------------------------------
# Stage-1 parameters
# --------------------------------------------------------------------------------------------------
@dataclass
class SharedSmileParams:
    """Per-tenor SVI with the **shape** ``(rho, m, sigma)`` frozen from the market fit and the
    **level / skew-scale** ``(a, b)`` free -- so a candidate slice is linear in the free variables
    and needs no inner fit. ``spread`` sets the regime vol dispersion."""

    tenors: np.ndarray
    shape: np.ndarray                # (N, 3): rho, m, sigma  (fixed)
    ab: np.ndarray                   # (N, 2): a, b           (free)
    spread: float
    ab0: np.ndarray                  # the market-fit (a, b), for the bounding box

    @property
    def n_buckets(self) -> int:
        return len(self.tenors)

    def slices(self) -> list[SVISlice]:
        return [
            SVISlice(float(self.tenors[b]), float(self.ab[b, 0]), float(self.ab[b, 1]),
                     float(self.shape[b, 0]), float(self.shape[b, 1]), float(self.shape[b, 2]))
            for b in range(self.n_buckets)
        ]

    def free_vector(self) -> np.ndarray:
        return self.ab.ravel()

    def with_free_vector(self, vector: np.ndarray) -> "SharedSmileParams":
        return replace(self, ab=np.asarray(vector, dtype=float).reshape(self.n_buckets, 2))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        a0, b0 = self.ab0[:, 0], self.ab0[:, 1]
        lo = np.stack([a0 - 0.02, np.maximum(b0 * 0.4, 1e-4)], axis=1).ravel()
        hi = np.stack([a0 + 0.02, b0 * 2.2 + 0.02], axis=1).ravel()
        return lo, hi

    def regime_slices(self, regime: int) -> list[SVISlice]:
        mult = _regime_variance_multiplier(self.spread, regime)
        return [s.scaled(mult) for s in self.slices()]

    @classmethod
    def from_surface(cls, surface: FXVolSurface, spread: float) -> "SharedSmileParams":
        shape = np.array([[s.rho, s.m, s.sigma] for s in surface.svi_slices], dtype=float)
        ab = np.array([[s.a, s.b] for s in surface.svi_slices], dtype=float)
        return cls(np.asarray(surface._tenors, dtype=float), shape, ab.copy(), float(spread), ab.copy())


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
    spot: float
    rate: float
    dividend_yield: float
    regimes: list[_LocalVolRegime]
    q: np.ndarray
    regime_weights: np.ndarray
    params: SharedSmileParams
    surface: FXVolSurface
    time_varying: bool = True

    def bumped_vol(self, dv: float) -> "CalibratedRegimeModel":
        return replace(self, regimes=[r.shifted(dv) for r in self.regimes])

    def regime_implied_vol(self, i: int, t: float, y: np.ndarray) -> np.ndarray:
        w, _, _, _ = svi_total_variance_and_derivs(self.params.regime_slices(i), max(float(t), 1e-8), y)
        return np.sqrt(np.maximum(w, 1e-12) / max(float(t), 1e-8))


def _regime_local_vol_grids(params, forward_of_t, x_grid, t_grid, vol_floor, vol_cap):
    grids = []
    for i in range(3):
        regime_slices = params.regime_slices(i)
        rows = np.empty((len(t_grid), len(x_grid)))
        for kk, t in enumerate(t_grid):
            k = x_grid - np.log(forward_of_t(float(t)))
            rows[kk] = svi_local_vol(regime_slices, float(max(t, 1e-4)), k, vol_floor=vol_floor, vol_cap=vol_cap)
        grids.append(rows)
    return grids


def _build_model(surface, params, q, weights, x_grid, t_grid, max_t):
    grids = _regime_local_vol_grids(params, surface.forward, x_grid, t_grid, surface.vol_floor, surface.vol_cap)
    front = params.slices()[0]
    front_atm = float(np.sqrt(max(front.total_variance(0.0), 1e-12) / front.tenor))
    regimes = [
        _LocalVolRegime(
            x_grid, t_grid, grids[i], surface.domestic_zero(max_t), surface.foreign_zero(max_t),
            front_atm * (1.0 + REGIME_MULTIPLIERS[i] * params.spread), surface.vol_floor,
        )
        for i in range(3)
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
            for i in range(3)
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
        sig_ref = float(max(model.regime_implied_vol(1, max_t, np.array([0.0]))[0], 0.05))
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
    switch_rate: float
    switch_rate_identified: bool
    n_pde_passes: int
    n_residual_evals: int


def _arbitrage_diagnostics(params: SharedSmileParams) -> tuple[float, float]:
    """Worst butterfly violation over the three regimes, and worst calendar crossing of the base."""
    bf = 0.0
    for i in range(3):
        for slc in params.regime_slices(i):
            bf = max(bf, float(np.max(np.maximum(-slc.durrleman_g(_ARB_GRID), 0.0))))
    cal = 0.0
    base = params.slices()
    for a, b in zip(base[:-1], base[1:]):
        cal = max(cal, float(np.max(np.maximum(a.total_variance(_ARB_GRID) - b.total_variance(_ARB_GRID), 0.0))))
    return bf, cal


def _infer_pi0(surface: FXVolSurface, weights: np.ndarray) -> np.ndarray:
    atm = np.array([s.atm for s in surface.smiles], dtype=float)
    slope = (atm[-1] - atm[0]) / max(atm[0], 1e-6)
    tilt = float(np.clip(-slope * 2.0, -0.45, 0.45))   # upward term structure -> weight the low regime
    raw = weights * np.array([1.0 - tilt, 1.0, 1.0 + tilt])
    return raw / raw.sum()


# --------------------------------------------------------------------------------------------------
# stage 2: switch rate from the 25d butterfly term structure
# --------------------------------------------------------------------------------------------------
def _generator(rate: float, stationary: np.ndarray) -> np.ndarray:
    return rate * (np.ones((3, 1)) @ stationary[None, :] - np.eye(3))


def calibrate_switch_rate(
    model: CalibratedRegimeModel,
    pi0: np.ndarray,
    *,
    rate_bounds: tuple[float, float] = (0.2, 12.0),
) -> tuple[float, bool]:
    """Fit the switch rate so the model's 25d butterfly term structure matches the market's.

    Returns ``(rate, identified)``; ``identified`` is ``False`` (and ``rate`` the persistence prior)
    when the butterfly term structure barely responds to the rate.
    """
    surface = model.surface
    tenors = np.array([s.tenor for s in surface.smiles], dtype=float)
    stationary = model.regime_weights
    knot_k = surface._knot_y

    market_bf = []
    for slc, k in zip(surface.svi_slices, knot_k):
        iv = np.asarray(slc.implied_vol(np.array([k[1], 0.0, k[-2]])), dtype=float)  # 25dP, ATM, 25dC
        market_bf.append(0.5 * (iv[0] + iv[2]) - iv[1])
    market_bf = np.array(market_bf)

    def model_bf(rate: float) -> np.ndarray:
        m = replace(model, q=_generator(rate, stationary))
        vals = np.empty(len(tenors))
        for j, (k, t) in enumerate(zip(knot_k, tenors)):
            pi_bar = _time_averaged_regime_probs(m.q, stationary, float(t))
            iv = _mixture_implied_vol(m, float(t), np.array([k[1], 0.0, k[-2]]), pi_bar)
            vals[j] = 0.5 * (iv[0] + iv[2]) - iv[1]
        return vals

    bf_lo, bf_hi = model_bf(rate_bounds[0]), model_bf(rate_bounds[1])
    if np.max(np.abs(bf_hi - bf_lo)) < 2e-5:            # < 0.2 bp swing -> not identified
        return _PRIOR_SWITCH_RATE, False

    def objective(rate: float) -> float:
        return float(np.sum((model_bf(rate) - market_bf) * tenors))

    g_lo, g_hi = objective(rate_bounds[0]), objective(rate_bounds[1])
    if np.sign(g_lo) == np.sign(g_hi):
        return (rate_bounds[0] if abs(g_lo) < abs(g_hi) else rate_bounds[1]), True
    return float(brentq(objective, *rate_bounds, xtol=1e-6)), True


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
    spread: float = 0.03,
    calibrate_q: bool = True,
    pi0: np.ndarray | list[float] | None = None,
    q: np.ndarray | None = None,
    target: str = "hybrid",
    n_pde_passes: int = 4,
    q_refit_passes: int = 2,
    num_x: int = 701,
    steps_per_year: int = 600,
    dupire_nt: int = 41,
    dupire_nx: int = 161,
    max_nfev_inner: int = 120,
    verbose: bool = False,
) -> tuple[CalibratedRegimeModel, SurfaceCalibrationReport]:
    """Calibrate the full three-regime model to ``surface``.

    ``target``: ``"hybrid"`` (default) fits a smooth mixture surrogate inside the optimiser and
    re-anchors it ``n_pde_passes`` times with the forward-PDE correction -- fast, robust, converges
    to a PDE-accurate fit. ``"pde"`` fits the forward PDE directly (literal but noisier).
    ``"mixture"`` is the surrogate only.
    """
    weights = np.asarray(regime_weights, dtype=float)
    if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("regime_weights must be a length-3 vector summing to 1")
    if target not in {"hybrid", "pde", "mixture"}:
        raise ValueError("target must be 'hybrid', 'pde' or 'mixture'")
    q = (_generator(_PRIOR_SWITCH_RATE, weights) if q is None else np.asarray(q, dtype=float))

    nodes = surface.market_nodes()
    market_iv = np.array([iv for _, _, iv, _ in nodes])
    vega_w = np.array([w for _, _, _, w in nodes])
    vega_w = vega_w / vega_w.mean()

    template = SharedSmileParams.from_surface(surface, spread)
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
                model = _build_model(surface, template.with_free_vector(vec), q_, weights, x_grid, t_grid, max_t)
                if target == "pde":
                    return (_pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                            - market_iv) * vega_w
                return (_mixture_implied_vol_nodes(model, nodes) - _ref) * vega_w

            sol = least_squares(residual, x, bounds=(lo, hi), method="trf", jac_sparsity=jac_sparsity,
                                x_scale="jac", max_nfev=max_nfev_inner, ftol=1e-8, xtol=1e-10, gtol=1e-10)
            x, ok = sol.x, bool(sol.success)

            if target == "hybrid" and outer < passes - 1:
                model = _build_model(surface, template.with_free_vector(x), q_, weights, x_grid, t_grid, max_t)
                correction = (_pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                              - _mixture_implied_vol_nodes(model, nodes))
            if verbose:
                model = _build_model(surface, template.with_free_vector(x), q_, weights, x_grid, t_grid, max_t)
                chk = _pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                print(f"  {label}pass {outer + 1}/{passes}  PDE rms="
                      f"{np.sqrt(np.nanmean((chk - market_iv) ** 2)) * 1e4:6.2f} bp  (evals {evals['n']})")
        return x, ok

    passes = 1 if target != "hybrid" else max(1, n_pde_passes)
    x, ok = run_stage1(template.free_vector(), q, passes, label="[stage1] ")

    switch_rate, identified = _PRIOR_SWITCH_RATE, False
    if calibrate_q:
        model = _build_model(surface, template.with_free_vector(x), q, weights, x_grid, t_grid, max_t)
        pi_now = _infer_pi0(surface, weights) if pi0 is None else np.asarray(pi0, dtype=float)
        switch_rate, identified = calibrate_switch_rate(model, pi_now)
        q = _generator(switch_rate, weights)
        if verbose:
            print(f"  [stage2] switch rate = {switch_rate:.3f} / year  (identified={identified})")
        if identified:
            x, ok = run_stage1(x, q, max(1, q_refit_passes), label="[refit]  ")

    params = template.with_free_vector(x)
    bf_viol, cal_viol = _arbitrage_diagnostics(params)
    if bf_viol > 5e-4 or cal_viol > 1e-3:                  # nudge the wing scale down to clear it
        params.ab[:, 1] = np.minimum(params.ab[:, 1], template.ab0[:, 1] * 1.4)
        bf_viol, cal_viol = _arbitrage_diagnostics(params)

    model = _build_model(surface, params, q, weights, x_grid, t_grid, max_t)
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
        switch_rate=switch_rate,
        switch_rate_identified=identified,
        n_pde_passes=passes,
        n_residual_evals=evals["n"],
    )
    return model, report


def calibrate_regime_surface(surface, regime_weights=(0.25, 0.5, 0.25), q=None, **kwargs):
    """Backward-compatible alias: stage 1 only unless ``calibrate_q=True`` is passed."""
    kwargs.setdefault("calibrate_q", False)
    return calibrate_regime_model(surface, regime_weights, q=q, **kwargs)
