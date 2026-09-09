"""Full FX-surface calibration for the three-regime local-vol model.

Two stages, kept independent (as in :mod:`tarf_rslv.calibration`):

**Stage 1 -- static smile, to the whole surface.** A shared base implied smile per tenor
(``atm``, ``rr25``, ``bf25``, ``rr10``, ``bf10`` -- five numbers, one per market quote) plus a
single regime vol-dispersion ``spread``. The three regimes are the base smile scaled by
``(1 - spread, 1, 1 + spread)``. Each regime's implied smile is turned into a **Dupire local-vol
surface**, and the regime-switching model's own implied-vol surface -- computed with the forward
PDE of :mod:`tarf_rslv.vanilla` -- is driven onto the market surface by least squares over the
shared-smile parameters. ``spread`` and the generator ``Q`` are held fixed here (stage 2 / a prior).

**Stage 2 -- switching speed.** ``calibrate_switch_rate_to_term_structure`` fits the scalar rate of
``Q = rate * (1 pi^T - I)`` so the model's ATM **forward-variance** term structure matches the one
implied by the market ATM curve -- the quantity that carries switching-speed information that static
smiles cannot pin down. Needs the current regime distribution ``pi0`` to differ from ``stationary``.

The result of stage 1 is a :class:`CalibratedRegimeModel` that plugs straight into
``ThreeRegimePricer`` (it is time-dependent, so the pricer rebuilds its operators per segment).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, least_squares

from .model import SingleRegimeLocalVolModel
from .vanilla import RegimeForwardPDE
from .vol_surface import (
    FXVolSurface,
    bs_forward_price,
    implied_vol_from_forward_price,
    local_vol_from_total_variance,
)

REGIME_MULTIPLIERS = np.array([-1.0, 0.0, 1.0])  # base * (1 + m * spread)


# --------------------------------------------------------------------------------------------------
# Stage-1 parameters
# --------------------------------------------------------------------------------------------------
@dataclass
class SharedSmileParams:
    """Shared base implied smile per tenor bucket, plus one regime dispersion."""

    tenors: np.ndarray            # (N,)
    knot_y: list[np.ndarray]      # per bucket: 5 fixed log-moneyness pillars (10dP..10dC)
    atm: np.ndarray               # (N,)
    rr25: np.ndarray              # (N,)
    bf25: np.ndarray              # (N,)  smile butterfly (not market strangle)
    rr10: np.ndarray              # (N,)
    bf10: np.ndarray              # (N,)
    spread: float

    @property
    def n_buckets(self) -> int:
        return len(self.tenors)

    def base_knot_vols(self, b: int) -> np.ndarray:
        """Base implied vols at ``knot_y[b]``: 10dP, 25dP, ATM, 25dC, 10dC."""
        return np.array([
            self.atm[b] + self.bf10[b] - 0.5 * self.rr10[b],
            self.atm[b] + self.bf25[b] - 0.5 * self.rr25[b],
            self.atm[b],
            self.atm[b] + self.bf25[b] + 0.5 * self.rr25[b],
            self.atm[b] + self.bf10[b] + 0.5 * self.rr10[b],
        ])

    # -- flat vector <-> params (for least_squares) ----------------------------------------
    def free_vector(self) -> np.ndarray:
        return np.concatenate([self.atm, self.rr25, self.bf25, self.rr10, self.bf10])

    def with_free_vector(self, vector: np.ndarray) -> "SharedSmileParams":
        n = self.n_buckets
        a, r25, b25, r10, b10 = (vector[i * n:(i + 1) * n] for i in range(5))
        return replace(self, atm=a, rr25=r25, bf25=b25, rr10=r10, bf10=b10)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_buckets
        lo = np.concatenate([np.full(n, 1e-3), np.full(4 * n, -2.0)])
        hi = np.concatenate([np.full(n, 5.0), np.full(4 * n, 2.0)])
        return lo, hi

    @classmethod
    def from_surface(cls, surface: FXVolSurface, spread: float) -> "SharedSmileParams":
        tenors = np.array([s.tenor for s in surface.smiles], dtype=float)
        knot_y, atm, rr25, bf25, rr10, bf10 = [], [], [], [], [], []
        for smile, y_knots, smile_fn in zip(surface.smiles, surface._knot_y, surface._smile_of_y):
            vols = np.asarray(smile_fn(y_knots), dtype=float)  # 10dP,25dP,ATM,25dC,10dC in strike order
            knot_y.append(np.asarray(y_knots, dtype=float))
            atm.append(float(vols[2]))
            rr25.append(float(vols[3] - vols[1]))
            bf25.append(float(0.5 * (vols[3] + vols[1]) - vols[2]))
            rr10.append(float(vols[4] - vols[0]))
            bf10.append(float(0.5 * (vols[4] + vols[0]) - vols[2]))
        return cls(
            tenors=tenors, knot_y=knot_y,
            atm=np.array(atm), rr25=np.array(rr25), bf25=np.array(bf25),
            rr10=np.array(rr10), bf10=np.array(bf10), spread=float(spread),
        )


# --------------------------------------------------------------------------------------------------
# Calibrated model
# --------------------------------------------------------------------------------------------------
def _tv_linear(tenors: np.ndarray, values_by_bucket: np.ndarray, t: float) -> np.ndarray:
    """Total-variance-linear interpolation across tenor buckets at fixed ``y``.

    ``values_by_bucket`` has shape ``(N, ...)`` holding sigma(y) per bucket; returns sigma(y) at ``t``
    with constant-vol extension outside ``[tenors[0], tenors[-1]]``.
    """
    if t <= tenors[0]:
        return values_by_bucket[0]
    if t >= tenors[-1]:
        return values_by_bucket[-1]
    hi = int(np.searchsorted(tenors, t, side="right"))
    lo = hi - 1
    t_lo, t_hi = tenors[lo], tenors[hi]
    w_lo = values_by_bucket[lo] ** 2 * t_lo
    w_hi = values_by_bucket[hi] ** 2 * t_hi
    frac = (t - t_lo) / (t_hi - t_lo)
    w = (1.0 - frac) * w_lo + frac * w_hi
    return np.sqrt(np.maximum(w, 1e-12) / t)


class _LocalVolRegime:
    """One regime for ``ThreeRegimePricer``: a bilinear-interpolated Dupire local-vol grid."""

    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        sigma_grid: np.ndarray,
        rate: float,
        dividend_yield: float,
        atm_level: float,
        vol_floor: float,
    ) -> None:
        self._x = x_grid
        self._t = t_grid
        self._sigma = sigma_grid  # (n_t, n_x)
        self.rate = float(rate)
        self.dividend_yield = float(dividend_yield)
        self.local_vol = float(atm_level)
        self.vol_floor = float(vol_floor)
        # neutral smile attributes so the greeks bump-and-revalue path stays valid
        self.skew = 0.0
        self.curvature = 0.0
        self.smile_ref = float(np.exp(x_grid[len(x_grid) // 2]))
        self.sigma_ref = float(atm_level)
        self.spot = self.smile_ref

    def local_volatility(self, spot: np.ndarray | float, t: float = 0.0) -> np.ndarray:
        x = np.log(np.maximum(np.asarray(spot, dtype=float), 1e-300))
        it = int(np.clip(np.searchsorted(self._t, t) - 1, 0, len(self._t) - 2))
        t0, t1 = self._t[it], self._t[it + 1]
        wt = 0.0 if t1 == t0 else np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
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
    """Stage-1 output. Time-dependent, so ``ThreeRegimePricer`` treats it as such."""

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
        p = self.params
        mult = 1.0 + REGIME_MULTIPLIERS[i] * p.spread
        per_bucket = np.stack([
            np.interp(y, p.knot_y[b], p.base_knot_vols(b)) * mult for b in range(p.n_buckets)
        ])
        return _tv_linear(p.tenors, per_bucket, t)


def _build_regime_local_vol_grids(
    params: SharedSmileParams,
    forward_of_t: Callable[[float], float],
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    vol_floor: float,
    vol_cap: float,
) -> list[np.ndarray]:
    """For each regime, the Dupire local vol on ``(t_grid, x_grid)`` (shape ``(n_t, n_x)``)."""
    grids: list[np.ndarray] = []
    for i in range(3):
        mult = 1.0 + REGIME_MULTIPLIERS[i] * params.spread

        def total_variance(t: float, y: np.ndarray, _mult: float = mult) -> np.ndarray:
            per_bucket = np.stack([
                np.interp(np.atleast_1d(y), params.knot_y[b], params.base_knot_vols(b)) * _mult
                for b in range(params.n_buckets)
            ])
            sig = _tv_linear(params.tenors, per_bucket, t)
            return sig * sig * t

        sigma_rows = np.empty((len(t_grid), len(x_grid)))
        for k, t in enumerate(t_grid):
            y = x_grid - np.log(forward_of_t(t))
            sigma_rows[k] = local_vol_from_total_variance(
                total_variance, float(max(t, 1e-4)), y, vol_floor=vol_floor, vol_cap=vol_cap
            )
        grids.append(sigma_rows)
    return grids


# --------------------------------------------------------------------------------------------------
# Model-implied vanilla surface: a smooth mixture surrogate, and the accurate forward PDE
# --------------------------------------------------------------------------------------------------
def _time_averaged_regime_probs(q: np.ndarray, weights: np.ndarray, t: float) -> np.ndarray:
    grid = np.linspace(0.0, t, 17)
    probs = np.array([expm(q.T * u) @ weights for u in grid])
    return np.sum(0.5 * (probs[1:] + probs[:-1]) * np.diff(grid)[:, None], axis=0) / t


def _mixture_implied_vol_nodes(model: CalibratedRegimeModel, nodes) -> np.ndarray:
    """Smooth surrogate: price = sum_i pi_bar_i(T) BS(F, K, sigma_i^impl(k, T)); invert.

    ``pi_bar`` is the time-averaged regime distribution, so switching enters at first order. Smooth
    and fast in the calibration parameters (no PDE noise), used as the optimiser's inner target.
    """
    out = np.empty(len(nodes))
    prob_cache: dict[float, np.ndarray] = {}
    for n, (t, y, _, _) in enumerate(nodes):
        if t not in prob_cache:
            prob_cache[t] = _time_averaged_regime_probs(model.q, model.regime_weights, t)
        pi_bar = prob_cache[t]
        forward = model.spot * np.exp((model.rate - model.dividend_yield) * t)
        strike = forward * np.exp(y)
        price = sum(
            pi_bar[i] * bs_forward_price(forward, strike, float(model.regime_implied_vol(i, t, np.array([y]))[0]), t, True)
            for i in range(3)
        )
        out[n] = implied_vol_from_forward_price(price, forward, strike, t, True)
    return out


# Below this tenor the regime distribution has not moved (Q * T is tiny) and the forward PDE cannot
# resolve the near-degenerate density, so the mixture surrogate is used -- and is essentially exact.
PDE_MIN_TENOR = 1.0 / 26.0  # ~2 weeks


def _pde_implied_vol_nodes(
    model: CalibratedRegimeModel, nodes, *, num_x: int, steps_per_year: int
) -> np.ndarray:
    """Best model implied vol at every node: forward PDE for tenors >= ``PDE_MIN_TENOR``, the mixture
    surrogate (exact in the no-switching limit) below it."""
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
        mix = _mixture_implied_vol_nodes(model, short_nodes)
        for nd, iv in zip(short_nodes, mix):
            result[(nd[0], float(nd[1]))] = float(iv)

    return np.array([result[(t, float(y))] for t, y, _, _ in nodes])


# --------------------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------------------
@dataclass
class SurfaceCalibrationReport:
    success: bool
    rms_vol_error: float          # from the accurate forward PDE
    max_vol_error: float
    node_errors: np.ndarray       # (tenor, y, pde_iv - market_iv) rows
    n_pde_passes: int
    n_residual_evals: int


def calibrate_regime_surface(
    surface: FXVolSurface,
    regime_weights: np.ndarray | list[float] = (0.25, 0.5, 0.25),
    q: np.ndarray | None = None,
    *,
    spread: float = 0.03,
    target: str = "hybrid",
    n_pde_passes: int = 4,
    num_x: int = 701,
    steps_per_year: int = 600,
    dupire_nt: int = 41,
    dupire_nx: int = 161,
    max_nfev_inner: int = 60,
    verbose: bool = False,
) -> tuple[CalibratedRegimeModel, SurfaceCalibrationReport]:
    """Fit the shared base smile so the three-regime model reproduces ``surface``.

    ``target``:

    * ``"hybrid"`` (default, recommended) -- least squares against the smooth mixture surrogate,
      re-anchored ``n_pde_passes`` times by the forward-PDE-minus-mixture correction. Fast and
      robust; converges to a PDE-accurate fit.
    * ``"pde"`` -- least squares directly against the forward PDE every evaluation. Most literal,
      but the PDE discretisation noise makes the optimiser sensitive near the optimum.
    * ``"mixture"`` -- the surrogate only; fastest, ~few-bp accurate.

    ``spread`` and ``q`` are inputs -- calibrate ``q`` with ``calibrate_switch_rate_to_term_structure``.
    """
    weights = np.asarray(regime_weights, dtype=float)
    if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("regime_weights must be a length-3 vector summing to 1")
    if target not in {"hybrid", "pde", "mixture"}:
        raise ValueError("target must be 'hybrid', 'pde' or 'mixture'")
    if q is None:
        q = np.array([[-0.5, 0.25, 0.25], [0.25, -0.5, 0.25], [0.25, 0.25, -0.5]])
    q = np.asarray(q, dtype=float)

    nodes = surface.market_nodes()
    market_iv = np.array([iv for _, _, iv, _ in nodes])
    vega_w = np.array([w for _, _, _, w in nodes])
    vega_w = vega_w / vega_w.mean()

    template = SharedSmileParams.from_surface(surface, spread)
    max_t = float(template.tenors[-1])
    n_buckets = template.n_buckets

    node_bucket = np.array([int(np.argmin(np.abs(template.tenors - t))) for t, _, _, _ in nodes])
    jac_sparsity = np.zeros((len(nodes), 5 * n_buckets), dtype=bool)
    for row, b in enumerate(node_bucket):
        for group in range(5):
            for bb in (b - 1, b, b + 1):
                if 0 <= bb < n_buckets:
                    jac_sparsity[row, group * n_buckets + bb] = True

    max_atm = float(max(template.atm.max(), 0.05))
    half = max(8.0 * max_atm * np.sqrt(max_t), 0.35)
    x_grid = np.log(surface.spot) + np.linspace(-half, half, int(dupire_nx) | 1)
    t_grid = np.concatenate([[1e-4], np.linspace(max_t / dupire_nt, max_t, dupire_nt)])

    def build_model(params: SharedSmileParams) -> CalibratedRegimeModel:
        grids = _build_regime_local_vol_grids(
            params, surface.forward, x_grid, t_grid, surface.vol_floor, surface.vol_cap
        )
        regimes = [
            _LocalVolRegime(
                x_grid, t_grid, grids[i], surface.domestic_zero(max_t), surface.foreign_zero(max_t),
                float(params.atm[0] * (1.0 + REGIME_MULTIPLIERS[i] * params.spread)), surface.vol_floor,
            )
            for i in range(3)
        ]
        return CalibratedRegimeModel(
            spot=surface.spot, rate=surface.domestic_zero(max_t), dividend_yield=surface.foreign_zero(max_t),
            regimes=regimes, q=q, regime_weights=weights, params=params, surface=surface,
        )

    lo, hi = template.bounds()
    x = np.clip(template.free_vector(), lo + 1e-9, hi - 1e-9)
    correction = np.zeros(len(nodes))
    evals = {"n": 0}
    passes = 1 if target != "hybrid" else max(1, n_pde_passes)

    for outer in range(passes):
        target_iv = market_iv - correction

        def residual(vector: np.ndarray, _ref: np.ndarray = target_iv) -> np.ndarray:
            evals["n"] += 1
            model = build_model(template.with_free_vector(vector))
            if target == "pde":
                return (_pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
                        - market_iv) * vega_w
            return (_mixture_implied_vol_nodes(model, nodes) - _ref) * vega_w

        sol = least_squares(
            residual, x, bounds=(lo, hi), method="trf", jac_sparsity=jac_sparsity,
            x_scale="jac", max_nfev=max_nfev_inner, ftol=1e-8, xtol=1e-10, gtol=1e-10,
        )
        x = sol.x

        if target == "hybrid" and outer < passes - 1:
            model = build_model(template.with_free_vector(x))
            pde_iv = _pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
            mix_iv = _mixture_implied_vol_nodes(model, nodes)
            correction = pde_iv - mix_iv
        if verbose:
            model = build_model(template.with_free_vector(x))
            check = _pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
            print(f"  pass {outer + 1}/{passes}  PDE rms={np.sqrt(np.mean((check - market_iv) ** 2)) * 1e4:6.2f} bp"
                  f"  (residual evals so far: {evals['n']})")

    model = build_model(template.with_free_vector(x))
    final_iv = _pde_implied_vol_nodes(model, nodes, num_x=num_x, steps_per_year=steps_per_year)
    errors = final_iv - market_iv
    node_errors = np.array([(t, y, e) for (t, y, _, _), e in zip(nodes, errors)])
    report = SurfaceCalibrationReport(
        success=bool(sol.success),
        rms_vol_error=float(np.sqrt(np.mean(errors ** 2))),
        max_vol_error=float(np.max(np.abs(errors))),
        node_errors=node_errors,
        n_pde_passes=passes,
        n_residual_evals=evals["n"],
    )
    return model, report


# --------------------------------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------------------------------
def _model_forward_variance_curve(
    pi0: np.ndarray, q: np.ndarray, regime_atm: np.ndarray, tenors: np.ndarray
) -> np.ndarray:
    """E[ integral_0^T sigma_ATM(u)^2 du ] under the switching mixture, at each tenor."""
    sig2 = regime_atm ** 2
    out = np.empty(len(tenors))
    for k, t in enumerate(tenors):
        grid = np.linspace(0.0, t, 64)
        vals = np.array([(expm(q.T * u) @ pi0) @ sig2 for u in grid])
        out[k] = float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(grid)))
    return out


def calibrate_switch_rate_to_term_structure(
    surface: FXVolSurface,
    regime_atm_levels: np.ndarray | list[float],
    pi0: np.ndarray | list[float],
    stationary: np.ndarray | list[float] | None = None,
    *,
    max_rate: float = 40.0,
) -> dict[str, object]:
    """Fit the scalar switch rate of ``Q = rate (1 stationary^T - I)`` so the model ATM
    forward-variance term structure matches the market ATM curve.

    ``regime_atm_levels`` are the three regime ATM vols (e.g. ``params.atm[0] * (1 +- spread)``,
    or a front-tenor slice). ``pi0`` must differ from ``stationary`` for the rate to be identified.
    """
    regime_atm = np.asarray(regime_atm_levels, dtype=float)
    pi0 = np.asarray(pi0, dtype=float)
    stationary = pi0 if stationary is None else np.asarray(stationary, dtype=float)
    if regime_atm.shape != (3,) or pi0.shape != (3,) or stationary.shape != (3,):
        raise ValueError("regime_atm_levels, pi0, stationary must all be length 3")
    if np.allclose(pi0, stationary, atol=1e-9):
        raise ValueError("pi0 must differ from stationary for the switch rate to be identifiable")

    tenors = np.array([s.tenor for s in surface.smiles], dtype=float)
    market_atm = np.array([s.atm for s in surface.smiles], dtype=float)
    market_fwd_var = market_atm ** 2 * tenors  # total variance to each tenor

    def gap(rate: float) -> float:
        q = rate * (np.ones((3, 1)) @ stationary[None, :] - np.eye(3))
        model_curve = _model_forward_variance_curve(pi0, q, regime_atm, tenors)
        return float(np.sum((model_curve - market_fwd_var) * tenors))  # tenor-weighted

    lo, hi = gap(1e-8), gap(max_rate)
    if np.sign(lo) == np.sign(hi):
        rate = 0.0 if abs(lo) < abs(hi) else max_rate
    else:
        rate = float(brentq(gap, 1e-8, max_rate, xtol=1e-8))

    q = rate * (np.ones((3, 1)) @ stationary[None, :] - np.eye(3))
    return {
        "switch_rate": rate,
        "q": q,
        "model_forward_variance": _model_forward_variance_curve(pi0, q, regime_atm, tenors),
        "market_forward_variance": market_fwd_var,
        "tenors": tenors,
    }
