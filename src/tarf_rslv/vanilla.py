"""Model-implied vanilla volatilities for the regime-switching local-vol model, via a forward
(Fokker-Planck) PDE for the joint (log-spot, regime) density.

This is the calibration target: given three regime local-volatility functions, a generator ``Q``
and an initial regime distribution, it returns the model's European implied-vol surface, which the
calibrator drives onto the market surface.

The density ``p_i(x, t)`` (``x = ln S``, regime ``i``) evolves by

    dp_i/dt = d^2/dx^2[ a_i(x,t) p_i ] - d/dx[ b_i(x,t) p_i ] + sum_j (Q^T)_ij p_j ,
    a_i = 1/2 sigma_i^2 ,   b_i = r_d - r_f - 1/2 sigma_i^2 ,

discretised with conservative central differences on a uniform ``x`` grid, a theta-scheme with a
fully-implicit Rannacher startup for the point-mass initial condition, and the regime coupling done
exactly per step with ``exp(Q^T dt)`` (operator splitting) -- the forward-equation transpose of the
backward coupling used in ``solver.py``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.linalg import expm, solve_banded

from .vol_surface import implied_vol_from_forward_price

RANNACHER_STEPS = 4

RegimeLocalVol = Callable[[np.ndarray, float], np.ndarray]


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral -- ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0, and this
    package targets NumPy 1.22, so neither name is used directly."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


class RegimeForwardPDE:
    """Forward density solver for the three-regime local-vol model."""

    def __init__(
        self,
        spot: float,
        domestic_rate: float,
        foreign_rate: float,
        regime_local_vols: Sequence[RegimeLocalVol],
        q: np.ndarray,
        weights: Sequence[float],
        *,
        max_t: float,
        sigma_ref: float = 0.1,
        num_x: int = 401,
        n_std: float = 6.0,
        steps_per_year: int = 400,
        min_sub_steps: int = 12,
    ) -> None:
        self.regime_local_vols = list(regime_local_vols)
        self.n_regimes = len(self.regime_local_vols)
        if self.n_regimes < 1:
            raise ValueError("need at least one regime local-vol function")
        self.rate = float(domestic_rate)
        self.dividend_yield = float(foreign_rate)

        self.q_T = np.asarray(q, dtype=float).T
        if self.q_T.shape != (self.n_regimes, self.n_regimes):
            raise ValueError(f"q must be {self.n_regimes}x{self.n_regimes}")

        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (self.n_regimes,) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"weights must be a length-{self.n_regimes} vector summing to 1")

        num_x = int(num_x) | 1  # force odd so the spot sits on the centre node
        half_width = max(n_std * sigma_ref * np.sqrt(max(max_t, 1e-6)), 0.2)
        self.log_spot = np.log(spot)
        self.x = self.log_spot + np.linspace(-half_width, half_width, num_x)
        self.dx = float(self.x[1] - self.x[0])
        self.spot_index = num_x // 2
        self.s = np.exp(self.x)
        self.sigma_ref = float(sigma_ref)

        self.max_t = float(max_t)
        self.steps_per_year = int(steps_per_year)
        self.min_sub_steps = int(min_sub_steps)

    # ---------------------------------------------------------------------------------------
    def _step_operator_bands(self, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = 0.5 * sigma * sigma
        b = self.rate - self.dividend_yield - a
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_2dx = 1.0 / (2.0 * self.dx)
        sub = np.roll(a, 1) * inv_dx2 + np.roll(b, 1) * inv_2dx      # coeff of p[k-1] in (L p)[k]
        diag = -2.0 * a * inv_dx2
        sup = np.roll(a, -1) * inv_dx2 - np.roll(b, -1) * inv_2dx    # coeff of p[k+1] in (L p)[k]
        return sub, diag, sup

    def _theta_solve(
        self, p: np.ndarray, sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, dt: float, theta: float
    ) -> np.ndarray:
        n = p.size
        # explicit part: rhs = p + (1-theta) dt L p
        lp = np.zeros_like(p)
        lp[1:-1] = sub[1:-1] * p[:-2] + diag[1:-1] * p[1:-1] + sup[1:-1] * p[2:]
        rhs = p + (1.0 - theta) * dt * lp
        rhs[0] = 0.0
        rhs[-1] = 0.0

        ab = np.zeros((3, n))
        ab[0, 2:] = -theta * dt * sup[1:-1]        # super-diagonal  A[k, k+1]
        ab[1, 1:-1] = 1.0 - theta * dt * diag[1:-1]
        ab[2, :-2] = -theta * dt * sub[1:-1]       # sub-diagonal    A[k, k-1]
        ab[1, 0] = 1.0
        ab[1, -1] = 1.0
        return solve_banded((1, 1), ab, rhs)

    # ---------------------------------------------------------------------------------------
    def solve(self, expiries: Sequence[float]) -> dict[float, np.ndarray]:
        """March the density forward, returning the marginal spot density at each expiry."""
        checkpoints = sorted(float(t) for t in expiries if t > 0.0)
        if not checkpoints:
            return {}

        # Seed at a small t_eps with the analytic per-regime lognormal density (exact for flat vol
        # over one short step, and switching is still negligible), rather than a raw point mass.
        t_eps = min(checkpoints[0] / 20.0, 1.0 / 365.0)
        p = np.empty((self.n_regimes, self.x.size))
        for i, local_vol in enumerate(self.regime_local_vols):
            sig = float(np.asarray(local_vol(np.array([self.log_spot]), 0.5 * t_eps))[0])
            mean = self.log_spot + (self.rate - self.dividend_yield - 0.5 * sig * sig) * t_eps
            var = max(sig * sig * t_eps, (2.0 * self.dx) ** 2)
            dens = np.exp(-0.5 * (self.x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)
            p[i] = self.weights[i] * dens / max(_trapz(dens, self.x), 1e-300)

        edges = [t_eps, *checkpoints]
        densities: dict[float, np.ndarray] = {}
        step_counter = 0
        for seg in range(len(edges) - 1):
            t0, t1 = edges[seg], edges[seg + 1]
            n_sub = max(self.min_sub_steps, int(round(self.steps_per_year * (t1 - t0))))
            if seg == 0:
                n_sub = max(n_sub, 20)
            times = np.linspace(t0, t1, n_sub + 1)
            for s in range(n_sub):
                dt = float(times[s + 1] - times[s])
                t_mid = 0.5 * float(times[s] + times[s + 1])
                theta = 1.0 if step_counter < RANNACHER_STEPS else 0.5
                step_counter += 1

                diffused = np.empty_like(p)
                for i, local_vol in enumerate(self.regime_local_vols):
                    sigma = np.asarray(local_vol(self.x, t_mid), dtype=float)
                    sub, diag, sup = self._step_operator_bands(sigma)
                    diffused[i] = self._theta_solve(p[i], sub, diag, sup, dt, theta)
                p = expm(self.q_T * dt) @ diffused

            marginal = np.clip(p.sum(axis=0), 0.0, None)
            mass = _trapz(marginal, self.x)
            densities[t1] = marginal / mass if mass > 0 else marginal
        return densities

    # ---------------------------------------------------------------------------------------
    def implied_vols(
        self, expiries: Sequence[float], strikes_by_expiry: dict[float, np.ndarray]
    ) -> dict[float, np.ndarray]:
        densities = self.solve(expiries)
        out: dict[float, np.ndarray] = {}
        for t, density in densities.items():
            forward = self.s[self.spot_index] * np.exp((self.rate - self.dividend_yield) * t)
            strikes = np.atleast_1d(np.asarray(strikes_by_expiry[t], dtype=float))
            ivs = np.empty(strikes.size)
            for j, strike in enumerate(strikes):
                call = _trapz(np.maximum(self.s - strike, 0.0) * density, self.x)
                ivs[j] = implied_vol_from_forward_price(call, forward, float(strike), t, True)
            out[t] = ivs
        return out

    def implied_vol_grid(
        self, expiries: Sequence[float], log_moneyness: np.ndarray
    ) -> dict[float, np.ndarray]:
        """Convenience: implied vols on a shared ``y = ln(K / F_t)`` grid for every expiry."""
        y = np.asarray(log_moneyness, dtype=float)
        strikes_by_expiry = {}
        for t in sorted(float(t) for t in expiries if t > 0.0):
            forward = self.s[self.spot_index] * np.exp((self.rate - self.dividend_yield) * t)
            strikes_by_expiry[t] = forward * np.exp(y)
        return self.implied_vols(expiries, strikes_by_expiry)
