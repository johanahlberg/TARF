"""Barrier and touch options under the (single- or three-) regime local-vol model.

A backward log-space PDE, one slice per regime, coupled by ``exp(Q dt)`` -- the same engine as the
TARF pricer but without the accumulated-amount dimension. The barrier is pinned to a grid node and
the knock region is overwritten every sub-step (continuous monitoring); a Crank-Nicolson
theta-scheme with a fully-implicit Rannacher startup damps the payoff kink at the barrier.

These are the instruments that carry the forward-smile / path-dependence information the vanilla
surface lacks, so they are what ``calibrate_regime_model(barrier_quotes=...)`` fits the regime
structural parameters (``level_spread``, ``skew_spread``, switch rate) to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm

from .solver import RANNACHER_STEPS, _LogVolOperator, _log_spot_grid

# Broadie-Glasserman-Kou-style continuity correction: a barrier monitored on the discrete time grid
# is harder to breach than a continuously-monitored one, so the effective barrier is shifted towards
# the spot by exp(+- beta sigma sqrt(dt)). The theoretical beta = 0.5826 assumes discrete monitoring
# with no intra-interval dynamics; here the Crank-Nicolson diffusion within a step already captures
# some sub-step crossing, so beta is tuned down to ~0.30 (one-touch error < 5 bp vs the
# Rubinstein-Reiner closed form on a flat-vol single regime).
_BGK = 0.30


def _barrier_grid(spot: float, barrier: float, sigma_ref: float, maturity: float, num_spot: int) -> np.ndarray:
    """Log-spot grid with both ``spot`` and ``barrier`` on nodes (reuses the TARF grid builder,
    which pins the 'strike' -- here the barrier)."""
    return np.exp(_log_spot_grid(spot, barrier, sigma_ref, maturity, num_spot, n_std=6.0))


def _effective_barrier(barrier: float, direction: str, sigma_ref: float, dt: float) -> float:
    shift = _BGK * sigma_ref * np.sqrt(max(dt, 0.0))
    return barrier * np.exp(shift if direction == "down" else -shift)


def _knock_mask(s_grid: np.ndarray, barrier: float, direction: str) -> np.ndarray:
    if direction == "up":
        return s_grid >= barrier * (1.0 - 1e-12)
    if direction == "down":
        return s_grid <= barrier * (1.0 + 1e-12)
    raise ValueError("direction must be 'up' or 'down'")


class RegimeBarrierPricer:
    """Barrier / touch pricer for a :class:`~tarf_rslv.calibration_surface.CalibratedRegimeModel`
    (or any object exposing ``spot``, ``rate``, ``dividend_yield``, ``regimes`` with
    ``local_volatility(s, t)``, ``q``, ``regime_weights``, ``time_varying``)."""

    def __init__(self, model, *, num_spot: int = 401, steps_per_year: int = 600, min_steps: int = 120) -> None:
        self.model = model
        self.num_spot = max(81, int(num_spot))
        self.steps_per_year = int(steps_per_year)
        self.min_steps = int(min_steps)

    def _num_steps(self, maturity: float) -> int:
        return max(self.min_steps, int(round(self.steps_per_year * maturity)))

    # ---------------------------------------------------------------------------------------
    def _solve(
        self,
        barrier: float,
        maturity: float,
        direction: str,
        terminal: Callable[[np.ndarray], np.ndarray],
        knock_value: float,
    ) -> float:
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")
        model = self.model
        n_reg = len(model.regimes)
        sigma_ref = float(getattr(model.regimes[0], "sigma_ref", 0.1) or 0.1)
        n_steps = self._num_steps(maturity)
        b_eff = _effective_barrier(barrier, direction, sigma_ref, maturity / n_steps)
        s_grid = _barrier_grid(model.spot, b_eff, sigma_ref, maturity, self.num_spot)
        x_grid = np.log(s_grid)
        knock = _knock_mask(s_grid, b_eff, direction)
        time_varying = bool(getattr(model, "time_varying", False))

        q = np.asarray(model.q if model.q is not None else np.zeros((n_reg, n_reg)), dtype=float)
        couple = (lambda dt: expm(q * dt)) if n_reg > 1 else None

        values = np.repeat(np.asarray(terminal(s_grid), dtype=float)[None, :], n_reg, axis=0)
        values[:, knock] = knock_value

        times = np.linspace(maturity, 0.0, n_steps + 1)
        for step in range(n_steps):
            t_hi, t_lo = float(times[step]), float(times[step + 1])
            dt = t_hi - t_lo
            theta = 1.0 if step < RANNACHER_STEPS else 0.5
            t_mid = 0.5 * (t_hi + t_lo)
            operators = [
                _LogVolOperator(
                    x_grid,
                    regime.local_volatility(s_grid, t_mid) if time_varying else regime.local_volatility(s_grid),
                    regime.rate,
                    regime.dividend_yield,
                )
                for regime in model.regimes
            ]
            diffused = np.stack([operators[r].step(values[r], dt, theta) for r in range(n_reg)])
            values = np.einsum("ij,jk->ik", couple(dt), diffused) if couple is not None else diffused
            values[:, knock] = knock_value

        idx = int(np.argmin(np.abs(s_grid - model.spot)))
        return float(np.asarray(model.regime_weights, dtype=float) @ values[:, idx])

    # ---------------------------------------------------------------------------------------
    def one_touch(
        self, barrier: float, maturity: float, *, direction: str, rebate: float = 1.0,
        payment: str = "hit",
    ) -> float:
        """Value of a one-touch: pays ``rebate`` if ``spot`` reaches ``barrier`` before ``maturity``.

        ``payment="hit"`` -- paid at the hit (barrier value = rebate, discounted through the
        operator to the hit time). ``payment="expiry"`` -- paid at maturity if touched (barrier
        value = rebate * DF(t, T); handled by pricing the deferred claim)."""
        if payment == "hit":
            return self._solve(barrier, maturity, direction, lambda s: np.zeros_like(s), rebate)
        if payment == "expiry":
            # value(touched, t) = rebate * exp(-r_d (T - t)); march that boundary back
            return self._solve_deferred(barrier, maturity, direction, rebate)
        raise ValueError("payment must be 'hit' or 'expiry'")

    def no_touch(
        self, barrier: float, maturity: float, *, direction: str, rebate: float = 1.0,
    ) -> float:
        """Value of a no-touch (pays ``rebate`` at maturity if the barrier is never reached).
        Discounting is in the operator, so the terminal condition is the undiscounted payoff."""
        return self._solve(barrier, maturity, direction, lambda s: np.full_like(s, rebate), 0.0)

    def knock_out(
        self, strike: float, barrier: float, maturity: float, *, option_type: str, direction: str,
        rebate: float = 0.0,
    ) -> float:
        """European knock-out option: the vanilla payoff at maturity unless the barrier is touched."""
        cp = 1.0 if option_type == "call" else -1.0
        return self._solve(
            barrier, maturity, direction, lambda s: np.maximum(cp * (s - strike), 0.0), rebate,
        )

    def knock_in(
        self, strike: float, barrier: float, maturity: float, *, option_type: str, direction: str,
    ) -> float:
        """Knock-in by parity: vanilla - knock-out (same strike / barrier, zero KO rebate)."""
        vanilla = self._vanilla(strike, maturity, option_type)
        knockout = self.knock_out(strike, barrier, maturity, option_type=option_type, direction=direction)
        return vanilla - knockout

    # ---------------------------------------------------------------------------------------
    def _domestic_rate(self, t: float) -> float:
        return float(self.model.regimes[0].rate)

    def _vanilla(self, strike: float, maturity: float, option_type: str) -> float:
        """Plain European from the same backward engine (barrier far out of the way)."""
        far = self.model.spot * (100.0 if option_type == "call" else 0.01)
        cp = 1.0 if option_type == "call" else -1.0
        return self._solve(far, maturity, "up" if cp > 0 else "down",
                           lambda s: np.maximum(cp * (s - strike), 0.0), 0.0)

    def _solve_deferred(self, barrier: float, maturity: float, direction: str, rebate: float) -> float:
        model = self.model
        n_reg = len(model.regimes)
        sigma_ref = float(getattr(model.regimes[0], "sigma_ref", 0.1) or 0.1)
        n_steps = self._num_steps(maturity)
        b_eff = _effective_barrier(barrier, direction, sigma_ref, maturity / n_steps)
        s_grid = _barrier_grid(model.spot, b_eff, sigma_ref, maturity, self.num_spot)
        x_grid = np.log(s_grid)
        knock = _knock_mask(s_grid, b_eff, direction)
        time_varying = bool(getattr(model, "time_varying", False))
        rate = self._domestic_rate(maturity)
        q = np.asarray(model.q if model.q is not None else np.zeros((n_reg, n_reg)), dtype=float)
        couple = (lambda dt: expm(q * dt)) if n_reg > 1 else None

        values = np.zeros((n_reg, s_grid.size))
        values[:, knock] = rebate  # value at T for a touched path

        times = np.linspace(maturity, 0.0, n_steps + 1)
        for step in range(n_steps):
            t_hi, t_lo = float(times[step]), float(times[step + 1])
            dt = t_hi - t_lo
            theta = 1.0 if step < RANNACHER_STEPS else 0.5
            t_mid = 0.5 * (t_hi + t_lo)
            operators = [
                _LogVolOperator(
                    x_grid,
                    regime.local_volatility(s_grid, t_mid) if time_varying else regime.local_volatility(s_grid),
                    regime.rate, regime.dividend_yield,
                )
                for regime in model.regimes
            ]
            diffused = np.stack([operators[r].step(values[r], dt, theta) for r in range(n_reg)])
            values = np.einsum("ij,jk->ik", couple(dt), diffused) if couple is not None else diffused
            values[:, knock] = rebate * np.exp(-rate * (maturity - t_lo))

        idx = int(np.argmin(np.abs(s_grid - model.spot)))
        return float(np.asarray(model.regime_weights, dtype=float) @ values[:, idx])


# --------------------------------------------------------------------------------------------------
# calibration quote
# --------------------------------------------------------------------------------------------------
@dataclass
class OneTouchQuote:
    """A market one-touch price (as a fraction of the rebate, 0..1)."""

    barrier: float
    maturity: float
    direction: str                 # "up" | "down"
    market_price: float
    rebate: float = 1.0
    payment: str = "hit"           # "hit" | "expiry"
    weight: float = 1.0

    def model_price(self, pricer: RegimeBarrierPricer) -> float:
        return pricer.one_touch(
            self.barrier, self.maturity, direction=self.direction, rebate=self.rebate, payment=self.payment
        ) / self.rebate
