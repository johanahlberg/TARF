"""Calibration for the regime-switching local-vol model, kept as two independent stages (design brief):

1. Per-regime smile calibration -- a well-posed low-dimensional fit that can run daily.
2. Switching-matrix (Q) calibration -- run less often, against a target that carries *forward* smile
   information, since static smile marginals under-determine the switching speed.

The lightweight helpers ``calibrate_regime_smile`` and ``fit_switching_matrix`` from the first draft
are retained; the new ``calibrate_regime_smiles`` / ``calibrate_switching_rate`` are the real thing.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq, least_squares


# --------------------------------------------------------------------------------------------------
# Stage 1: per-regime smile
# --------------------------------------------------------------------------------------------------
def calibrate_regime_smile(strikes: np.ndarray | list[float], implied_vols: np.ndarray | list[float]) -> dict[str, float]:
    """Fit a single compact smile ``atm_vol + skew * x + curvature * x**2`` (x = log-moneyness).

    Lightweight single-smile helper retained for the interface tests; see ``calibrate_regime_smiles``
    for the multi-regime fit.
    """
    k = np.asarray(strikes, dtype=float)
    vols = np.asarray(implied_vols, dtype=float)

    if k.ndim != 1 or vols.ndim != 1:
        raise ValueError("strikes and implied_vols must be 1D arrays")
    if k.size != vols.size:
        raise ValueError("strikes and implied_vols must have the same length")
    if k.size < 3:
        raise ValueError("at least three smile points are required")

    x = np.log(k / np.median(k))
    design = np.column_stack([np.ones_like(x), x, x * x])
    coeffs, *_ = np.linalg.lstsq(design, vols, rcond=None)
    fitted = design @ coeffs
    return {
        "atm_vol": float(coeffs[0]),
        "skew": float(coeffs[1]),
        "curvature": float(coeffs[2]),
        "fit_error": float(np.sqrt(np.mean((vols - fitted) ** 2))),
    }


def _mixture_vol(log_moneyness: np.ndarray, atm: float, skew: float, curv: float, spread: float, weights: np.ndarray) -> np.ndarray:
    """Variance-weighted implied vol of the regime mixture: sqrt(sum_i w_i * sigma_i(k)^2), with the
    regimes a symmetric vol-dispersion ``(-spread, 0, +spread)`` around a shared base smile."""
    base = atm + skew * log_moneyness + curv * log_moneyness * log_moneyness
    shifts = np.array([-abs(spread), 0.0, abs(spread)])
    regime_vols = np.maximum(1e-6, base[None, :] + shifts[:, None])
    return np.sqrt(weights @ (regime_vols * regime_vols))


def calibrate_regime_smiles(
    log_moneyness: np.ndarray | list[float],
    market_vols: np.ndarray | list[float],
    regime_weights: np.ndarray | list[float],
    regime_spread: float = 0.04,
    smile_ref: float = 1.0,
    fit_spread: bool = False,
    initial: dict[str, float] | None = None,
) -> dict[str, object]:
    """Fit a shared base smile so the regime-weighted model reproduces the market vanilla smile.

    Free parameters: ``atm_vol / skew / curvature`` of the base smile. The regime vol-dispersion
    ``regime_spread`` (regimes at ``base - spread`` / ``base`` / ``base + spread``) is held fixed --
    at the money it is degenerate with the ATM level, so vanillas alone cannot pin it stably; it is
    a switching-structure parameter, set by stage 2 or a realized-vol prior. Pass ``fit_spread=True``
    to co-fit it anyway (accepting the weaker conditioning).

    Returns per-regime ``{local_vol, skew, curvature, smile_ref}`` dicts ready for
    ``SingleRegimeLocalVolModel``.
    """
    k = np.asarray(log_moneyness, dtype=float)
    vols = np.asarray(market_vols, dtype=float)
    weights = np.asarray(regime_weights, dtype=float)
    if k.shape != vols.shape or k.ndim != 1:
        raise ValueError("log_moneyness and market_vols must be matching 1D arrays")
    if weights.shape != (3,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("regime_weights must be a length-3 vector summing to 1")
    if k.size < 3:
        raise ValueError("at least three smile points are required")
    if regime_spread < 0.0:
        raise ValueError("regime_spread must be non-negative")

    guess = {"atm_vol": float(np.mean(vols)), "skew": 0.0, "curvature": 0.0, "spread": regime_spread}
    if initial:
        guess.update(initial)

    if fit_spread:
        p0 = [guess["atm_vol"], guess["skew"], guess["curvature"], abs(guess["spread"])]
        lo, hi = [1e-3, -5.0, -5.0, 0.0], [5.0, 5.0, 5.0, 2.0]
        residual = lambda p: _mixture_vol(k, p[0], p[1], p[2], p[3], weights) - vols  # noqa: E731
    else:
        p0 = [guess["atm_vol"], guess["skew"], guess["curvature"]]
        lo, hi = [1e-3, -5.0, -5.0], [5.0, 5.0, 5.0]
        residual = lambda p: _mixture_vol(k, p[0], p[1], p[2], regime_spread, weights) - vols  # noqa: E731

    sol = least_squares(residual, p0, method="trf", bounds=(lo, hi))
    atm, skew, curv = sol.x[0], sol.x[1], sol.x[2]
    spread = sol.x[3] if fit_spread else regime_spread
    shifts = [-abs(float(spread)), 0.0, abs(float(spread))]
    fitted = _mixture_vol(k, atm, skew, curv, spread, weights)

    regimes = [
        {
            "local_vol": float(max(1e-6, atm + d)),
            "skew": float(skew),
            "curvature": float(curv),
            "smile_ref": float(smile_ref),
        }
        for d in shifts
    ]
    return {
        "regimes": regimes,
        "base": {"atm_vol": float(atm), "skew": float(skew), "curvature": float(curv)},
        "spread": float(abs(spread)),
        "fit_error": float(np.sqrt(np.mean((fitted - vols) ** 2))),
        "success": bool(sol.success),
    }


# --------------------------------------------------------------------------------------------------
# Stage 2: switching matrix
# --------------------------------------------------------------------------------------------------
def fit_switching_matrix(weights: np.ndarray | list[list[float]]) -> np.ndarray:
    """Convert a row-stochastic transition matrix P to a generator Q = P - I (retained helper)."""
    matrix = np.asarray(weights, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("weights must have shape (3, 3)")
    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each row of the transition matrix must be positive and sum to a positive value")
    normalized = matrix / row_sums[:, None]
    return normalized - np.eye(3, dtype=float)


def build_generator(stationary: np.ndarray | list[float], switch_rate: float) -> np.ndarray:
    """Reversible 3-state generator with a prescribed stationary distribution and overall speed:

        Q = switch_rate * (1 pi^T - I)

    Rows sum to zero, off-diagonals are non-negative, and ``pi`` is the stationary distribution.
    A single scalar ``switch_rate`` controls how fast regime probabilities relax toward ``pi`` --
    exactly the degree of freedom that static smiles cannot pin down.
    """
    pi = np.asarray(stationary, dtype=float)
    if pi.shape != (3,) or np.any(pi < 0) or not np.isclose(pi.sum(), 1.0):
        raise ValueError("stationary must be a length-3 probability vector")
    if switch_rate < 0.0:
        raise ValueError("switch_rate must be non-negative")
    return switch_rate * (np.ones((3, 1)) @ pi[None, :] - np.eye(3))


def regime_probabilities(pi0: np.ndarray | list[float], q: np.ndarray, t: float) -> np.ndarray:
    """Regime distribution at time ``t`` given the initial distribution ``pi0``: pi0 @ exp(Q t)."""
    pi0 = np.asarray(pi0, dtype=float)
    return pi0 @ expm(np.asarray(q, dtype=float) * float(t))


def model_total_variance(pi0: np.ndarray | list[float], q: np.ndarray, regime_vols: np.ndarray | list[float], t: float) -> float:
    """E[ integral_0^t sigma(u)^2 du ] under the regime-switching mixture (no vol smile dependence)."""
    sig2 = np.asarray(regime_vols, dtype=float) ** 2
    integrand = lambda u: float(regime_probabilities(pi0, q, u) @ sig2)  # noqa: E731
    value, _ = quad(integrand, 0.0, float(t), limit=200)
    return value


def model_forward_variance(
    pi0: np.ndarray | list[float], q: np.ndarray, regime_vols: np.ndarray | list[float], t1: float, t2: float
) -> float:
    """Forward variance over ``[t1, t2]`` -- the simplest model quantity that carries switching-speed
    (forward-smile) information and is monotone in the switch rate when the regimes differ."""
    if not 0.0 <= t1 < t2:
        raise ValueError("require 0 <= t1 < t2")
    return model_total_variance(pi0, q, regime_vols, t2) - model_total_variance(pi0, q, regime_vols, t1)


def calibrate_switching_rate(
    pi0: np.ndarray | list[float],
    stationary: np.ndarray | list[float],
    regime_vols: np.ndarray | list[float],
    t1: float,
    t2: float,
    target_forward_variance: float,
    max_rate: float = 50.0,
) -> dict[str, object]:
    """Calibrate the scalar switch rate so the model forward variance over ``[t1, t2]`` matches a
    target (e.g. implied by forward-starting vanillas or a risk-reversal/butterfly term structure).

    The current regime distribution ``pi0`` must differ from the long-run ``stationary`` for the
    switch rate to be identified: it governs how fast ``pi0`` relaxes toward ``stationary``, and if
    they are equal the regime probabilities -- and hence the forward variance -- never move.

    Returns the rate and the corresponding generator ``Q``. Raises if the target is not bracketed
    by ``switch_rate in [0, max_rate]``.
    """
    pi0 = np.asarray(pi0, dtype=float)
    pi_star = np.asarray(stationary, dtype=float)
    if pi0.shape != (3,) or not np.isclose(pi0.sum(), 1.0):
        raise ValueError("pi0 must be a length-3 probability vector")
    if pi_star.shape != (3,) or not np.isclose(pi_star.sum(), 1.0):
        raise ValueError("stationary must be a length-3 probability vector")
    if np.allclose(pi0, pi_star, atol=1e-9):
        raise ValueError("pi0 must differ from stationary for the switch rate to be identifiable")

    def gap(rate: float) -> float:
        q = build_generator(pi_star, rate)
        return model_forward_variance(pi0, q, regime_vols, t1, t2) - target_forward_variance

    lo, hi = gap(0.0), gap(max_rate)
    if lo == 0.0:
        rate = 0.0
    elif np.sign(lo) == np.sign(hi):
        raise ValueError("target_forward_variance is not reachable for switch_rate in [0, max_rate]")
    else:
        rate = float(brentq(gap, 0.0, max_rate, xtol=1e-10, rtol=1e-12))

    q = build_generator(pi_star, rate)
    return {
        "switch_rate": float(rate),
        "q": q,
        "forward_variance": model_forward_variance(pi0, q, regime_vols, t1, t2),
    }
