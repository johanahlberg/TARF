from __future__ import annotations

import numpy as np


def calibrate_regime_smile(strikes: np.ndarray | list[float], implied_vols: np.ndarray | list[float]) -> dict[str, float]:
    """Fit a compact SABR-like smile approximation using a quadratic in log-moneyness.

    The fit is intentionally lightweight and robust: we approximate the smile as
    atm_vol + skew * x + curvature * x^2, where x = log(S/K). This follows the design brief's
    preference for smooth, compact per-regime smile calibration over a noisy non-parametric surface.
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
    fit_error = float(np.sqrt(np.mean((vols - fitted) ** 2)))

    return {
        "atm_vol": float(coeffs[0]),
        "skew": float(coeffs[1]),
        "curvature": float(coeffs[2]),
        "fit_error": fit_error,
    }


def fit_switching_matrix(weights: np.ndarray | list[list[float]]) -> np.ndarray:
    """Convert a row-stochastic regime transition matrix to a generator matrix Q.

    For a row-stochastic matrix P, the generator matrix is Q = P - I, which guarantees a row sum of
    zero, a negative diagonal, and positive off-diagonal entries. This keeps the fitting structure
    stable and makes Q directly usable in a regime-switching PDE.
    """
    matrix = np.asarray(weights, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("weights must have shape (3, 3)")

    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each row of the transition matrix must be positive and sum to a positive value")
    normalized = matrix / row_sums[:, None]
    return normalized - np.eye(3, dtype=float)
