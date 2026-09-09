"""Arbitrage-free SVI parameterisation of the vol surface, with analytic Dupire local volatility.

Each tenor slice is a raw-SVI total-variance curve in log-moneyness ``k = ln(K / F_t)``:

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

Fitting is a single joint least squares over every slice's ``(a, b, rho, m, sigma)`` with:

* per-slice **butterfly** (Durrleman) constraint ``g(k) >= 0`` on a dense grid,
* adjacent-slice **calendar** constraint ``w_i(k) <= w_{i+1}(k)`` (no crossing),
* Roger Lee wing bound ``b (1 + |rho|) <= 2`` and positivity ``a + b sigma sqrt(1 - rho^2) > 0``.

Because SVI is smooth and defined for all ``k``, the wings are the parameterisation's own
asymptotically-linear-in-``|k|`` continuation -- no separate extrapolation rule -- and every
derivative the Dupire formula needs is closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# Log-moneyness grid for the arbitrage penalties -- covers well beyond the 10-delta wings of any
# traded FX smile; the SVI parameterisation is asymptotically linear and safe past this.
_ARB_GRID = np.linspace(-0.9, 0.9, 73)


@dataclass(frozen=True)
class SVISlice:
    """One maturity's raw-SVI total-variance curve."""

    tenor: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray | float) -> np.ndarray:
        d = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d * d + self.sigma * self.sigma))

    def dw_dk(self, k: np.ndarray | float) -> np.ndarray:
        d = np.asarray(k, dtype=float) - self.m
        return self.b * (self.rho + d / np.sqrt(d * d + self.sigma * self.sigma))

    def d2w_dk2(self, k: np.ndarray | float) -> np.ndarray:
        d = np.asarray(k, dtype=float) - self.m
        r = np.sqrt(d * d + self.sigma * self.sigma)
        return self.b * self.sigma * self.sigma / (r * r * r)

    def implied_vol(self, k: np.ndarray | float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k), 1e-12) / self.tenor)

    def min_total_variance(self) -> float:
        return float(self.a + self.b * self.sigma * np.sqrt(max(1.0 - self.rho * self.rho, 0.0)))

    def durrleman_g(self, k: np.ndarray | float) -> np.ndarray:
        """Durrleman's function; ``>= 0`` everywhere iff the slice has no butterfly arbitrage."""
        w = np.maximum(self.total_variance(k), 1e-12)
        wp = self.dw_dk(k)
        wpp = self.d2w_dk2(k)
        return (1.0 - np.asarray(k, dtype=float) * wp / (2.0 * w)) ** 2 \
            - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0

    def scaled(self, variance_multiplier: float) -> "SVISlice":
        """Slice with total variance scaled by ``variance_multiplier`` (``a`` and ``b`` scale; the
        shape parameters ``rho, m, sigma`` are unchanged) -- used for the regime level shift."""
        return SVISlice(self.tenor, self.a * variance_multiplier, self.b * variance_multiplier,
                        self.rho, self.m, self.sigma)


# --------------------------------------------------------------------------------------------------
# per-slice quasi-explicit initial fit (Zeliade): for fixed (m, sigma), w is linear in (a, b*rho, b)
# --------------------------------------------------------------------------------------------------
def _quasi_explicit_slice(tenor: float, k: np.ndarray, w: np.ndarray) -> SVISlice:
    best: tuple[float, SVISlice] | None = None
    span = float(max(np.ptp(k), 0.1))
    for m in np.linspace(k.min() - 0.1 * span, k.max() + 0.1 * span, 11):
        for sigma in np.linspace(0.02, max(0.6, span), 12):
            d = k - m
            r = np.sqrt(d * d + sigma * sigma)
            design = np.column_stack([np.ones_like(k), d, r])          # w = a + p*d + q*r
            coef, *_ = np.linalg.lstsq(design, w, rcond=None)
            a, p, q = coef
            if q <= 1e-8 or abs(p) >= q:
                continue
            b = float(q)
            rho = float(np.clip(p / q, -0.999, 0.999))
            slc = SVISlice(tenor, float(a), b, rho, float(m), float(sigma))
            err = float(np.sum((design @ coef - w) ** 2))
            if best is None or err < best[0]:
                best = (err, slc)
    if best is None:
        atm = float(np.interp(0.0, k, w))
        return SVISlice(tenor, max(atm, 1e-6), 0.1, -0.2, 0.0, 0.1)
    return best[1]


# --------------------------------------------------------------------------------------------------
# joint arbitrage-constrained fit
# --------------------------------------------------------------------------------------------------
@dataclass
class SVIFitReport:
    rms_vol_error: float
    max_vol_error: float
    max_butterfly_violation: float      # max(0, -g) over the dense grid, all slices
    max_calendar_violation: float       # max(0, w_i - w_{i+1}) over the dense grid
    per_slice_rms: np.ndarray
    success: bool


def _params_to_slices(x: np.ndarray, tenors: np.ndarray) -> list[SVISlice]:
    return [SVISlice(float(t), *x[5 * i:5 * i + 5]) for i, t in enumerate(tenors)]


def fit_svi_surface(
    tenors: np.ndarray,
    k_by_tenor: list[np.ndarray],
    iv_by_tenor: list[np.ndarray],
    weight_by_tenor: list[np.ndarray] | None = None,
    *,
    butterfly_weight: float = 50.0,
    calendar_weight: float = 50.0,
    wing_weight: float = 10.0,
    max_nfev: int = 4000,
) -> tuple[list[SVISlice], SVIFitReport]:
    """Fit one SVI slice per tenor jointly, enforcing static arbitrage bounds by penalty."""
    tenors = np.asarray(tenors, dtype=float)
    n = len(tenors)
    if weight_by_tenor is None:
        weight_by_tenor = [np.ones_like(k) for k in k_by_tenor]

    w_by_tenor = [iv ** 2 * t for iv, t in zip(iv_by_tenor, tenors)]
    init_slices = [_quasi_explicit_slice(t, k_by_tenor[i], w_by_tenor[i]) for i, t in enumerate(tenors)]
    x0 = np.concatenate([np.array([s.a, s.b, s.rho, s.m, s.sigma]) for s in init_slices])

    lo = np.tile([-5.0, 0.0, -0.999, -2.0, 1e-3], n)
    hi = np.tile([5.0, 5.0, 0.999, 2.0, 2.0], n)
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

    def residuals(x: np.ndarray) -> np.ndarray:
        slices = _params_to_slices(x, tenors)
        parts: list[np.ndarray] = []
        for i, slc in enumerate(slices):
            model_iv = slc.implied_vol(k_by_tenor[i])
            parts.append((model_iv - iv_by_tenor[i]) * weight_by_tenor[i])
        for slc in slices:                                   # butterfly
            g = slc.durrleman_g(_ARB_GRID)
            parts.append(butterfly_weight * np.minimum(g, 0.0))
            parts.append(np.array([wing_weight * max(0.0, slc.b * (1.0 + abs(slc.rho)) - 2.0)]))
            parts.append(np.array([wing_weight * max(0.0, 1e-6 - slc.min_total_variance())]))
        for i in range(n - 1):                               # calendar (no crossing)
            gap = slices[i].total_variance(_ARB_GRID) - slices[i + 1].total_variance(_ARB_GRID)
            parts.append(calendar_weight * np.maximum(gap, 0.0))
        return np.concatenate(parts)

    sol = least_squares(residuals, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev)
    slices = _params_to_slices(sol.x, tenors)

    per_slice_rms = np.array([
        float(np.sqrt(np.mean((slices[i].implied_vol(k_by_tenor[i]) - iv_by_tenor[i]) ** 2)))
        for i in range(n)
    ])
    all_err = np.concatenate([slices[i].implied_vol(k_by_tenor[i]) - iv_by_tenor[i] for i in range(n)])
    bfly = max((float(np.max(np.maximum(-slc.durrleman_g(_ARB_GRID), 0.0))) for slc in slices), default=0.0)
    cal = 0.0
    for i in range(n - 1):
        gap = slices[i].total_variance(_ARB_GRID) - slices[i + 1].total_variance(_ARB_GRID)
        cal = max(cal, float(np.max(np.maximum(gap, 0.0))))

    report = SVIFitReport(
        rms_vol_error=float(np.sqrt(np.mean(all_err ** 2))),
        max_vol_error=float(np.max(np.abs(all_err))),
        max_butterfly_violation=bfly,
        max_calendar_violation=cal,
        per_slice_rms=per_slice_rms,
        success=bool(sol.success),
    )
    return slices, report


# --------------------------------------------------------------------------------------------------
# analytic Dupire from a stack of SVI slices
# --------------------------------------------------------------------------------------------------
def _slice_stack(slices: list[SVISlice]) -> np.ndarray:
    return np.array([s.tenor for s in slices], dtype=float)


def svi_total_variance_and_derivs(
    slices: list[SVISlice], t: float, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(w, w_k, w_kk, w_t)`` at ``(t, k)``: analytic in ``k``, total-variance-linear in ``t``
    between slices (constant forward variance before the first / after the last)."""
    tenors = _slice_stack(slices)
    k = np.atleast_1d(np.asarray(k, dtype=float))

    if t <= tenors[0]:
        s0 = slices[0]
        scale = t / tenors[0]
        return (s0.total_variance(k) * scale, s0.dw_dk(k) * scale, s0.d2w_dk2(k) * scale,
                s0.total_variance(k) / tenors[0])
    if t >= tenors[-1]:
        sl = slices[-1]
        scale = t / tenors[-1]
        return (sl.total_variance(k) * scale, sl.dw_dk(k) * scale, sl.d2w_dk2(k) * scale,
                sl.total_variance(k) / tenors[-1])

    hi = int(np.searchsorted(tenors, t, side="right"))
    lo = hi - 1
    t_lo, t_hi = tenors[lo], tenors[hi]
    frac = (t - t_lo) / (t_hi - t_lo)
    s_lo, s_hi = slices[lo], slices[hi]
    w = (1.0 - frac) * s_lo.total_variance(k) + frac * s_hi.total_variance(k)
    w_k = (1.0 - frac) * s_lo.dw_dk(k) + frac * s_hi.dw_dk(k)
    w_kk = (1.0 - frac) * s_lo.d2w_dk2(k) + frac * s_hi.d2w_dk2(k)
    w_t = (s_hi.total_variance(k) - s_lo.total_variance(k)) / (t_hi - t_lo)
    return w, w_k, w_kk, w_t


def svi_local_vol(
    slices: list[SVISlice], t: float, k: np.ndarray, *, vol_floor: float = 1e-3, vol_cap: float = 5.0
) -> np.ndarray:
    """Dupire local volatility ``sigma_loc(k, t)``, ``k = ln(S / F_t)``, from the SVI stack."""
    w, w_k, w_kk, w_t = svi_total_variance_and_derivs(slices, max(t, 1e-6), k)
    w = np.maximum(w, 1e-8)
    w_t = np.maximum(w_t, 1e-8)  # calendar monotonicity
    denom = (
        1.0
        - (k / w) * w_k
        + 0.25 * (-0.25 - 1.0 / w + (k * k) / (w * w)) * (w_k * w_k)
        + 0.5 * w_kk
    )
    denom = np.maximum(denom, 0.02)
    return np.sqrt(np.clip(w_t / denom, vol_floor * vol_floor, vol_cap * vol_cap))
