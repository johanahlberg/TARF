"""FX volatility-surface layer: reconstruct a full smile from broker quotes, hold a term
structure of them, and produce implied / local (Dupire) volatilities anywhere on the surface.

Broker quotes per tenor are the market standard five: the ATM volatility plus the 25- and
10-delta risk reversals and butterflies,

    RR_d  = sigma(d-call)  - sigma(d-put)
    BF_d  = 0.5 * (sigma(d-call) + sigma(d-put)) - sigma_ATM            (smile butterfly)

but the *quoted* butterfly is the **market strangle** (``BF_d^MS``): the single volatility to
add to the ATM level so that a strangle struck at the resulting ``+-d`` strikes has the same
price as the true smile strangle. Recovering the smile from the market strangle is a small
root-find per wing per tenor (Reiswich & Wystup 2010), done in ``SmileQuotes.knots``.

Per-currency-pair market conventions (``DeltaConvention``; build from strings with
``DeltaConvention.from_market(delta_type, atm_convention, premium)``):

* ``delta_type``      -- "spot" (default) or "forward" ("fwd") Black-Scholes delta,
* ``atm_convention``  -- "dns" delta-neutral straddle (default) or "forward" ("fwd" / "atmf", K = F),
* ``premium``         -- how the option price is quoted: "domestic" (pips of the terms currency;
  delta not premium-adjusted -- e.g. CHF for USDCHF) or "foreign" (% of the base currency;
  premium-adjusted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .svi import SVIFitReport, SVISlice, fit_svi_surface, svi_local_vol, svi_total_variance_and_derivs

SQRT_EPS = 1e-12


# --------------------------------------------------------------------------------------------------
# Black-Scholes building blocks (forward / undiscounted)
# --------------------------------------------------------------------------------------------------
def _d1_d2(forward: float, strike: float, sigma: float, t: float) -> tuple[float, float]:
    vol_sqrt_t = max(sigma * np.sqrt(t), SQRT_EPS)
    d1 = (np.log(forward / strike) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_forward_price(forward: float, strike: float, sigma: float, t: float, is_call: bool) -> float:
    """Undiscounted Black-Scholes price (multiply by the domestic discount factor for PV)."""
    if sigma <= 0.0 or t <= 0.0:
        intrinsic = (forward - strike) if is_call else (strike - forward)
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(forward, strike, sigma, t)
    if is_call:
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)


def bs_forward_vega(forward: float, strike: float, sigma: float, t: float) -> float:
    d1, _ = _d1_d2(forward, strike, sigma, t)
    return forward * norm.pdf(d1) * np.sqrt(t)


def implied_vol_from_forward_price(
    price: float, forward: float, strike: float, t: float, is_call: bool
) -> float:
    """Invert ``bs_forward_price`` for the volatility. Returns ``nan`` outside the no-arbitrage band."""
    intrinsic = max((forward - strike) if is_call else (strike - forward), 0.0)
    upper = forward if is_call else strike
    if not (intrinsic - 1e-14 <= price <= upper + 1e-14):
        return float("nan")
    price = min(max(price, intrinsic + 1e-15), upper - 1e-15)

    def objective(sigma: float) -> float:
        return bs_forward_price(forward, strike, sigma, t, is_call) - price

    try:
        return float(brentq(objective, 1e-6, 8.0, xtol=1e-12, rtol=1e-12, maxiter=200))
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------------------------------
# FX delta <-> strike
# --------------------------------------------------------------------------------------------------
_DELTA_TYPE_ALIASES = {
    "spot": "spot", "s": "spot", "spot_delta": "spot",
    "forward": "forward", "fwd": "forward", "f": "forward", "forward_delta": "forward",
}
_ATM_ALIASES = {
    "forward": "forward", "fwd": "forward", "atmf": "forward", "atm_forward": "forward", "f": "forward",
    "dns": "dns", "delta_neutral": "dns", "delta_neutral_straddle": "dns", "delta-neutral": "dns",
    "delta_neutral_atm": "dns", "straddle": "dns",
    "spot": "spot",
}
# how the option price / premium is quoted -> whether delta is premium-adjusted
_PREMIUM_ADJUSTED = {
    "domestic": False, "%domestic": False, "domestic_pips": False, "pips": False,
    "unadjusted": False, "excluded": False, "premium_excluded": False, "terms": False, "false": False,
    "foreign": True, "%foreign": True, "foreign_pct": True, "pct": True, "%f": True,
    "adjusted": True, "included": True, "premium_included": True, "base": True, "true": True,
}


@dataclass(frozen=True)
class DeltaConvention:
    """FX market conventions for the smile: how delta is defined, what "ATM" means, and whether delta
    is premium-adjusted (which follows from the currency the option price is quoted in)."""

    delta_type: str = "spot"          # "spot" | "forward"
    premium_adjusted: bool = False
    atm_convention: str = "dns"        # "dns" | "forward" | "spot"

    def __post_init__(self) -> None:
        if self.delta_type not in {"spot", "forward"}:
            raise ValueError("delta_type must be 'spot' or 'forward'")
        if self.atm_convention not in {"dns", "forward", "spot"}:
            raise ValueError("atm_convention must be 'dns', 'forward' or 'spot'")

    @classmethod
    def from_market(
        cls, delta_type: str = "spot", atm_convention: str = "dns", premium: str | bool = "domestic",
    ) -> "DeltaConvention":
        """Build from per-currency-pair strings (as configured in Front Arena):

        * ``delta_type``     -- "spot" or "forward" ("fwd");
        * ``atm_convention`` -- "forward" ("fwd" / "atmf") or "dns" (delta-neutral straddle);
        * ``premium``        -- the price/premium convention: "domestic" (pips of the terms currency;
          delta not premium-adjusted) or "foreign" (% of the base currency; premium-adjusted). A
          bool is also accepted directly as ``premium_adjusted``.
        """
        dt = _DELTA_TYPE_ALIASES.get(str(delta_type).strip().lower())
        atm = _ATM_ALIASES.get(str(atm_convention).strip().lower())
        if dt is None:
            raise ValueError(f"unknown delta_type {delta_type!r}")
        if atm is None:
            raise ValueError(f"unknown atm_convention {atm_convention!r}")
        if isinstance(premium, bool):
            pa = premium
        else:
            pa = _PREMIUM_ADJUSTED.get(str(premium).strip().lower())
            if pa is None:
                raise ValueError(f"unknown premium convention {premium!r} (use 'domestic' or 'foreign')")
        return cls(delta_type=dt, premium_adjusted=pa, atm_convention=atm)


def atm_strike(
    spot: float, forward: float, sigma: float, t: float, foreign_df: float, convention: DeltaConvention
) -> float:
    if convention.atm_convention == "forward":
        return forward
    if convention.atm_convention == "spot":
        return spot
    # delta-neutral straddle
    if convention.premium_adjusted:
        return forward * np.exp(-0.5 * sigma * sigma * t)
    return forward * np.exp(0.5 * sigma * sigma * t)


def strike_from_delta(
    signed_delta: float,
    sigma: float,
    t: float,
    spot: float,
    forward: float,
    foreign_df: float,
    convention: DeltaConvention,
) -> float:
    """Strike whose Black-Scholes delta equals ``signed_delta`` (>0 call wing, <0 put wing)."""
    is_call = signed_delta > 0.0
    vol_sqrt_t = sigma * np.sqrt(t)

    if not convention.premium_adjusted:
        # delta = [fd] * phi * N(phi d1);  fd = foreign_df (spot delta) or 1 (forward delta)
        fd = foreign_df if convention.delta_type == "spot" else 1.0
        n_arg = abs(signed_delta) / fd
        n_arg = min(max(n_arg, 1e-10), 1.0 - 1e-10)
        d1 = norm.ppf(n_arg) if is_call else -norm.ppf(n_arg)
        return forward * np.exp(-d1 * vol_sqrt_t + 0.5 * vol_sqrt_t * vol_sqrt_t)

    # premium-adjusted: delta = [fd] * phi * (K / F) * N(phi d2)  -> implicit in K
    fd = foreign_df if convention.delta_type == "spot" else 1.0
    target = abs(signed_delta)

    def objective(strike: float) -> float:
        _, d2 = _d1_d2(forward, strike, sigma, t)
        model = fd * (strike / forward) * (norm.cdf(d2) if is_call else norm.cdf(-d2))
        return model - target

    lo = forward * np.exp(-8.0 * vol_sqrt_t)
    hi = forward * np.exp(8.0 * vol_sqrt_t)
    return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


# --------------------------------------------------------------------------------------------------
# One tenor's smile: broker quotes -> five (strike, vol) knots
# --------------------------------------------------------------------------------------------------
@dataclass
class SmileQuotes:
    """Quotes for a single tenor.

    ``bf_convention="market_strangle"`` (default) -- ``bf25`` / ``bf10`` are broker **market
    strangles**; the smile is recovered by the price-matching root-find. ``"smile"`` -- ``bf25`` /
    ``bf10`` are already smile butterflies (e.g. read straight off a parametric surface), used as-is.
    """

    tenor: float                       # year fraction
    atm: float
    rr25: float
    bf25: float
    rr10: float
    bf10: float
    bf_convention: str = "market_strangle"

    def __post_init__(self) -> None:
        if self.bf_convention not in {"market_strangle", "smile"}:
            raise ValueError("bf_convention must be 'market_strangle' or 'smile'")

    def _wing_smile_vols(
        self,
        pillar_delta: float,
        rr: float,
        bf_market: float,
        spot: float,
        forward: float,
        foreign_df: float,
        domestic_df: float,
        convention: DeltaConvention,
    ) -> tuple[float, float, float, float]:
        """Solve the market-strangle constraint for the smile butterfly, then return
        ``(K_put, sigma_put, K_call, sigma_call)`` for this wing."""
        t = self.tenor

        if self.bf_convention == "smile":
            sig_call = self.atm + bf_market + 0.5 * rr
            sig_put = self.atm + bf_market - 0.5 * rr
            k_call = strike_from_delta(pillar_delta, sig_call, t, spot, forward, foreign_df, convention)
            k_put = strike_from_delta(-pillar_delta, sig_put, t, spot, forward, foreign_df, convention)
            return k_put, sig_put, k_call, sig_call

        sigma_ms = self.atm + bf_market
        k_call_ms = strike_from_delta(pillar_delta, sigma_ms, t, spot, forward, foreign_df, convention)
        k_put_ms = strike_from_delta(-pillar_delta, sigma_ms, t, spot, forward, foreign_df, convention)
        target_price = (
            bs_forward_price(forward, k_call_ms, sigma_ms, t, True)
            + bs_forward_price(forward, k_put_ms, sigma_ms, t, False)
        )

        def strangle_gap(bf_smile: float) -> float:
            sig_call = self.atm + bf_smile + 0.5 * rr
            sig_put = self.atm + bf_smile - 0.5 * rr
            k_call = strike_from_delta(pillar_delta, sig_call, t, spot, forward, foreign_df, convention)
            k_put = strike_from_delta(-pillar_delta, sig_put, t, spot, forward, foreign_df, convention)
            price = (
                bs_forward_price(forward, k_call, sig_call, t, True)
                + bs_forward_price(forward, k_put, sig_put, t, False)
            )
            return price - target_price

        bf_smile = float(brentq(strangle_gap, bf_market - 0.05, bf_market + 0.05, xtol=1e-12, maxiter=200))
        sig_call = self.atm + bf_smile + 0.5 * rr
        sig_put = self.atm + bf_smile - 0.5 * rr
        k_call = strike_from_delta(pillar_delta, sig_call, t, spot, forward, foreign_df, convention)
        k_put = strike_from_delta(-pillar_delta, sig_put, t, spot, forward, foreign_df, convention)
        return k_put, sig_put, k_call, sig_call

    def knots(
        self,
        spot: float,
        forward: float,
        foreign_df: float,
        domestic_df: float,
        convention: DeltaConvention,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(strikes, implied_vols)`` sorted by strike: 10dP, 25dP, ATM, 25dC, 10dC."""
        k_atm = atm_strike(spot, forward, self.atm, self.tenor, foreign_df, convention)
        k25p, s25p, k25c, s25c = self._wing_smile_vols(
            0.25, self.rr25, self.bf25, spot, forward, foreign_df, domestic_df, convention
        )
        k10p, s10p, k10c, s10c = self._wing_smile_vols(
            0.10, self.rr10, self.bf10, spot, forward, foreign_df, domestic_df, convention
        )
        strikes = np.array([k10p, k25p, k_atm, k25c, k10c])
        vols = np.array([s10p, s25p, self.atm, s25c, s10c])
        order = np.argsort(strikes)
        return strikes[order], vols[order]


# --------------------------------------------------------------------------------------------------
# Full surface (arbitrage-free SVI per slice, analytic Dupire)
# --------------------------------------------------------------------------------------------------
@dataclass
class FXVolSurface:
    """A term structure of :class:`SmileQuotes`, fitted with one arbitrage-free SVI slice per tenor.

    ``domestic_zero`` / ``foreign_zero`` are callables ``t -> continuously-compounded Act/365
    zero rate``. The surface pins ``spot`` and derives every forward as ``spot * exp((r_d - r_f) t)``.
    Log-moneyness ``y = ln(K / F_t)``; total variance is linear in ``t`` between slices and grows
    linearly (constant forward vol) outside the quoted range. ``svi_report`` carries the fit error
    and the residual butterfly / calendar arbitrage.
    """

    spot: float
    smiles: list[SmileQuotes]
    domestic_zero: Callable[[float], float]
    foreign_zero: Callable[[float], float]
    convention: DeltaConvention = field(default_factory=DeltaConvention)
    vol_floor: float = 1e-3
    vol_cap: float = 5.0

    svi_slices: list[SVISlice] = field(default_factory=list, init=False)
    svi_report: SVIFitReport | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.smiles = sorted(self.smiles, key=lambda s: s.tenor)
        self._tenors = np.array([s.tenor for s in self.smiles], dtype=float)
        if self._tenors[0] <= 0.0:
            raise ValueError("all tenors must be positive")
        if len(np.unique(self._tenors)) != len(self._tenors):
            raise ValueError("duplicate tenor in the surface")

        self._knot_y: list[np.ndarray] = []
        knot_iv: list[np.ndarray] = []
        for smile in self.smiles:
            fwd = self.forward(smile.tenor)
            strikes, vols = smile.knots(
                self.spot, fwd, self.foreign_df(smile.tenor), self.domestic_df(smile.tenor), self.convention
            )
            self._knot_y.append(np.log(strikes / fwd))
            knot_iv.append(np.asarray(vols, dtype=float))

        self.svi_slices, self.svi_report = fit_svi_surface(self._tenors, self._knot_y, knot_iv)

    # -- curves --------------------------------------------------------------------------------
    def domestic_df(self, t: float) -> float:
        return float(np.exp(-self.domestic_zero(t) * t))

    def foreign_df(self, t: float) -> float:
        return float(np.exp(-self.foreign_zero(t) * t))

    def forward(self, t: float) -> float:
        return self.spot * self.foreign_df(t) / self.domestic_df(t)

    # -- implied vol / total variance --------------------------------------------------------
    def total_variance(self, t: float, y: np.ndarray | float) -> np.ndarray:
        w, _, _, _ = svi_total_variance_and_derivs(self.svi_slices, max(float(t), 1e-8), y)
        return np.maximum(w, 1e-12)

    def implied_vol(self, t: float, y: np.ndarray | float) -> np.ndarray:
        """Implied volatility at maturity ``t`` and log-moneyness ``y = ln(K / F_t)``."""
        return np.clip(np.sqrt(self.total_variance(t, y) / max(float(t), 1e-8)), self.vol_floor, self.vol_cap)

    def market_nodes(self) -> list[tuple[float, float, float, float]]:
        """``(tenor, y, implied_vol, vega_weight)`` for every quoted (tenor, delta) point (SVI values,
        which reproduce the quotes to ``svi_report.rms_vol_error``)."""
        nodes: list[tuple[float, float, float, float]] = []
        for slc, y_knots in zip(self.svi_slices, self._knot_y):
            fwd = self.forward(slc.tenor)
            vols = np.asarray(slc.implied_vol(y_knots), dtype=float)
            for y_val, vol in zip(y_knots, vols):
                vega = bs_forward_vega(fwd, fwd * np.exp(y_val), float(vol), slc.tenor)
                nodes.append((slc.tenor, float(y_val), float(vol), max(vega, 1e-6)))
        return nodes

    # -- Dupire local volatility ----------------------------------------------------------
    def local_vol(self, t: float, spot_level: np.ndarray | float) -> np.ndarray:
        """Dupire local volatility ``sigma_loc(S, t)`` -- analytic from the SVI stack (Gatheral
        total-variance form; ``k = ln(S / F_t)``)."""
        k = np.log(np.asarray(spot_level, dtype=float) / self.forward(t))
        return svi_local_vol(self.svi_slices, float(t), k, vol_floor=self.vol_floor, vol_cap=self.vol_cap)


def local_vol_from_total_variance(
    total_variance: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray | float,
    *,
    vol_floor: float = 1e-3,
    vol_cap: float = 5.0,
    dy: float = 6e-3,
    dt: float = 8e-3,
) -> np.ndarray:
    """Gatheral's local variance from a total-variance surface ``w(t, y)``, ``y = ln(K / F_t)``.

        v_L = (dw/dt) / [ 1 - (y/w) w_y + 0.25 (-0.25 - 1/w + y^2/w^2) w_y^2 + 0.5 w_yy ]
    """
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    t_up = t + dt
    t_dn = max(t - dt, 0.5 * dt)

    w = np.asarray(total_variance(t, y_arr), dtype=float)
    w = np.maximum(w, 1e-8)
    w_yp = np.asarray(total_variance(t, y_arr + dy), dtype=float)
    w_ym = np.asarray(total_variance(t, y_arr - dy), dtype=float)
    w_tp = np.asarray(total_variance(t_up, y_arr), dtype=float)
    w_tm = np.asarray(total_variance(t_dn, y_arr), dtype=float)

    w_y = (w_yp - w_ym) / (2.0 * dy)
    w_yy = (w_yp - 2.0 * w + w_ym) / (dy * dy)
    w_t = (w_tp - w_tm) / (t_up - t_dn)
    w_t = np.maximum(w_t, 1e-8)  # enforce calendar monotonicity

    denom = (
        1.0
        - (y_arr / w) * w_y
        + 0.25 * (-0.25 - 1.0 / w + (y_arr * y_arr) / (w * w)) * (w_y * w_y)
        + 0.5 * w_yy
    )
    denom = np.maximum(denom, 0.05)
    local_var = np.clip(w_t / denom, vol_floor * vol_floor, vol_cap * vol_cap)
    return np.sqrt(local_var)
