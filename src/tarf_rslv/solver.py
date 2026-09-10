from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm, solve_banded
from scipy.stats import norm

from .model import RegimeSwitchingLocalVolModel, SingleRegimeLocalVolModel
from .product import TARFAccumulator

# Rannacher startup: number of fully-implicit steps applied immediately after every non-smooth event
# (the terminal condition and each fixing jump) before switching to Crank-Nicolson. This damps the
# CN oscillation that a discontinuous payoff/knockout would otherwise excite near the kink.
RANNACHER_STEPS = 2


def _log_spot_grid(
    spot: float,
    strike: float,
    sigma_ref: float,
    maturity: float,
    num_spot: int,
    n_std: float = 5.0,
) -> np.ndarray:
    """Uniform grid in ``x = ln S`` (paper Section 4.4), with the spot pinned to a node and the strike
    pinned as well whenever that does not blow up the node count.

    A uniform log grid is what makes the central-difference stencil second-order consistent -- the
    previous implementation used a uniform-grid stencil on a geometrically-spaced grid, which is only
    first-order and biased the price by several percent with no convergence under refinement.
    """
    num_spot = max(41, int(num_spot))
    if num_spot % 2 == 0:
        num_spot += 1

    center = np.log(spot)
    log_mny = np.log(strike / spot)
    half_width = max(n_std * sigma_ref * np.sqrt(max(maturity, 1e-6)), abs(log_mny) * 1.5, 0.1)
    dx = 2.0 * half_width / (num_spot - 1)

    # Pin the strike too when it sits at least half a step away and doing so keeps the grid bounded.
    steps_to_strike = int(round(log_mny / dx))
    if abs(log_mny) >= 0.5 * dx and steps_to_strike != 0:
        dx = log_mny / steps_to_strike  # same sign as log_mny, so dx > 0
        half_pts = int(np.ceil(half_width / dx))
        if 2 * half_pts + 1 <= 6 * num_spot:
            return center + dx * np.arange(-half_pts, half_pts + 1)

    half_pts = (num_spot - 1) // 2
    return center + dx * np.arange(-half_pts, half_pts + 1)


class _LogVolOperator:
    """Spatial operator L for the log-space pricing PDE (paper eq. 17)

        dV/dt + 1/2 sigma(x)^2 V_xx + nu(x) V_x - r_d V = 0,   nu = r_d - r_f - 1/2 sigma^2

    discretised with central differences on a uniform ``x`` grid. Interior rows are the standard
    second-order stencil; the two boundary rows impose d^2V/dS^2 = 0 (paper Section 4.3) as a plain
    linear-in-S extrapolation, which for a payoff at most linear in S at the far field is exact.
    The linear-extrapolation rows reach three nodes deep, so the linear system is pentadiagonal.
    L is time-independent here, so it is built once and reused across every theta-scheme step.
    """

    def __init__(self, x_grid: np.ndarray, sigma_nodes: np.ndarray, rate: float, dividend_yield: float) -> None:
        n = len(x_grid)
        dx = float(x_grid[1] - x_grid[0])
        s_grid = np.exp(x_grid)
        sigma = np.asarray(sigma_nodes, dtype=float)
        nu = rate - dividend_yield - 0.5 * sigma * sigma

        diff = 0.5 * sigma * sigma / (dx * dx)
        adv = nu / (2.0 * dx)

        # Interior stencil coefficients (rows 1 .. n-2); boundary rows are algebraic, handled in step().
        self.n = n
        self.rate = float(rate)
        self.sub = diff - adv
        self.diag = -2.0 * diff - rate
        self.sup = diff + adv
        # Linear-in-S extrapolation weights: V_0 = (1+r0) V_1 - r0 V_2, V_{n-1} = (1+rN) V_{n-2} - rN V_{n-3}.
        self.r0 = float((s_grid[0] - s_grid[1]) / (s_grid[1] - s_grid[2]))
        self.rN = float((s_grid[-1] - s_grid[-2]) / (s_grid[-2] - s_grid[-3]))

    def matvec(self, values: np.ndarray) -> np.ndarray:
        """Apply the interior part of L; boundary rows return 0 (they are algebraic constraints)."""
        out = np.zeros_like(values)
        if values.ndim == 2:
            out[1:-1] = (
                self.sub[1:-1, None] * values[:-2]
                + self.diag[1:-1, None] * values[1:-1]
                + self.sup[1:-1, None] * values[2:]
            )
        else:
            out[1:-1] = self.sub[1:-1] * values[:-2] + self.diag[1:-1] * values[1:-1] + self.sup[1:-1] * values[2:]
        return out

    def step(self, values: np.ndarray, dt: float, theta: float) -> np.ndarray:
        """One backward theta-scheme step: (I - theta dt L) V^n = (I + (1-theta) dt L) V^{n+1}."""
        rhs = values + (1.0 - theta) * dt * self.matvec(values)
        rhs[0] = 0.0
        rhs[-1] = 0.0

        ab = np.zeros((5, self.n))  # solve_banded (l=2, u=2) layout: ab[2 + i - j, j] = A[i, j]
        ab[1, 2:] = -theta * dt * self.sup[1:-1]          # A[i, i+1], interior rows i = 1 .. n-2
        ab[2, 1:-1] = 1.0 - theta * dt * self.diag[1:-1]  # A[i, i]
        ab[3, :-2] = -theta * dt * self.sub[1:-1]         # A[i, i-1]

        # Row 0: V_0 - (1 + r0) V_1 + r0 V_2 = 0
        ab[2, 0] = 1.0
        ab[1, 1] = -(1.0 + self.r0)
        ab[0, 2] = self.r0
        # Row n-1: V_{n-1} - (1 + rN) V_{n-2} + rN V_{n-3} = 0
        ab[2, -1] = 1.0
        ab[3, -2] = -(1.0 + self.rN)
        ab[4, -3] = self.rN

        return solve_banded((2, 2), ab, rhs)


def _interp_along_accumulation(a_grid: np.ndarray, column_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate a value curve in the accumulated-amount coordinate.

    The paper stresses that the jump-condition interpolation must be smooth: linear/quadratic schemes
    can converge to the wrong answer for discretely-sampled path dependence (Forsyth et al. 2002).
    A natural cubic spline is used, with a linear fallback for grids too small to spline.
    """
    clipped = np.clip(query, a_grid[0], a_grid[-1])
    if len(a_grid) >= 4:
        return CubicSpline(a_grid, column_values, bc_type="natural")(clipped)
    return np.interp(clipped, a_grid, column_values)


def _apply_tarf_fixing(
    tarf: TARFAccumulator,
    fixing_index: int,
    s_grid: np.ndarray,
    a_grid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Apply one fixing-date jump of the TARF payoff to a (spot, accumulated) value grid.

    This is the paper's forward jump condition (eq. 10), explicit in the on-grid accumulated amount:

        V(S, t_k^-, A_j) = V(S, t_k, A_j + C_k(S, A_j)) + C_k(S, A_j)

    ``values`` is the continuation value immediately after this fixing; the return is the value
    immediately before it. On the ITM side ``A`` advances by a spot-dependent increment (handled by the
    spline interpolation above) and may breach the target, triggering the knockout settlement. On the
    OTM side ``A`` is unchanged and the leveraged / barrier-gated cashflow is added directly.
    """
    intrinsic = (s_grid - tarf.strike) * tarf.call_or_put
    itm_mask = intrinsic > 0.0
    notional1 = float(tarf.notional1[fixing_index])
    notional2 = float(tarf.notional2[fixing_index])

    result = np.empty_like(values)

    otm_spot = s_grid[~itm_mask]
    if otm_spot.size:
        hit = np.array([tarf.barrier_is_hit(float(s)) for s in otm_spot])
        otm_cashflow = notional2 * intrinsic[~itm_mask] * hit
        result[~itm_mask, :] = otm_cashflow[:, None] + values[~itm_mask, :]

    for i in np.nonzero(itm_mask)[0]:
        spot = float(s_grid[i])
        increment = tarf.accumulation_increment(spot)
        accumulated_new = a_grid + increment
        terminated = (tarf.target_level - accumulated_new) < tarf.zero_comparison

        settlement_notional = np.full(a_grid.shape, notional1)
        settlement_intrinsic = np.full(a_grid.shape, intrinsic[i])
        if np.any(terminated):
            remaining = tarf.target_level - a_grid
            if tarf.target_adjustment == 1:
                if tarf.inverted_target:
                    adjusted_strike = 1.0 / ((1.0 / spot) - remaining * tarf.call_or_put * -1.0)
                else:
                    adjusted_strike = spot - remaining * tarf.call_or_put
                settlement_intrinsic = np.where(terminated, (spot - adjusted_strike) * tarf.call_or_put, settlement_intrinsic)
            elif tarf.target_adjustment == 2:
                settlement_notional = np.where(terminated, notional1 * (remaining / increment), settlement_notional)
            elif tarf.target_adjustment == 3:
                settlement_notional = np.where(terminated, 0.0, settlement_notional)

        cashflow = settlement_notional * settlement_intrinsic
        continuation = _interp_along_accumulation(a_grid, values[i, :], accumulated_new)
        result[i, :] = cashflow + np.where(terminated, 0.0, continuation)

    return result


# A fixing within this many year-fractions of the valuation date (~0.03 s) is read as
# "fixes today, not yet observed": it is kept and snapped to t = 0 so its jump is applied
# at the front of the backward march (undiscounted, at the known current spot). Fixings
# before that are treated as already observed -- their realised amount belongs in
# ``TARFAccumulator.accumulated_value`` and they must not be re-applied here.
_FIXING_TIME_TOL = 1e-9


def _tarf_fixing_schedule(tarf: TARFAccumulator, maturity: float) -> list[tuple[int, float]]:
    schedule = []
    for k, raw in enumerate(tarf.fixing_dates):
        d = float(raw)
        if d < -_FIXING_TIME_TOL or d > maturity + _FIXING_TIME_TOL:
            continue
        schedule.append((k, max(d, 0.0)))
    return schedule


def _event_times(fixing_schedule: list[tuple[int, float]], maturity: float) -> np.ndarray:
    return np.array(sorted({0.0, float(maturity), *(d for _, d in fixing_schedule)}))


def _segment_steps(segment_length: float, maturity: float, n_steps: int) -> int:
    return max(RANNACHER_STEPS + 1, int(round(n_steps * segment_length / maturity)))


def _march_segment(operators, values: np.ndarray, t_hi: float, t_lo: float, n_sub: int, couple=None) -> np.ndarray:
    """March one inter-event segment backward from ``t_hi`` to ``t_lo`` with Rannacher startup.

    ``operators`` is a single ``_LogVolOperator`` (single regime) or a list of three (coupled). When
    coupled, ``couple(dt)`` returns exp(Q dt) and the regimes are recombined after each diffusion
    sub-step -- operator splitting, with the Q-coupling done exactly via the matrix exponential.
    """
    step_times = np.linspace(t_hi, t_lo, n_sub + 1)
    for s in range(n_sub):
        dt = float(step_times[s] - step_times[s + 1])
        theta = 1.0 if s < RANNACHER_STEPS else 0.5
        if couple is None:
            values = operators.step(values, dt, theta)
        else:
            diffused = np.stack([op.step(values[r], dt, theta) for r, op in enumerate(operators)])
            values = np.einsum("ij,jkl->ikl", couple(dt), diffused)
    return values


class SingleRegimePricer:
    """Single-regime local-vol pricer with a finite-difference TARF valuation, following the scheme of
    Luo & Shevchenko, *Pricing TARN Using a Finite Difference Method* (arXiv:1304.7563).

    Tracks ``num_target`` one-dimensional log-space PDE solutions (one per accumulated-amount node),
    couples them only through the fixing-date jump condition, uses a Crank-Nicolson theta-scheme with
    Rannacher startup, and interpolates the jump with a natural cubic spline.
    """

    def __init__(
        self,
        model: SingleRegimeLocalVolModel,
        num_spot: int = 121,
        num_target: int = 80,
        n_steps: int = 80,
    ) -> None:
        self.model = model
        self.num_spot = max(41, int(num_spot))
        self.num_target = max(4, int(num_target))
        self.n_steps = max(20, int(n_steps))

    def price_european(self, strike: float, maturity: float, option_type: str = "call") -> float:
        """Closed-form European price in the flat-vol limit (``model.local_vol`` as the ATM level)."""
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        s = self.model.spot
        r = self.model.rate
        q = self.model.dividend_yield
        sigma = self.model.local_vol

        sqrt_t = np.sqrt(maturity)
        d1 = (np.log(s / strike) + (r - q + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        if option_type == "call":
            price = s * np.exp(-q * maturity) * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * maturity) * norm.cdf(-d2) - s * np.exp(-q * maturity) * norm.cdf(-d1)
        return float(price)

    def _grids(self, tarf: TARFAccumulator, maturity: float) -> tuple[np.ndarray, np.ndarray]:
        s_grid = np.exp(
            _log_spot_grid(self.model.spot, tarf.strike, self.model.sigma_ref, maturity, self.num_spot)
        )
        a_lo = min(0.0, tarf.accumulated_value)
        a_grid = np.linspace(a_lo, tarf.target_level, self.num_target)
        return s_grid, a_grid

    def solve_tarf_pde(self, tarf: TARFAccumulator, maturity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward finite-difference TARF solve on a (spot, accumulated) grid.

        Zero terminal condition at ``T``; the final fixing's jump is applied first (T -> T^-), then the
        theta-scheme marches each accumulated-amount slice back to ``t_0``, re-applying the jump at
        every fixing date. Discounting is embedded in L, so the price is read straight off the grid.
        """
        if maturity <= 0:
            raise ValueError("maturity must be positive")

        s_grid, a_grid = self._grids(tarf, maturity)
        x_grid = np.log(s_grid)
        time_varying = bool(getattr(self.model, "time_varying", False))

        def operator_at(t_mid: float) -> _LogVolOperator:
            sigma_nodes = self.model.local_volatility(s_grid, t_mid) if time_varying else self.model.local_volatility(s_grid)
            return _LogVolOperator(x_grid, sigma_nodes, self.model.rate, self.model.dividend_yield)

        operator = None if time_varying else operator_at(0.0)

        schedule = _tarf_fixing_schedule(tarf, maturity)
        fixing_at = {round(d, 12): k for k, d in schedule}
        events = _event_times(schedule, maturity)

        values = np.zeros((len(s_grid), len(a_grid)), dtype=float)

        applied: set[int] = set()
        top_key = round(float(events[-1]), 12)
        if top_key in fixing_at:
            values = _apply_tarf_fixing(tarf, fixing_at[top_key], s_grid, a_grid, values)
            applied.add(fixing_at[top_key])

        for seg in range(len(events) - 2, -1, -1):
            t_hi, t_lo = float(events[seg + 1]), float(events[seg])
            n_sub = _segment_steps(t_hi - t_lo, maturity, self.n_steps)
            seg_operator = operator_at(0.5 * (t_hi + t_lo)) if time_varying else operator
            values = _march_segment(seg_operator, values, t_hi, t_lo, n_sub)

            key = round(t_lo, 12)
            if key in fixing_at and fixing_at[key] not in applied:
                values = _apply_tarf_fixing(tarf, fixing_at[key], s_grid, a_grid, values)
                applied.add(fixing_at[key])

        return s_grid, a_grid, values

    def price_tarf(self, tarf: TARFAccumulator, maturity: float) -> float:
        s_grid, a_grid, values = self.solve_tarf_pde(tarf=tarf, maturity=maturity)
        idx_spot = int(np.argmin(np.abs(s_grid - self.model.spot)))
        return float(_interp_along_accumulation(a_grid, values[idx_spot, :], np.array([tarf.accumulated_value]))[0])

    def tarf_greeks(self, tarf: TARFAccumulator, maturity: float, vega_bump: float = 1e-4) -> dict[str, float]:
        """Delta/gamma read directly off the PDE grid (no bump-and-revalue); vega by a single vol bump.

        On the uniform ``x = ln S`` grid, with V_x and V_xx by central differences at the pinned spot
        node: delta = V_x / S, gamma = (V_xx - V_x) / S^2.
        """
        s_grid, a_grid, values = self.solve_tarf_pde(tarf=tarf, maturity=maturity)
        idx = int(np.argmin(np.abs(s_grid - self.model.spot)))
        idx = min(max(idx, 1), len(s_grid) - 2)
        s0 = float(s_grid[idx])
        dx = float(np.log(s_grid[1]) - np.log(s_grid[0]))

        curve = np.array(
            [
                _interp_along_accumulation(a_grid, values[j, :], np.array([tarf.accumulated_value]))[0]
                for j in (idx - 1, idx, idx + 1)
            ]
        )
        price = float(curve[1])
        v_x = (curve[2] - curve[0]) / (2.0 * dx)
        v_xx = (curve[2] - 2.0 * curve[1] + curve[0]) / (dx * dx)
        delta = v_x / s0
        gamma = (v_xx - v_x) / (s0 * s0)

        if getattr(self.model, "time_varying", False):
            # calibrated (Dupire) model: bump the whole local-vol surface by a parallel shift
            bumped = self.model.bumped_vol(vega_bump)
        else:
            bumped = SingleRegimeLocalVolModel(
                spot=self.model.spot,
                rate=self.model.rate,
                dividend_yield=self.model.dividend_yield,
                local_vol=self.model.local_vol + vega_bump,
                strike=self.model.strike,
                skew=self.model.skew,
                curvature=self.model.curvature,
                smile_ref=self.model.smile_ref,
                vol_floor=self.model.vol_floor,
            )
        price_up = SingleRegimePricer(bumped, self.num_spot, self.num_target, self.n_steps).price_tarf(tarf, maturity)
        vega = (price_up - price) / vega_bump * 0.01

        return {"price": price, "delta": float(delta), "gamma": float(gamma), "vega": float(vega)}

    def compute_greeks(self, strike: float, maturity: float) -> dict[str, float]:
        """European delta/gamma/vega scaffold retained for the calibration/interface tests."""
        base_price = self.price_european(strike=strike, maturity=maturity)
        hp = 1e-3

        original_spot = self.model.spot
        self.model.spot = original_spot * (1.0 + hp)
        price_up = self.price_european(strike=strike, maturity=maturity)
        self.model.spot = original_spot * (1.0 - hp)
        price_down = self.price_european(strike=strike, maturity=maturity)
        self.model.spot = original_spot

        delta = (price_up - price_down) / (2.0 * hp * original_spot)
        gamma = (price_up - 2.0 * base_price + price_down) / (hp * original_spot) ** 2
        vega = 0.01 * base_price
        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


class ThreeRegimePricer:
    """Three coupled regime layers sharing one (spot, accumulated) grid.

    Between fixings each regime diffuses independently via its own ``_LogVolOperator`` and the three
    are recombined by exp(Q dt) after every sub-step (operator splitting). At each fixing the jump
    condition is applied per regime. In the single-regime limit (identical regimes) exp(Q dt) acts as
    the identity on the common value, so this reduces exactly to ``SingleRegimePricer``.
    """

    def __init__(
        self,
        model: RegimeSwitchingLocalVolModel,
        regime_weights: np.ndarray | None = None,
        num_spot: int = 121,
        num_target: int = 80,
        n_steps: int = 80,
    ):
        self.model = model
        self.num_spot = max(41, int(num_spot))
        self.num_target = max(4, int(num_target))
        self.n_steps = max(20, int(n_steps))
        if regime_weights is None:
            regime_weights = np.ones(3, dtype=float) / 3.0
        regime_weights = np.asarray(regime_weights, dtype=float)
        if regime_weights.shape != (3,):
            raise ValueError("regime_weights must be a 1D array of length 3")
        if not np.all(regime_weights >= 0.0):
            raise ValueError("regime weights must be non-negative")
        if not np.isclose(regime_weights.sum(), 1.0):
            raise ValueError("regime weights must sum to 1")
        self.regime_weights = regime_weights

    def solve_tarf_coupled_pde(self, tarf: TARFAccumulator, maturity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")

        reference = SingleRegimePricer(self.model.regimes[0], self.num_spot, self.num_target, self.n_steps)
        s_grid, a_grid = reference._grids(tarf, maturity)
        x_grid = np.log(s_grid)

        time_varying = bool(getattr(self.model, "time_varying", False))

        def operators_at(t_mid: float) -> list:
            return [
                _LogVolOperator(
                    x_grid,
                    regime.local_volatility(s_grid, t_mid) if time_varying else regime.local_volatility(s_grid),
                    regime.rate,
                    regime.dividend_yield,
                )
                for regime in self.model.regimes
            ]

        operators = None if time_varying else operators_at(0.0)
        q = np.asarray(self.model.q if self.model.q is not None else np.zeros((3, 3)), dtype=float)
        if q.shape != (3, 3):
            raise ValueError("Generator matrix must be 3x3")
        couple = lambda dt: expm(q * dt)  # noqa: E731

        schedule = _tarf_fixing_schedule(tarf, maturity)
        fixing_at = {round(d, 12): k for k, d in schedule}
        events = _event_times(schedule, maturity)

        values = np.zeros((3, len(s_grid), len(a_grid)), dtype=float)

        applied: set[int] = set()
        top_key = round(float(events[-1]), 12)
        if top_key in fixing_at:
            for r in range(3):
                values[r] = _apply_tarf_fixing(tarf, fixing_at[top_key], s_grid, a_grid, values[r])
            applied.add(fixing_at[top_key])

        for seg in range(len(events) - 2, -1, -1):
            t_hi, t_lo = float(events[seg + 1]), float(events[seg])
            n_sub = _segment_steps(t_hi - t_lo, maturity, self.n_steps)
            seg_operators = operators_at(0.5 * (t_hi + t_lo)) if time_varying else operators
            values = _march_segment(seg_operators, values, t_hi, t_lo, n_sub, couple=couple)

            key = round(t_lo, 12)
            if key in fixing_at and fixing_at[key] not in applied:
                for r in range(3):
                    values[r] = _apply_tarf_fixing(tarf, fixing_at[key], s_grid, a_grid, values[r])
                applied.add(fixing_at[key])

        return s_grid, a_grid, values

    def _single_regime_value(self, strike: float, maturity: float, option_type: str = "call") -> np.ndarray:
        return np.asarray(
            [
                SingleRegimePricer(model=regime).price_european(strike=strike, maturity=maturity, option_type=option_type)
                for regime in self.model.regimes
            ],
            dtype=float,
        )

    def _apply_regime_transition(self, regime_values: np.ndarray, maturity: float, n_steps: int | None = None) -> np.ndarray:
        values = np.asarray(regime_values, dtype=float)
        if values.shape != (3,):
            raise ValueError("regime_values must be a length-3 vector")

        q = self.model.q if self.model.q is not None else np.zeros((3, 3), dtype=float)
        if np.allclose(q, 0.0):
            return values.copy()

        if n_steps is None:
            n_steps = max(8, min(80, int(max(maturity, 0.0) * 40) + 1))
        dt = maturity / max(n_steps, 1)
        propagated = values.copy()
        for _ in range(max(n_steps, 1)):
            propagated = expm(q * dt) @ propagated
        return propagated

    def price_european(self, strike: float, maturity: float, option_type: str = "call") -> float:
        values = self._single_regime_value(strike=strike, maturity=maturity, option_type=option_type)
        coupled = self._apply_regime_transition(values, maturity=maturity)
        return float(np.dot(self.regime_weights, coupled))

    def _regime_curve(self, tarf: TARFAccumulator, maturity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_grid, a_grid, values = self.solve_tarf_coupled_pde(tarf=tarf, maturity=maturity)
        idx_spot = int(np.argmin(np.abs(s_grid - self.model.spot)))
        regime_values = np.array(
            [
                _interp_along_accumulation(a_grid, values[r, idx_spot, :], np.array([tarf.accumulated_value]))[0]
                for r in range(3)
            ]
        )
        return s_grid, a_grid, regime_values

    def price_tarf(self, tarf: TARFAccumulator, maturity: float) -> float:
        _, _, regime_values = self._regime_curve(tarf, maturity)
        return float(np.dot(self.regime_weights, regime_values))

    def price_tarf_coupled(self, tarf: TARFAccumulator, maturity: float) -> float:
        """Generator-matrix-coupled TARF price (kept as an explicit name for the interface)."""
        return self.price_tarf(tarf=tarf, maturity=maturity)

    def tarf_greeks(self, tarf: TARFAccumulator, maturity: float, vega_bump: float = 1e-4) -> dict[str, float]:
        """Aggregate delta/gamma from the regime-weighted grid, plus per-regime and aggregate vega."""
        s_grid, a_grid, values = self.solve_tarf_coupled_pde(tarf=tarf, maturity=maturity)
        blended = np.einsum("r,rij->ij", self.regime_weights, values)

        idx = int(np.argmin(np.abs(s_grid - self.model.spot)))
        idx = min(max(idx, 1), len(s_grid) - 2)
        s0 = float(s_grid[idx])
        dx = float(np.log(s_grid[1]) - np.log(s_grid[0]))
        curve = np.array(
            [_interp_along_accumulation(a_grid, blended[j, :], np.array([tarf.accumulated_value]))[0] for j in (idx - 1, idx, idx + 1)]
        )
        price = float(curve[1])
        v_x = (curve[2] - curve[0]) / (2.0 * dx)
        v_xx = (curve[2] - 2.0 * curve[1] + curve[0]) / (dx * dx)

        if getattr(self.model, "time_varying", False):
            # A calibrated (Dupire) model bumps its whole local-vol surface by a parallel shift.
            bumped_price = ThreeRegimePricer(
                self.model.bumped_vol(vega_bump), self.regime_weights, self.num_spot, self.num_target, self.n_steps
            ).price_tarf(tarf, maturity)
            total_vega = (bumped_price - price) / vega_bump * 0.01
            return {
                "price": price,
                "delta": float(v_x / s0),
                "gamma": float((v_xx - v_x) / (s0 * s0)),
                "vega": float(total_vega),
                "regime_vega": [float(total_vega)],
            }

        regime_vega: list[float] = []
        for r, regime in enumerate(self.model.regimes):
            bumped_regimes = list(self.model.regimes)
            bumped_regimes[r] = SingleRegimeLocalVolModel(
                spot=regime.spot,
                rate=regime.rate,
                dividend_yield=regime.dividend_yield,
                local_vol=regime.local_vol + vega_bump,
                strike=regime.strike,
                skew=regime.skew,
                curvature=regime.curvature,
                smile_ref=regime.smile_ref,
                vol_floor=regime.vol_floor,
            )
            bumped_model = RegimeSwitchingLocalVolModel(
                spot=self.model.spot,
                rate=self.model.rate,
                dividend_yield=self.model.dividend_yield,
                regimes=bumped_regimes,
                q=self.model.q,
            )
            bumped_price = ThreeRegimePricer(
                bumped_model, self.regime_weights, self.num_spot, self.num_target, self.n_steps
            ).price_tarf(tarf, maturity)
            regime_vega.append((bumped_price - price) / vega_bump * 0.01)

        return {
            "price": price,
            "delta": float(v_x / s0),
            "gamma": float((v_xx - v_x) / (s0 * s0)),
            "vega": float(sum(regime_vega)),
            "regime_vega": [float(v) for v in regime_vega],
        }
