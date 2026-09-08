from dataclasses import dataclass, field

import numpy as np


@dataclass
class SingleRegimeLocalVolModel:
    """Single-regime local-vol model.

    ``local_vol`` is the ATM level. ``skew`` and ``curvature`` add a compact parametric smile in
    log-moneyness ``k = ln(S / smile_ref)``:

        sigma(S) = max(vol_floor, local_vol + skew * k + curvature * k**2)

    This is the "SABR-style, 3-4 parameter" per-regime smile the design brief asks for, used directly
    as the regime's local-volatility function. Mapping a market *implied* smile to a *local* vol via
    Dupire is a deliberate future extension: the calibration layer fits these parameters so the
    regime-weighted model reproduces the market vanilla smile, which is self-consistent without it.
    """

    spot: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    local_vol: float = 0.2
    strike: float = 100.0
    skew: float = 0.0
    curvature: float = 0.0
    smile_ref: float | None = None
    vol_floor: float = 1e-3
    sigma_ref: float | None = None

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.local_vol <= 0:
            raise ValueError("local_vol must be positive")
        if self.smile_ref is None:
            self.smile_ref = self.strike
        if self.sigma_ref is None:
            self.sigma_ref = self.local_vol

    def local_volatility(self, spot: np.ndarray | float, t: float = 0.0) -> np.ndarray:
        """Local volatility at the given spot level(s). ``t`` is accepted for interface symmetry."""
        s = np.asarray(spot, dtype=float)
        k = np.log(np.maximum(s, 1e-300) / self.smile_ref)
        vol = self.local_vol + self.skew * k + self.curvature * k * k
        return np.maximum(self.vol_floor, vol)


@dataclass
class RegimeSwitchingLocalVolModel:
    """Three-regime switching local-vol model: three ``SingleRegimeLocalVolModel`` layers coupled by a
    3x3 generator matrix ``q``.

    Each regime carries its own smile parameters. The default construction is a level-shifted smile
    (low / mid / high vol) which is the common regime interpretation and keeps calibration well-posed.
    """

    spot: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    regimes: list[SingleRegimeLocalVolModel] = field(default_factory=list)
    q: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if len(self.regimes) == 0:
            self.regimes = [
                SingleRegimeLocalVolModel(
                    spot=self.spot,
                    rate=self.rate,
                    dividend_yield=self.dividend_yield,
                    local_vol=level,
                    strike=100.0,
                )
                for level in (0.15, 0.20, 0.30)
            ]
        if self.q is None:
            self.q = build_default_regime_matrix()

        if len(self.regimes) != 3:
            raise ValueError("Three regimes are required for the default RSLV extension")
        self.q = np.asarray(self.q, dtype=float)
        if self.q.shape != (3, 3):
            raise ValueError("Generator matrix must have shape (3, 3)")


def build_default_regime_matrix() -> np.ndarray:
    """Return a simple generator matrix that is conservative and has a negative diagonal."""
    q = np.array(
        [
            [-0.5, 0.25, 0.25],
            [0.25, -0.5, 0.25],
            [0.25, 0.25, -0.5],
        ],
        dtype=float,
    )
    return q
