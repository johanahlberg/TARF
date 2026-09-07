from dataclasses import dataclass, field

import numpy as np


@dataclass
class SingleRegimeLocalVolModel:
    """Minimal single-regime local-vol model object used by the initial solver baseline."""

    spot: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    local_vol: float = 0.2
    strike: float = 100.0
    sigma_ref: float | None = None

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.local_vol <= 0:
            raise ValueError("local_vol must be positive")
        if self.sigma_ref is None:
            self.sigma_ref = self.local_vol


@dataclass
class RegimeSwitchingLocalVolModel:
    """Small 3-regime switching local-vol model used as the next extension point.

    Each regime is represented by a local-vol surface parameter but the pricing layer currently uses the
    regime-weighted constant-vol approximation so the architecture stays tractable and testable.
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
                    local_vol=0.15,
                    strike=100.0,
                ),
                SingleRegimeLocalVolModel(
                    spot=self.spot,
                    rate=self.rate,
                    dividend_yield=self.dividend_yield,
                    local_vol=0.20,
                    strike=100.0,
                ),
                SingleRegimeLocalVolModel(
                    spot=self.spot,
                    rate=self.rate,
                    dividend_yield=self.dividend_yield,
                    local_vol=0.30,
                    strike=100.0,
                ),
            ]
        if self.q is None:
            self.q = build_default_regime_matrix()

        if len(self.regimes) != 3:
            raise ValueError("Three regimes are required for the default RSLV extension")
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
