from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class TARFAccumulator:
    """Baseline TARF accumulation logic. The implementation intentionally keeps the model simple."""

    target_level: float
    coupon: float
    fixing_dates: Sequence[float] = field(default_factory=lambda: (0.5, 1.0))
    trigger_mode: str = "accumulation"

    def __post_init__(self) -> None:
        if self.target_level <= 0:
            raise ValueError("target_level must be positive")
        if self.coupon < 0:
            raise ValueError("coupon must be non-negative")
        if self.trigger_mode not in {"accumulation", "knockout"}:
            raise ValueError("Unsupported trigger_mode")
        if len(self.fixing_dates) == 0:
            raise ValueError("fixing_dates must not be empty")

    def accumulated_value(self, current_level: float) -> float:
        return float(current_level + self.coupon)

    def payoff(self, spot: float, accumulated_level: float) -> float:
        """A lightweight payoff approximation for the initial PDE baseline.

        The product is not yet a fully coupled 2D TARF. This keeps the code easy to extend while
        matching the design intent: at each fixing, the accumulation variable moves upward and the
        target triggers when the level is reached or exceeded.
        """
        if accumulated_level >= self.target_level:
            return max(spot - self.target_level, 0.0)
        return 0.0

    def projected_level(self, current_level: float) -> float:
        return current_level + self.coupon
