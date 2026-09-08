from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class FixingOutcome(NamedTuple):
    """Result of one fixing, mirroring one MAIN LOOP iteration of the udmcTRFPayoff ESL script."""

    cashflow: float
    accumulated: float
    terminated: bool


@dataclass
class TARFAccumulator:
    """Target Redemption Forward accumulator, ported from the ``udmcTRFPayoff`` ESL payoff script.

    On an ITM fixing the realized intrinsic value (or its inverted-quote equivalent) accrues toward
    ``target_level``; once the target is reached the deal terminates, with the final cashflow shaped by
    ``target_adjustment`` (0 full gain, 1 strike-adjusted, 2 notional-adjusted / part gain,
    3 no gain -- matching the three knockout types of Luo & Shevchenko plus the ESL variants).
    On an OTM fixing the leveraged ``notional2`` side pays out, unless a KI
    ``barrier`` is enabled and this fixing's spot lands beyond it. The barrier check has no memory
    across fixings: each OTM fixing re-evaluates it fresh from the current spot alone.
    """

    target_level: float
    strike: float
    fixing_dates: Sequence[float]
    is_call_option: bool = True
    notional1: Sequence[float] | float = 1.0
    notional2: Sequence[float] | float = 1.0
    barrier: float = 0.0
    target_adjustment: int = 0
    inverted_target: bool = False
    accumulated_value: float = 0.0
    zero_comparison: float = 1e-8

    def __post_init__(self) -> None:
        if self.target_level <= 0:
            raise ValueError("target_level must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.target_adjustment not in (0, 1, 2, 3):
            raise ValueError(
                "target_adjustment must be 0 (none / full gain), 1 (strike-adjusted), "
                "2 (notional-adjusted / part gain) or 3 (no gain)"
            )
        if len(self.fixing_dates) == 0:
            raise ValueError("fixing_dates must not be empty")
        if list(self.fixing_dates) != sorted(self.fixing_dates):
            raise ValueError("fixing_dates must be sorted ascending")
        if self.barrier < 0:
            raise ValueError("barrier must be non-negative (use 0 to disable)")
        if self.accumulated_value < 0:
            raise ValueError("accumulated_value must be non-negative")

        self.notional1 = self._broadcast_notional(self.notional1, "notional1")
        self.notional2 = self._broadcast_notional(self.notional2, "notional2")

    def _broadcast_notional(self, notional: Sequence[float] | float, name: str) -> np.ndarray:
        n = len(self.fixing_dates)
        array = np.atleast_1d(np.asarray(notional, dtype=float))
        if array.shape == (1,):
            array = np.full(n, float(array[0]))
        if array.shape != (n,):
            raise ValueError(f"{name} must be a scalar or have one entry per fixing date")
        return array

    @property
    def call_or_put(self) -> float:
        return 1.0 if self.is_call_option else -1.0

    def intrinsic(self, spot: float) -> float:
        return (spot - self.strike) * self.call_or_put

    def accumulation_increment(self, spot: float) -> float:
        """Amount added to ``accumulated`` on an ITM fixing, in either quoting convention."""
        if self.inverted_target:
            return (1.0 / spot - 1.0 / self.strike) * self.call_or_put * -1.0
        return self.intrinsic(spot)

    def barrier_is_hit(self, spot: float) -> bool:
        """True unless the KI barrier is enabled and this fixing's spot lands beyond it."""
        if self.barrier <= self.zero_comparison:
            return True
        barrier_intrinsic = (self.barrier - spot) * self.call_or_put
        return not (barrier_intrinsic < 0.0)

    def settle(self, spot: float, previous_accumulated: float, fixing_index: int) -> FixingOutcome:
        """Replicate one MAIN LOOP iteration of the ESL script for a single fixing."""
        intrinsic = self.intrinsic(spot)
        notional1 = float(self.notional1[fixing_index])
        notional2 = float(self.notional2[fixing_index])

        if intrinsic > 0.0:
            increment = self.accumulation_increment(spot)
            accumulated = previous_accumulated + increment
            terminated = (self.target_level - accumulated) < self.zero_comparison

            settlement_notional = notional1
            settlement_intrinsic = intrinsic
            if terminated:
                remaining = self.target_level - previous_accumulated
                if self.target_adjustment == 1:
                    if self.inverted_target:
                        adjusted_strike = 1.0 / ((1.0 / spot) - remaining * self.call_or_put * -1.0)
                    else:
                        adjusted_strike = spot - remaining * self.call_or_put
                    settlement_intrinsic = (spot - adjusted_strike) * self.call_or_put
                elif self.target_adjustment == 2:
                    settlement_notional = notional1 * (remaining / increment)
                elif self.target_adjustment == 3:
                    settlement_notional = 0.0

            return FixingOutcome(
                cashflow=settlement_notional * settlement_intrinsic,
                accumulated=accumulated,
                terminated=terminated,
            )

        hit = self.barrier_is_hit(spot)
        cashflow = notional2 * intrinsic * (1.0 if hit else 0.0)
        return FixingOutcome(cashflow=cashflow, accumulated=previous_accumulated, terminated=False)
