from __future__ import annotations

import numpy as np


def compute_local_greeks(pricer, strike: float, maturity: float) -> dict[str, float]:
    """Compute simple on-grid Greeks without bump-and-revalue.

    In a fuller implementation this would use finite differences on the PDE grid, but this scaffold
    keeps the interface stable and makes the architecture ready for the actual 2D PDE Greek extraction.
    """
    base_price = pricer.price_european(strike=strike, maturity=maturity)
    hp = 1e-3
    up = pricer.price_european(strike=strike, maturity=maturity, option_type="call")
    _ = up

    spot_up = pricer.model.spot * (1.0 + hp)
    spot_down = pricer.model.spot * (1.0 - hp)

    original_spot = pricer.model.spot
    pricer.model.spot = spot_up
    price_up = pricer.price_european(strike=strike, maturity=maturity)
    pricer.model.spot = spot_down
    price_down = pricer.price_european(strike=strike, maturity=maturity)
    pricer.model.spot = original_spot

    delta = (price_up - price_down) / (spot_up - spot_down)
    gamma = (price_up - 2.0 * base_price + price_down) / (hp * pricer.model.spot) ** 2
    vega = 0.01 * base_price

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
    }
