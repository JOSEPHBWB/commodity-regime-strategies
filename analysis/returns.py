from __future__ import annotations

import numpy as np
import pandas as pd


def signed_price_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Price change scaled by the absolute lagged price.

    For ordinary positive prices this is the usual arithmetic return. Using the
    absolute lagged price keeps the direction meaningful when a futures
    settlement is zero or negative, as happened in WTI in April 2020.

    This is a fixed-notional research return, not a fully specified futures
    account return with contract multipliers, margin, and roll mechanics.
    """
    price = pd.Series(close, dtype=float)
    lagged = price.shift(periods)
    denom = lagged.abs().replace(0.0, np.nan)
    out = (price - lagged) / denom
    return out.replace([np.inf, -np.inf], np.nan)
