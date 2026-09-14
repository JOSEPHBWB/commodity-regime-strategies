from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Metrics for a fixed-notional daily P&L return series.

    The series is treated additively rather than compounded. This avoids
    meaningless wealth compounding when a futures price can be zero/negative.
    """
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "period_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    period_return = r.sum()
    annual_return = r.mean() * periods_per_year
    annual_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = np.nan if annual_vol == 0 else annual_return / annual_vol

    cumulative_pnl = r.cumsum()
    drawdown = cumulative_pnl - cumulative_pnl.cummax()

    return {
        "period_return": float(period_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
