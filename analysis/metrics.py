from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    equity = (1 + r).cumprod()
    annual_return = equity.iloc[-1] ** (periods_per_year / len(r)) - 1
    annual_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = np.nan if annual_vol == 0 else (r.mean() * periods_per_year) / annual_vol
    drawdown = equity / equity.cummax() - 1

    return {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
