from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_regime(
    df: pd.DataFrame,
    vol_window: int = 20,
    min_history: int = 120,
    low_q: float = 0.30,
    high_q: float = 0.70,
) -> pd.DataFrame:
    out = df.copy()
    ret = out["close"].pct_change()
    out["rolling_vol"] = ret.rolling(vol_window).std() * np.sqrt(252)

    low_cut = out["rolling_vol"].expanding(min_periods=min_history).quantile(low_q)
    high_cut = out["rolling_vol"].expanding(min_periods=min_history).quantile(high_q)

    out["regime"] = "normal"
    out.loc[out["rolling_vol"] < low_cut, "regime"] = "low_vol"
    out.loc[out["rolling_vol"] > high_cut, "regime"] = "high_vol"
    out.loc[low_cut.isna() | high_cut.isna(), "regime"] = pd.NA
    return out
