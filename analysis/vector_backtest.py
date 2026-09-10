from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(close: pd.Series, lookback: int = 60) -> pd.Series:
    return np.sign(close.pct_change(lookback)).fillna(0.0)


def mean_reversion_signal(close: pd.Series, window: int = 20, z_entry: float = 1.25) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std
    signal = pd.Series(0.0, index=close.index)
    signal[z < -z_entry] = 1.0
    signal[z > z_entry] = -1.0
    return signal


def vol_scaled_momentum_signal(
    close: pd.Series,
    lookback: int = 60,
    vol_window: int = 20,
    target_vol: float = 0.15,
    max_leverage: float = 1.5,
) -> pd.Series:
    direction = momentum_signal(close, lookback)
    vol = close.pct_change().rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / vol).clip(upper=max_leverage)
    return (direction * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_vector_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    out = df[["date", "close"]].copy()
    out["asset_return"] = out["close"].pct_change().fillna(0.0)

    # Signal from day t is applied to day t+1 return.
    out["position"] = pd.Series(signal, index=df.index).shift(1).fillna(0.0)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover"] * (cost_bps / 10_000.0)
    out["strategy_return"] = out["position"] * out["asset_return"] - out["cost"]
    return out
