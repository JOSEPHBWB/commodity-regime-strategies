from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["ret_1", "ret_5", "momentum_20", "vol_20", "zscore_20"]


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a small set of causal features from daily prices."""
    out = df[["date", "close"]].copy()
    close = out["close"].astype(float)
    log_close = np.log(close)

    out["ret_1"] = log_close.diff()
    out["ret_5"] = log_close.diff(5)
    out["momentum_20"] = log_close.diff(20)
    out["vol_20"] = out["ret_1"].rolling(20).std()

    mean_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out["zscore_20"] = (close - mean_20) / std_20

    # Target at row t is the close-to-close return from t to t+1.
    out["target_return"] = log_close.diff().shift(-1)
    return out


def fit_standardizer(features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = features.mean()
    std = features.std(ddof=0).replace(0.0, 1.0)
    return mean, std


def apply_standardizer(
    features: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    return (features - mean) / std
