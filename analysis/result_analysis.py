"""Helpers for summarising walk-forward commodity results."""

import numpy as np
import pandas as pd

ENERGY = {"CL=F", "NG=F"}
METALS = {"GC=F", "SI=F", "PL=F"}


def asset_group(symbol):
    if symbol in ENERGY:
        return "Energy"
    if symbol in METALS:
        return "Precious metals"
    return "Other"


def load_lstm_results(path="results/lstm_walk_forward.csv"):
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError("Expected a symbol column in LSTM results.")
    df["group"] = df["symbol"].map(asset_group)
    return df


def _pick(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def aggregate_by_asset(df):
    lstm_sh = _pick(df, ["lstm_sharpe"])
    mom_sh = _pick(df, ["momentum_sharpe", "mom_sharpe"])
    da = _pick(df, ["directional_accuracy", "direction_accuracy"])
    rmse = _pick(df, ["forecast_rmse", "rmse"])
    mae = _pick(df, ["forecast_mae", "mae"])

    rows = []
    for symbol, g in df.groupby("symbol"):
        row = {"symbol": symbol, "group": asset_group(symbol), "windows": len(g)}
        if da:
            row["directional_accuracy"] = g[da].mean()
        if rmse:
            row["forecast_rmse"] = g[rmse].mean()
        if mae:
            row["forecast_mae"] = g[mae].mean()
        if lstm_sh:
            row["lstm_sharpe_mean"] = g[lstm_sh].mean()
            row["lstm_positive_sharpe_share"] = (g[lstm_sh] > 0).mean()
        if mom_sh:
            row["momentum_sharpe_mean"] = g[mom_sh].mean()
            row["momentum_positive_sharpe_share"] = (g[mom_sh] > 0).mean()
        if lstm_sh and mom_sh:
            row["lstm_minus_momentum_sharpe"] = (g[lstm_sh] - g[mom_sh]).mean()
            row["lstm_win_share"] = (g[lstm_sh] > g[mom_sh]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_group(df):
    asset = aggregate_by_asset(df)

    metric_cols = [
        c for c in asset.columns
        if c not in {"symbol", "group", "windows"}
    ]

    summary = asset.groupby("group", as_index=False)[metric_cols].mean()

    n_assets = (
        asset.groupby("group")
        .size()
        .rename("n_assets")
        .reset_index()
    )

    grouped = df.copy()
    grouped["group"] = grouped["symbol"].map(asset_group)
    n_windows = (
        grouped.groupby("group")
        .size()
        .rename("n_windows")
        .reset_index()
    )

    summary = summary.merge(n_assets, on="group")
    summary = summary.merge(n_windows, on="group")
    return summary


def wti_robustness(df):
    wti = df[df["symbol"] == "CL=F"].copy()
    if wti.empty:
        return pd.DataFrame()

    date_col = _pick(wti, ["test_start", "start", "window_start"])
    if date_col is None:
        return pd.DataFrame([{"sample": "full", "windows": len(wti)}])

    dates = pd.to_datetime(wti[date_col], errors="coerce", utc=True)
    stress = dates.dt.year.eq(2020)

    lstm_sh = _pick(wti, ["lstm_sharpe"])
    mom_sh = _pick(wti, ["momentum_sharpe", "mom_sharpe"])

    rows = []
    for label, g in [("full", wti), ("excluding_2020", wti.loc[~stress])]:
        row = {"sample": label, "windows": len(g)}
        if lstm_sh:
            row["lstm_sharpe_mean"] = g[lstm_sh].mean()
        if mom_sh:
            row["momentum_sharpe_mean"] = g[mom_sh].mean()
        if lstm_sh and mom_sh:
            row["lstm_minus_momentum_sharpe"] = (
                g[lstm_sh] - g[mom_sh]
            ).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_value_relationship(df):
    da = _pick(df, ["directional_accuracy", "direction_accuracy"])
    lstm_sh = _pick(df, ["lstm_sharpe"])
    if not da or not lstm_sh:
        return pd.DataFrame()

    x = pd.to_numeric(df[da], errors="coerce")
    y = pd.to_numeric(df[lstm_sh], errors="coerce")
    valid = x.notna() & y.notna()

    corr = np.nan if valid.sum() < 3 else x[valid].corr(y[valid])

    return pd.DataFrame([{
        "windows": int(valid.sum()),
        "corr_directional_accuracy_lstm_sharpe": corr,
    }])
