from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.features import FEATURE_COLUMNS, apply_standardizer, fit_standardizer, make_price_features
from analysis.metrics import performance_metrics
from analysis.vector_backtest import momentum_signal, run_vector_backtest
from models.lstm_forecaster import predict, train_model

TRAIN_DAYS = 756
TEST_DAYS = 126
SEQUENCE_LENGTH = 20


def make_sequences(
    feature_values: np.ndarray,
    targets: np.ndarray,
    row_indices: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, kept_rows = [], [], []

    for row in row_indices:
        start = row - sequence_length + 1
        if start < 0:
            continue

        window = feature_values[start : row + 1]
        target = targets[row]
        if np.isnan(window).any() or np.isnan(target):
            continue

        x.append(window)
        y.append(target)
        kept_rows.append(row)

    if not x:
        return (
            np.empty((0, sequence_length, feature_values.shape[1])),
            np.empty(0),
            np.empty(0, dtype=int),
        )

    return np.asarray(x), np.asarray(y), np.asarray(kept_rows, dtype=int)


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    directional_accuracy = float(np.mean(np.sign(actual) == np.sign(predicted)))
    return {"forecast_rmse": rmse, "forecast_mae": mae, "directional_accuracy": directional_accuracy}


def run_symbol(
    df: pd.DataFrame,
    symbol: str,
    epochs: int,
    hidden_size: int,
    cost_bps: float,
) -> list[dict]:
    feature_df = make_price_features(df)
    raw_features = feature_df[FEATURE_COLUMNS]
    targets = feature_df["target_return"].to_numpy(dtype=float)

    rows = []
    fold = 1
    test_start = TRAIN_DAYS

    while test_start + TEST_DAYS <= len(df):
        train_start = test_start - TRAIN_DAYS
        test_end = test_start + TEST_DAYS

        # Standardization is fitted only on the current training window.
        train_feature_rows = raw_features.iloc[train_start:test_start]
        mean, std = fit_standardizer(train_feature_rows)
        scaled = apply_standardizer(raw_features, mean, std).to_numpy(dtype=float)

        train_rows = np.arange(train_start, test_start - 1)
        test_rows = np.arange(test_start - 1, test_end - 1)

        x_train, y_train, _ = make_sequences(scaled, targets, train_rows, SEQUENCE_LENGTH)
        x_test, y_test, test_prediction_rows = make_sequences(scaled, targets, test_rows, SEQUENCE_LENGTH)

        if len(x_train) == 0 or len(x_test) == 0:
            test_start += TEST_DAYS
            fold += 1
            continue

        # Daily returns are small, so standardize the target using training data as well.
        target_mean = float(y_train.mean())
        target_std = float(y_train.std(ddof=0))
        if target_std == 0:
            target_std = 1.0
        y_train_scaled = (y_train - target_mean) / target_std

        model = train_model(
            x_train,
            y_train_scaled,
            hidden_size=hidden_size,
            epochs=epochs,
            seed=7 + fold,
        )
        y_pred_scaled = predict(model, x_test)
        y_pred = y_pred_scaled * target_std + target_mean
        f_metrics = forecast_metrics(y_test, y_pred)

        # Prediction made at row t is used as the signal for the t -> t+1 return.
        lstm_signal = pd.Series(0.0, index=df.index)
        lstm_signal.iloc[test_prediction_rows] = np.sign(y_pred)
        lstm_bt = run_vector_backtest(df, lstm_signal, cost_bps=cost_bps)

        momentum = momentum_signal(df["close"], 60)
        momentum_bt = run_vector_backtest(df, momentum, cost_bps=cost_bps)

        first_trade_row = test_start
        last_trade_row = test_end - 1
        test_mask = (df.index >= first_trade_row) & (df.index <= last_trade_row)

        lstm_perf = performance_metrics(lstm_bt.loc[test_mask, "strategy_return"])
        momentum_perf = performance_metrics(momentum_bt.loc[test_mask, "strategy_return"])

        rows.append(
            {
                "symbol": symbol,
                "fold": fold,
                "test_start": df.iloc[first_trade_row]["date"],
                "test_end": df.iloc[last_trade_row]["date"],
                **f_metrics,
                "lstm_sharpe": lstm_perf["sharpe"],
                "lstm_period_return": lstm_perf["period_return"],
                "lstm_annualized_pnl": lstm_perf["annual_return"],
                "lstm_max_drawdown": lstm_perf["max_drawdown"],
                "lstm_turnover": float(lstm_bt.loc[test_mask, "turnover"].mean()),
                "momentum_sharpe": momentum_perf["sharpe"],
                "momentum_period_return": momentum_perf["period_return"],
                "momentum_annualized_pnl": momentum_perf["annual_return"],
                "momentum_max_drawdown": momentum_perf["max_drawdown"],
            }
        )

        test_start += TEST_DAYS
        fold += 1

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward LSTM return forecasting experiment")
    parser.add_argument("--symbol", default=None, help="Run one symbol, e.g. CL=F")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted((ROOT / "data" / "raw").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("No data found. Run data/make_demo_data.py or data/download_data.py first.")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        symbol = str(df["symbol"].iloc[0])
        if args.symbol and symbol != args.symbol:
            continue

        print(f"running {symbol}...")
        rows.extend(run_symbol(df, symbol, args.epochs, args.hidden_size, args.cost_bps))

    if not rows:
        raise ValueError("No matching symbol or no usable walk-forward windows.")

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    out = ROOT / "results" / "lstm_walk_forward.csv"
    result.to_csv(out, index=False)

    summary = (
        result.groupby("symbol")
        .agg(
            folds=("fold", "count"),
            direction_accuracy=("directional_accuracy", "mean"),
            forecast_rmse=("forecast_rmse", "mean"),
            lstm_sharpe_mean=("lstm_sharpe", "mean"),
            lstm_sharpe_median=("lstm_sharpe", "median"),
            momentum_sharpe_mean=("momentum_sharpe", "mean"),
            momentum_sharpe_median=("momentum_sharpe", "median"),
            lstm_period_return_mean=("lstm_period_return", "mean"),
            momentum_period_return_mean=("momentum_period_return", "mean"),
            lstm_max_drawdown_mean=("lstm_max_drawdown", "mean"),
            momentum_max_drawdown_mean=("momentum_max_drawdown", "mean"),
        )
        .reset_index()
    )
    summary_path = ROOT / "results" / "lstm_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nsaved: {out}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
