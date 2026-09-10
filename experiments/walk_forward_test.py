from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import performance_metrics
from analysis.vector_backtest import momentum_signal, run_vector_backtest

LOOKBACKS = [20, 60, 120]
TRAIN_DAYS = 504
TEST_DAYS = 126


def score_window(df: pd.DataFrame, lookback: int) -> float:
    signal = momentum_signal(df["close"], lookback)
    bt = run_vector_backtest(df, signal, cost_bps=5)
    return performance_metrics(bt["strategy_return"])["sharpe"]


def walk_forward(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    start = TRAIN_DAYS

    while start + TEST_DAYS <= len(df):
        train = df.iloc[start - TRAIN_DAYS : start].reset_index(drop=True)
        # Keep lookback history before the test section so the first signal is not artificially blank.
        history_start = max(0, start - max(LOOKBACKS) - 1)
        combined = df.iloc[history_start : start + TEST_DAYS].reset_index(drop=True)
        test_start_date = df.iloc[start]["date"]
        test_end_date = df.iloc[start + TEST_DAYS - 1]["date"]

        train_scores = {lb: score_window(train, lb) for lb in LOOKBACKS}
        valid_scores = {k: v for k, v in train_scores.items() if pd.notna(v)}
        if not valid_scores:
            start += TEST_DAYS
            continue
        best_lb = max(valid_scores, key=valid_scores.get)

        signal = momentum_signal(combined["close"], best_lb)
        bt = run_vector_backtest(combined, signal, cost_bps=5)
        mask = (bt["date"] >= test_start_date) & (bt["date"] <= test_end_date)
        metrics = performance_metrics(bt.loc[mask, "strategy_return"])

        rows.append(
            {
                "symbol": symbol,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "chosen_lookback": best_lb,
                "train_sharpe": valid_scores[best_lb],
                "test_sharpe": metrics["sharpe"],
                "test_return": metrics["annual_return"],
                "test_max_drawdown": metrics["max_drawdown"],
            }
        )
        start += TEST_DAYS

    return rows


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        symbol = str(df["symbol"].iloc[0])
        rows.extend(walk_forward(df, symbol))

    if not rows:
        raise FileNotFoundError("No usable data found. Run data/make_demo_data.py first.")

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    out = ROOT / "results" / "walk_forward_summary.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
