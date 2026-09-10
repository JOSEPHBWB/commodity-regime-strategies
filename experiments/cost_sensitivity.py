from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import performance_metrics
from analysis.vector_backtest import momentum_signal, run_vector_backtest

COSTS_BPS = [0, 2, 5, 10, 20]


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        symbol = str(df["symbol"].iloc[0])
        signal = momentum_signal(df["close"], lookback=60)

        for cost in COSTS_BPS:
            bt = run_vector_backtest(df, signal, cost_bps=cost)
            metrics = performance_metrics(bt["strategy_return"])
            rows.append({"symbol": symbol, "cost_bps": cost, **metrics})

    if not rows:
        raise FileNotFoundError("No data found. Run data/make_demo_data.py first.")

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    out = ROOT / "results" / "cost_sensitivity.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
