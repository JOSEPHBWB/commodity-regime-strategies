from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import performance_metrics
from analysis.vector_backtest import (
    mean_reversion_signal,
    momentum_signal,
    run_vector_backtest,
    vol_scaled_momentum_signal,
)


def load_all() -> list[tuple[str, pd.DataFrame]]:
    raw = ROOT / "data" / "raw"
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No data found. Run data/make_demo_data.py first.")

    out = []
    for path in files:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        symbol = str(df["symbol"].iloc[0])
        out.append((symbol, df))
    return out


def main() -> None:
    rows = []
    for symbol, df in load_all():
        tests = {
            "momentum_60": momentum_signal(df["close"], 60),
            "mean_reversion_20": mean_reversion_signal(df["close"], 20, 1.25),
            "vol_scaled_momentum": vol_scaled_momentum_signal(df["close"], 60, 20, 0.15),
        }

        for name, signal in tests.items():
            bt = run_vector_backtest(df, signal, cost_bps=5)
            metrics = performance_metrics(bt["strategy_return"])
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": name,
                    **metrics,
                    "avg_turnover": float(bt["turnover"].mean()),
                }
            )

    result = pd.DataFrame(rows).sort_values(["symbol", "strategy"])
    print(result.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    out = ROOT / "results" / "baseline_summary.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
