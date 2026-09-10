from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import performance_metrics
from analysis.regimes import add_volatility_regime
from analysis.vector_backtest import momentum_signal, run_vector_backtest


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        symbol = str(df["symbol"].iloc[0])

        labeled = add_volatility_regime(df)
        signal = momentum_signal(labeled["close"], lookback=60)
        bt = run_vector_backtest(labeled, signal, cost_bps=5)
        bt["regime"] = labeled["regime"]

        for regime in ["low_vol", "normal", "high_vol"]:
            sample = bt.loc[bt["regime"] == regime, "strategy_return"]
            metrics = performance_metrics(sample)
            rows.append({"symbol": symbol, "regime": regime, "n_days": len(sample), **metrics})

    if not rows:
        raise FileNotFoundError("No data found. Run data/make_demo_data.py first.")

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    out = ROOT / "results" / "regime_summary.csv"
    result.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
