from __future__ import annotations

import argparse
from pathlib import Path

import akquant as aq
import pandas as pd

from strategies.momentum import MomentumStrategy


def load_symbol(symbol: str) -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "raw" / f"{symbol.replace('=', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run data/make_demo_data.py or data/download_data.py first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "open", "high", "low", "close", "volume", "symbol"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="CL=F")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--commission", type=float, default=0.0005)
    args = parser.parse_args()

    df = load_symbol(args.symbol)

    class ThisMomentum(MomentumStrategy):
        def __init__(self):
            super().__init__(lookback=args.lookback)

    result = aq.run_backtest(
        data=df,
        strategy=ThisMomentum,
        symbols=args.symbol,
        initial_cash=100_000.0,
        commission_rate=args.commission,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    print(result)

    out = Path(__file__).resolve().parent / "results" / f"akquant_{args.symbol.replace('=', '_')}.html"
    result.viz.report(filename=str(out), show=False, compact_currency=True)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
