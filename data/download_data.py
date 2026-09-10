from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_SYMBOLS = ["CL=F", "NG=F", "COAL", "GC=F", "SI=F", "PL=F"]


def download_one(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["symbol"] = symbol
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public commodity market data")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        try:
            df = download_one(symbol, args.start, args.end)
            safe_name = symbol.replace("=", "_")
            path = out_dir / f"{safe_name}.csv"
            df.to_csv(path, index=False)
            print(f"{symbol:6s} -> {path.name:12s}  {len(df):5d} rows")
        except Exception as exc:
            print(f"{symbol:6s} -> failed: {exc}")


if __name__ == "__main__":
    main()
