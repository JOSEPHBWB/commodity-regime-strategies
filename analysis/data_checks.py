from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.returns import signed_price_return


def audit_market_data(df: pd.DataFrame, symbol: str) -> dict[str, object]:
    close = pd.to_numeric(df["close"], errors="coerce")
    ret = signed_price_return(close)
    abs_ret = ret.abs()

    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "start": df["date"].min(),
        "end": df["date"].max(),
        "missing_close": int(close.isna().sum()),
        "nonpositive_close": int((close <= 0).sum()),
        "min_close": float(close.min()),
        "max_abs_daily_return": float(abs_ret.max()),
        "days_over_50pct": int((abs_ret > 0.50).sum()),
    }


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        symbol = str(df["symbol"].iloc[0])
        rows.append(audit_market_data(df, symbol))

    if not rows:
        raise FileNotFoundError("No data found in data/raw.")

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
