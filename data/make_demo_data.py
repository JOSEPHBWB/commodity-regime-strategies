from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = {
    "CL=F": (75.0, 0.021),
    "NG=F": (3.4, 0.035),
    "COAL": (25.0, 0.018),
    "GC=F": (1900.0, 0.012),
    "SI=F": (24.0, 0.019),
    "PL=F": (950.0, 0.017),
}


def make_series(symbol: str, start_price: float, base_vol: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n, tz="UTC")

    # Piecewise regimes give the tests something non-trivial to find.
    vol_mult = np.ones(n)
    vol_mult[n // 3 : n // 2] = 1.8
    vol_mult[3 * n // 4 :] = 1.35

    drift = np.zeros(n)
    drift[: n // 4] = 0.0004
    drift[n // 2 : 3 * n // 4] = -0.00025

    shocks = rng.normal(drift, base_vol * vol_mult)
    close = start_price * np.exp(np.cumsum(shocks))
    overnight = rng.normal(0, base_vol * 0.15, n)
    open_ = close * np.exp(overnight)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, base_vol * 0.25, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, base_vol * 0.25, n)))
    volume = rng.integers(5_000, 100_000, n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
        }
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (symbol, (price, vol)) in enumerate(SYMBOLS.items()):
        df = make_series(symbol, price, vol, n=2200, seed=42 + i)
        path = out_dir / f"{symbol.replace('=', '_')}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
