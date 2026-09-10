from __future__ import annotations

import numpy as np
from akquant import Strategy


class VolScaledMomentumStrategy(Strategy):
    def __init__(
        self,
        lookback: int = 60,
        vol_window: int = 20,
        target_vol: float = 0.15,
        max_target_percent: float = 0.9,
    ):
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_target_percent = max_target_percent
        self.warmup_period = max(lookback + 1, vol_window + 1)

    def on_bar(self, bar) -> None:
        n = max(self.lookback + 1, self.vol_window + 1)
        closes = self.get_history(count=n, symbol=bar.symbol, field="close")
        if len(closes) < n:
            return

        direction = np.sign(closes[-1] / closes[-1 - self.lookback] - 1.0)
        daily_returns = np.diff(closes[-1 - self.vol_window :]) / closes[-1 - self.vol_window : -1]
        annual_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
        if annual_vol <= 0 or not np.isfinite(annual_vol):
            return

        size = min(self.target_vol / annual_vol, self.max_target_percent)
        self.order_target_percent(bar.symbol, float(direction * size))


__all__ = ["VolScaledMomentumStrategy"]
