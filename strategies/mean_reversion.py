from __future__ import annotations

import numpy as np
from akquant import Strategy


class MeanReversionStrategy(Strategy):
    def __init__(self, window: int = 20, z_entry: float = 1.25, target_percent: float = 0.6):
        self.window = window
        self.z_entry = z_entry
        self.target_percent = target_percent
        self.warmup_period = window

    def on_bar(self, bar) -> None:
        closes = self.get_history(count=self.window, symbol=bar.symbol, field="close")
        if len(closes) < self.window:
            return

        mean = float(np.mean(closes))
        std = float(np.std(closes, ddof=1))
        if std == 0:
            return

        z = (bar.close - mean) / std
        pos = self.get_position(bar.symbol)

        if z < -self.z_entry and pos <= 0:
            self.order_target_percent(bar.symbol, self.target_percent)
        elif z > self.z_entry and pos >= 0:
            self.order_target_percent(bar.symbol, -self.target_percent)
        elif abs(z) < 0.25 and pos != 0:
            self.close_position(bar.symbol)


__all__ = ["MeanReversionStrategy"]
