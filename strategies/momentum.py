from __future__ import annotations

import numpy as np
from akquant import Strategy


class MomentumStrategy(Strategy):
    def __init__(self, lookback: int = 60, target_percent: float = 0.8):
        self.lookback = lookback
        self.target_percent = target_percent
        self.warmup_period = lookback + 1

    def on_bar(self, bar) -> None:
        closes = self.get_history(count=self.lookback + 1, symbol=bar.symbol, field="close")
        if len(closes) < self.lookback + 1:
            return

        momentum = closes[-1] / closes[0] - 1.0
        pos = self.get_position(bar.symbol)

        if momentum > 0 and pos <= 0:
            self.order_target_percent(bar.symbol, self.target_percent)
        elif momentum < 0 and pos >= 0:
            self.order_target_percent(bar.symbol, -self.target_percent)


__all__ = ["MomentumStrategy"]
