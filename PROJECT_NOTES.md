# Project notes

These notes are intentionally short. I want the repository to read like a research notebook, not a software product.

## Why these markets?

I am more interested in physical commodity markets than single-stock prediction. Oil and gas are directly relevant to energy trading; coal is included through a listed sector proxy because continuous public coal-futures data are harder to obtain cleanly; gold, silver and platinum give a comparison group with different demand and volatility drivers.

## Why not machine learning in version 1?

Before adding a model, I want to know whether the basic backtest is trustworthy. It is easy to get a better-looking result by adding parameters, but that also makes leakage and overfitting harder to notice. The first version therefore uses only simple price signals.

## Things I would check before taking any result seriously

- contract rolling rules and whether the public futures series introduces artificial jumps;
- realistic commodity-specific commissions and slippage;
- whether a signal survives subperiods rather than only the full sample;
- whether parameter choices remain stable in walk-forward testing;
- futures curve / carry information, which is missing from the current price-only version;
- the fact that the coal ETF is not a futures contract.

## Possible version 2

Add curve slope and inventory variables for crude oil and natural gas, then ask whether the same momentum signal behaves differently when the market is in backwardation versus contango. That extension would be more commodity-specific than simply adding a neural network.

## Phase 2: LSTM baseline

I added the LSTM after the first signal tests rather than starting with it. The point is to compare forecasting accuracy with trading usefulness, not to claim that a neural network is automatically better.

Current choices:
- one LSTM layer, 24 hidden units by default;
- 20-day input sequence;
- only causal price features;
- training-window-only standardization;
- chronological 3-year train / 6-month test windows;
- 5 bps default transaction cost;
- 60-day momentum kept as the simple trading benchmark.

A useful result would include cases where the LSTM forecasts slightly better but does not improve Sharpe after costs. That is more informative than tuning until every market looks profitable.
