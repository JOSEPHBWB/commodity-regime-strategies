# Project notes

These notes are intentionally short. I want the repository to read like a research notebook, not a software product.

## Why these markets?

I am more interested in physical commodity markets than single-stock prediction. Oil and gas are directly relevant to energy trading; gold, silver and platinum give a comparison group with different demand and volatility drivers. Coal is intentionally kept out of the default universe until I have a proper thermal-coal benchmark rather than an equity proxy.

## Things I would check before taking any result seriously

- contract rolling rules and whether the public futures series introduces artificial jumps;
- realistic commodity-specific commissions and slippage;
- whether a signal survives subperiods rather than only the full sample;
- whether parameter choices remain stable in walk-forward testing;
- futures curve / carry information, which is missing from the current price-only version;
- whether the coal series is actually a traded coal benchmark rather than a sector ETF.

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

A useful result includes cases where the LSTM forecasts slightly better but does not improve Sharpe after costs. That is more informative than tuning until every market looks profitable.

## Phase 3: methodology cleanup

The first real-data run exposed two useful problems.

First, WTI's April 2020 negative settlement makes log returns invalid. I replaced the log-return feature pipeline with a fixed-notional signed price return that remains defined when prices cross zero. The raw negative-price observation stays in the dataset.

Second, compounding futures P&L as if it were an ETF wealth series can create misleading annualized returns, especially in volatile natural gas windows. Performance is now treated additively: period P&L is the sum of daily normalized P&L and annualized P&L is the daily mean times 252. Sharpe is still based on annualized mean over annualized volatility.

These choices are still approximations. Public continuous futures do not reproduce a fully specified tradable roll strategy, so I treat the results as comparative research evidence rather than realized strategy returns.

## Coal data

The earlier `COAL` symbol is a listed coal-sector ETF, not thermal coal futures. It is no longer downloaded by default. A better energy-market extension is a Newcastle thermal-coal benchmark, stored locally if the data license does not allow redistribution.
