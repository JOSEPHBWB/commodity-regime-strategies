# Commodity Regime Strategies

A small research project on a question I kept coming back to while learning quantitative trading:

> Do simple commodity trading signals still work after market regimes and trading costs are taken seriously?

The project focuses on energy and metals because those are the markets I am most interested in. I use **AKQuant** as the event-driven backtesting framework and keep the strategy logic deliberately simple so that the experiments are easy to inspect.

## Markets

The default public-data universe is:

| Group | Symbol | Market |
|---|---|---|
| Oil | `CL=F` | WTI crude oil futures |
| Natural gas | `NG=F` | Henry Hub natural gas futures |
| Coal | `COAL` | Range Global Coal ETF (used as a coal-sector proxy) |
| Gold | `GC=F` | Gold futures |
| Silver | `SI=F` | Silver futures |
| Platinum | `PL=F` | Platinum futures |

`COAL` is an equity ETF rather than a coal futures contract, so I report it separately as a proxy rather than treating it as directly comparable to the futures series. The loader can also read a user-supplied coal futures CSV if better data are available.

## Questions

I am testing three basic ideas:

1. Does momentum behave differently in low- and high-volatility periods?
2. How much performance disappears once transaction costs are added?
3. Does simple volatility scaling improve drawdowns enough to justify the extra turnover?

This is not meant to be an alpha-production system. The goal is to practice clean backtesting and understand when apparently good results are fragile.

## Strategies

- **Momentum:** long when the lookback return is positive, short when it is negative.
- **Mean reversion:** trade against unusually large deviations from a rolling mean.
- **Volatility-scaled momentum:** same momentum direction, but target a smaller position when recent volatility is high.

The regime label is intentionally simple: a 20-day rolling volatility estimate is compared with expanding 30th/70th percentile thresholds. Using expanding thresholds avoids classifying an old observation with information from the future.

## Repository layout

```text
.
|-- data/
|   |-- download_data.py
|   `-- make_demo_data.py
|-- strategies/
|   |-- momentum.py
|   |-- mean_reversion.py
|   `-- vol_scaled_momentum.py
|-- analysis/
|   |-- regimes.py
|   |-- metrics.py
|   `-- vector_backtest.py
|-- experiments/
|   |-- baseline_test.py
|   |-- regime_test.py
|   |-- cost_sensitivity.py
|   `-- walk_forward_test.py
|-- tests/
|-- results/
|-- run_akquant_demo.py
`-- requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

AKQuant is published on PyPI, so a Rust toolchain is not required for the normal Python install.

## Quick start

First create an offline demo dataset:

```bash
python data/make_demo_data.py
```

Run the simple research experiments:

```bash
python experiments/baseline_test.py
python experiments/regime_test.py
python experiments/cost_sensitivity.py
python experiments/walk_forward_test.py
```

To download current public market data through Yahoo Finance:

```bash
python data/download_data.py --start 2015-01-01
```

Then run the AKQuant event-driven example:

```bash
python run_akquant_demo.py --symbol CL=F
```

## Backtesting choices

A few choices matter more to me here than adding a complicated model:

- signals are lagged by one day before returns are applied;
- regime thresholds are expanding rather than full-sample quantiles;
- transaction costs are charged when the position changes;
- parameter selection in the walk-forward experiment only uses the training window;
- futures and the coal ETF proxy are not pooled into one claim without noting the difference.

## What I would extend next

The current version only studies price-based signals. A more serious commodity project would add futures curve information (backwardation/contango), inventories, weather for natural gas, and possibly macro variables. I left those out of the first version because I wanted a small experiment where I could understand every assumption before adding more data.

## Framework attribution

This project uses the open-source [AKQuant](https://github.com/akfamily/akquant) backtesting framework for event-driven execution. AKQuant is distributed under the MIT License. Strategy definitions, regime analysis, cost tests, and walk-forward experiments in this repository are separate project code.

## Disclaimer

For research and educational use only. This is not investment advice.
