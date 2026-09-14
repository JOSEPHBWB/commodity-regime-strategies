import numpy as np
import pandas as pd

from analysis.metrics import performance_metrics
from analysis.regimes import add_volatility_regime
from analysis.vector_backtest import momentum_signal, run_vector_backtest


def sample_df(n=400):
    dates = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    close = 100 * np.exp(np.linspace(0, 0.4, n))
    return pd.DataFrame({"date": dates, "close": close})


def test_momentum_eventually_long():
    df = sample_df()
    sig = momentum_signal(df["close"], 20)
    assert sig.iloc[-1] == 1


def test_position_is_lagged_one_day():
    df = sample_df(80)
    sig = pd.Series(1.0, index=df.index)
    bt = run_vector_backtest(df, sig, cost_bps=0)
    assert bt.loc[0, "position"] == 0
    assert bt.loc[1, "position"] == 1


def test_regime_uses_missing_warmup():
    df = sample_df()
    out = add_volatility_regime(df, min_history=60)
    assert out["regime"].iloc[:59].isna().all()


def test_metrics_drawdown_non_positive():
    m = performance_metrics(pd.Series([0.01, -0.02, 0.01, 0.0]))
    assert m["max_drawdown"] <= 0


def test_price_features_only_use_past_prices():
    from analysis.features import make_price_features

    df = sample_df(100)
    first = make_price_features(df)

    changed = df.copy()
    changed.loc[80:, "close"] = changed.loc[80:, "close"] * 2
    second = make_price_features(changed)

    # Altering future prices must not change features at an earlier date.
    cols = ["ret_1", "ret_5", "momentum_20", "vol_20", "zscore_20"]
    pd.testing.assert_series_equal(first.loc[70, cols], second.loc[70, cols])


def test_signed_return_handles_negative_price():
    from analysis.returns import signed_price_return

    close = pd.Series([20.0, -10.0, 15.0])
    r = signed_price_return(close)
    assert np.isfinite(r.iloc[1])
    assert np.isfinite(r.iloc[2])
    assert r.iloc[1] < 0
    assert r.iloc[2] > 0


def test_metrics_are_additive_not_compounded():
    m = performance_metrics(pd.Series([0.10, -0.20, 0.05]))
    assert np.isclose(m["period_return"], -0.05)
