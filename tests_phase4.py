import pandas as pd

from analysis.result_analysis import aggregate_by_asset, accuracy_value_relationship


def test_asset_aggregation_compares_lstm_and_momentum():
    df = pd.DataFrame({
        "symbol": ["CL=F", "CL=F", "GC=F"],
        "directional_accuracy": [0.52, 0.48, 0.55],
        "lstm_sharpe": [1.0, -0.5, 0.4],
        "momentum_sharpe": [0.2, 0.1, 0.6],
    })
    out = aggregate_by_asset(df)
    cl = out[out["symbol"] == "CL=F"].iloc[0]
    assert cl["windows"] == 2
    assert cl["lstm_win_share"] == 0.5


def test_accuracy_value_relationship_returns_correlation():
    df = pd.DataFrame({
        "symbol": ["CL=F", "NG=F", "GC=F"],
        "directional_accuracy": [0.45, 0.50, 0.60],
        "lstm_sharpe": [-1.0, 0.0, 2.0],
    })
    out = accuracy_value_relationship(df)
    assert out.loc[0, "windows"] == 3
    assert out.loc[0, "corr_directional_accuracy_lstm_sharpe"] > 0.9
