"""Phase 4: aggregate the real walk-forward results.

This script does not train a new model. It summarises the existing LSTM and
momentum results and writes small tables and figures for the project analysis.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.result_analysis import (
    load_lstm_results,
    aggregate_by_asset,
    aggregate_by_group,
    wti_robustness,
    accuracy_value_relationship,
)


def save_bar(asset, out):
    needed = {"lstm_sharpe_mean", "momentum_sharpe_mean"}
    if not needed.issubset(asset.columns):
        return

    plot = asset.set_index("symbol")[
        ["lstm_sharpe_mean", "momentum_sharpe_mean"]
    ]
    ax = plot.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Mean walk-forward Sharpe")
    ax.set_xlabel("")
    ax.set_title("LSTM vs momentum across commodities")
    ax.axhline(0, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out / "sharpe_by_asset.png", dpi=160)
    plt.close()


def save_scatter(df, out):
    da = next(
        (c for c in ["directional_accuracy", "direction_accuracy"] if c in df.columns),
        None,
    )
    if da is None or "lstm_sharpe" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for symbol, g in df.groupby("symbol"):
        ax.scatter(g[da], g["lstm_sharpe"], label=symbol, alpha=0.75)

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Directional accuracy")
    ax.set_ylabel("LSTM Sharpe")
    ax.set_title("Forecast accuracy vs trading performance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "accuracy_vs_sharpe.png", dpi=160)
    plt.close()


def main():
    result_file = ROOT / "results" / "lstm_walk_forward.csv"
    if not result_file.exists():
        raise FileNotFoundError(
            "results/lstm_walk_forward.csv not found. "
            "Run the real-data LSTM walk-forward experiment first."
        )

    out = ROOT / "results" / "phase4"
    out.mkdir(parents=True, exist_ok=True)

    df = load_lstm_results(result_file)

    # Keep the final study focused on the five actual commodity series.
    df = df[
        df["symbol"].isin(["CL=F", "NG=F", "GC=F", "SI=F", "PL=F"])
    ].copy()

    asset = aggregate_by_asset(df)
    group = aggregate_by_group(df)
    wti = wti_robustness(df)
    relation = accuracy_value_relationship(df)

    asset.to_csv(out / "asset_summary.csv", index=False)
    group.to_csv(out / "group_summary.csv", index=False)
    wti.to_csv(out / "wti_robustness.csv", index=False)
    relation.to_csv(out / "accuracy_value_relationship.csv", index=False)

    save_bar(asset, out)
    save_scatter(df, out)

    print("\nAsset summary")
    print(asset.round(3).to_string(index=False))

    print("\nEnergy vs precious metals")
    print(group.round(3).to_string(index=False))

    print("\nWTI robustness")
    print(wti.round(3).to_string(index=False))

    print("\nForecast accuracy vs trading value")
    print(relation.round(3).to_string(index=False))

    print(f"\nsaved Phase 4 outputs to: {out}")


if __name__ == "__main__":
    main()
