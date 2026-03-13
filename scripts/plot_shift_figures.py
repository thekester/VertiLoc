#!/usr/bin/env python3
"""Create publication-ready figures/tables for shift-vs-performance analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shift quantification figures.")
    parser.add_argument(
        "--shift-csv",
        type=Path,
        default=REPORT_DIR / "shift_loco_e102.csv",
        help="Shift metrics per held-out split.",
    )
    parser.add_argument(
        "--corr-csv",
        type=Path,
        default=REPORT_DIR / "shift_loco_e102_perf_corr.csv",
        help="Correlation table shift vs perf drops.",
    )
    parser.add_argument(
        "--perf-csv",
        type=Path,
        default=REPORT_DIR / "loco_e102.csv",
        help="LOCO performance details.",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=REPORT_DIR / "shift_loco_e102_overview.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=REPORT_DIR / "shift_loco_e102_top_correlations.csv",
        help="Output top-correlation table path.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Rows in top-correlation table.")
    return parser.parse_args()


def _normalize(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def main() -> None:
    args = parse_args()
    shift_df = pd.read_csv(args.shift_csv)
    corr_df = pd.read_csv(args.corr_csv)
    perf_df = pd.read_csv(args.perf_csv)

    # Export compact table of strongest correlations (absolute Pearson).
    top_corr = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index).head(args.top_k)
    top_corr.to_csv(args.out_table, index=False)

    # Build a readable "shift severity" score for campaigns.
    plot_df = shift_df.copy()
    plot_df["shift_score_z"] = (
        _normalize(plot_df["mmd_rbf"])
        + _normalize(plot_df["wasserstein_mean"])
        + _normalize(plot_df["psi_mean"])
    ) / 3.0
    plot_df = plot_df.sort_values("shift_score_z", ascending=False)

    # Pick strongest positive err_increase relation for a scatter panel.
    pos_err = corr_df[(corr_df["target"] == "err_increase") & (corr_df["pearson_r"] > 0)].copy()
    if pos_err.empty:
        pos_err = corr_df[corr_df["target"] == "err_increase"].copy()
    best = pos_err.iloc[pos_err["pearson_r"].abs().argmax()]
    model = str(best["model"])
    metric = str(best["shift_metric"])
    pearson = float(best["pearson_r"])

    merged = perf_df.merge(shift_df[["held_out", metric]], on="held_out", how="inner")
    err_col = f"{model}_mean_error_m"
    merged["err_increase"] = merged[err_col] - merged[err_col].min()

    # Figure layout.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # Panel A: shift severity by held-out campaign.
    ax = axes[0]
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["shift_score_z"], color="#2b7a78", alpha=0.85, label="Shift score (z)")
    ax.plot(x, plot_df["mmd_rbf"], marker="o", color="#17252a", linewidth=1.8, label="MMD RBF")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["held_out"], rotation=35, ha="right")
    ax.set_title("A) Shift severity per LOCO split (E102)")
    ax.set_ylabel("Normalized severity / MMD")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    # Panel B: strongest shift->error relation.
    ax2 = axes[1]
    xx = merged[metric].to_numpy(dtype=float)
    yy = merged["err_increase"].to_numpy(dtype=float)
    ax2.scatter(xx, yy, s=70, color="#fe4a49", edgecolors="black", linewidth=0.4, alpha=0.9)
    for _, row in merged.iterrows():
        ax2.annotate(str(row["held_out"]), (row[metric], row["err_increase"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    if len(xx) >= 2 and np.std(xx) > 1e-12:
        coeffs = np.polyfit(xx, yy, 1)
        xline = np.linspace(xx.min(), xx.max(), 100)
        yline = coeffs[0] * xline + coeffs[1]
        ax2.plot(xline, yline, color="#2f4858", linewidth=1.8)
    ax2.set_title(
        f"B) Strongest relation: {model} err_increase vs {metric}\nPearson r={pearson:.3f}"
    )
    ax2.set_xlabel(metric)
    ax2.set_ylabel(f"{model} err_increase (m)")
    ax2.grid(alpha=0.25)

    fig.suptitle("Train/Test Shift Quantification and Performance Degradation (LOCO E102)", fontsize=12)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.out_fig}")
    print(f"Saved table:  {args.out_table}")


if __name__ == "__main__":
    main()
