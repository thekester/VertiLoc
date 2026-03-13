#!/usr/bin/env python3
"""Quantify train/test distribution shift and correlate with performance drops.

Default workflow:
1) Build LOCO splits inside E102 campaigns.
2) Compute shift metrics between train (5 campaigns) and test (held-out campaign):
   - MMD (RBF kernel),
   - Wasserstein-1 (per feature + mean/max aggregate),
   - PSI (per feature + mean/max aggregate).
3) Merge with model performance from reports/benchmarks/loco_e102.csv.
4) Correlate shift severity with performance degradation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"

_E102_ROOT = PROJECT_ROOT / "data" / "E102"
CAMPAIGNS: dict[str, CampaignSpec] = {
    "exp1_back_right": CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0),
    "exp2_front_right": CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0),
    "exp3_front_left": CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0),
    "exp4_back_left": CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0),
    "exp5_ground": CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6_1m50": CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}


def _load_campaign_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        df["campaign"] = name
        frames[name] = df
    return frames


def _sample_matrix(x: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= max_samples:
        return x
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx]


def _rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    # ||a-b||^2 = a^2 + b^2 - 2ab
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    sq_dists = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.exp(-gamma * sq_dists)


def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    x_small = _sample_matrix(x, max_samples=min(400, len(x)), rng=rng)
    y_small = _sample_matrix(y, max_samples=min(400, len(y)), rng=rng)
    z = np.vstack([x_small, y_small])
    if len(z) < 2:
        return 1.0 / max(x.shape[1], 1)
    z2 = np.sum(z * z, axis=1, keepdims=True)
    sq = np.maximum(z2 + z2.T - 2.0 * (z @ z.T), 0.0)
    tri = sq[np.triu_indices(len(z), k=1)]
    tri = tri[tri > 0]
    if len(tri) == 0:
        return 1.0 / max(x.shape[1], 1)
    median_sq = float(np.median(tri))
    return 1.0 / max(2.0 * median_sq, 1e-12)


def _mmd_rbf_unbiased(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_samples: int,
) -> float:
    x = _sample_matrix(x, max_samples=max_samples, rng=rng)
    y = _sample_matrix(y, max_samples=max_samples, rng=rng)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan")

    gamma = _median_heuristic_gamma(x, y, rng=rng)
    k_xx = _rbf_kernel(x, x, gamma=gamma)
    k_yy = _rbf_kernel(y, y, gamma=gamma)
    k_xy = _rbf_kernel(x, y, gamma=gamma)

    term_x = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    term_y = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    term_xy = k_xy.mean()
    mmd2 = term_x + term_y - 2.0 * term_xy
    return float(max(mmd2, 0.0))


def _stable_bin_edges(train_col: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(train_col, np.linspace(0.0, 1.0, bins + 1))
    edges = np.asarray(edges, dtype=float)
    edges = np.unique(edges)
    if len(edges) < 2:
        c = float(train_col[0]) if len(train_col) else 0.0
        edges = np.array([c - 1.0, c + 1.0], dtype=float)
    return edges


def _psi_1d(train_col: np.ndarray, test_col: np.ndarray, bins: int, eps: float = 1e-6) -> float:
    edges = _stable_bin_edges(train_col, bins=bins)
    train_hist, _ = np.histogram(train_col, bins=edges)
    test_hist, _ = np.histogram(test_col, bins=edges)
    p = train_hist.astype(float)
    q = test_hist.astype(float)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def compute_shift_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    bins: int,
    mmd_max_samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    metrics: dict[str, float] = {
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
    }
    metrics["mmd_rbf"] = _mmd_rbf_unbiased(x_train, x_test, rng=rng, max_samples=mmd_max_samples)

    w_vals: list[float] = []
    psi_vals: list[float] = []
    for i, col in enumerate(FEATURE_COLUMNS):
        w = float(wasserstein_distance(x_train[:, i], x_test[:, i]))
        psi = _psi_1d(x_train[:, i], x_test[:, i], bins=bins)
        metrics[f"wasserstein_{col}"] = w
        metrics[f"psi_{col}"] = psi
        w_vals.append(w)
        psi_vals.append(psi)

    metrics["wasserstein_mean"] = float(np.mean(w_vals))
    metrics["wasserstein_max"] = float(np.max(w_vals))
    metrics["psi_mean"] = float(np.mean(psi_vals))
    metrics["psi_max"] = float(np.max(psi_vals))
    return metrics


def build_shift_table(*, bins: int, mmd_max_samples: int, seed: int) -> pd.DataFrame:
    frames = _load_campaign_frames()
    rows: list[dict] = []
    campaign_names = list(CAMPAIGNS.keys())
    for held_out in campaign_names:
        test_df = frames[held_out]
        train_df = pd.concat([frames[name] for name in campaign_names if name != held_out], ignore_index=True)
        row = {"held_out": held_out}
        row.update(
            compute_shift_metrics(
                train_df,
                test_df,
                bins=bins,
                mmd_max_samples=mmd_max_samples,
                seed=seed,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    merged = pd.concat([a, b], axis=1).dropna()
    if len(merged) < 3:
        return float("nan")
    if merged.iloc[:, 0].std(ddof=0) <= 1e-12 or merged.iloc[:, 1].std(ddof=0) <= 1e-12:
        return float("nan")
    return float(merged.iloc[:, 0].corr(merged.iloc[:, 1], method=method))


def correlate_shift_with_performance(shift_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    merged = perf_df.merge(shift_df, on="held_out", how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between shift table and performance table.")

    shift_cols = ["mmd_rbf", "wasserstein_mean", "wasserstein_max", "psi_mean", "psi_max"]
    model_prefixes = sorted(
        col[: -len("_cell_accuracy")] for col in merged.columns if col.endswith("_cell_accuracy")
    )
    rows: list[dict] = []
    for model in model_prefixes:
        acc_col = f"{model}_cell_accuracy"
        err_col = f"{model}_mean_error_m"
        if acc_col not in merged.columns or err_col not in merged.columns:
            continue
        acc_drop = merged[acc_col].max() - merged[acc_col]
        err_increase = merged[err_col] - merged[err_col].min()
        for metric in shift_cols:
            rows.append(
                {
                    "model": model,
                    "target": "acc_drop",
                    "shift_metric": metric,
                    "pearson_r": _safe_corr(merged[metric], acc_drop, method="pearson"),
                    "spearman_r": _safe_corr(merged[metric], acc_drop, method="spearman"),
                    "n_points": int(pd.concat([merged[metric], acc_drop], axis=1).dropna().shape[0]),
                }
            )
            rows.append(
                {
                    "model": model,
                    "target": "err_increase",
                    "shift_metric": metric,
                    "pearson_r": _safe_corr(merged[metric], err_increase, method="pearson"),
                    "spearman_r": _safe_corr(merged[metric], err_increase, method="spearman"),
                    "n_points": int(pd.concat([merged[metric], err_increase], axis=1).dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "model", "shift_metric"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify train/test shift (MMD, Wasserstein, PSI) and correlate with perf drop."
    )
    parser.add_argument(
        "--perf-csv",
        type=Path,
        default=REPORT_DIR / "loco_e102.csv",
        help="Performance CSV with held_out splits and model metrics.",
    )
    parser.add_argument("--bins", type=int, default=10, help="PSI bins per feature.")
    parser.add_argument(
        "--mmd-max-samples",
        type=int,
        default=1200,
        help="Max samples per side for MMD computation.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--out-prefix",
        default="shift_loco_e102",
        help="Output prefix under reports/benchmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.perf_csv.exists():
        raise FileNotFoundError(f"Performance CSV not found: {args.perf_csv}")
    perf_df = pd.read_csv(args.perf_csv)
    if "held_out" not in perf_df.columns:
        raise ValueError("Performance CSV must contain a 'held_out' column.")

    shift_df = build_shift_table(
        bins=args.bins,
        mmd_max_samples=args.mmd_max_samples,
        seed=args.seed,
    )
    corr_df = correlate_shift_with_performance(shift_df=shift_df, perf_df=perf_df)

    shift_path = REPORT_DIR / f"{args.out_prefix}.csv"
    corr_path = REPORT_DIR / f"{args.out_prefix}_perf_corr.csv"
    summary_path = REPORT_DIR / f"{args.out_prefix}_perf_corr.json"

    shift_df.to_csv(shift_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    summary = {
        "perf_csv": str(args.perf_csv),
        "n_splits": int(shift_df["held_out"].nunique()),
        "shift_metrics": ["mmd_rbf", "wasserstein_mean", "wasserstein_max", "psi_mean", "psi_max"],
        "top_abs_pearson": corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index)
        .head(10)
        .to_dict(orient="records"),
        "top_abs_spearman": corr_df.reindex(corr_df["spearman_r"].abs().sort_values(ascending=False).index)
        .head(10)
        .to_dict(orient="records"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved shift metrics to: {shift_path}")
    print(f"Saved correlations to: {corr_path}")
    print(f"Saved summary JSON to: {summary_path}")
    if not corr_df.empty:
        top = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index).head(5)
        print("\nTop 5 |pearson| correlations:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
