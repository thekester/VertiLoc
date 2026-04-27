"""Few-shot calibration by number of target cells.

This benchmark answers a stricter question than K-shot-per-class calibration:
what happens if only M cells of the target room are calibrated, then the model
must localize the rest of the target room?

Protocol:
  - LORO split: the target room is absent from source training.
  - The scaler is fitted on source rooms only.
  - For each target fold, randomly select M calibrated target cells.
  - Draw K labeled support samples per calibrated cell.
  - Evaluate on:
      * all remaining target samples, excluding supports
      * unseen target cells only, i.e. cells with no target support

Important interpretation:
  TargetOnly KNN can only predict labels seen in the support set. Its exact
  accuracy on unseen cells is therefore expected to be 0. The Source+Target
  hybrid keeps source-room labels so it can still attempt all cell labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42

warnings.filterwarnings("ignore", message=".*unique classes.*")


def load_multiroom() -> pd.DataFrame:
    campaigns = filter_room_campaigns()
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = room
                frames.append(df)
    if not frames:
        raise RuntimeError("No benchmark campaign data found.")
    return pd.concat(frames, ignore_index=True)


def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, lookup: pd.DataFrame) -> dict:
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    pred_c = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    true_c = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    mask = ~(np.isnan(pred_c).any(axis=1) | np.isnan(true_c).any(axis=1))
    if mask.any():
        errors = np.linalg.norm(pred_c[mask] - true_c[mask], axis=1)
        err = float(np.mean(errors))
        p90 = float(np.percentile(errors, 90))
    else:
        err = float("nan")
        p90 = float("nan")
    return {"cell_acc": acc, "mean_error_m": err, "p90_error_m": p90, "n": int(len(y_true))}


def _fit_knn(X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(
        n_neighbors=min(n_neighbors, len(X_train)),
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    return knn


def _sample_supports_for_cells(
    y_target: np.ndarray,
    calibrated_cells: np.ndarray,
    shots_per_cell: int,
    rng: np.random.Generator,
) -> np.ndarray:
    support_idx: list[int] = []
    for cell in calibrated_cells:
        cell_idx = np.where(y_target == cell)[0]
        if len(cell_idx) == 0:
            continue
        k_eff = min(shots_per_cell, len(cell_idx))
        support_idx.extend(rng.choice(cell_idx, size=k_eff, replace=False).tolist())
    return np.asarray(support_idx, dtype=int)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanstd(arr))


def _aggregate_metric(rows: list[dict], prefix: str, metric: str) -> tuple[float, float]:
    values = [row[prefix][metric] for row in rows if prefix in row]
    return _nanmean(values), _nanstd(values)


def run(args: argparse.Namespace) -> dict:
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)

    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()
    room_arr = df["room"].to_numpy()

    out: dict = {
        "config": vars(args),
        "protocol": (
            "LORO cell-coverage few-shot: choose M calibrated cells in the held-out "
            "target room, draw K labeled support samples per calibrated cell, then "
            "evaluate on all remaining target samples and on unseen cells only."
        ),
        "interpretation_note": (
            "TargetOnly KNN cannot predict labels for cells absent from support; "
            "its exact accuracy on unseen cells should be 0. Source+Target keeps "
            "source-room labels and can attempt uncalibrated cell labels."
        ),
        "folds": {},
        "summary": [],
    }

    print("\n=== Few-shot cell coverage LORO ===")
    print(f"  K={args.shots_per_cell} support(s) per calibrated cell")

    for fold_idx, held_out in enumerate(rooms):
        train_mask = room_arr != held_out
        target_mask = room_arr == held_out

        X_src = X_all[train_mask]
        y_src = y_all[train_mask]
        X_tgt = X_all[target_mask]
        y_tgt = y_all[target_mask]
        cells = np.unique(y_tgt)
        source_labels = set(np.unique(y_src).tolist())
        source_covered = float(np.mean([cell in source_labels for cell in cells]))

        scaler = StandardScaler().fit(X_src)
        X_src_s = scaler.transform(X_src).astype(np.float32)
        X_tgt_s = scaler.transform(X_tgt).astype(np.float32)

        source_knn = _fit_knn(X_src_s, y_src, args.source_neighbors)
        source_pred_all = source_knn.predict(X_tgt_s)

        fold: dict = {
            "n_target_samples": int(len(y_tgt)),
            "n_target_cells": int(len(cells)),
            "source_label_coverage_of_target_cells": source_covered,
            "source_knn_all_target": _metrics(y_tgt, source_pred_all, lookup),
            "by_cell_count": {},
        }

        print(
            f"\n  Fold {held_out}: {len(y_tgt)} samples, {len(cells)} cells, "
            f"source label coverage={source_covered:.1%}"
        )
        print(f"    SourceKNN all target: {fold['source_knn_all_target']['cell_acc']:.4f}")

        for requested_m in args.cell_counts:
            if requested_m < 1:
                continue
            m = min(int(requested_m), len(cells))
            fold_key = str(m)
            if fold_key in fold["by_cell_count"]:
                continue
            trial_rows: list[dict] = []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + fold_idx * 100_000 + m * 1_000 + trial)
                calibrated_cells = rng.choice(cells, size=m, replace=False)
                calibrated_set = set(calibrated_cells.tolist())

                support_idx = _sample_supports_for_cells(
                    y_tgt,
                    calibrated_cells,
                    args.shots_per_cell,
                    rng,
                )
                support_set = set(support_idx.tolist())
                all_query_idx = np.asarray(
                    [i for i in range(len(y_tgt)) if i not in support_set],
                    dtype=int,
                )
                unseen_query_idx = np.asarray(
                    [i for i, label in enumerate(y_tgt) if label not in calibrated_set],
                    dtype=int,
                )

                X_sup_s = X_tgt_s[support_idx]
                y_sup = y_tgt[support_idx]

                target_knn = _fit_knn(X_sup_s, y_sup, args.target_neighbors)
                target_all_pred = target_knn.predict(X_tgt_s[all_query_idx])
                target_unseen_pred = (
                    target_knn.predict(X_tgt_s[unseen_query_idx])
                    if len(unseen_query_idx)
                    else np.asarray([], dtype=object)
                )

                X_sup_rep = np.repeat(X_sup_s, args.target_replicas, axis=0)
                y_sup_rep = np.repeat(y_sup, args.target_replicas)
                X_hybrid = np.vstack([X_src_s, X_sup_rep])
                y_hybrid = np.concatenate([y_src, y_sup_rep])
                hybrid_knn = _fit_knn(X_hybrid, y_hybrid, args.hybrid_neighbors)
                hybrid_all_pred = hybrid_knn.predict(X_tgt_s[all_query_idx])
                hybrid_unseen_pred = (
                    hybrid_knn.predict(X_tgt_s[unseen_query_idx])
                    if len(unseen_query_idx)
                    else np.asarray([], dtype=object)
                )

                source_all_pred = source_pred_all[all_query_idx]
                source_unseen_pred = source_pred_all[unseen_query_idx]

                trial_rows.append(
                    {
                        "m": int(m),
                        "coverage": float(m / len(cells)),
                        "n_support": int(len(support_idx)),
                        "n_query_all": int(len(all_query_idx)),
                        "n_query_unseen": int(len(unseen_query_idx)),
                        "TargetOnly_all": _metrics(y_tgt[all_query_idx], target_all_pred, lookup),
                        "TargetOnly_unseen": _metrics(y_tgt[unseen_query_idx], target_unseen_pred, lookup),
                        "SourceKNN_all": _metrics(y_tgt[all_query_idx], source_all_pred, lookup),
                        "SourceKNN_unseen": _metrics(y_tgt[unseen_query_idx], source_unseen_pred, lookup),
                        "SourceTarget_all": _metrics(y_tgt[all_query_idx], hybrid_all_pred, lookup),
                        "SourceTarget_unseen": _metrics(y_tgt[unseen_query_idx], hybrid_unseen_pred, lookup),
                    }
                )

            fold["by_cell_count"][fold_key] = {
                "requested_cell_count": int(requested_m),
                "effective_cell_count": int(m),
                "coverage_mean": _nanmean([r["coverage"] for r in trial_rows]),
                "n_support_mean": _nanmean([r["n_support"] for r in trial_rows]),
                "n_query_all_mean": _nanmean([r["n_query_all"] for r in trial_rows]),
                "n_query_unseen_mean": _nanmean([r["n_query_unseen"] for r in trial_rows]),
                "TargetOnly_all": {},
                "TargetOnly_unseen": {},
                "SourceKNN_all": {},
                "SourceKNN_unseen": {},
                "SourceTarget_all": {},
                "SourceTarget_unseen": {},
            }
            for method in (
                "TargetOnly_all",
                "TargetOnly_unseen",
                "SourceKNN_all",
                "SourceKNN_unseen",
                "SourceTarget_all",
                "SourceTarget_unseen",
            ):
                for metric in ("cell_acc", "mean_error_m", "p90_error_m", "n"):
                    mean, std = _aggregate_metric(trial_rows, method, metric)
                    fold["by_cell_count"][fold_key][method][f"{metric}_mean"] = mean
                    fold["by_cell_count"][fold_key][method][f"{metric}_std"] = std

            ta = fold["by_cell_count"][fold_key]["TargetOnly_all"]["cell_acc_mean"]
            tu = fold["by_cell_count"][fold_key]["TargetOnly_unseen"]["cell_acc_mean"]
            hu = fold["by_cell_count"][fold_key]["SourceTarget_unseen"]["cell_acc_mean"]
            print(
                f"    M={m:>3} ({m / len(cells):>5.1%})  "
                f"TargetOnly all={ta:.4f} unseen={tu:.4f}  "
                f"Source+Target unseen={hu:.4f}"
            )

        out["folds"][held_out] = fold

    # Cross-room summary by requested/effective M. For each requested M, average
    # folds using their effective M, because rooms do not all have the same cell count.
    print("\n  Cross-room summary:")
    for requested_m in args.cell_counts:
        rows_for_m = []
        for room in rooms:
            fold = out["folds"][room]
            effective_m = min(int(requested_m), fold["n_target_cells"])
            key = str(effective_m)
            if key in fold["by_cell_count"]:
                row = {
                    "room": room,
                    "requested_cell_count": int(requested_m),
                    "effective_cell_count": int(effective_m),
                    "n_target_cells": fold["n_target_cells"],
                    **fold["by_cell_count"][key],
                }
                rows_for_m.append(row)
        if not rows_for_m:
            continue

        summary_row = {
            "requested_cell_count": int(requested_m),
            "coverage_mean": _nanmean([r["coverage_mean"] for r in rows_for_m]),
            "n_support_mean": _nanmean([r["n_support_mean"] for r in rows_for_m]),
        }
        for method in (
            "TargetOnly_all",
            "TargetOnly_unseen",
            "SourceKNN_all",
            "SourceKNN_unseen",
            "SourceTarget_all",
            "SourceTarget_unseen",
        ):
            for metric in ("cell_acc", "mean_error_m", "p90_error_m"):
                summary_row[f"{method}_{metric}_mean"] = _nanmean(
                    [r[method][f"{metric}_mean"] for r in rows_for_m]
                )
                summary_row[f"{method}_{metric}_std_cross_room"] = _nanstd(
                    [r[method][f"{metric}_mean"] for r in rows_for_m]
                )
        out["summary"].append(summary_row)
        print(
            f"    M={requested_m:>3} coverage={summary_row['coverage_mean']:>5.1%} "
            f"TargetOnly_all={summary_row['TargetOnly_all_cell_acc_mean']:.4f} "
            f"TargetOnly_unseen={summary_row['TargetOnly_unseen_cell_acc_mean']:.4f} "
            f"Source+Target_unseen={summary_row['SourceTarget_unseen_cell_acc_mean']:.4f}"
        )

    return out


def save_summary_csv(results: dict, path: Path) -> None:
    pd.DataFrame(results["summary"]).to_csv(path, index=False)


def save_accuracy_plot(results: dict, path: Path) -> None:
    summary = pd.DataFrame(results["summary"])
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = summary["coverage_mean"] * 100.0
    series = [
        ("SourceKNN_unseen_cell_acc_mean", "SourceKNN, cellules non calibrees", "#8c8c8c", "--"),
        ("SourceTarget_unseen_cell_acc_mean", "Source+Target, cellules non calibrees", "#1f77b4", "-"),
        ("TargetOnly_all_cell_acc_mean", "TargetOnly, toute la cible restante", "#2ca02c", "-"),
        ("TargetOnly_unseen_cell_acc_mean", "TargetOnly, cellules non calibrees", "#d62728", ":"),
    ]
    for col, label, color, linestyle in series:
        ax.plot(x, summary[col] * 100.0, marker="o", linewidth=2, color=color, linestyle=linestyle, label=label)

    ax.set_xlabel("Cellules cible calibrees (%)")
    ax.set_ylabel("Accuracy cellule (%)")
    ax.set_title("Few-shot par couverture de cellules cible")
    ax.set_xlim(0, min(100, max(5.0, float(x.max()) + 3.0)))
    ymax = max(5.0, float((summary[[s[0] for s in series]].max().max() * 100.0) + 2.0))
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot by number of calibrated target cells")
    parser.add_argument("--cell-counts", nargs="+", type=int, default=[1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80])
    parser.add_argument("--shots-per-cell", type=int, default=1, help="K labeled supports per calibrated cell.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--target-neighbors", type=int, default=1)
    parser.add_argument("--source-neighbors", type=int, default=7)
    parser.add_argument("--hybrid-neighbors", type=int, default=5)
    parser.add_argument("--target-replicas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="fewshot_cell_coverage.json")
    parser.add_argument("--summary-csv-name", default="fewshot_cell_coverage_summary.csv")
    parser.add_argument("--plot-name", default="fewshot_cell_coverage_accuracy.png")
    args = parser.parse_args()

    results = run(args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / args.output_name
    csv_path = REPORT_DIR / args.summary_csv_name
    plot_path = REPORT_DIR / args.plot_name

    json_path.write_text(json.dumps(results, indent=2, default=str))
    save_summary_csv(results, csv_path)
    save_accuracy_plot(results, plot_path)

    print(f"\nResults saved to {json_path}")
    print(f"Summary saved to {csv_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
