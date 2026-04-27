"""Few-shot spatial radio-map interpolation.

This benchmark tests whether a target room can be partially calibrated, then
the missing cells inferred from spatial interpolation.

Compared with `fewshot_cell_coverage.py`, this script does not train a KNN only
on observed target labels. Instead, it builds a synthetic RSSI prototype for
every target cell from the M calibrated cells:

  - NearestSpatial: copy the nearest calibrated cell prototype.
  - IDW: inverse-distance weighted interpolation of RSSI prototypes.
  - GP_RBF: Gaussian-process/RBF interpolation, equivalent in spirit to
    ordinary kriging with a fixed kernel.

Localization then predicts the cell whose interpolated RSSI prototype is
closest to the query RSSI vector.
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
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


def _cell_table(df_target: pd.DataFrame) -> pd.DataFrame:
    return (
        df_target[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .sort_values(["coord_y_m", "coord_x_m", "grid_cell"])
        .reset_index(drop=True)
    )


def _support_prototypes(
    X_support_s: np.ndarray,
    y_support: np.ndarray,
    cell_coords: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.asarray(sorted(np.unique(y_support).tolist()))
    proto = np.vstack([X_support_s[y_support == cell].mean(axis=0) for cell in cells])
    coords_by_cell = cell_coords.set_index("grid_cell").loc[cells, ["coord_x_m", "coord_y_m"]]
    return cells, coords_by_cell.to_numpy(dtype=float), proto


def _nearest_spatial_prototypes(
    support_coords: np.ndarray,
    support_proto: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    d = np.linalg.norm(target_coords[:, None, :] - support_coords[None, :, :], axis=2)
    return support_proto[np.argmin(d, axis=1)]


def _idw_prototypes(
    support_coords: np.ndarray,
    support_proto: np.ndarray,
    target_coords: np.ndarray,
    *,
    power: float,
    eps: float = 1e-9,
) -> np.ndarray:
    d = np.linalg.norm(target_coords[:, None, :] - support_coords[None, :, :], axis=2)
    exact = d < eps
    weights = 1.0 / np.maximum(d, eps) ** power
    weights = weights / weights.sum(axis=1, keepdims=True)
    pred = weights @ support_proto
    if exact.any():
        rows, cols = np.where(exact)
        pred[rows] = support_proto[cols]
    return pred


def _gp_rbf_prototypes(
    support_coords: np.ndarray,
    support_proto: np.ndarray,
    target_coords: np.ndarray,
    *,
    length_scale: float,
    noise_level: float,
) -> np.ndarray:
    if len(support_coords) < 2:
        return np.repeat(support_proto.mean(axis=0, keepdims=True), len(target_coords), axis=0)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed")
    )
    preds = []
    for feature_idx in range(support_proto.shape[1]):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer=None,
            normalize_y=True,
            random_state=SEED,
        )
        gp.fit(support_coords, support_proto[:, feature_idx])
        preds.append(gp.predict(target_coords))
    return np.vstack(preds).T


def _predict_from_prototypes(
    X_query_s: np.ndarray,
    cell_labels: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    d = np.linalg.norm(X_query_s[:, None, :] - prototypes[None, :, :], axis=2)
    return cell_labels[np.argmin(d, axis=1)]


def _force_observed_support_prototypes(
    all_cells: np.ndarray,
    prototypes: np.ndarray,
    support_cells: np.ndarray,
    support_proto: np.ndarray,
) -> np.ndarray:
    out = prototypes.copy()
    positions = {cell: idx for idx, cell in enumerate(all_cells)}
    for support_idx, cell in enumerate(support_cells):
        out[positions[cell]] = support_proto[support_idx]
    return out


def _evaluate_interpolator(
    name: str,
    prototypes: np.ndarray,
    all_cells: np.ndarray,
    X_query_all_s: np.ndarray,
    y_query_all: np.ndarray,
    X_query_unseen_s: np.ndarray,
    y_query_unseen: np.ndarray,
    lookup: pd.DataFrame,
) -> dict:
    pred_all = _predict_from_prototypes(X_query_all_s, all_cells, prototypes)
    pred_unseen = (
        _predict_from_prototypes(X_query_unseen_s, all_cells, prototypes)
        if len(y_query_unseen)
        else np.asarray([], dtype=object)
    )
    return {
        "method": name,
        "all": _metrics(y_query_all, pred_all, lookup),
        "unseen": _metrics(y_query_unseen, pred_unseen, lookup),
    }


def run(args: argparse.Namespace) -> dict:
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)

    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()
    room_arr = df["room"].to_numpy()

    methods = ["NearestSpatial", "IDW", "GP_RBF"]
    out: dict = {
        "config": vars(args),
        "protocol": (
            "LORO partial target calibration with spatial radio-map interpolation. "
            "M target cells are calibrated with K support samples per cell; RSSI "
            "prototypes for all target cells are spatially interpolated and used "
            "for nearest-prototype localization."
        ),
        "methods": {
            "NearestSpatial": "Copy the RSSI prototype of the nearest calibrated cell.",
            "IDW": "Inverse-distance weighted RSSI interpolation.",
            "GP_RBF": "Gaussian-process/RBF interpolation, kriging-like with fixed kernel.",
        },
        "folds": {},
        "summary": [],
    }

    print("\n=== Few-shot spatial interpolation LORO ===")
    print(f"  K={args.shots_per_cell} support(s) per calibrated cell")

    for fold_idx, held_out in enumerate(rooms):
        train_mask = room_arr != held_out
        target_mask = room_arr == held_out

        X_src = X_all[train_mask]
        y_src = y_all[train_mask]
        X_tgt = X_all[target_mask]
        y_tgt = y_all[target_mask]
        df_tgt = df[target_mask].reset_index(drop=True)

        scaler = StandardScaler().fit(X_src)
        X_src_s = scaler.transform(X_src).astype(np.float32)
        X_tgt_s = scaler.transform(X_tgt).astype(np.float32)

        source_knn = _fit_knn(X_src_s, y_src, args.source_neighbors)
        source_pred_all = source_knn.predict(X_tgt_s)

        cell_df = _cell_table(df_tgt)
        all_cells = cell_df["grid_cell"].to_numpy()
        all_coords = cell_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
        n_cells = len(all_cells)

        fold: dict = {
            "n_target_samples": int(len(y_tgt)),
            "n_target_cells": int(n_cells),
            "source_knn_all_target": _metrics(y_tgt, source_pred_all, lookup),
            "by_cell_count": {},
        }

        print(f"\n  Fold {held_out}: {len(y_tgt)} samples, {n_cells} cells")
        print(f"    SourceKNN all target: {fold['source_knn_all_target']['cell_acc']:.4f}")

        for requested_m in args.cell_counts:
            if requested_m < 1:
                continue
            m = min(int(requested_m), n_cells)
            fold_key = str(m)
            if fold_key in fold["by_cell_count"]:
                continue

            trial_rows: list[dict] = []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + fold_idx * 100_000 + m * 1_000 + trial)
                calibrated_cells = rng.choice(all_cells, size=m, replace=False)
                calibrated_set = set(calibrated_cells.tolist())

                support_idx = _sample_supports_for_cells(y_tgt, calibrated_cells, args.shots_per_cell, rng)
                support_set = set(support_idx.tolist())
                all_query_idx = np.asarray([i for i in range(len(y_tgt)) if i not in support_set], dtype=int)
                unseen_query_idx = np.asarray(
                    [i for i, label in enumerate(y_tgt) if label not in calibrated_set],
                    dtype=int,
                )

                support_cells, support_coords, support_proto = _support_prototypes(
                    X_tgt_s[support_idx],
                    y_tgt[support_idx],
                    cell_df,
                )

                nearest_proto = _nearest_spatial_prototypes(support_coords, support_proto, all_coords)
                idw_proto = _idw_prototypes(
                    support_coords,
                    support_proto,
                    all_coords,
                    power=args.idw_power,
                )
                gp_proto = _gp_rbf_prototypes(
                    support_coords,
                    support_proto,
                    all_coords,
                    length_scale=args.gp_length_scale,
                    noise_level=args.gp_noise,
                )

                proto_by_method = {
                    "NearestSpatial": nearest_proto,
                    "IDW": idw_proto,
                    "GP_RBF": gp_proto,
                }
                trial_result = {
                    "m": int(m),
                    "coverage": float(m / n_cells),
                    "n_support": int(len(support_idx)),
                    "n_query_all": int(len(all_query_idx)),
                    "n_query_unseen": int(len(unseen_query_idx)),
                    "SourceKNN_all": _metrics(y_tgt[all_query_idx], source_pred_all[all_query_idx], lookup),
                    "SourceKNN_unseen": _metrics(
                        y_tgt[unseen_query_idx],
                        source_pred_all[unseen_query_idx],
                        lookup,
                    ),
                }

                for method_name, proto in proto_by_method.items():
                    proto = _force_observed_support_prototypes(all_cells, proto, support_cells, support_proto)
                    trial_result[method_name] = _evaluate_interpolator(
                        method_name,
                        proto,
                        all_cells,
                        X_tgt_s[all_query_idx],
                        y_tgt[all_query_idx],
                        X_tgt_s[unseen_query_idx],
                        y_tgt[unseen_query_idx],
                        lookup,
                    )

                trial_rows.append(trial_result)

            by_m = {
                "requested_cell_count": int(requested_m),
                "effective_cell_count": int(m),
                "coverage_mean": _nanmean([r["coverage"] for r in trial_rows]),
                "n_support_mean": _nanmean([r["n_support"] for r in trial_rows]),
                "n_query_all_mean": _nanmean([r["n_query_all"] for r in trial_rows]),
                "n_query_unseen_mean": _nanmean([r["n_query_unseen"] for r in trial_rows]),
                "SourceKNN_all": {},
                "SourceKNN_unseen": {},
            }

            for source_key in ("SourceKNN_all", "SourceKNN_unseen"):
                for metric in ("cell_acc", "mean_error_m", "p90_error_m"):
                    vals = [r[source_key][metric] for r in trial_rows]
                    by_m[source_key][f"{metric}_mean"] = _nanmean(vals)
                    by_m[source_key][f"{metric}_std"] = _nanstd(vals)

            for method_name in methods:
                by_m[method_name] = {"all": {}, "unseen": {}}
                for split in ("all", "unseen"):
                    for metric in ("cell_acc", "mean_error_m", "p90_error_m"):
                        vals = [r[method_name][split][metric] for r in trial_rows]
                        by_m[method_name][split][f"{metric}_mean"] = _nanmean(vals)
                        by_m[method_name][split][f"{metric}_std"] = _nanstd(vals)

            fold["by_cell_count"][fold_key] = by_m
            print(
                f"    M={m:>3} ({m / n_cells:>5.1%})  "
                f"Nearest unseen={by_m['NearestSpatial']['unseen']['cell_acc_mean']:.4f}  "
                f"IDW unseen={by_m['IDW']['unseen']['cell_acc_mean']:.4f}  "
                f"GP unseen={by_m['GP_RBF']['unseen']['cell_acc_mean']:.4f}"
            )

        out["folds"][held_out] = fold

    print("\n  Cross-room summary:")
    for requested_m in args.cell_counts:
        rows_for_m = []
        for room in rooms:
            fold = out["folds"][room]
            effective_m = min(int(requested_m), fold["n_target_cells"])
            key = str(effective_m)
            if key in fold["by_cell_count"]:
                rows_for_m.append(fold["by_cell_count"][key])
        if not rows_for_m:
            continue

        row = {
            "requested_cell_count": int(requested_m),
            "coverage_mean": _nanmean([r["coverage_mean"] for r in rows_for_m]),
            "n_support_mean": _nanmean([r["n_support_mean"] for r in rows_for_m]),
        }
        for source_key in ("SourceKNN_all", "SourceKNN_unseen"):
            for metric in ("cell_acc", "mean_error_m", "p90_error_m"):
                values = [r[source_key][f"{metric}_mean"] for r in rows_for_m]
                row[f"{source_key}_{metric}_mean"] = _nanmean(values)
                row[f"{source_key}_{metric}_std_cross_room"] = _nanstd(values)

        for method_name in methods:
            for split in ("all", "unseen"):
                for metric in ("cell_acc", "mean_error_m", "p90_error_m"):
                    values = [r[method_name][split][f"{metric}_mean"] for r in rows_for_m]
                    row[f"{method_name}_{split}_{metric}_mean"] = _nanmean(values)
                    row[f"{method_name}_{split}_{metric}_std_cross_room"] = _nanstd(values)
        out["summary"].append(row)
        print(
            f"    M={requested_m:>3} coverage={row['coverage_mean']:>5.1%}  "
            f"Source unseen={row['SourceKNN_unseen_cell_acc_mean']:.4f}  "
            f"Nearest={row['NearestSpatial_unseen_cell_acc_mean']:.4f}  "
            f"IDW={row['IDW_unseen_cell_acc_mean']:.4f}  "
            f"GP={row['GP_RBF_unseen_cell_acc_mean']:.4f}"
        )

    return out


def save_summary_csv(results: dict, path: Path) -> None:
    pd.DataFrame(results["summary"]).to_csv(path, index=False)


def save_plot(results: dict, path: Path) -> None:
    summary = pd.DataFrame(results["summary"])
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = summary["coverage_mean"] * 100.0
    series = [
        ("SourceKNN_unseen_cell_acc_mean", "SourceKNN", "#8c8c8c", "--"),
        ("NearestSpatial_unseen_cell_acc_mean", "NearestSpatial", "#9467bd", "-"),
        ("IDW_unseen_cell_acc_mean", "IDW", "#1f77b4", "-"),
        ("GP_RBF_unseen_cell_acc_mean", "GP/RBF", "#2ca02c", "-"),
    ]
    for col, label, color, linestyle in series:
        ax.plot(
            x,
            summary[col] * 100.0,
            marker="o",
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xlabel("Cellules cible calibrees (%)")
    ax.set_ylabel("Accuracy sur cellules non calibrees (%)")
    ax.set_title("Interpolation spatiale few-shot des cellules non calibrees")
    ax.set_xlim(0, min(100, max(5.0, float(x.max()) + 3.0)))
    ymax = max(5.0, float((summary[[s[0] for s in series]].max().max() * 100.0) + 2.0))
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot target-cell spatial interpolation")
    parser.add_argument("--cell-counts", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20, 40, 60, 80])
    parser.add_argument("--shots-per-cell", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--source-neighbors", type=int, default=7)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--gp-length-scale", type=float, default=0.75)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="fewshot_spatial_interpolation.json")
    parser.add_argument("--summary-csv-name", default="fewshot_spatial_interpolation_summary.csv")
    parser.add_argument("--plot-name", default="fewshot_spatial_interpolation_unseen_accuracy.png")
    args = parser.parse_args()

    results = run(args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / args.output_name
    csv_path = REPORT_DIR / args.summary_csv_name
    plot_path = REPORT_DIR / args.plot_name

    json_path.write_text(json.dumps(results, indent=2, default=str))
    save_summary_csv(results, csv_path)
    save_plot(results, plot_path)

    print(f"\nResults saved to {json_path}")
    print(f"Summary saved to {csv_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
