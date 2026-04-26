"""Few-shot target-room calibration baselines.

This benchmark isolates the useful part observed in ProtoNet experiments:
having a small number of labeled samples in the target room. It evaluates
simple non-parametric baselines with K support samples per cell:

  - RawCentroid: nearest centroid in standardized RSSI space
  - TargetKNN: KNN trained only on target-room support samples

The protocol is LORO-like: each room is held out, the source rooms only provide
the scaler and the no-calibration KNN baseline, while support/query samples are
drawn from the held-out room.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

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

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
)


def load_multiroom() -> pd.DataFrame:
    campaigns = filter_room_campaigns()
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = room
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, lookup: pd.DataFrame) -> dict:
    acc = float((y_true == y_pred).mean())
    pred_c = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    true_c = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pred_c).any(1) | np.isnan(true_c).any(1))
    err = float(np.linalg.norm(pred_c[mask] - true_c[mask], axis=1).mean()) if mask.sum() else float("nan")
    return {"cell_acc": acc, "mean_error_m": err, "n": int(len(y_true))}


def raw_centroid_predict(
    scaler: StandardScaler,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    X_sup_s = scaler.transform(X_support).astype(np.float32)
    X_q_s = scaler.transform(X_query).astype(np.float32)
    classes = np.unique(y_support)
    prototypes = np.stack([X_sup_s[y_support == c].mean(axis=0) for c in classes])
    dists = np.linalg.norm(X_q_s[:, None, :] - prototypes[None, :, :], axis=2)
    return classes[dists.argmin(axis=1)]


def target_knn_predict(
    scaler: StandardScaler,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    *,
    n_neighbors: int = 3,
) -> np.ndarray:
    X_sup_s = scaler.transform(X_support).astype(np.float32)
    X_q_s = scaler.transform(X_query).astype(np.float32)
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(X_sup_s)), weights="distance")
    knn.fit(X_sup_s, y_support)
    return knn.predict(X_q_s)


def run(args: argparse.Namespace) -> dict:
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()
    room_arr = df["room"].to_numpy()

    results: dict[str, dict] = {"config": vars(args), "folds": {}, "averages": {}}

    print("\n=== Few-shot target calibration LORO ===")
    for fold_idx, held_out in enumerate(rooms):
        print(f"\n  Fold: held_out={held_out}")
        train_mask = room_arr != held_out
        test_mask = room_arr == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        knn_source = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn_source.fit(X_train_s, y_train)
        fold_results: dict[str, dict] = {
            "KNN_vanilla_LORO": _metrics(y_test, knn_source.predict(X_test_s), lookup)
        }
        print(f"    KNN no target labels: {fold_results['KNN_vanilla_LORO']['cell_acc']:.4f}")

        cells = np.unique(y_test)
        for k in args.k_values:
            centroid_accs, centroid_errs = [], []
            knn_accs, knn_errs = [], []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + fold_idx * 10000 + trial * 1000)
                sup_idx, qry_idx = [], []
                for cell in cells:
                    cell_idx = np.where(y_test == cell)[0]
                    if len(cell_idx) <= 1:
                        continue
                    k_eff = min(k, len(cell_idx) - 1)
                    perm = rng.permutation(len(cell_idx))
                    sup_idx.extend(cell_idx[perm[:k_eff]])
                    qry_idx.extend(cell_idx[perm[k_eff:]])

                X_sup, y_sup = X_test[sup_idx], y_test[sup_idx]
                X_q, y_q = X_test[qry_idx], y_test[qry_idx]

                pred_c = raw_centroid_predict(scaler, X_sup, y_sup, X_q)
                m_c = _metrics(y_q, pred_c, lookup)
                centroid_accs.append(m_c["cell_acc"])
                centroid_errs.append(m_c["mean_error_m"])

                pred_k = target_knn_predict(scaler, X_sup, y_sup, X_q, n_neighbors=args.knn_neighbors)
                m_k = _metrics(y_q, pred_k, lookup)
                knn_accs.append(m_k["cell_acc"])
                knn_errs.append(m_k["mean_error_m"])

            fold_results[f"RawCentroid_K{k}"] = {
                "cell_acc": float(np.mean(centroid_accs)),
                "cell_acc_std": float(np.std(centroid_accs)),
                "mean_error_m": float(np.mean(centroid_errs)),
            }
            fold_results[f"TargetKNN_K{k}"] = {
                "cell_acc": float(np.mean(knn_accs)),
                "cell_acc_std": float(np.std(knn_accs)),
                "mean_error_m": float(np.mean(knn_errs)),
            }
            print(
                f"    K={k:>2}  centroid={fold_results[f'RawCentroid_K{k}']['cell_acc']:.4f}"
                f"  targetKNN={fold_results[f'TargetKNN_K{k}']['cell_acc']:.4f}"
                f"  err={fold_results[f'TargetKNN_K{k}']['mean_error_m']:.3f}m"
            )

        results["folds"][held_out] = fold_results

    for method in ("RawCentroid", "TargetKNN"):
        for k in args.k_values:
            key = f"{method}_K{k}"
            accs = [results["folds"][r][key]["cell_acc"] for r in rooms]
            errs = [results["folds"][r][key]["mean_error_m"] for r in rooms]
            results["averages"][key] = {
                "cell_acc_mean": float(np.mean(accs)),
                "cell_acc_std": float(np.std(accs)),
                "mean_error_m": float(np.mean(errs)),
            }

    source_accs = [results["folds"][r]["KNN_vanilla_LORO"]["cell_acc"] for r in rooms]
    results["knn_loro_baseline_acc_mean"] = float(np.mean(source_accs))

    print("\n  Cross-room averages:")
    print(f"    KNN no target labels: {results['knn_loro_baseline_acc_mean']:.4f}")
    for k in args.k_values:
        v = results["averages"][f"TargetKNN_K{k}"]
        print(f"    TargetKNN_K{k:<2} acc={v['cell_acc_mean']:.4f}  err={v['mean_error_m']:.3f}m")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot target-room calibration baselines")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 3, 5, 10, 15, 20, 25])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="fewshot_target_calibration.json")
    args = parser.parse_args()

    results = run(args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / args.output_name
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
