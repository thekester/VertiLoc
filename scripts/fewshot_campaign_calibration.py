"""Few-shot leave-one-campaign-out calibration baselines.

This is the LOCO counterpart of `fewshot_target_calibration.py`.
For each room and held-out campaign:

  - KNN_source is trained on the other campaigns of the same room.
  - K labeled samples per cell are drawn from the held-out campaign.
  - RawCentroid and TargetKNN are trained only on those support samples.
  - The remaining held-out samples are used as query/test samples.

This answers whether few-shot local calibration helps under campaign shift
inside the same room, not only under leave-one-room-out shift.
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


def load_room(room: str) -> pd.DataFrame:
    campaigns = filter_room_campaigns(room_filter=[room])
    frames = []
    for r, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = r
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
    n_neighbors: int,
) -> np.ndarray:
    X_sup_s = scaler.transform(X_support).astype(np.float32)
    X_q_s = scaler.transform(X_query).astype(np.float32)
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(X_sup_s)), weights="distance")
    knn.fit(X_sup_s, y_support)
    return knn.predict(X_q_s)


def run_room(room: str, args: argparse.Namespace) -> dict:
    print(f"\n=== Few-shot LOCO calibration — {room} ===")
    df = load_room(room)
    campaigns = sorted(df["campaign"].unique())
    lookup = _cell_lookup(df)

    room_results: dict[str, dict] = {}
    for fold_idx, held_out in enumerate(campaigns):
        train_df = df[df["campaign"] != held_out].copy()
        test_df = df[df["campaign"] == held_out].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["grid_cell"].to_numpy()
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test_df["grid_cell"].to_numpy()

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        short_name = str(held_out).split("/")[-1]
        print(f"\n  Held-out campaign: {short_name}  train={len(X_train)} test={len(X_test)}")

        knn_source = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn_source.fit(X_train_s, y_train)
        fold_results: dict[str, dict] = {
            "KNN_source_LOCO": _metrics(y_test, knn_source.predict(X_test_s), lookup)
        }
        print(f"    KNN source campaigns: {fold_results['KNN_source_LOCO']['cell_acc']:.4f}")

        cells = np.unique(y_test)
        for k in args.k_values:
            centroid_accs, centroid_errs = [], []
            knn_accs, knn_errs = [], []
            n_supports, n_queries = [], []
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

                if not qry_idx:
                    continue

                X_sup, y_sup = X_test[sup_idx], y_test[sup_idx]
                X_q, y_q = X_test[qry_idx], y_test[qry_idx]
                n_supports.append(len(sup_idx))
                n_queries.append(len(qry_idx))

                pred_c = raw_centroid_predict(scaler, X_sup, y_sup, X_q)
                m_c = _metrics(y_q, pred_c, lookup)
                centroid_accs.append(m_c["cell_acc"])
                centroid_errs.append(m_c["mean_error_m"])

                pred_k = target_knn_predict(
                    scaler, X_sup, y_sup, X_q, n_neighbors=args.knn_neighbors
                )
                m_k = _metrics(y_q, pred_k, lookup)
                knn_accs.append(m_k["cell_acc"])
                knn_errs.append(m_k["mean_error_m"])

            fold_results[f"RawCentroid_K{k}"] = {
                "cell_acc": float(np.mean(centroid_accs)) if centroid_accs else float("nan"),
                "cell_acc_std": float(np.std(centroid_accs)) if centroid_accs else float("nan"),
                "mean_error_m": float(np.mean(centroid_errs)) if centroid_errs else float("nan"),
                "n_support_mean": float(np.mean(n_supports)) if n_supports else 0.0,
                "n_query_mean": float(np.mean(n_queries)) if n_queries else 0.0,
            }
            fold_results[f"TargetKNN_K{k}"] = {
                "cell_acc": float(np.mean(knn_accs)) if knn_accs else float("nan"),
                "cell_acc_std": float(np.std(knn_accs)) if knn_accs else float("nan"),
                "mean_error_m": float(np.mean(knn_errs)) if knn_errs else float("nan"),
                "n_support_mean": float(np.mean(n_supports)) if n_supports else 0.0,
                "n_query_mean": float(np.mean(n_queries)) if n_queries else 0.0,
            }
            print(
                f"    K={k:>2}  centroid={fold_results[f'RawCentroid_K{k}']['cell_acc']:.4f}"
                f"  targetKNN={fold_results[f'TargetKNN_K{k}']['cell_acc']:.4f}"
                f"  err={fold_results[f'TargetKNN_K{k}']['mean_error_m']:.3f}m"
            )

        room_results[short_name] = fold_results

    return room_results


def summarize(all_results: dict[str, dict], k_values: list[int]) -> dict:
    summary: dict[str, dict] = {}
    for room, folds in all_results.items():
        summary[room] = {}
        source_accs = [
            fold["KNN_source_LOCO"]["cell_acc"]
            for fold in folds.values()
            if "KNN_source_LOCO" in fold
        ]
        summary[room]["KNN_source_LOCO"] = {
            "cell_acc_mean": float(np.mean(source_accs)) if source_accs else float("nan")
        }
        for method in ("RawCentroid", "TargetKNN"):
            for k in k_values:
                key = f"{method}_K{k}"
                vals = [fold[key] for fold in folds.values() if key in fold]
                accs = [v["cell_acc"] for v in vals if not np.isnan(v["cell_acc"])]
                errs = [v["mean_error_m"] for v in vals if not np.isnan(v["mean_error_m"])]
                summary[room][key] = {
                    "cell_acc_mean": float(np.mean(accs)) if accs else float("nan"),
                    "cell_acc_std": float(np.std(accs)) if accs else float("nan"),
                    "mean_error_m": float(np.mean(errs)) if errs else float("nan"),
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot LOCO campaign calibration baselines")
    parser.add_argument("--rooms", nargs="+", default=["E102"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 3, 5, 10, 15, 20, 25])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="fewshot_campaign_calibration.json")
    args = parser.parse_args()

    all_results: dict[str, dict] = {
        "config": vars(args),
        "folds": {},
        "summary": {},
    }
    for room in args.rooms:
        all_results["folds"][room] = run_room(room, args)
    all_results["summary"] = summarize(all_results["folds"], args.k_values)

    print("\n=== Summary ===")
    for room, summary in all_results["summary"].items():
        print(f"\n  {room}")
        print(f"    KNN source: {summary['KNN_source_LOCO']['cell_acc_mean']:.4f}")
        for k in args.k_values:
            v = summary[f"TargetKNN_K{k}"]
            print(f"    TargetKNN_K{k:<2} acc={v['cell_acc_mean']:.4f}  err={v['mean_error_m']:.3f}m")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / args.output_name
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
