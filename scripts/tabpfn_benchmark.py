"""TabNet benchmark for RSSI indoor localization.

TabNet (Arik & Pfister, 2021) uses sequential attention to select relevant
features at each decision step — well suited for small tabular datasets.

Compares TabNet against KNN and RF baselines on:
  - Room-aware multi-room (80/20, stratified by cell)
  - Intra-room E102 (80/20)
  - LORO (room-agnostic, generalization)

Also optionally benchmarks TabPFN v2 if TABPFN_TOKEN env var is set.

Usage:
    .venv/bin/python scripts/tabpfn_benchmark.py
    .venv/bin/python scripts/tabpfn_benchmark.py --skip-loro
    TABPFN_TOKEN=<token> .venv/bin/python scripts/tabpfn_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

warnings.filterwarnings("ignore")
REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


def _localization_metrics(y_true, y_pred, lookup: pd.DataFrame) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    pred_coords = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    true_coords = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pred_coords).any(axis=1) | np.isnan(true_coords).any(axis=1))
    errors = np.linalg.norm(pred_coords[mask] - true_coords[mask], axis=1)
    return {
        "cell_acc": acc,
        "mean_error_m": float(errors.mean()) if len(errors) else float("nan"),
        "p90_error_m": float(np.percentile(errors, 90)) if len(errors) else float("nan"),
        "n_test": int(len(y_true)),
    }


def _subsample_train(
    X: np.ndarray,
    y: np.ndarray,
    max_n: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_n:
        return X, y
    try:
        _, idx = train_test_split(np.arange(len(y)), train_size=max_n,
                                  random_state=random_state, stratify=y)
    except ValueError:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx]


def load_multiroom(rooms: list[str] | None = None) -> pd.DataFrame:
    campaigns = filter_room_campaigns(room_filter=rooms)
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = room
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    room_ohe = pd.get_dummies(df["room"], prefix="room")
    return pd.concat([df, room_ohe], axis=1)


def build_X(df: pd.DataFrame, include_room: bool = True) -> np.ndarray:
    base = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not include_room:
        return base
    room_cols = sorted(c for c in df.columns if c.startswith("room_"))
    if not room_cols:
        return base
    return np.hstack([base, df[room_cols].to_numpy(dtype=float)])


# ---------------------------------------------------------------------------
# TabNet wrapper
# ---------------------------------------------------------------------------

def tabnet_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    max_epochs: int = 100,
    patience: int = 15,
    n_steps: int = 3,
    n_d: int = 16,
    n_a: int = 16,
    random_state: int = SEED,
) -> np.ndarray:
    """Fit TabNetClassifier and return string predictions for X_test."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    # Small validation split for early stopping
    X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
        X_tr, y_enc, test_size=0.1, random_state=random_state, stratify=y_enc
    ) if len(np.unique(y_enc)) < len(y_enc) * 0.9 else (X_tr, X_tr[:10], y_enc, y_enc[:10])

    clf = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        seed=random_state,
        verbose=0,
    )
    clf.fit(
        X_tr_fit, y_tr_fit,
        eval_set=[(X_val, y_val)],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=256,
        virtual_batch_size=128,
    )
    pred_enc = clf.predict(X_te)
    return le.inverse_transform(pred_enc)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

def run_protocol(
    label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lookup: pd.DataFrame,
    *,
    max_train_tabpfn: int = 2000,
    tabnet_epochs: int = 100,
    n_estimators_tabpfn: int = 4,
    skip_tabpfn: bool = False,
) -> dict:
    results = {}
    n_tr = len(X_train)

    print(f"\n  [{label}] n_train={n_tr}  n_test={len(X_test)}")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, y_train)
    results["KNN"] = _localization_metrics(y_test, knn.predict(X_test), lookup)
    print(f"    KNN:    acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # RF
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    results["RF"] = _localization_metrics(y_test, rf.predict(X_test), lookup)
    print(f"    RF:     acc={results['RF']['cell_acc']:.4f}  err={results['RF']['mean_error_m']:.3f}m")

    # TabNet
    if HAS_TABNET:
        try:
            pred_tn = tabnet_predict(X_train, y_train, X_test, max_epochs=tabnet_epochs)
            results["TabNet"] = _localization_metrics(y_test, pred_tn, lookup)
            print(f"    TabNet: acc={results['TabNet']['cell_acc']:.4f}  err={results['TabNet']['mean_error_m']:.3f}m")
        except Exception as exc:
            print(f"    TabNet: FAILED — {exc}")
            results["TabNet"] = {"cell_acc": float("nan"), "error": str(exc)}
    else:
        print("    TabNet: not installed")

    # TabPFN (requires TABPFN_TOKEN env var)
    if HAS_TABPFN and not skip_tabpfn and os.environ.get("TABPFN_TOKEN"):
        X_sub, y_sub = _subsample_train(X_train, y_train, max_train_tabpfn, SEED)
        try:
            clf = TabPFNClassifier(
                n_estimators=n_estimators_tabpfn,
                ignore_pretraining_limits=True,
                random_state=SEED,
                device="cpu",
            )
            clf.fit(X_sub, y_sub)
            pred_pf = clf.predict(X_test)
            results["TabPFN"] = _localization_metrics(y_test, pred_pf, lookup)
            results["TabPFN"]["n_train_used"] = int(len(X_sub))
            print(f"    TabPFN: acc={results['TabPFN']['cell_acc']:.4f}")
        except Exception as exc:
            print(f"    TabPFN: FAILED — {exc}")
            results["TabPFN"] = {"cell_acc": float("nan"), "error": str(exc)}
    elif HAS_TABPFN and not skip_tabpfn:
        print("    TabPFN: set TABPFN_TOKEN env var to enable (requires account at ux.priorlabs.ai)")

    return results


def run_room_aware(args) -> dict:
    print("\n=== Room-aware benchmark (multi-room, 80/20 stratified) ===")
    df = load_multiroom()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"]
    )
    lookup = _cell_lookup(df)
    return run_protocol(
        "room_aware",
        build_X(train_df, include_room=True),
        train_df["grid_cell"].to_numpy(),
        build_X(test_df, include_room=True),
        test_df["grid_cell"].to_numpy(),
        lookup,
        tabnet_epochs=args.tabnet_epochs,
    )


def run_intraroom_e102(args) -> dict:
    print("\n=== Intra-room E102 benchmark (80/20) ===")
    df = load_multiroom(rooms=["E102"])
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"]
    )
    lookup = _cell_lookup(df)
    return run_protocol(
        "E102_intraroom",
        build_X(train_df, include_room=False),
        train_df["grid_cell"].to_numpy(),
        build_X(test_df, include_room=False),
        test_df["grid_cell"].to_numpy(),
        lookup,
        tabnet_epochs=args.tabnet_epochs,
    )


def run_loro(args) -> dict:
    print("\n=== LORO benchmark (Leave-One-Room-Out) ===")
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    all_folds: dict[str, dict] = {}

    for held_out in rooms:
        train_df = df[df["room"] != held_out]
        test_df = df[df["room"] == held_out]
        fold = run_protocol(
            f"LORO held_out={held_out}",
            build_X(train_df, include_room=False),
            train_df["grid_cell"].to_numpy(),
            build_X(test_df, include_room=False),
            test_df["grid_cell"].to_numpy(),
            lookup,
            tabnet_epochs=max(30, args.tabnet_epochs // 3),
        )
        all_folds[held_out] = fold

    # Aggregate
    model_names = list(next(iter(all_folds.values())).keys())
    averages = {}
    print("\n  LORO averages:")
    for m in model_names:
        accs = [all_folds[r][m]["cell_acc"] for r in rooms if m in all_folds[r]]
        valid = [a for a in accs if not np.isnan(a)]
        avg_acc = float(np.mean(valid)) if valid else float("nan")
        averages[m] = {"cell_acc_mean": avg_acc, "cell_acc_std": float(np.std(valid)) if valid else float("nan")}
        print(f"    {m:<10}: {avg_acc:.4f} ± {averages[m]['cell_acc_std']:.4f}")

    return {"folds": all_folds, "averages": averages}


def main():
    parser = argparse.ArgumentParser(description="TabNet benchmark for RSSI localization")
    parser.add_argument("--tabnet-epochs", type=int, default=100,
                        help="Max training epochs for TabNet.")
    parser.add_argument("--skip-loro", action="store_true",
                        help="Skip LORO benchmark (slow for TabNet).")
    parser.add_argument("--skip-room-aware", action="store_true")
    parser.add_argument("--skip-e102", action="store_true")
    args = parser.parse_args()

    all_results = {}

    if not args.skip_room_aware:
        all_results["room_aware"] = run_room_aware(args)

    if not args.skip_e102:
        all_results["e102_intraroom"] = run_intraroom_e102(args)

    if not args.skip_loro:
        all_results["loro"] = run_loro(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "tabnet_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n=== SUMMARY ===")
    for protocol, res in all_results.items():
        if protocol == "loro":
            print(f"\n{protocol} (averages):")
            for m, v in res.get("averages", {}).items():
                print(f"  {m:<10} acc={v['cell_acc_mean']:.4f} ± {v['cell_acc_std']:.4f}")
        else:
            print(f"\n{protocol}:")
            for m, v in res.items():
                print(f"  {m:<10} acc={v.get('cell_acc', float('nan')):.4f}  "
                      f"err={v.get('mean_error_m', float('nan')):.3f}m  "
                      f"p90={v.get('p90_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
