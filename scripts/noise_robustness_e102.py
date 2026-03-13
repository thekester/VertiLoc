#!/usr/bin/env python3
"""Robustness benchmark: controlled synthetic Gaussian noise on RSSI features.

Default protocol:
- LOCO over E102 campaigns (train on 5 campaigns, test on held-out one),
- Add Gaussian noise N(0, sigma^2) to RSSI test vectors,
- Report performance-vs-noise curves for several models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS: dict[str, CampaignSpec] = {
    "exp1_back_right": CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0),
    "exp2_front_right": CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0),
    "exp3_front_left": CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0),
    "exp4_back_left": CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0),
    "exp5_ground": CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6_1m50": CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCO E102 robustness curve under synthetic Gaussian RSSI noise.")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0],
        help="Gaussian noise std values in dBm (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of random repeats per sigma and held-out campaign (default: 7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for reproducibility (default: 7).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["knn", "rf", "nn_lknn"],
        choices=["knn", "rf", "nn_lknn"],
        help="Models to evaluate (default: knn rf nn_lknn).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="noise_robustness_e102",
        help="Output file prefix in reports/benchmarks (default: noise_robustness_e102).",
    )
    return parser.parse_args()


def localization_metrics(
    y_true_cells: np.ndarray,
    y_pred_cells: np.ndarray,
    cell_lookup: pd.DataFrame,
) -> dict[str, float]:
    pred_coords = cell_lookup.loc[y_pred_cells][["coord_x_m", "coord_y_m"]].to_numpy()
    true_coords = cell_lookup.loc[y_true_cells][["coord_x_m", "coord_y_m"]].to_numpy()
    errors = np.linalg.norm(true_coords - pred_coords, axis=1)
    return {
        "cell_accuracy": float((y_pred_cells == y_true_cells).mean()),
        "mean_error_m": float(errors.mean()),
        "p90_error_m": float(np.percentile(errors, 90)),
    }


def fit_nn_lknn(X_train: np.ndarray, y_train: pd.Series, random_state: int) -> EmbeddingKnnLocalizer:
    cfg = EmbeddingKnnConfig(
        hidden_layer_sizes=(48, 24),
        k_neighbors=5,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=400,
        tol=1e-3,
        random_state=random_state,
    )
    model = EmbeddingKnnLocalizer(config=cfg)
    model.fit(X_train, y_train)
    return model


def build_model(model_name: str, random_state: int):
    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "nn_lknn":
        return "nn_lknn"
    raise ValueError(f"Unknown model: {model_name}")


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading E102 campaigns...")
    frames: dict[str, pd.DataFrame] = {}
    for name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        df["campaign"] = name
        frames[name] = df

    all_df = pd.concat(frames.values(), ignore_index=True)
    cell_lookup = (
        all_df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )

    campaign_names = list(CAMPAIGNS.keys())
    rows: list[dict] = []

    for held_out in campaign_names:
        print(f"Held-out campaign: {held_out}")
        train_df = pd.concat([df for n, df in frames.items() if n != held_out], ignore_index=True)
        test_df = frames[held_out]

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["grid_cell"]
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test_df["grid_cell"].to_numpy()

        trained_models: dict[str, object] = {}
        for model_name in args.models:
            model = build_model(model_name, random_state=args.seed)
            if model_name == "nn_lknn":
                trained_models[model_name] = fit_nn_lknn(X_train, y_train, random_state=args.seed)
            else:
                model.fit(X_train, y_train)
                trained_models[model_name] = model

        for sigma in sorted(args.sigmas):
            for rep in range(args.repeats):
                rng = np.random.default_rng(args.seed + 1_000 * rep + 10_000 * int(sigma * 100) + hash(held_out) % 997)
                if sigma <= 0:
                    X_noisy = X_test
                else:
                    X_noisy = X_test + rng.normal(0.0, sigma, size=X_test.shape)

                for model_name, model in trained_models.items():
                    if model_name == "nn_lknn":
                        y_pred = np.asarray(model.predict(X_noisy))
                    else:
                        y_pred = np.asarray(model.predict(X_noisy))
                    metrics = localization_metrics(y_test, y_pred, cell_lookup)
                    rows.append(
                        {
                            "held_out": held_out,
                            "model": model_name,
                            "sigma_dbm": float(sigma),
                            "repeat": rep,
                            "n_test": len(y_test),
                            **metrics,
                        }
                    )

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby(["model", "sigma_dbm"], as_index=False)
        .agg(
            mean_cell_accuracy=("cell_accuracy", "mean"),
            std_cell_accuracy=("cell_accuracy", "std"),
            mean_error_m=("mean_error_m", "mean"),
            std_error_m=("mean_error_m", "std"),
            mean_p90_error_m=("p90_error_m", "mean"),
            std_p90_error_m=("p90_error_m", "std"),
        )
        .sort_values(["model", "sigma_dbm"])
    )
    summary_df["delta_acc_vs_clean"] = summary_df.groupby("model")["mean_cell_accuracy"].transform(
        lambda s: s - float(s.iloc[0])
    )
    return detail_df, summary_df


def save_curve(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.2))
    for model, sub in summary_df.groupby("model"):
        x = sub["sigma_dbm"].to_numpy()
        y = sub["mean_cell_accuracy"].to_numpy()
        yerr = sub["std_cell_accuracy"].fillna(0.0).to_numpy()
        plt.plot(x, y, marker="o", linewidth=2, label=model)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    plt.xlabel("Bruit gaussien RSSI (sigma, dBm)")
    plt.ylabel("Cell accuracy")
    plt.title("Robustesse au bruit synthétique (LOCO E102)")
    plt.grid(alpha=0.3)
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detail_df, summary_df = evaluate(args)

    prefix = args.out_prefix
    detail_path = REPORT_DIR / f"{prefix}_detailed.csv"
    summary_path = REPORT_DIR / f"{prefix}_summary.csv"
    json_path = REPORT_DIR / f"{prefix}.json"
    fig_path = REPORT_DIR / f"{prefix}_curve.png"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    save_curve(summary_df, fig_path)

    payload = {
        "config": {
            "sigmas": [float(v) for v in sorted(args.sigmas)],
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "models": list(args.models),
            "protocol": "LOCO E102, train clean / test noisy",
        },
        "summary": summary_df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:")
    print(f"  - {detail_path}")
    print(f"  - {summary_path}")
    print(f"  - {json_path}")
    print(f"  - {fig_path}")

    print("\nSummary (mean cell accuracy):")
    pivot = summary_df.pivot(index="sigma_dbm", columns="model", values="mean_cell_accuracy")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
