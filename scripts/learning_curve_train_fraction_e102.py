#!/usr/bin/env python3
"""Learning curve on E102: performance vs training-set size (20/40/60/80/100%).

Goal: quantify calibration effort on terrain by measuring how performance evolves
when only a fraction of the calibration train set is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.data import CampaignSpec, load_measurements  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS = [
    CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0),
    CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0),
    CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0),
    CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0),
    CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
]

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def localization_metrics(y_true: np.ndarray, y_pred: np.ndarray, cell_lookup: pd.DataFrame) -> dict[str, float]:
    pred_meta = cell_lookup.loc[y_pred]
    true_meta = cell_lookup.loc[y_true]
    pred_xy = pred_meta[["coord_x_m", "coord_y_m"]].to_numpy()
    true_xy = true_meta[["coord_x_m", "coord_y_m"]].to_numpy()
    errors = np.linalg.norm(pred_xy - true_xy, axis=1)
    return {
        "cell_accuracy": float((y_pred == y_true).mean()),
        "mean_error_m": float(errors.mean()),
        "p90_error_m": float(np.percentile(errors, 90)),
    }


def make_models(
    seed: int,
    model_names: list[str],
    *,
    rf_estimators: int,
    histgb_max_iter: int,
) -> list[tuple[str, object]]:
    model_names = [name.strip().lower() for name in model_names if name.strip()]
    builders = {
        "knn": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "rf": lambda: RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=14,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "histgb": lambda: HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=8,
            max_iter=histgb_max_iter,
            random_state=seed,
        ),
    }
    pretty = {"knn": "KNN", "rf": "RF", "histgb": "HistGB"}
    unknown = [name for name in model_names if name not in builders]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {sorted(builders)}")
    return [(pretty[name], builders[name]()) for name in model_names]


def build_report_snippet(summary_df: pd.DataFrame, test_size: float, repeats: int) -> str:
    lines: list[str] = []
    lines.append("## Learning curve E102 (train fraction)")
    lines.append("")
    lines.append(
        f"- Protocol: stratified random split (test={int(test_size * 100)}%), "
        f"repeated {repeats} time(s)."
    )
    lines.append("- Fractions explored: 20/40/60/80/100% of the train split.")
    lines.append("")
    for model_name, sub in summary_df.groupby("model", sort=True):
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Train fraction | Acc (mean+-std) | Mean error m (mean+-std) |")
        lines.append("|---:|---:|---:|")
        for _, row in sub.iterrows():
            pct = int(round(100 * float(row["train_fraction"])))
            acc_mean = float(row["cell_accuracy_mean"])
            acc_std = float(row["cell_accuracy_std"])
            err_mean = float(row["mean_error_m_mean"])
            err_std = float(row["mean_error_m_std"])
            lines.append(
                f"| {pct}% | {acc_mean:.4f} +- {acc_std:.4f} | "
                f"{err_mean:.4f} +- {err_std:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_learning_curves(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax_acc, ax_err = axes
    for model_name, sub in summary_df.groupby("model", sort=True):
        x = (sub["train_fraction"] * 100.0).to_numpy(dtype=float)
        acc = sub["cell_accuracy_mean"].to_numpy(dtype=float)
        acc_std = sub["cell_accuracy_std"].to_numpy(dtype=float)
        err = sub["mean_error_m_mean"].to_numpy(dtype=float)
        err_std = sub["mean_error_m_std"].to_numpy(dtype=float)

        ax_acc.plot(x, acc, marker="o", linewidth=2.0, label=model_name)
        ax_acc.fill_between(x, np.maximum(0.0, acc - acc_std), np.minimum(1.0, acc + acc_std), alpha=0.16)

        ax_err.plot(x, err, marker="o", linewidth=2.0, label=model_name)
        ax_err.fill_between(x, np.maximum(0.0, err - err_std), err + err_std, alpha=0.16)

    ax_acc.set_title("Cell accuracy vs train fraction")
    ax_acc.set_xlabel("Train fraction (%)")
    ax_acc.set_ylabel("Cell accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_xticks([20, 40, 60, 80, 100])
    ax_acc.grid(alpha=0.25)
    ax_acc.legend()

    ax_err.set_title("Mean localization error vs train fraction")
    ax_err.set_xlabel("Train fraction (%)")
    ax_err.set_ylabel("Mean error (m)")
    ax_err.set_xticks([20, 40, 60, 80, 100])
    ax_err.grid(alpha=0.25)
    ax_err.legend()

    fig.suptitle("E102 calibration effort learning curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="E102 learning curve over train-size fractions.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Global holdout ratio (default: 0.2).")
    parser.add_argument("--repeats", type=int, default=5, help="Number of random repeats (default: 5).")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed (default: 7).")
    parser.add_argument(
        "--models",
        type=str,
        default="knn,rf,histgb",
        help="Comma-separated model list among: knn,rf,histgb (default: knn,rf,histgb).",
    )
    parser.add_argument("--rf-estimators", type=int, default=180, help="RF n_estimators (default: 180).")
    parser.add_argument("--histgb-max-iter", type=int, default=160, help="HistGB max_iter (default: 160).")
    args = parser.parse_args()
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    if not model_names:
        raise ValueError("No models selected. Example: --models knn,rf")

    print("Loading E102 campaigns...")
    df = load_measurements(CAMPAIGNS).copy()
    cell_lookup = (
        df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
        .sort_index()
    )

    print(f"Dataset size: {len(df)} samples, {df['grid_cell'].nunique()} cells")

    rows: list[dict] = []
    for repeat in range(args.repeats):
        split_seed = args.seed + repeat
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=split_seed,
            stratify=df["grid_cell"],
        )
        y_test = test_df["grid_cell"].to_numpy()
        X_test = test_df[FEATURE_COLUMNS].to_numpy()

        print(f"\nRepeat {repeat + 1}/{args.repeats} | train={len(train_df)} test={len(test_df)}")

        for frac in FRACTIONS:
            if frac < 1.0:
                sub_train_df, _ = train_test_split(
                    train_df,
                    train_size=frac,
                    random_state=split_seed * 10 + int(frac * 100),
                    stratify=train_df["grid_cell"],
                )
            else:
                sub_train_df = train_df

            X_train = sub_train_df[FEATURE_COLUMNS].to_numpy()
            y_train = sub_train_df["grid_cell"]
            n_train = len(sub_train_df)
            calibration_effort_pct = 100.0 * n_train / len(train_df)

            for model_name, model in make_models(
                split_seed,
                model_names,
                rf_estimators=args.rf_estimators,
                histgb_max_iter=args.histgb_max_iter,
            ):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = localization_metrics(y_test, y_pred, cell_lookup)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "split_seed": split_seed,
                        "model": model_name,
                        "train_fraction": frac,
                        "n_train_samples": n_train,
                        "n_test_samples": len(test_df),
                        "calibration_effort_pct": calibration_effort_pct,
                        **metrics,
                    }
                )
            print(f"  frac={int(frac*100):>3d}% | n_train={n_train}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(rows).sort_values(["model", "train_fraction", "repeat"])
    runs_path = REPORT_DIR / "learning_curve_train_fraction_e102_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    summary_df = (
        runs_df.groupby(["model", "train_fraction"], as_index=False)
        .agg(
            n_repeats=("repeat", "count"),
            n_train_samples_mean=("n_train_samples", "mean"),
            cell_accuracy_mean=("cell_accuracy", "mean"),
            cell_accuracy_std=("cell_accuracy", "std"),
            mean_error_m_mean=("mean_error_m", "mean"),
            mean_error_m_std=("mean_error_m", "std"),
            p90_error_m_mean=("p90_error_m", "mean"),
            p90_error_m_std=("p90_error_m", "std"),
        )
        .sort_values(["model", "train_fraction"])
    )
    std_cols = [col for col in summary_df.columns if col.endswith("_std")]
    if std_cols:
        summary_df.loc[:, std_cols] = summary_df.loc[:, std_cols].fillna(0.0)
    summary_path = REPORT_DIR / "learning_curve_train_fraction_e102_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    fig_path = REPORT_DIR / "learning_curve_train_fraction_e102_curve.png"
    plot_learning_curves(summary_df, fig_path)

    snippet = build_report_snippet(summary_df, test_size=args.test_size, repeats=args.repeats)
    snippet_path = REPORT_DIR / "learning_curve_train_fraction_e102_report_snippet.md"
    snippet_path.write_text(snippet, encoding="utf-8")

    payload = {
        "config": {
            "dataset": "E102",
            "fractions": FRACTIONS,
            "test_size": args.test_size,
            "repeats": args.repeats,
            "seed": args.seed,
            "models": model_names,
            "rf_estimators": args.rf_estimators,
            "histgb_max_iter": args.histgb_max_iter,
            "features": FEATURE_COLUMNS,
        },
        "summary": summary_df.to_dict(orient="records"),
    }
    json_path = REPORT_DIR / "learning_curve_train_fraction_e102.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  - {runs_path}")
    print(f"  - {summary_path}")
    print(f"  - {json_path}")
    print(f"  - {fig_path}")
    print(f"  - {snippet_path}")

    print("\nQuick view (mean over repeats):")
    for model_name in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_name]
        print(f"\n{model_name}:")
        for _, row in sub.iterrows():
            frac = int(row["train_fraction"] * 100)
            print(
                f"  {frac:>3d}% -> acc={row['cell_accuracy_mean']:.4f}, "
                f"err={row['mean_error_m_mean']:.4f}m"
            )


if __name__ == "__main__":
    main()
