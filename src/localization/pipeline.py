"""CLI pipeline for training/evaluating the RSSI localization model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from .data import (
    CampaignSpec,
    DEFAULT_CELL_HEIGHT_M,
    DEFAULT_CELL_WIDTH_M,
    default_campaign_specs,
    load_measurements,
)
from .embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer

# Only use RSSI-derived signals as inputs. Router distance stays out of features to
# avoid leaking the target context; the model must infer distance implicitly.
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
META_COLUMNS = ["grid_x", "grid_y", "coord_x_m", "coord_y_m"]
REPORT_DIR = Path("reports")
MODEL_PATH = REPORT_DIR / "localizer.joblib"
CONFUSION_PATH = REPORT_DIR / "confusion_matrix.png"
ROC_PATH = REPORT_DIR / "roc_micro_macro.png"
CONFUSION_CELL_DISTANCE_PATH = REPORT_DIR / "confusion_cell_distance_with_logreg.png"
CONFUSION_CELL_DISTANCE_BASELINE_PATH = REPORT_DIR / "confusion_cell_distance_without_logreg.png"
PREDICTIONS_PATH = REPORT_DIR / "predictions.csv"


def parse_campaign_args(entries: Iterable[str]) -> list[CampaignSpec]:
    """Translate CLI arguments into CampaignSpec objects."""
    specs: list[CampaignSpec] = []
    for entry in entries:
        if ":" in entry:
            folder, distance = entry.split(":", 1)
            specs.append(CampaignSpec(Path(folder), float(distance)))
        else:
            specs.append(CampaignSpec(Path(entry)))
    if not specs:
        specs = default_campaign_specs()
    return specs


def compute_metrics(
    y_true,
    y_pred,
    meta_test: np.ndarray,
    cell_lookup,
    *,
    y_proba=None,
    class_order=None,
    y_true_binarized=None,
) -> dict:
    """Return a dictionary summarising grid accuracy and metric distances."""
    accuracy = accuracy_score(y_true, y_pred)
    pred_meta = cell_lookup.loc[y_pred]
    pred_coords = pred_meta[["coord_x_m", "coord_y_m"]].to_numpy()
    pred_grid = pred_meta[["grid_x", "grid_y"]].to_numpy()

    true_coords = meta_test[:, 2:4]
    true_grid = meta_test[:, 0:2]

    errors_m = np.linalg.norm(true_coords - pred_coords, axis=1)
    grid_errors = np.linalg.norm(true_grid - pred_grid, axis=1)
    cell_hits = (grid_errors == 0).mean()

    percentiles = {
        "p50": float(np.percentile(errors_m, 50)),
        "p75": float(np.percentile(errors_m, 75)),
        "p90": float(np.percentile(errors_m, 90)),
        "p95": float(np.percentile(errors_m, 95)),
    }

    thresholds = [0.25, 0.5, 1.0]
    # Convenience metrics: share of samples that fall within pre-defined distance bands.
    within = {f"within_{thr:.2f}m": float((errors_m <= thr).mean()) for thr in thresholds}

    metrics = {}
    if y_proba is not None:
        confidences = y_proba.max(axis=1)
        metrics["avg_confidence"] = float(confidences.mean())

    if (
        y_proba is not None
        and class_order is not None
        and y_true_binarized is not None
        and len(class_order) > 1
    ):
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_true_binarized, y_proba, multi_class="ovr", average="macro")
        )
        metrics["roc_auc_micro"] = float(
            roc_auc_score(y_true_binarized, y_proba, multi_class="ovr", average="micro")
        )
        per_class_auc = roc_auc_score(
            y_true_binarized, y_proba, multi_class="ovr", average=None
        )
        metrics["roc_auc_per_class"] = {
            cls: float(score) for cls, score in zip(class_order, per_class_auc)
        }

    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "mean_distance_m": float(errors_m.mean()),
        "median_distance_m": percentiles["p50"],
        "percentiles": percentiles,
        "cell_hit_rate": float(cell_hits),
        "mean_grid_error_cells": float(grid_errors.mean()),
        **within,
        **metrics,
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def save_embedding_projection(localizer: EmbeddingKnnLocalizer, df, feature_cols, out_path: Path) -> None:
    """Persist a PCA projection of the embedding space for qualitative inspection."""
    embeddings = localizer.transform(df[feature_cols])
    reducer = PCA(n_components=2, random_state=0)
    proj = reducer.fit_transform(embeddings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=df["grid_y"], cmap="viridis", s=10, alpha=0.8)
    plt.colorbar(scatter, label="Grid column (Y index)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Embedding space projection (color = grid column)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str = "Confusion matrix (cells)") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_curves(y_true_binarized, y_proba, classes, out_path: Path) -> None:
    if len(classes) < 2:
        return

    # Micro-average considers every prediction, macro-average treats each class equally.
    fpr_micro, tpr_micro, _ = roc_curve(y_true_binarized.ravel(), y_proba.ravel())
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(mean_fpr)

    for idx in range(len(classes)):
        fpr_c, tpr_c, _ = roc_curve(y_true_binarized[:, idx], y_proba[:, idx])
        mean_tpr += np.interp(mean_fpr, fpr_c, tpr_c)

    mean_tpr /= len(classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr_micro,
        tpr_micro,
        label="Micro-average ROC",
        color="blue",
    )
    ax.plot(mean_fpr, mean_tpr, label="Macro-average ROC", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC micro/macro (multi-classe)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_prediction_report(
    y_true,
    y_pred,
    y_proba,
    meta_test,
    cell_lookup,
    neighbor_distances,
    neighbor_labels,
    true_distances,
    pred_distances,
    pred_distance_proba,
    out_path: Path,
) -> None:
    confidences = y_proba.max(axis=1)
    if y_proba.shape[1] > 1:
        sorted_proba = np.sort(y_proba, axis=1)
        confidence_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    else:
        confidence_margin = np.zeros_like(confidences)

    pred_meta = cell_lookup.loc[y_pred]
    pred_coords = pred_meta[["coord_x_m", "coord_y_m"]].to_numpy()
    true_coords = meta_test[:, 2:4]
    errors_m = np.linalg.norm(true_coords - pred_coords, axis=1)

    report_df = pd.DataFrame(
        {
            "true_cell": y_true,
            "pred_cell": y_pred,
            "true_grid_x": meta_test[:, 0],
            "true_grid_y": meta_test[:, 1],
            "pred_grid_x": pred_meta["grid_x"].to_numpy(),
            "pred_grid_y": pred_meta["grid_y"].to_numpy(),
            "true_coord_x_m": true_coords[:, 0],
            "true_coord_y_m": true_coords[:, 1],
            "pred_coord_x_m": pred_coords[:, 0],
            "pred_coord_y_m": pred_coords[:, 1],
            "error_m": errors_m,
            "confidence": confidences,
            "confidence_margin": confidence_margin,
            "true_router_distance_m": true_distances,
            "pred_router_distance_m": pred_distances,
            "distance_confidence": pred_distance_proba,
        }
    )

    if neighbor_distances is not None and neighbor_labels is not None:
        for k in range(neighbor_labels.shape[1]):
            report_df[f"neighbor_{k+1}_cell"] = neighbor_labels[:, k]
            report_df[f"neighbor_{k+1}_distance"] = neighbor_distances[:, k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)


def save_model(localizer: EmbeddingKnnLocalizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(localizer, path)


def _log_split_diagnostics(df: pd.DataFrame, train_idx, test_idx, labels: pd.Series) -> None:
    """Print a concise summary of the 80/20 split to spot leakage issues quickly."""
    train_cells = labels.loc[train_idx].value_counts().sort_index()
    test_cells = labels.loc[test_idx].value_counts().sort_index()
    shared = set(train_idx) & set(test_idx)

    print(f"Split sizes -> train: {len(train_idx)} | test: {len(test_idx)} (target ratio ~{len(test_idx) / len(df):.2f})")
    print(f"Unique cells -> train: {train_cells.shape[0]} | test: {test_cells.shape[0]} | union: {labels.nunique()}")
    print(f"Train/test index overlap: {len(shared)} (should be 0)")
    print("Per-cell counts (train | test):")
    for cell in sorted(labels.unique()):
        tr = int(train_cells.get(cell, 0))
        te = int(test_cells.get(cell, 0))
        print(f"  {cell}: {tr:3d} | {te:3d}")


def _log_embedding_preview(
    localizer: EmbeddingKnnLocalizer,
    sample_idx,
    feature_df: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Show a tiny example of raw features -> embedding for interpretability."""
    indices = list(sample_idx)
    if len(indices) == 0:
        return
    take = indices[:3]
    raw = feature_df.loc[take].to_numpy()
    emb = localizer.transform(raw)
    print("\nEmbedding preview (raw RSSI -> embedding vector):")
    for row_idx, raw_vec, emb_vec in zip(take, raw, emb):
        print(f"- sample idx {row_idx}, label={y.loc[row_idx]}:")
        print(f"  raw: {np.array2string(raw_vec, precision=2, separator=', ')}")
        print(f"  emb: {np.array2string(emb_vec, precision=3, separator=', ')}")


def run_pipeline(args: argparse.Namespace) -> dict:
    """Main orchestration: data load, split, fit, evaluation, and artifact export."""
    specs = parse_campaign_args(args.campaign)
    df = load_measurements(
        specs,
        cell_width_m=args.cell_width_m,
        cell_height_m=args.cell_height_m,
    )
    print(f"Loaded {len(df)} samples from {len(specs)} campaign(s).")

    # Print a quick overview of the grid to ensure every cell has data.
    summary = df.groupby("grid_cell").agg({"Signal": ["mean", "std"], "grid_x": "first", "grid_y": "first"})
    print("Grid coverage:\n", summary.head())

    feature_df = df[FEATURE_COLUMNS]
    labels = df["grid_cell"]
    distances = df["router_distance_m"]
    meta = df[META_COLUMNS]

    # Stratified split keeps each cell represented in both train and test folds.
    # Split on the dataframe index so we can explicitly verify the two subsets do not overlap.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    if set(train_idx) & set(test_idx):
        raise RuntimeError("Train/test split leakage detected: overlapping sample indices.")

    X_train = feature_df.loc[train_idx].to_numpy()
    X_test = feature_df.loc[test_idx].to_numpy()
    y_train = labels.loc[train_idx].to_numpy()
    y_test = labels.loc[test_idx].to_numpy()
    distance_train = distances.loc[train_idx].to_numpy()
    distance_test = distances.loc[test_idx].to_numpy()
    meta_test = meta.loc[test_idx].to_numpy()
    _log_split_diagnostics(df, train_idx, test_idx, labels)

    config = EmbeddingKnnConfig(
        hidden_layer_sizes=tuple(args.hidden_layers),
        k_neighbors=args.k_neighbors,
        activation=args.activation,
        max_iter=args.max_iter,
        learning_rate_init=args.learning_rate,
        alpha=args.alpha,
        random_state=args.random_state,
    )
    # Fit the encoder + KNN. The embeddings for the training set are cached and
    # reused later when we call explain() for every test point.
    localizer = EmbeddingKnnLocalizer(config=config).fit(X_train, y_train)
    _log_embedding_preview(localizer, test_idx if len(test_idx) > 0 else train_idx, feature_df, labels)

    # Auxiliary head: predict router distance (campagne 2 m vs 4 m) à partir des embeddings.
    # On recycle l'espace latent appris pour la localisation sans jamais réintroduire la distance comme feature brute.
    train_embeddings = localizer.train_embeddings_
    test_embeddings = localizer.transform(X_test)
    distance_clf = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=None)
    distance_clf.fit(train_embeddings, distance_train)
    distance_pred = distance_clf.predict(test_embeddings)
    distance_proba = distance_clf.predict_proba(test_embeddings)
    distance_confidence = distance_proba.max(axis=1)
    # Persist the distance head inside the localizer so downstream consumers can reuse it at inference time.
    localizer.distance_clf_ = distance_clf
    localizer.distance_classes_ = distance_clf.classes_

    y_pred = localizer.predict(X_test)
    y_proba = localizer.predict_proba(X_test)
    class_order = list(localizer.knn_.classes_)
    y_test_binarized = label_binarize(y_test, classes=class_order)
    neighbor_distances, neighbor_labels = localizer.explain(X_test, top_k=args.explain_k)

    # Baseline: distance = mode observée pour la cellule prédite (calculée sur le train).
    train_distance_mode = (
        pd.DataFrame({"cell": labels.loc[train_idx], "distance": distance_train})
        .groupby("cell")["distance"]
        .agg(lambda s: s.mode().iloc[0])
    )
    global_distance_mode = float(pd.Series(distance_train).mode().iloc[0])
    distance_pred_baseline = np.array(
        [train_distance_mode.get(cell, global_distance_mode) for cell in y_pred]
    )

    cell_lookup = (
        df[["grid_cell", "grid_x", "grid_y", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    metrics = compute_metrics(
        y_test,
        y_pred,
        meta_test,
        cell_lookup,
        y_proba=y_proba,
        class_order=class_order,
        y_true_binarized=y_test_binarized,
    )
    metrics["router_distance_accuracy"] = float(accuracy_score(distance_test, distance_pred))
    metrics["router_distance_accuracy_baseline"] = float(accuracy_score(distance_test, distance_pred_baseline))
    metrics["router_distance_classes"] = [float(c) for c in sorted(np.unique(distances))]
    print(json.dumps(metrics, indent=2))
    print(
        f"Router distance accuracy (LogReg on embeddings): {metrics['router_distance_accuracy']:.3f} | "
        f"baseline (mode per predicted cell): {metrics['router_distance_accuracy_baseline']:.3f}"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, REPORT_DIR / "latest_metrics.json")
    save_embedding_projection(localizer, df, FEATURE_COLUMNS, REPORT_DIR / "embedding_pca.png")
    save_confusion_matrix(y_test, y_pred, class_order, Path(args.confusion_matrix))
    # Confusion matrix that jointly evaluates predicted cell + predicted router distance.
    true_combo = np.array([f"{cell}|{dist}" for cell, dist in zip(y_test, distance_test)])
    pred_combo = np.array([f"{cell}|{dist}" for cell, dist in zip(y_pred, distance_pred)])
    combo_labels = sorted(np.unique(np.concatenate([true_combo, pred_combo])))
    save_confusion_matrix(
        true_combo,
        pred_combo,
        combo_labels,
        Path(args.confusion_cell_distance),
        title="Confusion matrix (cell + router distance)",
    )
    # Baseline confusion: reuse predicted cell, distance = mode for that cell.
    pred_combo_baseline = np.array([f"{cell}|{dist}" for cell, dist in zip(y_pred, distance_pred_baseline)])
    combo_labels_baseline = sorted(np.unique(np.concatenate([true_combo, pred_combo_baseline])))
    save_confusion_matrix(
        true_combo,
        pred_combo_baseline,
        combo_labels_baseline,
        Path(args.confusion_cell_distance_baseline),
        title="Confusion matrix (cell + router distance baseline)",
    )
    save_roc_curves(y_test_binarized, y_proba, class_order, Path(args.roc_curve))
    save_prediction_report(
        y_test,
        y_pred,
        y_proba,
        meta_test,
        cell_lookup,
        neighbor_distances,
        neighbor_labels,
        distance_test,
        distance_pred,
        distance_confidence,
        Path(args.predictions_report),
    )
    save_model(localizer, Path(args.model_output))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate NN + L-KNN localization pipeline.")
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        help="Path to a measurement folder optionally suffixed with :distance_m. "
        "Example: --campaign ddeuxmetres:2",
    )
    parser.add_argument("--cell-width-m", type=float, default=DEFAULT_CELL_WIDTH_M)
    parser.add_argument("--cell-height-m", type=float, default=DEFAULT_CELL_HEIGHT_M)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "logistic", "identity"])
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization for the encoder.")
    parser.add_argument("--explain-k", type=int, default=3, help="Number of nearest neighbors to log for explainability.")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_PATH,
        help="Path where the trained localizer will be serialized.",
    )
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=CONFUSION_PATH,
        help="Path of the confusion matrix PNG.",
    )
    parser.add_argument(
        "--roc-curve",
        type=Path,
        default=ROC_PATH,
        help="Path of the ROC micro/macro PNG.",
    )
    parser.add_argument(
        "--confusion-cell-distance",
        type=Path,
        default=CONFUSION_CELL_DISTANCE_PATH,
        help="Path of the confusion matrix combining cell and router-distance predictions.",
    )
    parser.add_argument(
        "--confusion-cell-distance-baseline",
        type=Path,
        default=CONFUSION_CELL_DISTANCE_BASELINE_PATH,
        help="Path of the confusion matrix combining cell and router-distance predictions (baseline: modal distance per predicted cell).",
    )
    parser.add_argument(
        "--predictions-report",
        type=Path,
        default=PREDICTIONS_PATH,
        help="CSV file containing per-sample predictions with confidences and neighbor details.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
