#!/usr/bin/env python3
"""Generate qualitative cell-confusion cases with radio hypotheses (E102).

This script runs a cross-validated classifier on E102 campaigns, extracts the
most meaningful true->predicted cell confusions, and writes:
- a detailed confusion table (CSV),
- a compact top-cases artifact (JSON),
- a report-ready French text block (TXT).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from localization.data import CampaignSpec, load_measurements  # noqa: E402

FEATURES = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS = {
    "exp1": CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0),
    "exp2": CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0),
    "exp3": CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0),
    "exp4": CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0),
    "exp5": CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6": CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}


def _edge_type(grid_x: int, grid_y: int) -> str:
    if grid_x in (0, 4) and grid_y in (-4, 10):
        return "corner"
    if grid_x in (0, 4) or grid_y in (-4, 10):
        return "edge"
    return "interior"


def _campaign_focus_label(name: str) -> str:
    if name in {"exp5", "exp6"}:
        return "elevation"
    if name in {"exp1", "exp2", "exp3", "exp4"}:
        return "orientation"
    return "mixed"


def _build_radio_hypothesis(row: pd.Series) -> str:
    parts: list[str] = []

    same_row = int(row["dx"]) == 0
    same_col = int(row["dy"]) == 0
    near = float(row["distance_m"]) <= 0.36
    far = float(row["distance_m"]) >= 0.90

    if near and (same_row or same_col):
        parts.append("gradient RSSI local faible entre cellules voisines")
    elif near:
        parts.append("empreintes radio quasi indiscernables sur un voisinage proche")

    if far:
        parts.append("aliasing multi-trajets probable (confusion non locale)")

    true_edge = row["true_edge"]
    pred_edge = row["pred_edge"]
    if true_edge in {"edge", "corner"}:
        parts.append("effet de bord (géométrie mur/tableau) sur la cellule vraie")
    if pred_edge in {"edge", "corner"} and true_edge == "interior":
        parts.append("attraction vers une signature de bord")

    dominant_share = float(row["dominant_campaign_share"])
    dominant_campaign = str(row["dominant_campaign"])
    if dominant_share >= 0.55:
        focus = _campaign_focus_label(dominant_campaign)
        if focus == "elevation":
            parts.append(f"sensibilité marquée à l'élévation ({dominant_campaign} dominant)")
        elif focus == "orientation":
            parts.append(f"sensibilité marquée à l'orientation ({dominant_campaign} dominant)")

    if not parts:
        parts.append("chevauchement de signatures RSSI dû au canal indoor fortement réfléchi")

    return "; ".join(parts)


def _compose_report_block(df_top: pd.DataFrame, *, model_name: str, n_splits: int) -> str:
    lines = []
    lines.append("Analyse d'erreurs qualitative (automatisée)")
    lines.append(
        f"Méthode: {model_name}, validation croisée stratifiée {n_splits} folds, E102 (6 campagnes)."
    )
    lines.append("Cas concrets de cellules confondues (top):")

    for _, row in df_top.iterrows():
        rate = 100.0 * float(row["confusion_rate"]) 
        true_cell = row["true_cell"]
        pred_cell = row["pred_cell"]
        dist = float(row["distance_m"])
        support = int(row["true_support"])
        count = int(row["count"])
        dom = row["dominant_campaign"]
        dom_share = 100.0 * float(row["dominant_campaign_share"])
        hypo = row["radio_hypothesis"]

        lines.append(
            f"- {true_cell} -> {pred_cell} ({rate:.1f} %, {count}/{support}, d={dist:.3f} m, "
            f"campagne dominante={dom} {dom_share:.1f} %)"
        )
        lines.append(f"  Hypothèse radio: {hypo}.")

    lines.append(
        "Note: les hypothèses sont des diagnostics guidés par la géométrie des cellules "
        "et la concentration des erreurs par campagne; elles doivent être validées "
        "par tests ciblés (obstruction/rotation/élévation contrôlées)."
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    *,
    model_name: str,
    n_splits: int,
    random_state_split: int,
    random_state_model: int,
    top_k: int,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for spec in CAMPAIGNS.values():
        frames.append(load_measurements([spec]))
    df = pd.concat(frames, ignore_index=True)

    X = df[FEATURES].to_numpy()
    y = df["grid_cell"].to_numpy()

    cell_meta = (
        df[["grid_cell", "grid_x", "grid_y", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )

    if model_name == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=random_state_model, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state_split)
    rows = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        proba = model.predict_proba(X_te)
        pred_index = {label: i for i, label in enumerate(model.classes_)}

        for i, (true_label, pred_label) in enumerate(zip(y_te, preds)):
            conf = float(proba[i, pred_index[pred_label]])
            rows.append(
                {
                    "fold": fold_idx,
                    "campaign": df.iloc[te[i]]["campaign"],
                    "true_cell": true_label,
                    "pred_cell": pred_label,
                    "is_error": int(true_label != pred_label),
                    "pred_confidence": conf,
                }
            )

    pred_df = pd.DataFrame(rows)
    errors_df = pred_df[pred_df["is_error"] == 1].copy()

    support = pred_df.groupby("true_cell").size().rename("true_support")
    pair_counts = (
        errors_df.groupby(["true_cell", "pred_cell"]).size().rename("count").reset_index()
    )
    pair_counts = pair_counts.merge(support, on="true_cell", how="left")
    pair_counts["confusion_rate"] = pair_counts["count"] / pair_counts["true_support"]

    camp_counts = (
        errors_df.groupby(["true_cell", "pred_cell", "campaign"]).size().rename("c").reset_index()
    )
    dom = (
        camp_counts.sort_values(["true_cell", "pred_cell", "c"], ascending=[True, True, False])
        .drop_duplicates(["true_cell", "pred_cell"])
        .rename(columns={"campaign": "dominant_campaign", "c": "dominant_campaign_count"})
    )
    pair_counts = pair_counts.merge(
        dom[["true_cell", "pred_cell", "dominant_campaign", "dominant_campaign_count"]],
        on=["true_cell", "pred_cell"],
        how="left",
    )
    pair_counts["dominant_campaign_share"] = (
        pair_counts["dominant_campaign_count"] / pair_counts["count"]
    )

    pair_counts = pair_counts.merge(
        cell_meta.add_prefix("true_"), left_on="true_cell", right_index=True, how="left"
    )
    pair_counts = pair_counts.merge(
        cell_meta.add_prefix("pred_"), left_on="pred_cell", right_index=True, how="left"
    )

    pair_counts["dx"] = (pair_counts["pred_grid_x"] - pair_counts["true_grid_x"]).abs().astype(int)
    pair_counts["dy"] = (pair_counts["pred_grid_y"] - pair_counts["true_grid_y"]).abs().astype(int)
    pair_counts["distance_m"] = np.sqrt(
        (pair_counts["pred_coord_x_m"] - pair_counts["true_coord_x_m"]) ** 2
        + (pair_counts["pred_coord_y_m"] - pair_counts["true_coord_y_m"]) ** 2
    )
    pair_counts["true_edge"] = [
        _edge_type(int(x), int(y)) for x, y in zip(pair_counts["true_grid_x"], pair_counts["true_grid_y"])
    ]
    pair_counts["pred_edge"] = [
        _edge_type(int(x), int(y)) for x, y in zip(pair_counts["pred_grid_x"], pair_counts["pred_grid_y"])
    ]
    pair_counts["radio_hypothesis"] = pair_counts.apply(_build_radio_hypothesis, axis=1)

    pair_counts["severity_score"] = (
        0.60 * pair_counts["confusion_rate"]
        + 0.25 * (pair_counts["distance_m"] / math.sqrt(3.5**2 + 1.2**2))
        + 0.15 * pair_counts["dominant_campaign_share"]
    )

    filtered = pair_counts[pair_counts["count"] >= min_count].copy()
    ranked = filtered.sort_values(
        ["confusion_rate", "count", "severity_score"], ascending=False
    ).copy()

    # Balanced qualitative set: keep both local and non-local confusions.
    local = ranked[ranked["distance_m"] <= 0.45]
    non_local = ranked[ranked["distance_m"] > 0.45]
    local_quota = max(1, top_k // 2)

    selected_parts: list[pd.DataFrame] = []
    if not local.empty:
        selected_parts.append(local.head(local_quota))

    remaining = top_k - sum(len(part) for part in selected_parts)
    if remaining > 0 and not non_local.empty:
        selected_parts.append(non_local.head(remaining))

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else ranked.head(top_k).copy()
    )

    # If duplicates or insufficient rows (edge case), backfill from ranked order.
    top = selected.drop_duplicates(["true_cell", "pred_cell"])
    if len(top) < top_k:
        already = set(zip(top["true_cell"], top["pred_cell"]))
        mask = ranked.apply(
            lambda r: (r["true_cell"], r["pred_cell"]) not in already,
            axis=1,
        )
        backfill = ranked[mask].head(top_k - len(top))
        top = pd.concat([top, backfill], ignore_index=True)

    return pair_counts.sort_values(["confusion_rate", "count"], ascending=False), top


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative analysis of cell confusions in E102.")
    parser.add_argument("--model", choices=["rf"], default="rf")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs, top = run_analysis(
        model_name=args.model,
        n_splits=args.splits,
        random_state_split=args.split_seed,
        random_state_model=args.model_seed,
        top_k=args.top_k,
        min_count=args.min_count,
    )

    csv_path = REPORT_DIR / "qualitative_confusions_e102.csv"
    json_path = REPORT_DIR / "qualitative_cases_e102.json"
    txt_path = REPORT_DIR / "qualitative_cases_e102.txt"

    all_pairs.to_csv(csv_path, index=False)

    case_columns = [
        "true_cell",
        "pred_cell",
        "count",
        "true_support",
        "confusion_rate",
        "distance_m",
        "dx",
        "dy",
        "true_edge",
        "pred_edge",
        "dominant_campaign",
        "dominant_campaign_share",
        "radio_hypothesis",
        "severity_score",
    ]
    cases = top[case_columns].to_dict(orient="records")
    json_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    block = _compose_report_block(top, model_name=args.model.upper(), n_splits=args.splits)
    txt_path.write_text(block, encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")

    print("\nTop cases:")
    for row in cases:
        print(
            f"- {row['true_cell']} -> {row['pred_cell']} | rate={row['confusion_rate']:.3f} "
            f"| count={row['count']} | d={row['distance_m']:.3f}m | dom={row['dominant_campaign']}"
        )


if __name__ == "__main__":
    main()
