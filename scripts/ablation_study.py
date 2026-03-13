"""Run feature and room-onehot ablation studies on the NN+L-KNN localizer.

The study evaluates leave-one-room-out (LORO) localization performance for
multiple feature subsets and with/without room one-hot features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (SRC_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import benchmark_models as bm  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
ALL_FEATURES = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]


@dataclass(frozen=True)
class AblationScenario:
    name: str
    features: tuple[str, ...]
    include_room_onehot: bool


def _scenario_grid() -> list[AblationScenario]:
    no_room_common = [
        AblationScenario("all5_no_room_ohe", tuple(ALL_FEATURES), False),
        AblationScenario(
            "drop_noise_no_room_ohe",
            tuple(f for f in ALL_FEATURES if f != "Noise"),
            False,
        ),
        AblationScenario(
            "drop_a1_no_room_ohe",
            tuple(f for f in ALL_FEATURES if f != "signal_A1"),
            False,
        ),
        AblationScenario(
            "drop_a2_no_room_ohe",
            tuple(f for f in ALL_FEATURES if f != "signal_A2"),
            False,
        ),
        AblationScenario(
            "drop_a3_no_room_ohe",
            tuple(f for f in ALL_FEATURES if f != "signal_A3"),
            False,
        ),
    ]
    room_aware_ref = [
        AblationScenario("all5_room_ohe", tuple(ALL_FEATURES), True),
        AblationScenario(
            "drop_noise_room_ohe",
            tuple(f for f in ALL_FEATURES if f != "Noise"),
            True,
        ),
    ]
    return room_aware_ref + no_room_common


def _evaluate_loro(
    df: pd.DataFrame,
    cell_lookup: pd.DataFrame,
    *,
    include_room_onehot: bool,
    random_state: int,
) -> dict[str, float]:
    rooms = sorted(df["room"].unique())
    rows: list[dict] = []
    for held_out in rooms:
        train_df = df[df["room"] != held_out]
        test_df = df[df["room"] == held_out]
        model = bm.fit_localizer(train_df, include_room=include_room_onehot, random_state=random_state)
        preds = model.predict(bm.build_features(test_df, include_room=include_room_onehot))
        summary = bm.localization_summary(test_df, preds, cell_lookup)
        rows.append(
            {
                "held_out_room": held_out,
                "cell_accuracy": summary.cell_accuracy,
                "mean_error_m": summary.mean_error_m,
                "p90_error_m": summary.p90_error_m,
            }
        )
    fold_df = pd.DataFrame(rows)
    return {
        "cell_accuracy": float(fold_df["cell_accuracy"].mean()),
        "mean_error_m": float(fold_df["mean_error_m"].mean()),
        "p90_error_m": float(fold_df["p90_error_m"].mean()),
        "n_folds": int(len(fold_df)),
        "per_room": rows,
    }


def _parse_scenarios(values: Iterable[str] | None) -> set[str] | None:
    if not values:
        return None
    cleaned = {value.strip() for value in values if value and value.strip()}
    return cleaned or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NN+L-KNN ablation study.")
    parser.add_argument(
        "--room",
        action="append",
        help="Filter to one or more rooms (repeatable). Example: --room D005 --room E101",
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        help="Filter to router distances in meters. Example: --distances 2 4",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only selected scenarios (repeatable). Use --list-scenarios to discover names.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenario names and exit.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for NN+L-KNN.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    all_scenarios = _scenario_grid()
    selected_names = _parse_scenarios(args.scenario)

    if args.list_scenarios:
        for scenario in all_scenarios:
            print(scenario.name)
        return

    if selected_names is None:
        scenarios = all_scenarios
    else:
        scenarios = [scenario for scenario in all_scenarios if scenario.name in selected_names]
        missing = sorted(selected_names - {scenario.name for scenario in scenarios})
        if missing:
            raise ValueError(f"Unknown scenarios: {missing}. Use --list-scenarios.")
        if not scenarios:
            raise ValueError("No scenario selected.")

    df = bm.load_cross_room(room_filter=args.room, distance_filter=args.distances)
    if df["room"].nunique() < 2:
        raise ValueError("Ablation LORO requires at least 2 rooms after filtering.")

    cell_lookup = (
        df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    original_features = list(bm.FEATURE_COLUMNS)
    try:
        for scenario in scenarios:
            bm.FEATURE_COLUMNS = list(scenario.features)
            metrics = _evaluate_loro(
                df,
                cell_lookup,
                include_room_onehot=scenario.include_room_onehot,
                random_state=args.seed,
            )
            results.append(
                {
                    "scenario": scenario.name,
                    "features": ",".join(scenario.features),
                    "include_room_onehot": scenario.include_room_onehot,
                    "cell_accuracy": metrics["cell_accuracy"],
                    "mean_error_m": metrics["mean_error_m"],
                    "p90_error_m": metrics["p90_error_m"],
                    "n_folds": metrics["n_folds"],
                }
            )
            print(
                f"{scenario.name:24s} | "
                f"acc={metrics['cell_accuracy']:.4f} | "
                f"mean={metrics['mean_error_m']:.4f} m | "
                f"p90={metrics['p90_error_m']:.4f} m"
            )
    finally:
        bm.FEATURE_COLUMNS = original_features

    summary_df = pd.DataFrame(results).sort_values("cell_accuracy", ascending=False)
    summary_path = REPORT_DIR / "ablation_study_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    details = {
        "filters": {"rooms": args.room or "all", "distances": args.distances or "all"},
        "seed": args.seed,
        "results": results,
    }
    details_path = REPORT_DIR / "ablation_study_summary.json"
    details_path.write_text(json.dumps(details, indent=2))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()
