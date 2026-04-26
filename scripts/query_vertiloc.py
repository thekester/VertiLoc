"""CLI helper to query the VertiLoc model with single or batched RSSI measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Ensure src/ is on sys.path when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization import DEFAULT_RUN_NAME, RSSI_FEATURE_COLUMNS, VertiLocInferenceModel

FEATURE_COLUMNS = list(RSSI_FEATURE_COLUMNS)
DEFAULT_MODEL = ROOT / "reports/localizer.joblib"
METRICS_PATH = ROOT / "reports/latest_metrics.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query VertiLoc with one RSSI vector or a CSV batch.")
    parser.add_argument("Signal", type=float, nargs="?", help="Global RSSI (dBm)")
    parser.add_argument("Noise", type=float, nargs="?", help="Noise floor (dBm)")
    parser.add_argument("signal_A1", type=float, nargs="?", help="Per-antenna RSSI A1 (dBm)")
    parser.add_argument("signal_A2", type=float, nargs="?", help="Per-antenna RSSI A2 (dBm)")
    parser.add_argument("signal_A3", type=float, nargs="?", help="Per-antenna RSSI A3 (dBm)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        type=Path,
        help="Path to the serialized EmbeddingKnnLocalizer (default: reports/localizer.joblib)",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        type=str,
        help="Run name shown in outputs when the serialized model does not already carry one.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to display.")
    parser.add_argument(
        "--vector",
        type=str,
        help="Comma-separated RSSI vector (Signal,Noise,signal_A1,signal_A2,signal_A3)",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="CSV file containing one or more rows with Signal,Noise,signal_A1,signal_A2,signal_A3.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional output CSV path for batched predictions.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional output JSON path for batched or single predictions.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Console output format. `json` is useful for integration scripts.",
    )
    parser.add_argument(
        "--room",
        choices=["auto", "B121", "D005", "E101", "E102"],
        default="auto",
        help="Room preset used to choose board projection mode.",
    )
    parser.add_argument(
        "--router-height-m",
        type=float,
        default=0.75,
        help="Router height used for board projection/clamping.",
    )
    parser.add_argument(
        "--batch-top-k",
        action="store_true",
        help="Include top-k neighbor details in the output CSV/JSON for each input row.",
    )
    parser.add_argument(
        "--example",
        "--examples",
        action="store_true",
        dest="example",
        help="Print usage examples and exit.",
    )
    return parser.parse_args(argv)


def _preprocess_argv(raw_argv: list[str]) -> list[str]:
    processed: list[str] = []
    skip_next = False
    for i, token in enumerate(raw_argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--vector" and i + 1 < len(raw_argv):
            processed.append(f"--vector={raw_argv[i + 1]}")
            skip_next = True
        else:
            processed.append(token)
    return processed


def _build_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv is not None:
        if not args.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in input CSV: {missing}")
        return df.copy()

    if args.vector:
        parts = [token.strip() for token in args.vector.split(",") if token.strip()]
        if len(parts) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(FEATURE_COLUMNS)} comma-separated values, got {len(parts)}: {args.vector}"
            )
        values = [float(part) for part in parts]
    else:
        values = []
        for col in FEATURE_COLUMNS:
            value = getattr(args, col)
            if value is None:
                raise ValueError(
                    "Provide either a full positional vector, --vector, or --input-csv."
                )
            values.append(value)
    return pd.DataFrame([dict(zip(FEATURE_COLUMNS, values, strict=False))])


def _load_latest_metrics() -> dict[str, object] | None:
    model = VertiLocInferenceModel.load(DEFAULT_MODEL, metrics_path=METRICS_PATH)
    return model.latest_metrics


def _print_examples() -> None:
    print("Usage examples:")
    print("")
    print("Single vector with positional values:")
    print("  PYTHONPATH=src python3 scripts/query_vertiloc.py -42 -95 -44 -41 -43")
    print("")
    print("Single vector with --vector:")
    print('  PYTHONPATH=src python3 scripts/query_vertiloc.py --vector "-42,-95,-44,-41,-43"')
    print("")
    print("JSON output:")
    print('  PYTHONPATH=src python3 scripts/query_vertiloc.py --vector "-42,-95,-44,-41,-43" --format json')
    print("")
    print("Batch mode from CSV:")
    print("  PYTHONPATH=src python3 scripts/query_vertiloc.py \\")
    print("    --input-csv reports/benchmarks/query_batch_example.csv \\")
    print("    --output-csv reports/benchmarks/query_batch_predictions.csv \\")
    print("    --output-json reports/benchmarks/query_batch_predictions.json")
    print("")
    print("Show these examples again:")
    print("  PYTHONPATH=src python3 scripts/query_vertiloc.py --example")


def _print_text_summary(pred_df: pd.DataFrame, *, top_k: int) -> None:
    if len(pred_df) == 1:
        row = pred_df.iloc[0]
        print(f"Run: {row['run_name']}")
        print("Hello, I'm VertiLoc. Thanks for the data!")
        print(f"Based on the RSSI vector, the predicted cell is: {row['pred_cell']} (confidence={row['confidence']:.3f})")
        if not pd.isna(row.get("pred_router_distance_m")):
            print(
                f"Predicted router distance: {row['pred_router_distance_m']:.1f} m "
                f"(confidence={row['distance_confidence']:.3f})"
            )
        if "zone_id" in row and pd.notna(row["zone_id"]):
            print(
                "Board geometry -> "
                f"raw=(x={row['board_x_m']:.3f} m, z={row['board_z_m']:.3f} m) | "
                f"clamped=(x={row['board_x_clamped_m']:.3f} m, z={row['board_z_clamped_m']:.3f} m) | "
                f"zone={row['zone_id']} | "
                f"within_board={bool(row['within_board'])} | "
                f"projection_distance={row['projection_distance_m']:.3f} m | "
                f"router_esp_3d_raw={row['router_esp_3d_raw_m']:.3f} m | "
                f"router_esp_3d_clamped={row['router_esp_3d_clamped_m']:.3f} m"
            )
        print(
            "OOD detector: "
            f"{'UNKNOWN' if bool(row['ood_is_unknown']) else 'KNOWN'} "
            f"(energy={row['ood_energy']:.4f}, nn_dist={row['ood_embedding_distance']:.4f})"
        )
        neighbor_cols = [f"neighbor_{i}_cell" for i in range(1, top_k + 1) if f"neighbor_{i}_cell" in pred_df.columns]
        if neighbor_cols:
            print("Top-K neighbors in the embedding space:")
            for i in range(1, top_k + 1):
                cell_key = f"neighbor_{i}_cell"
                dist_key = f"neighbor_{i}_embedding_distance"
                if cell_key in pred_df.columns:
                    print(f"  #{i}: cell={row[cell_key]} | embedding_distance={row[dist_key]:.4f}")
        return

    display_cols = [
        "sample_id",
        "pred_cell",
        "zone_id",
        "pred_router_distance_m",
        "confidence",
        "ood_is_unknown",
    ]
    display_cols = [col for col in display_cols if col in pred_df.columns]
    print(f"Processed {len(pred_df)} samples.")
    print(pred_df[display_cols].to_string(index=False))


def main() -> None:
    args = parse_args(_preprocess_argv(sys.argv[1:]))
    if args.example:
        _print_examples()
        return
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model {args.model} not found. Run `python -m localization.pipeline ...` to train it first."
        )

    inference_model = VertiLocInferenceModel.load(
        args.model,
        metrics_path=METRICS_PATH,
        run_name=str(args.run_name),
    )
    input_df = _build_input_dataframe(args)
    pred_df = inference_model.predict_dataframe(
        input_df,
        top_k=int(args.top_k),
        room=str(args.room),
        router_height_m=float(args.router_height_m),
        include_neighbors=bool(args.batch_top_k),
    )

    latest_metrics = inference_model.latest_metrics
    if latest_metrics is not None:
        if "router_distance_accuracy" in latest_metrics:
            pred_df.attrs["router_distance_accuracy"] = latest_metrics["router_distance_accuracy"]
        if "router_distance_accuracy_baseline" in latest_metrics:
            pred_df.attrs["router_distance_accuracy_baseline"] = latest_metrics["router_distance_accuracy_baseline"]

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.output_csv, index=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": inference_model.run_name,
            "n_samples": int(len(pred_df)),
            "predictions": pred_df.to_dict(orient="records"),
        }
        if latest_metrics is not None:
            payload["latest_training_metrics"] = {
                key: latest_metrics[key]
                for key in ["router_distance_accuracy", "router_distance_accuracy_baseline", "board_projection"]
                if key in latest_metrics
            }
        args.output_json.write_text(json.dumps(payload, indent=2))

    if args.format == "json":
        payload = {
            "run_name": inference_model.run_name,
            "n_samples": int(len(pred_df)),
            "predictions": pred_df.to_dict(orient="records"),
        }
        if latest_metrics is not None:
            payload["latest_training_metrics"] = {
                key: latest_metrics[key]
                for key in ["router_distance_accuracy", "router_distance_accuracy_baseline", "board_projection"]
                if key in latest_metrics
            }
        print(json.dumps(payload, indent=2))
    else:
        print(
            "Note: the model does NOT take router distance as input; "
            "distance is inferred from embeddings (LogReg head)."
        )
        _print_text_summary(pred_df, top_k=int(args.top_k))
        if latest_metrics is not None:
            acc = latest_metrics.get("router_distance_accuracy")
            acc_base = latest_metrics.get("router_distance_accuracy_baseline")
            if acc is not None and acc_base is not None:
                print(
                    f"Distance inference (last training run) -> "
                    f"LogReg head: {acc:.3f} | baseline (mode per cell): {acc_base:.3f}"
                )
        if args.output_csv is not None:
            print(f"Saved CSV predictions to {args.output_csv}")
        if args.output_json is not None:
            print(f"Saved JSON predictions to {args.output_json}")


if __name__ == "__main__":
    main()
