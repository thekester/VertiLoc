"""CLI helper to query the VertiLoc model with raw RSSI measurements."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

# Ensure src/ is on sys.path when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.embedding_knn import EmbeddingKnnLocalizer

# Mirror the training feature set: only RSSI-based fields, no router distance as input.
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
DEFAULT_MODEL = ROOT / "reports/localizer.joblib"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser and handle two input styles.

    Args:
        argv: Optional list of arguments (used after pre-processing the
            --vector option so values starting with '-' are accepted).

    Returns:
        Parsed arguments with either the positional fields populated or the
        comma-separated vector stored in `args.vector`.
    """

    parser = argparse.ArgumentParser(description="Hello VertiLoc! Identify the cell from RSSI readings.")
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
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest neighbors to display for explainability.",
    )
    parser.add_argument(
        "--vector",
        type=str,
        help="Comma-separated RSSI vector (Signal,Noise,signal_A1,signal_A2,signal_A3)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the VertiLoc CLI helper."""
    raw_argv = sys.argv[1:]
    processed = []
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

    args = parse_args(processed)
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model {args.model} not found. Run `python -m localization.pipeline ...` to train it first."
        )

    localizer: EmbeddingKnnLocalizer = joblib.load(args.model)
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
                    "Either provide all positional arguments or a --vector \"val1,...,val5\" string."
                )
            values.append(value)

    sample = np.array([values], dtype=float)

    proba = localizer.predict_proba(sample)[0]
    pred = localizer.predict(sample)[0]
    top_prob = proba.max()

    print("Hello, I'm VertiLoc. Thanks for the data!")
    print(f"Based on the RSSI vector, the predicted cell is: {pred} (confidence={top_prob:.3f})")

    distances, neighbors = localizer.explain(sample, top_k=args.top_k)
    print("Top-K neighbors in the embedding space:")
    for rank, (cell, dist) in enumerate(zip(neighbors[0], distances[0]), start=1):
        print(f"  #{rank}: cell={cell} | embedding_distance={dist:.4f}")

if __name__ == "__main__":
    main()
