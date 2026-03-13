#!/usr/bin/env python3
"""Interactive helper to run benchmark_models with room/distance selection."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_models as bm
from localization.data import infer_router_distance


def _split_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[,\s]+", text.strip()) if token]


def _format_distances(distances: list[float]) -> str:
    return ", ".join(f"{dist:g}" for dist in distances)


def _collect_available() -> dict[str, list[float]]:
    rooms: dict[str, list[float]] = {}
    for room, specs in bm.ROOM_CAMPAIGNS.items():
        distances = set()
        for spec in specs:
            if not spec.path.exists():
                continue
            try:
                distances.add(float(spec.resolved_distance()))
            except ValueError:
                pass
        if distances:
            rooms[room] = sorted(distances)
    return rooms


def _resolve_rooms(
    raw_rooms: list[str] | None,
    available_rooms: list[str],
    *,
    interactive: bool,
) -> tuple[list[str], bool]:
    if not available_rooms:
        raise ValueError("No rooms available to select.")
    if raw_rooms:
        tokens = []
        for entry in raw_rooms:
            tokens.extend(_split_tokens(entry))
        if not tokens:
            tokens = available_rooms
        return _match_rooms(tokens, available_rooms), False
    if not interactive:
        raise ValueError("Missing --room and interactive mode disabled.")
    return _prompt_rooms(available_rooms)


def _match_rooms(tokens: list[str], available_rooms: list[str]) -> list[str]:
    lookup = {room.lower(): room for room in available_rooms}
    resolved: list[str] = []
    missing: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key in ("all", "a", "*"):
            return available_rooms
        if key in lookup:
            resolved.append(lookup[key])
        else:
            missing.append(token)
    if missing:
        raise ValueError(f"Unknown rooms: {', '.join(missing)}")
    return sorted(set(resolved), key=available_rooms.index)


def _prompt_rooms(available_rooms: list[str]) -> tuple[list[str], bool]:
    print("Available rooms:")
    for room in available_rooms:
        print(f"- {room}")
    while True:
        raw = input("Rooms to include [all]: ").strip()
        if not raw or raw.lower() in ("all", "a", "*"):
            return available_rooms, True
        tokens = _split_tokens(raw)
        try:
            return _match_rooms(tokens, available_rooms), False
        except ValueError as exc:
            print(f"{exc}. Try again.")


def _resolve_distances(
    raw_distances: list[str] | None,
    available_distances: list[float],
    *,
    interactive: bool,
) -> tuple[list[float] | None, bool]:
    if not available_distances:
        raise ValueError("No distances available to select.")
    if raw_distances:
        tokens = []
        for entry in raw_distances:
            tokens.extend(_split_tokens(entry))
        if any(token.lower() in ("all", "a", "*") for token in tokens):
            return None, True
        values = _parse_distance_tokens(tokens)
        matched = _match_distances(values, available_distances)
        return matched, False
    if not interactive:
        raise ValueError("Missing --distances and interactive mode disabled.")
    return _prompt_distances(available_distances)


def _parse_distance_tokens(tokens: list[str]) -> list[float]:
    values: list[float] = []
    for token in tokens:
        for part in _split_tokens(token):
            values.append(float(part.replace(",", ".")))
    return values


def _match_distances(values: list[float], available: list[float]) -> list[float] | None:
    if not values:
        return None
    matched: list[float] = []
    for value in values:
        found = None
        for candidate in available:
            if abs(candidate - value) < 1e-6:
                found = candidate
                break
        if found is None:
            raise ValueError(f"Unknown distance: {value:g}")
        matched.append(found)
    return sorted(set(matched))


def _prompt_distances(available_distances: list[float]) -> tuple[list[float] | None, bool]:
    print(f"Available distances (m): {_format_distances(available_distances)}")
    while True:
        raw = input("Distances to include [all]: ").strip()
        if not raw or raw.lower() in ("all", "a", "*"):
            return None, True
        try:
            values = _parse_distance_tokens([raw])
            matched = _match_distances(values, available_distances)
            return matched, False
        except ValueError as exc:
            print(f"{exc}. Try again.")


def _resolve_holdout(
    raw_holdout: str | None,
    *,
    rooms_count: int,
    distances_count: int,
    interactive: bool,
) -> str:
    if raw_holdout:
        return raw_holdout
    if not interactive:
        return "auto"
    default = "auto"
    if rooms_count > 1:
        default = "room"
    elif distances_count > 1:
        default = "distance"
    print("Holdout options: auto, room, distance, none")
    raw = input(f"Holdout strategy [{default}]: ").strip().lower()
    if not raw:
        return default
    if raw in ("auto", "room", "distance", "none"):
        return raw
    print("Unknown holdout option, using auto.")
    return "auto"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive benchmark runner.")
    parser.add_argument(
        "--room",
        action="append",
        help="Room filter (repeatable). Example: --room D005 --room E101",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        help="Distance filter in meters. Example: --distances 2 4",
    )
    parser.add_argument(
        "--holdout",
        choices=["auto", "room", "distance", "none"],
        help="Holdout strategy to avoid near-duplicate leakage.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional/unstable models (Mahalanobis, GPC, CatBoost, LightGBM, XGBoost).",
    )
    parser.add_argument(
        "--run-gpc",
        action="store_true",
        help="Run the Gaussian Process Classifier stages (very slow on large datasets).",
    )
    parser.add_argument(
        "--gpc-max-samples",
        type=int,
        help="Optional cap on training samples for GPC to keep runtime reasonable.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all optional models, including GPC.",
    )
    parser.add_argument(
        "--no-stacking",
        action="store_true",
        help="Skip the Stacking classifier (very slow on large datasets).",
    )
    parser.add_argument(
        "--strict-anti-leakage",
        action="store_true",
        help="Use strict anti-leakage train/test splits in benchmark_models.",
    )
    parser.add_argument(
        "--quant-step-db",
        type=float,
        default=0.5,
        help="Quantization step (dB) used by strict anti-leakage signature checks.",
    )
    parser.add_argument(
        "--benchmark-seeds",
        default="21",
        help=(
            "Seed list for classic benchmarks (comma/space separated or ranges, "
            "e.g. '7,13,21' or '1-10')."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail if required inputs are missing instead of prompting.",
    )
    parser.add_argument(
        "--run-generative",
        action="store_true",
        help="Run GAN/CycleGAN low-data experiments after classic benchmarks.",
    )
    parser.add_argument(
        "--generative-profile",
        choices=["quick", "full"],
        default="quick",
        help="Preset run list for generative experiments.",
    )
    parser.add_argument(
        "--generative-seeds",
        default="7",
        help=(
            "Seed list for generative runs (comma/space separated or ranges, "
            "e.g. '7,13,21' or '1-10')."
        ),
    )
    parser.add_argument(
        "--generative-device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Device for generative runs.",
    )
    parser.add_argument(
        "--generative-torch-threads",
        type=int,
        default=4,
        help="CPU threads for torch in generative runs.",
    )
    return parser.parse_args(argv)


def _parse_seed_spec(seed_spec: str) -> list[int]:
    values: list[int] = []
    for raw in re.split(r"[,\s]+", seed_spec.strip()):
        if not raw:
            continue
        if "-" in raw:
            left, right = raw.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid seed range: {raw}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(raw))
    if not values:
        raise ValueError("No valid generative seeds provided.")
    return sorted(set(values))


def _with_seed(run: list[str], seed: int) -> list[str]:
    updated = list(run)
    if "--seed" in updated:
        idx = updated.index("--seed")
        if idx + 1 < len(updated):
            updated[idx + 1] = str(seed)
            return updated
    return [*updated, "--seed", str(seed)]


def _generative_runs(profile: str, seeds: list[int]) -> list[list[str]]:
    if profile == "full":
        base_runs = [
            [
                "--mode", "cgan",
                "--target", "data/E102/exp1:4",
                "--low-data-ratio", "0.1",
                "--epochs", "40",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cgan",
                "--target", "data/E102/exp1:4",
                "--low-data-ratio", "0.1",
                "--epochs", "40",
                "--synthetic-per-class", "20",
                "--seed", "42",
            ],
            [
                "--mode", "cgan",
                "--target", "data/E102/exp1:4",
                "--low-data-ratio", "0.2",
                "--epochs", "40",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cgan",
                "--target", "data/E102/exp3:4",
                "--low-data-ratio", "0.1",
                "--epochs", "40",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cgan",
                "--target", "data/E102/elevation/exp5:4",
                "--low-data-ratio", "0.1",
                "--epochs", "40",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cgan",
                "--target", "data/E102/elevation/exp5:4",
                "--low-data-ratio", "0.2",
                "--epochs", "40",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cyclegan",
                "--target", "data/E102/exp1:4",
                "--source", "data/E102/exp2:4",
                "--low-data-ratio", "0.1",
                "--epochs", "20",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cyclegan",
                "--target", "data/E102/exp3:4",
                "--source", "data/E102/exp2:4",
                "--low-data-ratio", "0.1",
                "--epochs", "20",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
            [
                "--mode", "cyclegan",
                "--target", "data/E102/elevation/exp5:4",
                "--source", "data/E102/exp2:4",
                "--low-data-ratio", "0.1",
                "--epochs", "20",
                "--synthetic-per-class", "20",
                "--seed", "7",
            ],
        ]
        return [_with_seed(run, seed) for run in base_runs for seed in seeds]

    base_runs = [
        [
            "--mode", "cgan",
            "--target", "data/E102/exp1:4",
            "--low-data-ratio", "0.1",
            "--epochs", "20",
            "--synthetic-per-class", "20",
            "--seed", "7",
        ],
        [
            "--mode", "cgan",
            "--target", "data/E102/elevation/exp5:4",
            "--low-data-ratio", "0.1",
            "--epochs", "20",
            "--synthetic-per-class", "20",
            "--seed", "7",
        ],
        [
            "--mode", "cyclegan",
            "--target", "data/E102/exp1:4",
            "--source", "data/E102/exp2:4",
            "--low-data-ratio", "0.1",
            "--epochs", "10",
            "--synthetic-per-class", "20",
            "--seed", "7",
        ],
    ]
    return [_with_seed(run, seed) for run in base_runs for seed in seeds]


def _write_generative_summary(report_dir: Path) -> Path:
    rows: list[dict] = []
    for path in sorted(report_dir.glob("generative_*_r*_e*_s*_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        b = data["metrics"]["baseline_low_data"]
        a = data["metrics"]["augmented_with_generative"]
        u = data["metrics"]["upper_bound_full_train"]
        rows.append(
            {
                "file": path.name,
                "mode": data["mode"],
                "target": data["target"],
                "sources": "|".join(data.get("sources", [])),
                "seed": data["seed"],
                "low_data_ratio": data["params"]["low_data_ratio"],
                "epochs": data["params"]["epochs"],
                "synthetic_per_class": data["params"]["synthetic_per_class"],
                "baseline_acc": b["cell_accuracy"],
                "aug_acc": a["cell_accuracy"],
                "delta_acc": a["cell_accuracy"] - b["cell_accuracy"],
                "baseline_err_m": b["mean_error_m"],
                "aug_err_m": a["mean_error_m"],
                "delta_err_m": a["mean_error_m"] - b["mean_error_m"],
                "upper_acc": u["cell_accuracy"],
                "upper_err_m": u["mean_error_m"],
            }
        )

    summary_path = report_dir / "generative_trials_summary.csv"
    if not rows:
        summary_path.write_text(
            "file,mode,target,sources,seed,low_data_ratio,epochs,synthetic_per_class,"
            "baseline_acc,aug_acc,delta_acc,baseline_err_m,aug_err_m,delta_err_m,upper_acc,upper_err_m\n",
            encoding="utf-8",
        )
        _write_generative_stats_from_empty(report_dir)
        return summary_path

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["mode", "target", "low_data_ratio", "seed"])
    df.to_csv(summary_path, index=False)
    _write_generative_stats(df, report_dir)
    return summary_path


def _t_critical_95(n: int) -> float:
    """Two-sided 95% Student-t critical value (small n), normal approx otherwise."""
    if n <= 1:
        return float("nan")
    # df = n - 1
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(n - 1, 1.96)


def _write_generative_stats(df, report_dir: Path) -> Path:
    """Aggregate per protocol across seeds with mean/std and 95% CI."""
    import pandas as pd

    group_cols = ["mode", "target", "sources", "low_data_ratio", "epochs", "synthetic_per_class"]
    metrics = ["baseline_acc", "aug_acc", "delta_acc", "baseline_err_m", "aug_err_m", "delta_err_m"]

    rows: list[dict] = []
    for key, g in df.groupby(group_cols, dropna=False):
        n = int(len(g))
        base = dict(zip(group_cols, key))
        base["n_seeds"] = n
        for metric in metrics:
            values = g[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            sem = std / math.sqrt(n) if n > 1 else 0.0
            ci95 = _t_critical_95(n) * sem if n > 1 else 0.0
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_ci95"] = ci95
            base[f"{metric}_ci95_low"] = mean - ci95
            base[f"{metric}_ci95_high"] = mean + ci95
        rows.append(base)

    stats_path = report_dir / "generative_trials_stats.csv"
    if not rows:
        pd.DataFrame(columns=group_cols + ["n_seeds"]).to_csv(stats_path, index=False)
        return stats_path
    stats_df = pd.DataFrame(rows).sort_values(["mode", "target", "low_data_ratio"])
    stats_df.to_csv(stats_path, index=False)
    return stats_path


def _write_generative_stats_from_empty(report_dir: Path) -> Path:
    import pandas as pd

    group_cols = ["mode", "target", "sources", "low_data_ratio", "epochs", "synthetic_per_class"]
    stats_path = report_dir / "generative_trials_stats.csv"
    pd.DataFrame(columns=group_cols + ["n_seeds"]).to_csv(stats_path, index=False)
    return stats_path


def _launch_generative(profile: str, device: str, torch_threads: int, seeds: list[int]) -> None:
    script = SCRIPT_DIR / "generative_radio_map.py"
    if not script.exists():
        print(f"Skipping generative runs: script not found at {script}")
        return

    print("\nLaunching generative experiments...")
    runs = _generative_runs(profile, seeds)
    for idx, run_args in enumerate(runs, start=1):
        cmd = [
            sys.executable,
            str(script),
            *run_args,
            "--device",
            device,
            "--torch-threads",
            str(max(1, int(torch_threads))),
        ]
        print(f"[generative {idx}/{len(runs)}] {' '.join(run_args)}")
        subprocess.run(cmd, check=True)

    report_dir = SCRIPT_DIR.parent / "reports" / "benchmarks"
    summary_path = _write_generative_summary(report_dir)
    stats_path = report_dir / "generative_trials_stats.csv"
    print(f"Generative summary written: {summary_path}")
    print(f"Generative inter-seed stats written: {stats_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rooms_map = _collect_available()
    if not rooms_map:
        raise RuntimeError("No campaign folders found under data/.")
    available_rooms = sorted(rooms_map.keys())

    rooms, rooms_all = _resolve_rooms(
        args.room,
        available_rooms,
        interactive=not args.non_interactive,
    )
    available_distances = sorted({d for room in rooms for d in rooms_map[room]})
    distances, _distances_all = _resolve_distances(
        args.distances,
        available_distances,
        interactive=not args.non_interactive,
    )
    holdout = _resolve_holdout(
        args.holdout,
        rooms_count=len(rooms),
        distances_count=len(available_distances if distances is None else distances),
        interactive=not args.non_interactive,
    )
    if args.all_models:
        skip_optional = False
        run_gpc = True
    elif args.skip_optional:
        skip_optional = True
        run_gpc = False
    elif args.non_interactive:
        skip_optional = False
        run_gpc = args.run_gpc
    else:
        raw = input("Mode all (run all optional models including GPC)? [no]: ").strip().lower()
        if raw in ("y", "yes", "o", "oui"):
            skip_optional = False
            run_gpc = True
        else:
            raw = input("Include optional models (Mahalanobis, GPC, CatBoost, LightGBM, XGBoost)? [no]: ").strip().lower()
            skip_optional = raw not in ("y", "yes", "o", "oui")
            if skip_optional:
                run_gpc = False
            elif args.run_gpc:
                run_gpc = True
            else:
                raw = input("Include GPC (very slow on large datasets)? [no]: ").strip().lower()
                run_gpc = raw in ("y", "yes", "o", "oui")
    gpc_max_samples = args.gpc_max_samples
    if run_gpc and not args.non_interactive:
        raw = input("GPC max samples [1400]: ").strip()
        if raw:
            try:
                gpc_max_samples = int(raw)
            except ValueError:
                print("Invalid integer, using default (1400).")
                gpc_max_samples = 1400
        elif gpc_max_samples is None:
            gpc_max_samples = 1400
    if run_gpc and gpc_max_samples is None:
        gpc_max_samples = 1400

    print("\nSelections:")
    print(f"- rooms: {', '.join(rooms)}")
    if distances is None:
        print(f"- distances: all ({_format_distances(available_distances)})")
    else:
        print(f"- distances: {_format_distances(distances)}")
    print(f"- holdout: {holdout}")
    print(f"- optional models: {'skipped' if skip_optional else 'included'}")
    print(f"- GPC: {'enabled' if run_gpc else 'disabled'}")
    if run_gpc and gpc_max_samples:
        print(f"- GPC max samples: {gpc_max_samples}")
    print(
        f"- strict anti-leakage: {'enabled' if args.strict_anti_leakage else 'disabled'} "
        f"(quant_step={float(args.quant_step_db):g} dB)"
    )
    run_generative = args.run_generative
    generative_profile = args.generative_profile
    generative_device = args.generative_device
    generative_threads = args.generative_torch_threads
    generative_seeds = _parse_seed_spec(args.generative_seeds)
    benchmark_seeds = _parse_seed_spec(args.benchmark_seeds)
    if not args.non_interactive and not run_generative:
        raw = input("Run generative GAN/CycleGAN experiments after benchmarks? [no]: ").strip().lower()
        run_generative = raw in ("y", "yes", "o", "oui")
    if run_generative:
        print(
            f"- generative: enabled ({generative_profile}, device={generative_device}, "
            f"threads={generative_threads}, seeds={','.join(str(s) for s in generative_seeds)})"
        )
    else:
        print("- generative: disabled")
    print(f"- benchmark seeds: {','.join(str(s) for s in benchmark_seeds)}")

    cli_args: list[str] = []
    if not rooms_all:
        for room in rooms:
            cli_args += ["--room", room]
    if distances is not None:
        cli_args += ["--distances"] + [f"{dist:g}" for dist in distances]
    if holdout:
        cli_args += ["--holdout", holdout]
    if skip_optional:
        cli_args += ["--skip-optional"]
    if run_gpc:
        cli_args += ["--run-gpc"]
    if gpc_max_samples:
        cli_args += ["--gpc-max-samples", str(gpc_max_samples)]
    if args.all_models:
        cli_args += ["--all-models"]
    if args.no_stacking:
        cli_args += ["--no-stacking"]
    if args.strict_anti_leakage:
        cli_args += ["--strict-anti-leakage"]
    if float(args.quant_step_db) != 0.5:
        cli_args += ["--quant-step-db", f"{float(args.quant_step_db):g}"]
    cli_args += ["--seeds", args.benchmark_seeds]

    print("\nLaunching benchmarks...")
    bm.main(cli_args)
    if run_generative:
        _launch_generative(generative_profile, generative_device, generative_threads, generative_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
