from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_DIR / "data_v5.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts" / "local_tuning"

LEARNING_RATES = [0.0000113, 0.0000123, 0.0000133]
LABEL_SMOOTHING = [0.1119, 0.1219, 0.1319]
REWRITE_BOOST = [1.454, 1.554, 1.654]
MAX_LENGTH = [88, 96, 104]
BASE_COMBINATION = (0.0000123, 0.1219, 1.554, 96)


def slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local grid tuning around the current BERT parameters."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--limit", type=int, default=0, help="Optional max run count for smoke tests.")
    parser.add_argument(
        "--mode",
        choices=("one-factor", "lr-smoothing-corners", "lr-smoothing-length", "full-grid"),
        default="one-factor",
        help=(
            "one-factor keeps the other tuned values at the current baseline; "
            "lr-smoothing-corners fixes rewrite_boost and max_length, then tunes lr and label_smoothing lower/upper values; "
            "lr-smoothing-length fixes rewrite_boost and tunes lr, label_smoothing, max_length; "
            "full-grid runs all combinations."
        ),
    )
    return parser


def build_runs(mode: str) -> list[tuple[float, float, float, int]]:
    if mode == "full-grid":
        return list(
            itertools.product(
                LEARNING_RATES,
                LABEL_SMOOTHING,
                REWRITE_BOOST,
                MAX_LENGTH,
            )
        )
    if mode == "lr-smoothing-length":
        return [
            (lr, smoothing, BASE_COMBINATION[2], max_length)
            for lr, smoothing, max_length in itertools.product(
                LEARNING_RATES,
                LABEL_SMOOTHING,
                [96, 104],
            )
        ]
    if mode == "lr-smoothing-corners":
        return [
            (lr, smoothing, BASE_COMBINATION[2], 104)
            for lr, smoothing in itertools.product(
                [LEARNING_RATES[0], LEARNING_RATES[-1]],
                [LABEL_SMOOTHING[0], LABEL_SMOOTHING[-1]],
            )
        ]

    base_lr, base_smoothing, base_rewrite_boost, base_max_length = BASE_COMBINATION
    runs = [BASE_COMBINATION]
    runs.extend((lr, base_smoothing, base_rewrite_boost, base_max_length) for lr in LEARNING_RATES if lr != base_lr)
    runs.extend((base_lr, smoothing, base_rewrite_boost, base_max_length) for smoothing in LABEL_SMOOTHING if smoothing != base_smoothing)
    runs.extend((base_lr, base_smoothing, rewrite_boost, base_max_length) for rewrite_boost in REWRITE_BOOST if rewrite_boost != base_rewrite_boost)
    runs.extend((base_lr, base_smoothing, base_rewrite_boost, max_length) for max_length in MAX_LENGTH if max_length != base_max_length)
    return runs


def read_result(artifact_dir: Path) -> dict[str, object]:
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = metadata.get("bert_metrics", {})
    config = metadata.get("training_config", {})
    return {
        "artifact_dir": str(artifact_dir),
        "macro_f1": metrics.get("macro_f1"),
        "accuracy": metrics.get("accuracy"),
        "learning_rate": config.get("learning_rate"),
        "label_smoothing": config.get("label_smoothing"),
        "rewrite_boost": metadata.get("rewrite_boost"),
        "max_length": metadata.get("bert_max_length"),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "device": config.get("device"),
        "fp16": config.get("fp16"),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"tuning_results_{args.mode}.csv"
    runs = build_runs(args.mode)
    if args.limit > 0:
        runs = runs[: args.limit]

    results: list[dict[str, object]] = []
    for index, (lr, smoothing, rewrite_boost, max_length) in enumerate(runs, start=1):
        run_name = (
            f"run_{index:03d}"
            f"_lr_{slug_float(lr)}"
            f"_ls_{slug_float(smoothing)}"
            f"_rb_{slug_float(rewrite_boost)}"
            f"_ml_{max_length}"
        )
        artifact_dir = args.output_dir / run_name
        metadata_path = artifact_dir / "metadata.json"
        if args.skip_existing and metadata_path.exists():
            print(f"[{index}/{len(runs)}] skip existing {run_name}", flush=True)
            result = read_result(artifact_dir)
            if result:
                results.append(result)
            continue

        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "train_model.py"),
            "train",
            "--dataset-path",
            str(args.dataset_path),
            "--artifact-dir",
            str(artifact_dir),
            "--learning-rate",
            str(lr),
            "--label-smoothing",
            str(smoothing),
            "--rewrite-boost",
            str(rewrite_boost),
            "--max-length",
            str(max_length),
            "--device",
            args.device,
            "--no-auto-resume",
        ]
        if args.fp16:
            command.append("--fp16")

        print(f"[{index}/{len(runs)}] start {run_name} at {datetime.now().isoformat(timespec='seconds')}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_DIR)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

        result = read_result(artifact_dir)
        if result:
            results.append(result)
            frame = pd.DataFrame(results).sort_values(
                ["macro_f1", "accuracy"],
                ascending=[False, False],
                na_position="last",
            )
            frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
            best = frame.iloc[0].to_dict()
            print(
                "[best] "
                f"macro_f1={best.get('macro_f1')} "
                f"accuracy={best.get('accuracy')} "
                f"lr={best.get('learning_rate')} "
                f"label_smoothing={best.get('label_smoothing')} "
                f"rewrite_boost={best.get('rewrite_boost')} "
                f"max_length={best.get('max_length')}",
                flush=True,
            )

    if results:
        frame = pd.DataFrame(results).sort_values(
            ["macro_f1", "accuracy"],
            ascending=[False, False],
            na_position="last",
        )
        frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
        (args.output_dir / "best_result.json").write_text(
            json.dumps(frame.iloc[0].to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
