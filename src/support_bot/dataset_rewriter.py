from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import get_config
from .rewrite_service import RewriteService


def rewrite_dataset(args: argparse.Namespace) -> None:
    config = get_config()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    if args.message_column not in df.columns:
        raise ValueError(f"Message column not found: {args.message_column}. Columns: {list(df.columns)}")
    df[args.rewrite_column] = ""

    checkpoint_path = checkpoint_dir / "rewrite_progress.csv"
    if checkpoint_path.exists() and args.resume:
        checkpoint = pd.read_csv(checkpoint_path)
        if len(checkpoint) == len(df):
            df[args.rewrite_column] = checkpoint[args.rewrite_column].fillna("").astype(str)

    service = RewriteService(config)
    started = time.perf_counter()
    rewritten_count = 0

    for idx, row in df.iterrows():
        current_rewrite = str(row.get(args.rewrite_column, "") or "").strip()
        if args.resume and current_rewrite:
            continue

        message = str(row[args.message_column] or "").strip()
        try:
            result = service.rewrite(message)
        except Exception:
            df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            raise
        df.at[idx, args.rewrite_column] = result.rewritten_text
        rewritten_count += 1

        if rewritten_count % args.save_every == 0:
            df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "processed_new_rows": rewritten_count,
                        "last_row": int(idx),
                        "elapsed_seconds": round(elapsed, 2),
                        "avg_seconds": round(elapsed / max(rewritten_count, 1), 4),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
    df.to_excel(output_path, index=False)
    print(
        json.dumps(
            {
                "done": True,
                "input": str(input_path),
                "output": str(output_path),
                "rows": int(len(df)),
                "rewritten_new_rows": int(rewritten_count),
                "total_seconds": round(time.perf_counter() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    config = get_config()
    parser = argparse.ArgumentParser(description="Rewrite dataset rewrite column with Gemma4 E4B via Ollama.")
    parser.add_argument("--input", default=str(config.dataset_path))
    parser.add_argument("--output", default=str(config.rewritten_dataset_path))
    parser.add_argument("--checkpoint-dir", default=str(config.rewrite_checkpoint_dir))
    parser.add_argument("--message-column", default=config.message_column)
    parser.add_argument("--rewrite-column", default=config.rewrite_column)
    parser.add_argument("--timeout", type=int, default=config.rewrite_timeout)
    parser.add_argument("--num-gpu", type=int, default=config.ollama_num_gpu, help="Ollama num_gpu option. Use 0 to force CPU.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.timeout:
        import os

        os.environ["SUPPORT_BOT_REWRITE_TIMEOUT"] = str(args.timeout)
    if args.num_gpu is not None:
        import os

        os.environ["SUPPORT_BOT_OLLAMA_NUM_GPU"] = str(args.num_gpu)
    rewrite_dataset(args)


if __name__ == "__main__":
    main()
