from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src klasorunu path'e ekle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Bot V7 API runner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "support_bot.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

