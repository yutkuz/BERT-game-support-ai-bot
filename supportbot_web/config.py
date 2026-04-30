from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
DEFAULT_ARTIFACT_DIR = Path(
    os.environ.get("SUPPORT_BOT_ARTIFACT_DIR", BASE_DIR / "artifacts" / "tr_bert_uncased_epoch9")
)
DEFAULT_DATASET_PATH = BASE_DIR / "data_v6.xlsx"
FEEDBACK_DATASET_PATH = BASE_DIR / "feedback_dataset.xlsx"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("SUPPORT_BOT_PORT", "8009"))
GEMMA_REWRITE_ENABLED = os.environ.get("GEMMA_REWRITE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
GEMMA_REWRITE_MODEL = os.environ.get("GEMMA_REWRITE_MODEL", "gemma4:e4b")
GEMMA_REWRITE_URL = os.environ.get("GEMMA_REWRITE_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
GEMMA_REWRITE_TIMEOUT_SECONDS = float(os.environ.get("GEMMA_REWRITE_TIMEOUT_SECONDS", "180"))
GEMMA_REWRITE_PROMPT_PATH = BASE_DIR / "gemma_rewrite_prompt.md"
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = float(os.environ.get("HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "0.60"))

MESSAGE_COLUMN_CANDIDATES = (
    "kullanici_mesaji",
    "original_text",
    "original_message",
    "mesaj",
    "message",
    "user_message",
    "kullanici_mesaji_original",
    "text",
)
