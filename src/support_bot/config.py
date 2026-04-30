from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppConfig:
    dataset_path: Path = PROJECT_ROOT / "data_v6.xlsx"
    rewritten_dataset_path: Path = PROJECT_ROOT / "data_v6.xlsx"
    artifact_dir: Path = PROJECT_ROOT / "artifacts" / "tr_bert_uncased_epoch9"
    rewrite_checkpoint_dir: Path = PROJECT_ROOT / "artifacts" / "rewrite_checkpoints"
    rewrite_model: str = os.environ.get("SUPPORT_BOT_REWRITE_MODEL", "gemma4:e4b")
    ollama_url: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    rewrite_timeout: int = int(os.environ.get("SUPPORT_BOT_REWRITE_TIMEOUT", "180"))
    ollama_num_gpu: int = int(os.environ.get("SUPPORT_BOT_OLLAMA_NUM_GPU", "-1"))
    message_column: str = "kullanici_mesaji"
    rewrite_column: str = "rewrite"
    label_column: str = "kategori"
    bert_model_name: str = "ytu-ce-cosmos/turkish-base-bert-uncased"
    max_length: int = 104
    rewrite_boost: float = 1.0
    human_review_confidence_threshold: float = float(
        os.environ.get("SUPPORT_BOT_HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "0.60")
    )


def get_config() -> AppConfig:
    return AppConfig()
