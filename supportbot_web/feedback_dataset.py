from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FEEDBACK_DATASET_PATH


FEEDBACK_COLUMNS = [
    "kullanici_mesaji",
    "rewrite",
    "kategori",
    "tahmin_kategori",
    "tahmin_guven",
    "otomatik_cevap",
    "dogru_cevap",
]


class FeedbackDatasetStore:
    def __init__(
        self,
        feedback_path: Path = FEEDBACK_DATASET_PATH,
    ) -> None:
        self.feedback_path = feedback_path
        self._lock = threading.Lock()

    def status(self) -> dict[str, object]:
        feedback_count = self._feedback_count()
        return {
            "feedbackPath": str(self.feedback_path),
            "feedbackCount": feedback_count,
            "trainingCommand": (
                f"python scripts/train_model.py train --dataset-path {self.feedback_path.name} "
                "--artifact-dir artifacts/xlm_roberta_feedback"
            ),
        }

    def append_many(self, records: list[dict[str, Any]]) -> dict[str, object]:
        cleaned = [self._clean_record(record) for record in records]
        cleaned = [record for record in cleaned if record]
        if not cleaned:
            raise ValueError("Eklenecek geçerli düzeltme bulunamadı.")

        with self._lock:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            existing_df = self._read_existing_feedback()
            new_df = pd.DataFrame(cleaned, columns=FEEDBACK_COLUMNS)
            feedback_df = pd.concat([existing_df, new_df], ignore_index=True)
            feedback_df.to_excel(self.feedback_path, index=False)

        return {
            "ok": True,
            "added": len(cleaned),
            "feedbackCount": self._feedback_count(),
            "feedbackPath": str(self.feedback_path),
        }

    def _clean_record(self, record: dict[str, Any]) -> dict[str, object] | None:
        message = str(record.get("message", "")).strip()
        label = str(record.get("correctLabel", "")).strip()
        if not message or not label:
            return None

        return {
            "kullanici_mesaji": message,
            "rewrite": str(record.get("rewrittenText", "")).strip(),
            "kategori": label,
            "tahmin_kategori": str(record.get("predictedLabel", "")).strip(),
            "tahmin_guven": record.get("confidence", ""),
            "otomatik_cevap": str(record.get("autoReply", "")).strip(),
            "dogru_cevap": str(record.get("correctReply", "")).strip(),
        }

    def _read_existing_feedback(self) -> pd.DataFrame:
        if not self.feedback_path.exists():
            return pd.DataFrame(columns=FEEDBACK_COLUMNS)
        df = pd.read_excel(self.feedback_path)
        for column in FEEDBACK_COLUMNS:
            if column not in df.columns:
                df[column] = ""
        return df[FEEDBACK_COLUMNS].copy()

    def _feedback_count(self) -> int:
        if not self.feedback_path.exists():
            return 0
        try:
            return int(len(pd.read_excel(self.feedback_path, usecols=["kategori"])))
        except Exception:
            return 0


feedback_dataset_store = FeedbackDatasetStore()
