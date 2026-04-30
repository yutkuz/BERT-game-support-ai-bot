from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.support_bot.training import BertPredictor

from .config import DEFAULT_ARTIFACT_DIR, HUMAN_REVIEW_CONFIDENCE_THRESHOLD
from .jobs import jobs
from .reply_templates import build_auto_reply
from .rewriter_service import rewriter_service


def normalize_prediction_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in items:
        message = str(item.get("message", "")).strip()
        if not message:
            continue
        normalized.append(
            {
                "message": message,
                "rewrittenText": str(item.get("rewrittenText", "")).strip(),
            }
        )
    return normalized


class PredictorService:
    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self._predictor: BertPredictor | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def metadata(self) -> dict[str, Any]:
        metadata_path = self.artifact_dir / "metadata.json"
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        metadata = self.metadata()
        metrics = metadata.get("bert_metrics", {})
        label_mapping_path = self.artifact_dir / "label_mapping.json"
        labels = json.loads(label_mapping_path.read_text(encoding="utf-8")).get("id_to_label", [])
        return {
            "artifactDir": str(self.artifact_dir),
            "modelName": metadata.get("bert_model_name"),
            "labelCount": metadata.get("label_count"),
            "labels": labels,
            "accuracy": metrics.get("accuracy"),
            "macroF1": metrics.get("macro_f1"),
            "loaded": self.loaded,
            "humanReviewConfidenceThreshold": HUMAN_REVIEW_CONFIDENCE_THRESHOLD,
        }

    def _load(self) -> BertPredictor:
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    self._predictor = BertPredictor(self.artifact_dir)
        return self._predictor

    def predict_items(
        self,
        items: list[dict[str, str]],
        job_id: str,
        rewrite_enabled: bool = False,
    ) -> list[dict[str, object]]:
        predictor = self._load()
        results: list[dict[str, object]] = []

        for index, item in enumerate(items, start=1):
            if jobs.is_cancelled(job_id):
                break

            message = item["message"]
            rewritten_text = ""
            rewrite_error = ""
            if rewrite_enabled and rewriter_service.enabled:
                try:
                    rewritten_text = rewriter_service.rewrite(message)
                except RuntimeError as exc:
                    rewrite_error = str(exc)
            elif rewrite_enabled:
                rewrite_error = "Gemma rewrite servisi aktif degil."
            outputs = predictor.predict_batch([message], [rewritten_text])
            proba = outputs["bert_proba"][0]
            top_indices = np.argsort(proba)[::-1][:3]
            prediction = predictor.decode_labels(np.array([top_indices[0]]))[0]
            confidence = round(float(proba[top_indices[0]]), 4)
            auto_reply = build_auto_reply(str(prediction), confidence)

            results.append(
                {
                    "row": index,
                    "message": message,
                    "rewrittenText": rewritten_text,
                    "rewriteUsed": bool(rewritten_text),
                    "rewriteRequested": bool(rewrite_enabled),
                    "rewriteError": rewrite_error,
                    "prediction": str(prediction),
                    "confidence": confidence,
                    "autoReply": auto_reply["reply"],
                    "requiresHumanReview": auto_reply["requiresHumanReview"],
                    "hasReplyTemplate": auto_reply["hasTemplate"],
                    "top3": [
                        {
                            "label": str(predictor.decode_labels(np.array([label_index]))[0]),
                            "confidence": round(float(proba[label_index]), 4),
                        }
                        for label_index in top_indices
                    ],
                }
            )

        return results


predictor_service = PredictorService()

