from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_PATH = PROJECT_DIR / "data_v5.xlsx"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "tr_bert_uncased_epoch9"
DEFAULT_TEST_SIZE = 0.15
DEFAULT_RANDOM_STATE = 42
DEFAULT_REWRITE_MODEL = os.environ.get("SUPPORT_BOT_REWRITE_MODEL", "gemma4:e4b")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# Project-tuned BERT training config. Keep these defaults in code so normal
# training does not depend on long command-line arguments or wrapper scripts.
DEFAULT_BERT_MODEL_NAME = "ytu-ce-cosmos/turkish-base-bert-uncased"
DEFAULT_EPOCHS = 9
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 96
DEFAULT_LEARNING_RATE = 1.22797e-5
DEFAULT_WEIGHT_DECAY = 0.00110172
DEFAULT_WARMUP_RATIO = 0.0158
DEFAULT_LABEL_SMOOTHING = 0.1219
DEFAULT_REWRITE_BOOST = 1.554
DEFAULT_SAVE_CHECKPOINTS = True
DEFAULT_AUTO_RESUME = True
DEFAULT_CHECKPOINT_STEPS = 500


def disable_transformers_torchvision() -> None:
    """Avoid importing a broken torchvision install for text-only BERT inference."""
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils

        import_utils.is_torchvision_available = lambda: False
        transformers_utils.is_torchvision_available = lambda: False
    except Exception:
        pass

MAIN_DATASET_CANDIDATES = (
    "data_v5.xlsx",
    "data_v5_gemma4_e4b.xlsx",
    "data_v4_guncel.xlsx",
    "data_v4.xlsx",
    "veriseti2.xlsx",
    "veriseti1.xlsx",
    "data.xlsx",
    "support_train_dataset.xlsx",
    "checkpoint.csv",
    "kullanici_mesajlari_qwen.xlsx",
    "rewrite_output.xlsx",
)

ORIGINAL_COLUMN_CANDIDATES = (
    "original_message",
    "mesaj",
    "message",
    "user_message",
    "kullanici_mesaji",
    "kullanici_mesaji_original",
)
REWRITTEN_COLUMN_CANDIDATES = (
    "rewritten_message",
    "rewrite",
    "rewritten_user_message",
    "qwen_rewritten_message",
    "duzeltilmis_mesaj",
    "normalized_message",
)
CONFIDENCE_COLUMN_CANDIDATES = (
    "rewrite_confidence",
    "qwen_confidence",
    "confidence",
    "duzeltme_confidence",
    "rewrite_score",
)
LABEL_COLUMN_CANDIDATES = (
    "label",
    "kategori",
    "Kategori",
    "category",
    "class",
    "sinif",
)

LABEL_MERGE_MAP: dict[str, str] = {
    "sans": "oyun_adalet_puan",
}


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def normalize_ollama_url(url: str) -> str:
    url = str(url or "http://127.0.0.1:11434").strip().rstrip("/")
    if not re.match(r"^https?://", url):
        url = f"http://{url}"
    return url


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def serialize_label(label: Any) -> str:
    return normalize_text(label).replace(" ", "_")


def discover_file(candidates: Iterable[str]) -> Optional[Path]:
    dataset_dir = PROJECT_DIR / "dataset"
    for name in candidates:
        script_path = SCRIPT_DIR / name
        if script_path.exists():
            return script_path
        path = PROJECT_DIR / name
        if path.exists():
            return path
        dataset_path = dataset_dir / name
        if dataset_path.exists():
            return dataset_path
    if dataset_dir.exists():
        for path in dataset_dir.glob("*.xlsx"):
            lowered = path.name.lower()
            if any(token.lower() in lowered for token in candidates):
                return path
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    exact_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in exact_map:
            return exact_map[candidate.lower()]

    normalized = {
        re.sub(r"[^a-z0-9]+", "", col.lower()): col
        for col in columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def coerce_confidence(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and numeric.quantile(0.5) > 1.0:
        numeric = numeric / 100.0
    return numeric.clip(0.0, 1.0).astype(float)


def build_sample_weight(rewrite_confidence: float) -> float:
    if pd.isna(rewrite_confidence):
        return 1.0
    confidence = min(max(float(rewrite_confidence), 0.0), 1.0)
    return 0.75 + (0.50 * confidence)


@dataclass
class PreparedDataset:
    df: pd.DataFrame
    main_dataset_path: str
    original_column: str
    rewritten_column: Optional[str]
    confidence_column: Optional[str]
    label_column: str


def expand_text_examples(df: pd.DataFrame, rewrite_boost: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        base_weight = float(getattr(row, "sample_weight", 1.0))
        original_text = str(getattr(row, "original_text", "") or "").strip()
        rewritten_text = str(getattr(row, "rewritten_text", "") or "").strip()
        original_norm = normalize_text(original_text)
        rewritten_norm = normalize_text(rewritten_text)

        if original_norm:
            records.append(
                {
                    "row_id": int(getattr(row, "row_id")),
                    "label": getattr(row, "label"),
                    "label_id": int(getattr(row, "label_id")),
                    "source": "original",
                    "training_text": original_text,
                    "sample_weight": base_weight,
                }
            )
        if rewritten_norm and rewritten_norm != original_norm:
            records.append(
                {
                    "row_id": int(getattr(row, "row_id")),
                    "label": getattr(row, "label"),
                    "label_id": int(getattr(row, "label_id")),
                    "source": "rewrite",
                    "training_text": rewritten_text,
                    "sample_weight": base_weight * rewrite_boost,
                }
            )
    return pd.DataFrame.from_records(records)


def combine_probabilities(
    original_proba: np.ndarray,
    rewritten_proba: np.ndarray,
    rewrite_mask: np.ndarray,
    rewrite_boost: float,
) -> np.ndarray:
    combined = original_proba.astype(float).copy()
    weights = np.ones((len(original_proba), 1), dtype=float)
    if rewrite_mask.any():
        combined[rewrite_mask] += rewritten_proba[rewrite_mask] * rewrite_boost
        weights[rewrite_mask] += rewrite_boost
    combined = combined / weights
    combined = combined / combined.sum(axis=1, keepdims=True)
    return combined


def clean_rewrite_response(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = text.strip("`\"' \n\t")
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if lines:
        text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def rewrite_with_ollama(
    text: str,
    model: str = DEFAULT_REWRITE_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
) -> str:
    original = str(text or "").strip()
    if not original:
        return ""

    prompt = (
        "Aşağıdaki Türkçe destek mesajını anlamını ve kullanıcının niyetini değiştirmeden "
        "temiz, düzgün ve kısa bir Türkçe cümleye çevir. Kategori tahmini yapma, cevap yazma, "
        "sadece düzeltilmiş mesajı döndür.\n\n"
        f"Mesaj: {original}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 160,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{normalize_ollama_url(ollama_url)}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama rewrite failed for model {model}: {exc}") from exc

    rewritten = clean_rewrite_response(str(result.get("response", "")))
    return rewritten or original


def prepare_dataset(
    dataset_path: Optional[Path] = None,
    merge_labels: bool = True,
) -> PreparedDataset:
    dataset_path = dataset_path or discover_file(MAIN_DATASET_CANDIDATES)
    if dataset_path is None:
        raise FileNotFoundError("Main dataset not found. Pass --dataset-path explicitly.")

    main_df = read_table(dataset_path).copy()
    original_col = find_column(main_df.columns, ORIGINAL_COLUMN_CANDIDATES)
    rewritten_col = find_column(main_df.columns, REWRITTEN_COLUMN_CANDIDATES)
    confidence_col = find_column(main_df.columns, CONFIDENCE_COLUMN_CANDIDATES)
    label_col = find_column(main_df.columns, LABEL_COLUMN_CANDIDATES)

    if original_col is None:
        raise ValueError(f"Original message column not found. Columns: {list(main_df.columns)}")
    if label_col is None:
        raise ValueError(f"Label column not found. Columns: {list(main_df.columns)}")

    df = pd.DataFrame()
    df["row_id"] = np.arange(len(main_df), dtype=int)
    df["original_text"] = main_df[original_col].fillna("").astype(str).str.strip()
    if rewritten_col:
        df["rewritten_text"] = main_df[rewritten_col].fillna("").astype(str).str.strip()
    else:
        df["rewritten_text"] = ""
    if confidence_col:
        df["rewrite_confidence"] = coerce_confidence(main_df[confidence_col])
    else:
        df["rewrite_confidence"] = 1.0

    df["label"] = main_df[label_col].map(serialize_label)
    if merge_labels:
        df["label"] = df["label"].replace(LABEL_MERGE_MAP)
    df["sample_weight"] = df["rewrite_confidence"].map(build_sample_weight)
    df = df[df["original_text"].map(normalize_text).ne("")]
    df = df[df["label"].astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)

    labels = sorted(df["label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    df["label_id"] = df["label"].map(label_to_id).astype(int)

    return PreparedDataset(
        df=df,
        main_dataset_path=str(dataset_path),
        original_column=original_col,
        rewritten_column=rewritten_col,
        confidence_column=confidence_col,
        label_column=label_col,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    return proba / proba.sum(axis=1, keepdims=True)


def _parse_version_tuple(version_text: str) -> tuple[int, ...]:
    main_part = version_text.split("+", 1)[0]
    parts: list[int] = []
    for chunk in main_part.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def train_bert(
    artifact_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    label_names: list[str],
    bert_model_name: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    save_checkpoints: bool,
    auto_resume: bool,
    rewrite_boost: float,
    label_smoothing: float,
    device: str,
    checkpoint_steps: int,
    fp16: bool,
    bf16: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    actual_device = device
    if device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print_json({"cuda_unavailable": True, "fallback_device": "cpu"})
        actual_device = "cpu"

    class SingleTextDataset(Dataset):
        def __init__(
            self,
            texts: list[str],
            labels: np.ndarray,
            weights: np.ndarray,
            tokenizer: Any,
            seq_max_length: int,
        ) -> None:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=seq_max_length,
            )
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.sample_weights = torch.tensor(weights, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
            item["labels"] = self.labels[idx]
            item["sample_weights"] = self.sample_weights[idx]
            return item

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            labels = inputs.pop("labels")
            sample_weights = inputs.pop("sample_weights", None)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
            losses = loss_fct(logits, labels)
            if sample_weights is not None:
                losses = losses * sample_weights.to(losses.device)
            loss = losses.mean()
            return (loss, outputs) if return_outputs else loss

    bert_dir = artifact_dir / "bert_model"
    checkpoint_dir = artifact_dir / "bert_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_strategy = "steps" if save_checkpoints and checkpoint_steps > 0 else ("epoch" if save_checkpoints else "no")
    trainer_output_dir = checkpoint_dir
    resume_checkpoint = None
    model_init_source = bert_model_name
    remaining_steps: Optional[int] = None
    resume_mode = "none"
    checkpoint_state: Optional[dict[str, Any]] = None

    if auto_resume and save_checkpoints:
        resume_checkpoint = get_last_checkpoint(str(checkpoint_dir))
        if resume_checkpoint:
            print_json({"resuming_from_checkpoint": resume_checkpoint})
            trainer_state_path = Path(resume_checkpoint) / "trainer_state.json"
            if trainer_state_path.exists():
                checkpoint_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            torch_version = _parse_version_tuple(torch.__version__)
            if checkpoint_state and torch_version < (2, 6):
                completed_steps = int(checkpoint_state.get("global_step", 0))
                total_steps = int(checkpoint_state.get("max_steps", 0))
                remaining_steps = max(total_steps - completed_steps, 0)
                if remaining_steps > 0:
                    resume_mode = "model_only"
                    model_init_source = resume_checkpoint
                    trainer_output_dir = artifact_dir / f"bert_training_resume_from_{Path(resume_checkpoint).name}"
                    trainer_output_dir.mkdir(parents=True, exist_ok=True)
                    resume_checkpoint = None
                else:
                    resume_mode = "full"
            elif resume_checkpoint:
                resume_mode = "full"

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_init_source,
        num_labels=len(label_names),
        id2label={idx: label for idx, label in enumerate(label_names)},
        label2id={label: idx for idx, label in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    expanded_train_df = expand_text_examples(train_df, rewrite_boost=rewrite_boost)
    train_ds = SingleTextDataset(
        expanded_train_df["training_text"].astype(str).tolist(),
        expanded_train_df["label_id"].to_numpy(dtype=int),
        expanded_train_df["sample_weight"].to_numpy(dtype=float),
        tokenizer,
        max_length,
    )
    val_ds = SingleTextDataset(
        val_df["original_text"].astype(str).tolist(),
        y_val,
        val_df["sample_weight"].to_numpy(dtype=float),
        tokenizer,
        max_length,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        }

    training_kwargs: dict[str, Any] = {
        "output_dir": str(trainer_output_dir),
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(batch_size, 8),
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "use_cpu": (actual_device == "cpu"),
        "logging_steps": 20,
        "logging_strategy": "steps",
        "eval_strategy": "steps" if checkpoint_strategy == "steps" else "epoch",
        "eval_steps": checkpoint_steps if checkpoint_strategy == "steps" else None,
        "save_strategy": checkpoint_strategy,
        "save_steps": checkpoint_steps if checkpoint_strategy == "steps" else None,
        "load_best_model_at_end": save_checkpoints,
        "metric_for_best_model": "macro_f1" if save_checkpoints else None,
        "save_total_limit": 3 if save_checkpoints else None,
        "report_to": "none",
        "fp16": bool(fp16),
        "bf16": bool(bf16),
    }
    if remaining_steps:
        training_kwargs["max_steps"] = int(remaining_steps)

    training_args = TrainingArguments(
        **training_kwargs,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    print_json(
        {
            "training_start": True,
            "requested_device": device,
            "device": actual_device,
            "checkpoint_dir": str(checkpoint_dir),
            "trainer_output_dir": str(trainer_output_dir),
            "save_checkpoints": bool(save_checkpoints),
            "checkpoint_strategy": checkpoint_strategy,
            "checkpoint_steps": int(checkpoint_steps),
            "fp16": bool(fp16),
            "bf16": bool(bf16),
            "auto_resume": bool(auto_resume),
            "resume_mode": resume_mode,
            "remaining_steps": int(remaining_steps) if remaining_steps is not None else None,
            "train_examples": int(len(train_ds)),
            "validation_examples": int(len(val_ds)),
        }
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    original_proba = _softmax(trainer.predict(val_ds).predictions)
    rewrite_mask = val_df["rewritten_text"].fillna("").astype(str).str.strip().ne("").to_numpy()
    rewritten_proba = np.zeros_like(original_proba)
    if rewrite_mask.any():
        rewrite_ds = SingleTextDataset(
            val_df.loc[rewrite_mask, "rewritten_text"].astype(str).tolist(),
            y_val[rewrite_mask],
            val_df.loc[rewrite_mask, "sample_weight"].to_numpy(dtype=float),
            tokenizer,
            max_length,
        )
        rewritten_proba[rewrite_mask] = _softmax(trainer.predict(rewrite_ds).predictions)

    val_proba = combine_probabilities(
        original_proba=original_proba,
        rewritten_proba=rewritten_proba,
        rewrite_mask=rewrite_mask,
        rewrite_boost=rewrite_boost,
    )

    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)

    val_pred = val_proba.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "bert_model_name": bert_model_name,
        "resume_mode": resume_mode,
        "resume_source_checkpoint": str(model_init_source) if model_init_source != bert_model_name else None,
    }
    return val_proba, metrics


class BertInferenceModel:
    def __init__(self, bert_dir: Path) -> None:
        import torch
        disable_transformers_torchvision()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(self.device)
        self.model.eval()

    def predict_texts(self, texts: list[str], max_length: int) -> np.ndarray:
        with self.torch.no_grad():
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.detach().cpu().numpy()
        return _softmax(logits)


class BertPredictor:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        with open(self.artifact_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(self.artifact_dir / "label_mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.labels = np.array(mapping["id_to_label"], dtype=object)
        self.rewrite_boost = float(self.metadata.get("rewrite_boost", 1.5))
        self.train_rewrite_boost = float(self.metadata.get("train_rewrite_boost", self.rewrite_boost))
        self.bert_model = BertInferenceModel(self.artifact_dir / "bert_model")

    def decode_labels(self, label_ids: np.ndarray) -> np.ndarray:
        return self.labels[label_ids.astype(int)]

    def predict_batch(
        self,
        originals: list[str],
        rewritten_texts: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        rewritten_texts = rewritten_texts or [""] * len(originals)
        original_proba = self.bert_model.predict_texts(
            originals,
            max_length=int(self.metadata["bert_max_length"]),
        )
        rewrite_mask = np.array([bool(str(text).strip()) for text in rewritten_texts], dtype=bool)
        rewritten_proba = np.zeros_like(original_proba)
        if rewrite_mask.any():
            rewrite_inputs = [rewritten_texts[i] for i, flag in enumerate(rewrite_mask) if flag]
            rewritten_proba[rewrite_mask] = self.bert_model.predict_texts(
                rewrite_inputs,
                max_length=int(self.metadata["bert_max_length"]),
            )
        bert_proba = combine_probabilities(
            original_proba=original_proba,
            rewritten_proba=rewritten_proba,
            rewrite_mask=rewrite_mask,
            rewrite_boost=self.rewrite_boost,
        )
        return {
            "bert_proba": bert_proba,
            "prediction_proba": bert_proba,
        }


def run_training(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataset(
        dataset_path=Path(args.dataset_path).resolve() if args.dataset_path else None,
        merge_labels=args.merge_labels,
    )
    df = prepared.df.copy()
    label_names = sorted(df["label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    df["label_id"] = df["label"].map(label_to_id).astype(int)
    sample_per_class = int(getattr(args, "sample_per_class", 0) or 0)
    if sample_per_class > 0:
        df = pd.concat(
            [
                group.sample(n=min(len(group), sample_per_class), random_state=args.random_state)
                for _, group in df.groupby("label_id", group_keys=False)
            ],
            ignore_index=True,
        )

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label_id"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    y_train = train_df["label_id"].to_numpy(dtype=int)
    y_val = val_df["label_id"].to_numpy(dtype=int)

    bert_val_proba, bert_metrics = train_bert(
        artifact_dir=artifact_dir,
        train_df=train_df,
        val_df=val_df,
        y_train=y_train,
        y_val=y_val,
        label_names=label_names,
        bert_model_name=args.bert_model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_checkpoints=args.save_checkpoints,
        auto_resume=args.auto_resume,
        rewrite_boost=args.rewrite_boost,
        label_smoothing=args.label_smoothing,
        device=args.device,
        checkpoint_steps=args.checkpoint_steps,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    pred_idx = bert_val_proba.argmax(axis=1)
    pred_labels = [label_names[int(idx)] for idx in pred_idx]
    actual_labels = [label_names[int(idx)] for idx in y_val]
    report = classification_report(actual_labels, pred_labels, output_dict=True, zero_division=0)

    df.to_csv(artifact_dir / "prepared_dataset.csv", index=False, encoding="utf-8-sig")
    val_df.assign(
        bert_pred=pred_labels,
        bert_confidence=bert_val_proba.max(axis=1).round(6),
    ).to_csv(artifact_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    label_mapping = {
        "id_to_label": label_names,
        "label_to_id": label_to_id,
    }
    with open(artifact_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    metadata = {
        "model_type": "bert_only",
        "main_dataset_path": prepared.main_dataset_path,
        "original_column": prepared.original_column,
        "rewritten_column": prepared.rewritten_column,
        "confidence_column": prepared.confidence_column,
        "label_column": prepared.label_column,
        "dataset_rows": int(len(df)),
        "sample_per_class": sample_per_class or None,
        "label_count": int(len(label_names)),
        "labels": label_names,
        "bert_model_name": args.bert_model_name,
        "bert_max_length": int(args.max_length),
        "train_rewrite_boost": float(args.rewrite_boost),
        "rewrite_boost": float(args.rewrite_boost),
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "label_smoothing": float(args.label_smoothing),
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "merge_labels": bool(args.merge_labels),
            "save_checkpoints": bool(args.save_checkpoints),
            "auto_resume": bool(args.auto_resume),
            "device": str(args.device),
            "checkpoint_steps": int(args.checkpoint_steps),
            "fp16": bool(args.fp16),
            "bf16": bool(args.bf16),
            "sample_per_class": sample_per_class or None,
        },
        "bert_metrics": bert_metrics,
        "classification_report": report,
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print_json(
        {
            "artifact_dir": str(artifact_dir),
            "dataset_rows": int(len(df)),
            "label_count": int(len(label_names)),
            "bert_metrics": bert_metrics,
        }
    )


def run_sample_test(args: argparse.Namespace) -> None:
    predictor = BertPredictor(args.artifact_dir)
    prepared_csv = Path(args.artifact_dir) / "prepared_dataset.csv"
    if not prepared_csv.exists():
        raise FileNotFoundError(f"Prepared dataset snapshot not found: {prepared_csv}")
    df = pd.read_csv(prepared_csv)
    sample = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).reset_index(drop=True)
    outputs = predictor.predict_batch(
        sample["original_text"].fillna("").astype(str).tolist(),
        sample["rewritten_text"].fillna("").astype(str).tolist(),
    )
    proba = outputs["bert_proba"]
    pred_idx = proba.argmax(axis=1)
    sample["bert_pred"] = predictor.decode_labels(pred_idx.astype(int))
    sample["bert_confidence"] = proba.max(axis=1).round(4)
    sample["bert_correct"] = sample["label"].astype(str) == sample["bert_pred"].astype(str)
    output_file = Path(args.output_file) if args.output_file else Path(args.artifact_dir) / f"sample_predictions_{len(sample)}.xlsx"
    sample.to_excel(output_file, index=False)
    print_json(
        {
            "output_file": str(output_file),
            "sample_size": int(len(sample)),
            "accuracy_on_sample": round(float(sample["bert_correct"].mean()), 4),
        }
    )


def run_predict(args: argparse.Namespace) -> None:
    predictor = BertPredictor(args.artifact_dir)
    rewritten_text = args.rewritten_text
    if args.auto_rewrite and not str(rewritten_text or "").strip():
        rewritten_text = rewrite_with_ollama(
            args.text,
            model=args.rewrite_model,
            ollama_url=args.ollama_url,
            timeout=args.rewrite_timeout,
        )
    outputs = predictor.predict_batch([args.text], [rewritten_text])
    proba = outputs["bert_proba"]
    pred_idx = proba.argmax(axis=1)
    result = {
        "prediction": predictor.decode_labels(pred_idx)[0],
        "confidence": round(float(proba.max(axis=1)[0]), 4),
        "model_type": "bert_only",
        "rewritten_text": rewritten_text,
        "rewrite_model": args.rewrite_model if args.auto_rewrite else None,
    }
    print_json(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a BERT-only support message classifier.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    train_parser = subparsers.add_parser("train", help="Train BERT and save artifacts.")
    train_parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH), help="Main dataset path.")
    train_parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Directory for trained artifacts.")
    train_parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Validation split ratio.")
    train_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed.")
    train_parser.add_argument("--bert-model-name", type=str, default=DEFAULT_BERT_MODEL_NAME, help="Hugging Face model name.")
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="BERT epoch count.")
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="BERT batch size.")
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="BERT max token length.")
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="BERT learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="BERT weight decay.")
    train_parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="BERT warmup ratio.")
    train_parser.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING, help="Label smoothing factor.")
    train_parser.add_argument("--rewrite-boost", type=float, default=DEFAULT_REWRITE_BOOST, help="Rewrite probability fusion/training weight.")
    train_parser.add_argument("--save-checkpoints", action="store_true", default=DEFAULT_SAVE_CHECKPOINTS, help="Save training checkpoints.")
    train_parser.add_argument("--no-save-checkpoints", action="store_false", dest="save_checkpoints", help="Disable epoch checkpoints.")
    train_parser.add_argument("--auto-resume", action="store_true", default=DEFAULT_AUTO_RESUME, help="Resume from the latest checkpoint when available.")
    train_parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume", help="Start from scratch even if checkpoints exist.")
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda", help="Training device. Defaults to cuda and falls back to cpu if CUDA is unavailable.")
    train_parser.add_argument("--checkpoint-steps", type=int, default=DEFAULT_CHECKPOINT_STEPS, help="Save/evaluate every N steps when checkpoints are enabled.")
    train_parser.add_argument("--fp16", action="store_true", help="Enable CUDA fp16 mixed precision training.")
    train_parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision training when supported.")
    train_parser.add_argument("--sample-per-class", type=int, default=0, help="Optional balanced sample limit per class for quick tuning/smoke runs. 0 uses all rows.")
    train_parser.add_argument("--merge-labels", action="store_true", default=True, help="Merge similar label categories.")
    train_parser.add_argument("--no-merge-labels", action="store_false", dest="merge_labels", help="Disable label category merging.")
    train_parser.set_defaults(func=run_training)

    sample_parser = subparsers.add_parser("sample-test", help="Run a random labeled sample through BERT.")
    sample_parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Trained artifact directory.")
    sample_parser.add_argument("--sample-size", type=int, default=50, help="How many random rows to export.")
    sample_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    sample_parser.add_argument("--output-file", type=str, default=None, help="Optional custom Excel output file path.")
    sample_parser.set_defaults(func=run_sample_test)

    predict_parser = subparsers.add_parser("predict", help="Predict one message with BERT.")
    predict_parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Trained artifact directory.")
    predict_parser.add_argument("--text", type=str, required=True, help="Original user message.")
    predict_parser.add_argument("--rewritten-text", type=str, default="", help="Optional rewritten message.")
    predict_parser.add_argument("--auto-rewrite", action="store_true", help="Rewrite the message with Ollama before prediction.")
    predict_parser.add_argument("--rewrite-model", type=str, default=DEFAULT_REWRITE_MODEL, help="Ollama model used for live rewrite.")
    predict_parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama base URL.")
    predict_parser.add_argument("--rewrite-timeout", type=int, default=120, help="Ollama rewrite timeout in seconds.")
    predict_parser.set_defaults(func=run_predict)

    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv:
        argv = ["train"]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.error(f"Unknown command: {args.command}")
    args.func(args)


if __name__ == "__main__":
    main()
