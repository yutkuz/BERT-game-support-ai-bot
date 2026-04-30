from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_config
from .rewrite_service import RewriteService
from .training import BertPredictor


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    use_rewrite: bool = True


class BatchPredictRequest(BaseModel):
    messages: list[str] = Field(..., min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=10)
    use_rewrite: bool = True


class CategoryScore(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    original_text: str
    rewritten_text: str
    prediction: str
    confidence: float
    top_categories: list[CategoryScore]
    rewrite_model: str
    rewrite_used: bool
    rewrite_error: str | None = None


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class DatasetSampleResponse(BaseModel):
    count: int
    total_rows: int
    message_column: str
    messages: list[str]


class LegacyPredictItem(BaseModel):
    message: str
    rewrittenText: str = ""


class LegacyPredictRequest(BaseModel):
    jobId: str | None = None
    items: list[LegacyPredictItem] | None = None
    messages: list[str] | None = None
    rewrittenTexts: list[str] | None = None
    rewriteEnabled: bool = False


class CorrectionRecord(BaseModel):
    message: str = ""
    rewrittenText: str = ""
    predictedLabel: str = ""
    correctLabel: str = ""
    confidence: float | None = None


app = FastAPI(title="Support Bot V7 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def json_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def feedback_path() -> Path:
    return get_config().artifact_dir.parent.parent / "feedback_dataset.csv"


def feedback_status() -> dict[str, Any]:
    path = feedback_path()
    if not path.exists():
        return {"feedbackPath": str(path), "feedbackCount": 0}
    try:
        return {"feedbackPath": str(path), "feedbackCount": max(0, len(pd.read_csv(path)))}
    except Exception:
        return {"feedbackPath": str(path), "feedbackCount": 0}


@lru_cache(maxsize=1)
def get_rewrite_service() -> RewriteService:
    return RewriteService(get_config())


@lru_cache(maxsize=1)
def get_predictor() -> BertPredictor:
    config = get_config()
    metadata_path = Path(config.artifact_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Trained model was not found at {config.artifact_dir}. Run scripts/run_pipeline.py first."
        )
    return BertPredictor(config.artifact_dir)


def build_response(message: str, top_k: int, use_rewrite: bool = True) -> PredictResponse:
    config = get_config()
    original = str(message or "").strip()
    rewritten = ""
    rewrite_error = None
    if use_rewrite:
        try:
            rewrite_result = get_rewrite_service().rewrite(original)
            original = rewrite_result.original_text
            rewritten = rewrite_result.rewritten_text
        except RuntimeError as exc:
            rewrite_error = str(exc)
    predictor = get_predictor()
    outputs = predictor.predict_batch(
        [original],
        [rewritten] if rewritten else None,
    )
    proba = outputs["prediction_proba"][0]
    sorted_indices = np.argsort(proba)[::-1][:top_k]
    top_categories = [
        CategoryScore(category=str(predictor.labels[index]), confidence=round(float(proba[index]), 4))
        for index in sorted_indices
    ]
    winner = top_categories[0]
    return PredictResponse(
        original_text=original,
        rewritten_text=rewritten,
        prediction=winner.category,
        confidence=winner.confidence,
        top_categories=top_categories,
        rewrite_model=config.rewrite_model,
        rewrite_used=bool(rewritten),
        rewrite_error=rewrite_error,
    )


def build_auto_reply(label: str, confidence: float | None = None) -> dict[str, object]:
    threshold = get_config().human_review_confidence_threshold
    human_review_labels = {
        "ban_ceza_itirazi",
        "giris_hesap_erisim",
        "hesap_islemleri",
        "hile_itirazi",
        "teknik_uygulama_sorunu",
    }
    replies = {
        "arkadas_adina_ban_itirazi": "Sistem logları ve kayıtları açıktır. Başkası adına itirazda bulunamazsınız. İlgili kullanıcının kendi hesabıyla destek talebi oluşturması gerekir.",
        "ban_ceza_itirazi": "Ceza itirazınız kayıt altına alınmıştır. Moderasyon ekibi sistem loglarını inceleyerek gerekli kontrolü yapacaktır.",
        "bilgi_sorusu": "Ana ekranda yer alan ? butonuna basarak oyun hakkında detaylı bilgi alabilirsiniz. Bilet sistemi için profilinizdeki Koleksiyon alanındaki i butonunu kullanabilirsiniz.",
        "davet_odulu": "Davet edilen kişinin daha önce hesap açmamış olması gerekir. Ödül, davet ettiğiniz kullanıcının ilk oyununu tamamlamasının ardından hesabınıza otomatik yüklenir.",
        "giris_hesap_erisim": "Giriş sorununuz kayıt altına alınmıştır. Google, Apple veya Facebook hesabınızla ilgili erişim kontrolü için destek ekibi inceleme yapacaktır.",
        "hediye_bonus_talebi": "Sistemin otomatik verdiği hediyeler dışında manuel Manc Altın, çip, üyelik paketi veya bonus tanımlaması yapılmamaktadır.",
        "hesap_islemleri": "Hesap işlemleri talebiniz kayıt altına alınmıştır. Güvenlik nedeniyle gerekli kontroller destek ekibi tarafından yapılacaktır.",
        "hesap_silme_kapatma": "Profilinize girip profil düzenleme ekranındaki Hesabı Sil seçeneğiyle hesabınızı silebilirsiniz. Hesaba tekrar giriş yapmazsanız hesap 7 gün içinde kalıcı olarak silinir.",
        "hile_itirazi": "Hile itirazınız kayıt altına alınmıştır. Sistem logları ve oyun kayıtları moderasyon ekibi tarafından incelenecektir.",
        "iade_tazmin_talebi": "Çip iadesi talebiniz kayıt altına alınmıştır. Sistem tarafından silinen Manc Altın/çip görünmüyorsa iade işlemi yapılamamaktadır. Çiplerinizin silindiğini düşünüyorsanız işlem zamanı, yaklaşık tutar ve varsa ekran görüntüsüyle birlikte destek@mancgames.com adresine iletebilirsiniz.",
        "kullanici_adi_degistirme": "Kullanıcı adı değişikliği talebiniz alınmıştır. Değiştirmek istediğiniz kullanıcı adını iletebilirsiniz; uygunluk kontrolünden sonra değerlendirilecektir.",
        "odeme_cip_yukleme": "Satın alımınıza ait Google Play GPA kodlu veya Apple Store sipariş numarasını/dekont görüntüsünü destek@mancgames.com adresine iletebilirsiniz. Sistemimize ulaşan satın almalar otomatik yüklenir.",
        "olumlu_geri_bildirim": "Geri bildiriminiz için teşekkür ederiz. İyi oyunlar dileriz.",
        "oyun_adalet_puan": "Oyuna müdahalemiz yoktur. Kurallar ve oyun mekanikleri tüm kullanıcılar için eşittir; satın alma işlemi kazanma/kaybetme durumunu değiştirmez.",
        "oyun_nasil_oynanir_yardim": "Ana ekranda yer alan ? butonundan oyun hakkında detaylı bilgi alabilirsiniz. Elinizi açmak için taşlarınızı kurala uygun dizip sıra size geldiğinde fazla taşı ortadaki kapalı taşların üzerine bırakmanız gerekir.",
        "oyuncu_sikayet_raporlama": "Oyuncu şikayetlerinizi, şikayet etmek istediğiniz kullanıcının profilindeki Şikayet Et düğmesini kullanarak iletebilirsiniz.",
        "ozel_direkt_mesaj": "Diğer kullanıcılara mesaj atabilmek için Altın, Platin veya Elmas Manc kullanıcı hesabına sahip olmanız gerekir.",
        "ozellik_degisim_talebi": "Bildiriminiz kayıt altına alınmıştır. Özellik değişikliği öneriniz ilgili ekip tarafından değerlendirilmek üzere iletilecektir.",
        "profil_fotografi_degistirme": "Kullanıcı adı ve profil resmi, Google, Apple veya Facebook hesabınızla giriş yaptığınızda hesabınıza eklenebilir/güncellenebilir.",
        "reklam_odulu": "Reklam yayınları Google AdMob tarafından sağlanmaktadır. AdMob size reklam sundukça reklam izleyebilirsiniz. Reklam ödülü yüklenmediyse kısa süre sonra tekrar kontrol ediniz.",
        "ses_sorunu": "Telefon ayarlarınızdan ses ve mikrofon izinlerini kontrol edin. Ardından oyun içindeki ayarlar menüsünden ses ve mikrofonun aktif olduğundan emin olun.",
        "sosyal_hesap_giris_gecis": "Google, Apple veya Facebook ile girişlerde cihazınızda aktif olan doğru hesabı seçtiğinizden emin olun. Farklı hesaba geçmek için önce mevcut oturumdan çıkış yapmanız gerekir.",
        "teknik_uygulama_sorunu": "Hata bildiriminiz sistemimiz tarafından kayıt altına alınmıştır. Teknik ekibimiz tarafından inceleme başlatılmıştır. İyi oyunlar.",
        "toplu_sikayet": "Kullanıcılarla ilgili şikayetlerinizi, şikayet etmek istediğiniz kullanıcının profil sayfasında yer alan Şikayet Et düğmesini kullanarak iletebilirsiniz. Toplu şikayet bildiriminiz ayrıca kayıt altına alınmıştır ve moderatörler tarafından incelenecektir.",
    }
    reply = replies.get(
        label,
        "Mesajınızı aldık. Talebinizi daha net inceleyebilmemiz için lütfen problemi kısa ve somut detaylarla tekrar iletin.",
    )
    return {
        "reply": reply,
        "requiresHumanReview": label in human_review_labels or (
            confidence is not None and confidence < threshold
        ),
        "hasTemplate": label in replies,
    }


def legacy_result(response: PredictResponse, row: int, message: str) -> dict[str, object]:
    auto_reply = build_auto_reply(response.prediction, response.confidence)
    return {
        "row": row,
        "message": message,
        "rewrittenText": response.rewritten_text,
        "rewriteUsed": response.rewrite_used,
        "rewriteRequested": bool(response.rewrite_used or response.rewrite_error),
        "rewriteError": response.rewrite_error or "",
        "prediction": response.prediction,
        "confidence": response.confidence,
        "autoReply": auto_reply["reply"],
        "requiresHumanReview": auto_reply["requiresHumanReview"],
        "hasReplyTemplate": auto_reply["hasTemplate"],
        "top3": [
            {"label": item.category, "confidence": item.confidence}
            for item in response.top_categories
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    config = get_config()
    return {
        "ok": True,
        "rewrite_model": config.rewrite_model,
        "artifact_dir": str(config.artifact_dir),
        "model_ready": (Path(config.artifact_dir) / "metadata.json").exists(),
        "dataset_ready": Path(config.dataset_path).exists(),
        "dataset_path": str(config.dataset_path),
    }


@app.get("/status")
def legacy_status() -> dict[str, Any]:
    config = get_config()
    metadata_path = Path(config.artifact_dir) / "metadata.json"
    label_mapping_path = Path(config.artifact_dir) / "label_mapping.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = metadata.get("bert_metrics", {})
    labels: list[str] = []
    if label_mapping_path.exists():
        label_mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
        labels = [str(label) for label in label_mapping.get("id_to_label", [])]
    else:
        try:
            labels = get_predictor().labels.tolist()
        except Exception:
            labels = []
    return {
        "artifactDir": str(config.artifact_dir),
        "modelName": metadata.get("bert_model_name", config.bert_model_name),
        "labelCount": len(labels),
        "labels": labels,
        "accuracy": metrics.get("accuracy"),
        "macroF1": metrics.get("macro_f1"),
        "loaded": True,
        "feedback": feedback_status(),
        "humanReviewConfidenceThreshold": config.human_review_confidence_threshold,
        "rewrite": {
            "enabled": True,
            "model": config.rewrite_model,
            "url": config.ollama_url,
        },
    }


@app.get("/labels")
def labels() -> dict[str, Any]:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"labels": predictor.labels.tolist()}


@app.get("/dataset/sample", response_model=DatasetSampleResponse)
def dataset_sample(count: int = 10) -> DatasetSampleResponse:
    config = get_config()
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset was not found at {dataset_path}")
    count = max(1, min(int(count or 10), 200))
    try:
        df = pd.read_excel(dataset_path, usecols=[config.message_column])
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Column '{config.message_column}' was not found in {dataset_path.name}",
        ) from exc
    messages = (
        df[config.message_column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    messages = messages[messages.astype(bool)]
    sampled = messages.sample(n=min(count, len(messages)), random_state=None).tolist()
    return DatasetSampleResponse(
        count=len(sampled),
        total_rows=int(len(messages)),
        message_column=config.message_column,
        messages=sampled,
    )


@app.post("/sample-upload")
async def legacy_sample_upload(file: UploadFile, count: int = 10) -> dict[str, Any]:
    count = max(1, min(int(count or 10), 200))
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    content = await file.read()
    import io

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))

    config = get_config()
    candidates = [
        config.message_column,
        "original_text",
        "original_message",
        "mesaj",
        "message",
        "user_message",
        "kullanici_mesaji_original",
        "text",
    ]
    column = next((name for name in candidates if name in df.columns), None)
    if column is None:
        column = str(df.columns[0])
    messages = df[column].dropna().astype(str).map(str.strip)
    messages = messages[messages.astype(bool)]
    sampled = messages.sample(n=min(count, len(messages)), random_state=None).tolist()
    rewrite_column = config.rewrite_column if config.rewrite_column in df.columns else None
    return {
        "filename": filename,
        "column": column,
        "rewriteColumn": rewrite_column,
        "rowCount": int(len(messages)),
        "sampleCount": len(sampled),
        "messages": sampled,
    }


@app.post("/cancel")
def legacy_cancel(request: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, "jobId": str((request or {}).get("jobId") or "")}


@app.post("/corrections")
def legacy_corrections(request: dict[str, Any]) -> dict[str, Any]:
    records = request.get("records")
    if isinstance(request.get("record"), dict):
        records = [request["record"]]
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="records listesi bekleniyor.")
    path = feedback_path()
    rows = [CorrectionRecord.model_validate(record).model_dump() for record in records]
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    status = feedback_status()
    return {"ok": True, **status}


@app.post("/predict")
def predict(request: dict[str, Any]) -> Any:
    try:
        if "items" in request or "messages" in request or "rewriteEnabled" in request:
            legacy_request = LegacyPredictRequest.model_validate(request)
            if legacy_request.items is not None:
                messages = [item.message for item in legacy_request.items if item.message.strip()]
            else:
                messages = [str(message).strip() for message in (legacy_request.messages or []) if str(message).strip()]
            if not messages:
                raise HTTPException(status_code=400, detail="Tahmin icin en az bir mesaj girin.")
            import time

            started = time.perf_counter()
            results = [
                legacy_result(
                    build_response(message, 3, legacy_request.rewriteEnabled),
                    index,
                    message,
                )
                for index, message in enumerate(messages, start=1)
            ]
            return {
                "jobId": legacy_request.jobId or "",
                "cancelled": False,
                "total": len(messages),
                "completed": len(results),
                "elapsedSeconds": round(time.perf_counter() - started, 2),
                "results": results,
            }
        normal_request = PredictRequest.model_validate(request)
        return build_response(normal_request.message, normal_request.top_k, normal_request.use_rewrite)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    try:
        messages = [message for message in request.messages if str(message).strip()]
        return BatchPredictResponse(
            results=[build_response(message, request.top_k, request.use_rewrite) for message in messages]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
