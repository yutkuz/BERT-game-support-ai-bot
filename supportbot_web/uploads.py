from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd

from src.support_bot.training import REWRITTEN_COLUMN_CANDIDATES, find_column

from .config import MESSAGE_COLUMN_CANDIDATES


def extract_multipart_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Dosya yukleme isteginde boundary bulunamadi.")

    boundary = ("--" + content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')).encode()
    for part in body.split(boundary):
        if b"Content-Disposition" not in part or b"filename=" not in part:
            continue

        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue

        header_text = header_blob.decode("utf-8", errors="ignore")
        disposition = next(
            (line for line in header_text.splitlines() if line.lower().startswith("content-disposition:")),
            "",
        )
        filename = _extract_filename(disposition)
        return filename, payload.rstrip(b"\r\n-")

    raise ValueError("Yuklenen dosya bulunamadi.")


def _extract_filename(disposition: str) -> str:
    for chunk in disposition.split(";"):
        chunk = chunk.strip()
        if chunk.startswith("filename="):
            return chunk.split("=", 1)[1].strip().strip('"') or "upload"
    return "upload"


def read_table_from_upload(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(payload)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)
    raise ValueError("Sadece CSV, XLSX veya XLS dosyalari destekleniyor.")


def sample_messages_from_upload(filename: str, payload: bytes, count: int) -> dict[str, object]:
    df = read_table_from_upload(filename, payload)
    message_col = find_column(df.columns, MESSAGE_COLUMN_CANDIDATES)
    if message_col is None:
        raise ValueError("Dosyada kullanici mesaji kolonu bulunamadi.")
    rewrite_col = find_column(df.columns, REWRITTEN_COLUMN_CANDIDATES)

    rows = []
    for _, row in df.iterrows():
        message = str(row.get(message_col, "")).strip()
        if not message:
            continue
        rows.append(
            {
                "message": message,
                "rewrittenText": str(row.get(rewrite_col, "")).strip() if rewrite_col else "",
            }
        )

    sample = random.sample(rows, k=min(count, len(rows))) if rows else []
    return {
        "filename": filename,
        "column": message_col,
        "rewriteColumn": rewrite_col,
        "rowCount": len(df),
        "sampleCount": len(sample),
        "items": sample,
        "messages": [item["message"] for item in sample],
        "rewrittenTexts": [item["rewrittenText"] for item in sample],
    }

