from __future__ import annotations

import json
import traceback
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import WEB_DIR
from .feedback_dataset import feedback_dataset_store
from .jobs import jobs
from .predictor_service import normalize_prediction_items, predictor_service
from .rewriter_service import rewriter_service
from .uploads import extract_multipart_file, sample_messages_from_upload


def json_payload(data: object, status: int = HTTPStatus.OK) -> tuple[int, bytes, str]:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return status, payload, "application/json; charset=utf-8"


class SupportBotHandler(BaseHTTPRequestHandler):
    server_version = "SupportBotV1/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            status = predictor_service.status()
            status["feedback"] = feedback_dataset_store.status()
            status["rewrite"] = rewriter_service.status()
            self.send_json(status)
            return

        self.serve_static_file(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/predict":
                self.handle_predict()
                return
            if parsed.path == "/api/cancel":
                self.handle_cancel()
                return
            if parsed.path == "/api/sample-upload":
                self.handle_sample_upload()
                return
            if parsed.path == "/api/corrections":
                self.handle_corrections()
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_payload(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data: object, status: int = HTTPStatus.OK) -> None:
        code, payload, content_type = json_payload(data, status)
        self.send_payload(code, payload, content_type)

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length)

    def serve_static_file(self, request_path: str) -> None:
        route = request_path.strip("/") or "index.html"
        target = (WEB_DIR / route).resolve()
        web_root = WEB_DIR.resolve()

        if not _is_child_path(target, web_root) or not target.exists() or target.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = _content_type_for(target)
        self.send_payload(HTTPStatus.OK, target.read_bytes(), content_type)

    def handle_predict(self) -> None:
        data = json.loads(self.read_body().decode("utf-8"))
        job_id = str(data.get("jobId") or f"job-{time.time_ns()}")
        items = self.read_prediction_items(data)
        rewrite_enabled = bool(data.get("rewriteEnabled"))
        if not items:
            self.send_json({"error": "Tahmin icin en az bir mesaj girin."}, HTTPStatus.BAD_REQUEST)
            return

        jobs.start(job_id)
        started = time.perf_counter()
        try:
            results = predictor_service.predict_items(items, job_id, rewrite_enabled=rewrite_enabled)
            cancelled = jobs.is_cancelled(job_id)
        finally:
            jobs.finish(job_id)

        self.send_json(
            {
                "jobId": job_id,
                "cancelled": cancelled,
                "total": len(items),
                "completed": len(results),
                "elapsedSeconds": round(time.perf_counter() - started, 2),
                "results": results,
            }
        )

    def read_prediction_items(self, data: dict[str, object]) -> list[dict[str, str]]:
        if isinstance(data.get("items"), list):
            return normalize_prediction_items(data["items"])

        messages = data.get("messages", [])
        rewrites = data.get("rewrittenTexts", [])
        if not isinstance(messages, list):
            return []
        if not isinstance(rewrites, list):
            rewrites = []

        items = [
            {
                "message": str(message),
                "rewrittenText": str(rewrites[index]) if index < len(rewrites) else "",
            }
            for index, message in enumerate(messages)
        ]
        return normalize_prediction_items(items)

    def handle_cancel(self) -> None:
        data = json.loads(self.read_body().decode("utf-8") or "{}")
        job_id = str(data.get("jobId") or "")
        jobs.cancel(job_id)
        self.send_json({"ok": True, "jobId": job_id})

    def handle_sample_upload(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        count = max(1, min(200, int(query.get("count", ["10"])[0] or 10)))
        filename, payload = extract_multipart_file(self.headers.get("Content-Type", ""), self.read_body())
        self.send_json(sample_messages_from_upload(filename, payload, count))

    def handle_corrections(self) -> None:
        data = json.loads(self.read_body().decode("utf-8") or "{}")
        records = data.get("records", [])
        if isinstance(data.get("record"), dict):
            records = [data["record"]]
        if not isinstance(records, list):
            self.send_json({"error": "records listesi bekleniyor."}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json(feedback_dataset_store.append_many(records))


def _is_child_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _content_type_for(path: Path) -> str:
    content_types = {
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".jsx": "text/babel; charset=utf-8",
    }
    return content_types.get(path.suffix.lower(), "application/octet-stream")

