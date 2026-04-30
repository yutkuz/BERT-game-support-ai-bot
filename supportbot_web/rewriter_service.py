from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.support_bot.prompts import REWRITE_SYSTEM_PROMPT

from .config import (
    GEMMA_REWRITE_ENABLED,
    GEMMA_REWRITE_MODEL,
    GEMMA_REWRITE_PROMPT_PATH,
    GEMMA_REWRITE_TIMEOUT_SECONDS,
    GEMMA_REWRITE_URL,
)


class GemmaRewriter:
    def __init__(
        self,
        enabled: bool = GEMMA_REWRITE_ENABLED,
        model: str = GEMMA_REWRITE_MODEL,
        url: str = GEMMA_REWRITE_URL,
        timeout_seconds: float = GEMMA_REWRITE_TIMEOUT_SECONDS,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.url = url
        self.timeout_seconds = timeout_seconds

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "url": self.url,
            "timeoutSeconds": self.timeout_seconds,
            "promptPath": str(GEMMA_REWRITE_PROMPT_PATH),
        }

    def rewrite(self, message: str) -> str:
        text = str(message or "").strip()
        if not self.enabled or not text:
            return ""

        payload = {
            "model": self.model,
            "messages": build_rewrite_messages(text),
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 180,
            },
        }
        request = Request(
            _chat_url(self.url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Gemma rewrite basarisiz: {exc}") from exc

        rewritten = _extract_chat_content(data)
        return _clean_rewrite(rewritten, fallback=text)


def build_rewrite_messages(message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_rewrite_system_prompt()},
        {"role": "user", "content": f"Orijinal kullanici mesaji:\n{message}"},
    ]


def load_rewrite_system_prompt() -> str:
    if GEMMA_REWRITE_PROMPT_PATH.exists():
        content = GEMMA_REWRITE_PROMPT_PATH.read_text(encoding="utf-8")
        extracted = _extract_markdown_prompt(content)
        if extracted:
            return extracted
    return REWRITE_SYSTEM_PROMPT


def _extract_markdown_prompt(content: str) -> str:
    lines: list[str] = []
    in_user_format = False
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        normalized = line.strip().lower()
        if normalized.startswith("## user prompt"):
            in_user_format = True
            continue
        if in_user_format:
            continue
        if normalized.startswith("#") or normalized.startswith("```"):
            continue
        if normalized.startswith("- "):
            line = line.strip()[2:]
        lines.append(line)
    prompt = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", prompt)


def _chat_url(url: str) -> str:
    normalized = str(url or "http://127.0.0.1:11434").strip().rstrip("/")
    if not re.match(r"^https?://", normalized):
        normalized = f"http://{normalized}"
    if normalized.endswith("/api/chat"):
        return normalized
    if normalized.endswith("/api/generate"):
        return normalized.rsplit("/", 1)[0] + "/chat"
    return normalized + "/api/chat"


def _extract_chat_content(data: dict[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(data.get("response", "")).strip()


def _clean_rewrite(value: str, fallback: str) -> str:
    cleaned = str(value or "").strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"`+", "", cleaned).strip()
    cleaned = re.sub(r"^rewrite\s*[:ï¼š]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or cleaned == fallback:
        return ""
    return cleaned


rewriter_service = GemmaRewriter()

