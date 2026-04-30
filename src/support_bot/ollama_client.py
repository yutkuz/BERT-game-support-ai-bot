from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any


def normalize_ollama_url(url: str) -> str:
    normalized = str(url or "http://127.0.0.1:11434").strip().rstrip("/")
    if not re.match(r"^https?://", normalized):
        normalized = f"http://{normalized}"
    return normalized


def clean_llm_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"`+", "", cleaned)
    cleaned = cleaned.strip(" \"'\n\t")
    return re.sub(r"\s+", " ", cleaned).strip()


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 180, num_gpu: int = -1) -> None:
        self.base_url = normalize_ollama_url(base_url)
        self.timeout = timeout
        self.num_gpu = num_gpu

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        num_predict: int = 180,
        think: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if self.num_gpu >= 0:
            payload["options"]["num_gpu"] = self.num_gpu
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Ollama request failed for model {model}: {exc}") from exc
        message = result.get("message") or {}
        return clean_llm_text(str(message.get("content", "")))
