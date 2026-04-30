from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .ollama_client import OllamaClient
from .prompts import build_rewrite_messages


@dataclass
class RewriteResult:
    original_text: str
    rewritten_text: str
    model: str


class RewriteService:
    def __init__(self, config: AppConfig, client: OllamaClient | None = None) -> None:
        self.config = config
        self.client = client or OllamaClient(
            config.ollama_url,
            timeout=config.rewrite_timeout,
            num_gpu=config.ollama_num_gpu,
        )

    def rewrite(self, message: str) -> RewriteResult:
        original = str(message or "").strip()
        if not original:
            return RewriteResult(original_text="", rewritten_text="", model=self.config.rewrite_model)
        rewritten = self.client.chat(
            model=self.config.rewrite_model,
            messages=build_rewrite_messages(original),
            temperature=0.0,
            num_predict=180,
            think=False,
        )
        return RewriteResult(
            original_text=original,
            rewritten_text=rewritten or original,
            model=self.config.rewrite_model,
        )
