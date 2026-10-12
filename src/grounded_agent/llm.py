"""Minimal client for a local Ollama server: chat with tools, and embeddings."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict]
    thinking: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    raw_message: dict = field(default_factory=dict)


class OllamaClient:
    """Talks to Ollama's /api/chat and /api/embed endpoints."""

    def __init__(self, model: str, embed_model: str, url: str = DEFAULT_URL,
                 think: bool = False, temperature: float = 0.0, num_ctx: int = 16384, timeout: float = 600.0):
        self.model = model
        self.embed_model = embed_model
        self.url = url.rstrip("/")
        self.think = think
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.http = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        payload = {
            "model": self.model, "messages": messages, "stream": False, "think": self.think,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        t0 = time.perf_counter()
        r = self.http.post(f"{self.url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        return ChatResult(
            content=msg.get("content", "") or "",
            tool_calls=msg.get("tool_calls") or [],
            thinking=msg.get("thinking", "") or "",
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            seconds=time.perf_counter() - t0,
            raw_message=msg,
        )

    def embed(self, texts: list[str], batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            r = self.http.post(f"{self.url}/api/embed",
                               json={"model": self.embed_model, "input": texts[i:i + batch], "truncate": True})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out
