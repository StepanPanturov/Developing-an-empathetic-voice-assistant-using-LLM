from __future__ import annotations

import httpx


class OllamaProvider:
    """Ollama локальный провайдер."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    async def generate(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        # Явно отключаем прокси для локальных запросов
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        