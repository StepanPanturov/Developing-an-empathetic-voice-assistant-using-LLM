from __future__ import annotations

import httpx


class YandexGPTProvider:
    """YandexGPT API провайдер."""

    def __init__(self, api_key: str, folder_id: str) -> None:
        self.api_key = api_key
        self.folder_id = folder_id
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    async def generate(self, messages: list[dict]) -> str:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": 1000,
            },
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.base_url,
                headers={"Authorization": f"Api-Key {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["result"]["alternatives"][0]["message"]["text"].strip()
        