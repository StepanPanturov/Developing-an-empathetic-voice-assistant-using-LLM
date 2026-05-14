from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..emotion import EmotionResult


@dataclass
class LLMResponse:
    text: str
    provider: str
    response_time_ms: int


class LLMRouter:
    def __init__(
        self,
        providers: dict[str, Any],
        default_provider: str = "gigachat",
        fallback_provider: str = "ollama",
    ) -> None:
        self.providers = providers
        self.default_provider = default_provider
        self.fallback_provider = fallback_provider

    async def generate(
        self,
        messages: list[dict],
        emotion_context: EmotionResult | None = None,
        provider: str = "auto",
    ) -> LLMResponse:
        if provider == "auto":
            provider = self.default_provider

        start_time = time.time()

        try:
            if provider in self.providers:
                response = await self.providers[provider].generate(messages)
                response_time = int((time.time() - start_time) * 1000)
                return LLMResponse(text=response, provider=provider, response_time_ms=response_time)
            raise ValueError(f"Provider {provider} not found")

        except Exception as e:
            print(f"Error with {provider}: {e}")
            if provider != self.fallback_provider and self.fallback_provider in self.providers:
                try:
                    response = await self.providers[self.fallback_provider].generate(messages)
                    response_time = int((time.time() - start_time) * 1000)
                    return LLMResponse(text=response, provider=self.fallback_provider, response_time_ms=response_time)
                except Exception as e2:
                    pass

            return LLMResponse(
                text="Извините, произошла ошибка. Попробуйте позже.",
                provider="error",
                response_time_ms=0,
            )
        