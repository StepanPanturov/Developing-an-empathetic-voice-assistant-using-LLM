from __future__ import annotations

import asyncio

from .bot import run_bot
from .config import AppSettings
from .llm.router import LLMRouter
from .llm.gigachat_provider import GigaChatProvider
from .llm.yandex_provider import YandexGPTProvider
from .llm.ollama_provider import OllamaProvider
from .memory.storage import HistoryStore
from .pipeline import AssistantPipeline
from .speech.asr import ASRService
from .speech.tts import TTSService


def build_pipeline(settings: AppSettings) -> AssistantPipeline:
    asr = ASRService(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
    )
    tts = TTSService(speaker=settings.tts_speaker)

    providers = {}
    if settings.gigachat_credentials:
        providers["gigachat"] = GigaChatProvider(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
        )
    if settings.yandex_api_key:
        providers["yandex"] = YandexGPTProvider(
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id,
        )
    providers["ollama"] = OllamaProvider(
        model=settings.ollama_model,
        host=settings.ollama_host,
    )

    llm = LLMRouter(
        providers=providers,
        default_provider=settings.default_llm_provider,
        fallback_provider=settings.fallback_llm_provider,
    )

    store = HistoryStore(path="data/history.json")

    return AssistantPipeline(asr=asr, llm=llm, tts=tts, store=store, empathy_enabled=settings.empathy_enabled)


def main() -> None:
    settings = AppSettings()
    pipeline = build_pipeline(settings)
    asyncio.run(run_bot(settings, pipeline))


if __name__ == "__main__":
    main()
    