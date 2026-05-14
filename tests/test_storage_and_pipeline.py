from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_bot.pipeline import AssistantPipeline
from assistant_bot.memory.storage import HistoryStore


class FakeASR:
    def speech_to_text(self, audio_path: str | Path) -> str:
        return "мне грустно"


class FakeLLM:
    async def generate(self, messages: list[dict]) -> str:
        assert any("user:" in m.get("content", "") for m in messages)
        return "Понимаю вас. Я рядом и готов помочь."


class FakeTTS:
    def synthesize_speech(self, text: str, emotion: str, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        return path


def test_store_append_and_context(tmp_path):
    store = HistoryStore(path=str(tmp_path / "history.json"))
    store.append("42", "user", "привет")
    store.append("42", "assistant", "здравствуйте")

    context = store.get_context("42")
    assert "user: привет" in context
    assert "assistant: здравствуйте" in context


@pytest.mark.asyncio
async def test_pipeline_text_flow(tmp_path):
    store = HistoryStore(path=str(tmp_path / "history.json"))
    pipeline = AssistantPipeline(
        asr=FakeASR(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        store=store,
    )

    answer, voice_path = await pipeline.handle_text(user_id="100", text="мне тяжело")

    assert "Я рядом" in answer
    assert voice_path.exists()


@pytest.mark.asyncio
async def test_pipeline_audio_flow(tmp_path):
    store = HistoryStore(path=str(tmp_path / "history.json"))
    pipeline = AssistantPipeline(
        asr=FakeASR(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        store=store,
    )

    # FakeASR всегда возвращает "мне грустно"
    answer, voice_path = await pipeline.handle_audio(
        user_id="200",
        audio_path=tmp_path / "fake.ogg",
    )

    assert "Я рядом" in answer
    assert voice_path.exists()


    [project.optional-dependencies]
    dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    ]