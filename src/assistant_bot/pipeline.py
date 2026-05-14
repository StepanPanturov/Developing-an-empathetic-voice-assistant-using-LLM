from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from .emotion.analyzer import EmotionAnalyzer
from .empathy.engine import EmpathyEngine


class ASRProtocol(Protocol):
    def speech_to_text(self, audio_path: str | Path) -> str: ...


class LLMProtocol(Protocol):
    async def generate(self, messages: list[dict]) -> str: ...


class TTSProtocol(Protocol):
    def synthesize_speech(self, text: str, emotion: str, output_path: str | Path) -> Path: ...


class StoreProtocol(Protocol):
    def append(self, user_id: str, role: str, content: str) -> None: ...
    def get_context(self, user_id: str, max_messages: int = 8) -> str: ...


class AssistantPipeline:
    def __init__(
        self,
        asr: ASRProtocol,
        llm: LLMProtocol,
        tts: TTSProtocol,
        store: StoreProtocol,
        empathy_enabled: bool = True,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.store = store
        self.empathy_enabled = empathy_enabled
        self.emotion_analyzer = EmotionAnalyzer()
        self.empathy_engine = EmpathyEngine()
        self.last_emotions: dict = {}
        self.user_personas: dict = {}
        self.user_profiles: dict = {}

    async def handle_audio(self, user_id: str, audio_path: str | Path) -> tuple[str, Path]:
        text = await asyncio.to_thread(self.asr.speech_to_text, audio_path)
        return await self._respond(user_id=user_id, user_text=text)

    async def handle_text(self, user_id: str, text: str) -> tuple[str, Path]:
        return await self._respond(user_id=user_id, user_text=text)

    async def _respond(self, user_id: str, user_text: str) -> tuple[str, Path]:
        # Получаем историю
        self.store.append(user_id, "user", user_text)
        conversation_history = self.store.get_context(user_id)

        if self.empathy_enabled:
            # Режим A — с эмпатией
            emotion = await asyncio.to_thread(self.emotion_analyzer.analyze, user_text)
            self.last_emotions[user_id] = emotion
            user_profile = self.user_profiles.get(user_id, {
                "persona": self.user_personas.get(user_id, "supportive")
            })
            system_prompt = self.empathy_engine.process(
                user_id=user_id,
                text=user_text,
                emotion_result=emotion,
                conversation_history=conversation_history,
                user_profile=user_profile,
            )
            #print(f"=== SYSTEM PROMPT ===\n{system_prompt}\n====================")
        else:
            # Режим B — без эмпатии
            emotion = None
            system_prompt = "Ты голосовой ассистент. Отвечай дружелюбно и помогай пользователю."

        # Формируем messages для LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Получаем ответ от LLM
        llm_response = await self.llm.generate(messages, emotion_context=emotion)
        answer = llm_response.text

        self.store.append(user_id, "assistant", answer)

        # Синтезируем речь
        voice_path = Path("data/audio") / f"{user_id}_last_reply.wav"
        emotion_label = emotion.primary_emotion if emotion else "neutral"
        audio_file = await asyncio.to_thread(
            self.tts.synthesize_speech,
            answer,
            emotion_label,
            voice_path,
        )

        return answer, audio_file
    