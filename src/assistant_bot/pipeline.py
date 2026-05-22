from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from .emotion.analyzer import EmotionAnalyzer
from .empathy.engine import EmpathyEngine
from .speech.voice_emotion import VoiceEmotionAnalyzer
from .emotion.fusion import EmotionFusion, FusedEmotionResult


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
        self.voice_emotion_analyzer = VoiceEmotionAnalyzer()
        self.emotion_fusion = EmotionFusion()
        self.last_emotions: dict = {}
        self.user_personas: dict = {}
        self.user_profiles: dict = {}

    async def handle_audio(self, user_id: str, audio_path: str | Path) -> tuple[str, Path]:
        text = await asyncio.to_thread(self.asr.speech_to_text, audio_path)
        voice_emotion = await asyncio.to_thread(
            self.voice_emotion_analyzer.analyze, audio_path
        )
        return await self._respond(
            user_id=user_id,
            user_text=text,
            voice_emotion=voice_emotion,
        )

    async def handle_text(self, user_id: str, text: str) -> tuple[str, Path]:
        return await self._respond(user_id=user_id, user_text=text, voice_emotion=None)

    async def _respond(
        self,
        user_id: str,
        user_text: str,
        voice_emotion=None,
    ) -> tuple[str, Path]:
        self.store.append(user_id, "user", user_text)
        conversation_history = self.store.get_context(user_id)

        if self.empathy_enabled:
            text_emotion = await asyncio.to_thread(
                self.emotion_analyzer.analyze, user_text
            )

            fused = self.emotion_fusion.fuse(text_emotion, voice_emotion)

            # Отладочный вывод
            print("─" * 40)
            print(f"📝 Текст:  {text_emotion.primary_emotion} ({text_emotion.confidence:.2f})")
            voice_conf = voice_emotion.confidence if voice_emotion else 0.0
            voice_label = voice_emotion.primary_emotion if voice_emotion else "нет"
            print(f"🎙️ Голос:  {voice_label} ({voice_conf:.2f})")
            print(f"🔀 Итого:  {fused.primary_emotion} | источник: {fused.source} | диссонанс: {fused.emotional_dissonance}")
            print("─" * 40)

            self.last_emotions[user_id] = fused

            user_profile = self.user_profiles.get(user_id, {
                "persona": self.user_personas.get(user_id, "supportive")
            })

            system_prompt = self.empathy_engine.process(
                user_id=user_id,
                text=user_text,
                emotion_result=text_emotion,
                conversation_history=conversation_history,
                user_profile=user_profile,
            )

            # Добавляем голосовой контекст в промпт
            if fused.emotional_dissonance:
                system_prompt += (
                    "\n\nВАЖНО: Пользователь говорит позитивные слова, но голос звучит "
                    "грустно или напряжённо. Это может означать что человек скрывает "
                    "своё состояние. Мягко спроси как он на самом деле себя чувствует — "
                    "например 'Ты говоришь что всё хорошо, но я чувствую что что-то "
                    "тебя беспокоит. Хочешь поговорить об этом?'. Не давай советов, "
                    "просто покажи что ты замечаешь и готов выслушать."
                )
            elif fused.source == "voice":
                if fused.primary_emotion in ["грусть", "страх"]:
                    system_prompt += (
                        f"\n\nВАЖНО: Голосовой анализ определил что пользователь звучит "
                        f"грустно (уверенность {fused.confidence:.2f}). "
                        f"Начни ответ с вопроса 'Всё ли у тебя хорошо?' или "
                        f"'Я слышу что тебе немного грустно — хочешь поговорить об этом?'. "
                        f"Прояви заботу и участие."
                    )
                elif fused.primary_emotion == "злость":
                    system_prompt += (
                        f"\n\nВАЖНО: Голосовой анализ определил раздражение в голосе "
                        f"(уверенность {fused.confidence:.2f}). "
                        f"Говори спокойно, деэскалируй напряжение, не спорь."
                    )
                elif fused.primary_emotion == "радость":
                    system_prompt += (
                        f"\n\nВАЖНО: Пользователь звучит радостно "
                        f"(уверенность {fused.confidence:.2f}). "
                        f"Разделяй позитив, отвечай энергично."
                    )
            elif fused.source == "fused":
                system_prompt += (
                    f"\n\nВАЖНО: Текстовый и голосовой анализ подтвердили "
                    f"эмоцию '{fused.primary_emotion}'. Высокая уверенность — "
                    f"реагируй соответственно."
                )

            emotion_label = fused.primary_emotion

        else:
            fused = None
            system_prompt = "Ты голосовой ассистент. Отвечай дружелюбно и помогай пользователю."
            emotion_label = "neutral"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        llm_response = await self.llm.generate(messages, emotion_context=None)
        answer = llm_response.text

        self.store.append(user_id, "assistant", answer)

        voice_path = Path("data/audio") / f"{user_id}_last_reply.wav"
        audio_file = await asyncio.to_thread(
            self.tts.synthesize_speech,
            answer,
            emotion_label,
            voice_path,
        )

        return answer, audio_file
    