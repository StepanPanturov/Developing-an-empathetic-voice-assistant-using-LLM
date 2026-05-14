from __future__ import annotations

from ..emotion import EmotionResult
from .persona import PERSONAS


class PromptBuilder:
    def __init__(self) -> None:
        self.personas = PERSONAS

    def build(
        self,
        emotion: EmotionResult,
        user_name: str = "пользователь",
        persona: str = "supportive",
        communication_style: str = "informal",
        interests: list[str] | None = None,
        session_summary: str = "",
        conversation_history: str = "",
    ) -> str:
        persona_data = self.personas.get(persona, self.personas["supportive"])

        prompt_parts = [
            f"Ты — {persona_data['description']}.",
            f"Ты общаешься с {user_name}.",
            "",
            "ТЕКУЩЕЕ СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ:",
            f"- Эмоция: {emotion.primary_emotion} (интенсивность: {emotion.intensity:.2f})",
            f"- Тональность: {emotion.sentiment}",
            f"- Требуется поддержка: {'Да' if emotion.needs_support else 'Нет'}",
            "",
            "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:",
            f"- Стиль общения: {communication_style}",
            f"- Интересы: {', '.join(interests) if interests else 'не указаны'}",
            "",
        ]

        if session_summary:
            prompt_parts.extend([
                "ИСТОРИЯ ПРЕДЫДУЩИХ СЕССИЙ:",
                session_summary,
                "",
            ])

        prompt_parts.extend([
            "ПРАВИЛА ЭМПАТИЧНОГО ОТВЕТА:",
            "1. Если needs_support=True — СНАЧАЛА признай чувства пользователя, потом помогай",
            "2. Если emotion=грусть — используй мягкий, поддерживающий тон без советов",
            "3. Если emotion=злость — не спорь, деэскалируй, используй нейтральный тон",
            "4. Если emotion=радость — разделяй позитив, отвечай энергично",
            "5. Адаптируй длину ответа: при стрессе — короче и яснее",
            "6. Никогда не игнорируй эмоциональный контекст",
            "",
            "ТЕКУЩИЙ ДИАЛОГ:",
            conversation_history,
        ])

        return "\n".join(prompt_parts)
    