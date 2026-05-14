from __future__ import annotations

from typing import Any

from ..emotion import EmotionResult
from .prompt_builder import PromptBuilder


class EmpathyEngine:
    def __init__(self) -> None:
        self.prompt_builder = PromptBuilder()

    def process(
        self,
        user_id: str,
        text: str,
        emotion_result: EmotionResult,
        conversation_history: str = "",
        user_profile: dict[str, Any] | None = None,
        session_summary: str = "",
    ) -> str:
        user_name = user_profile.get("name", "пользователь") if user_profile else "пользователь"
        persona = user_profile.get("persona", "supportive") if user_profile else "supportive"
        communication_style = user_profile.get("communication_style", "informal") if user_profile else "informal"
        interests = user_profile.get("interests", []) if user_profile else []

        return self.prompt_builder.build(
            emotion=emotion_result,
            user_name=user_name,
            persona=persona,
            communication_style=communication_style,
            interests=interests,
            session_summary=session_summary,
            conversation_history=conversation_history,
        )
    