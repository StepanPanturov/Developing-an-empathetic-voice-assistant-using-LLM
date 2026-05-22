from __future__ import annotations

from dataclasses import dataclass, field

from . import EmotionResult
from ..speech.voice_emotion import VoiceEmotionResult


@dataclass
class FusedEmotionResult:
    primary_emotion: str
    sentiment: str
    intensity: float
    confidence: float
    needs_support: bool
    source: str  # "text", "voice", "fused"
    text_emotion: str
    voice_emotion: str
    emotional_dissonance: bool = False


NEGATIVE_EMOTIONS = {"грусть", "злость", "страх", "отвращение", "sadness", "anger", "fear", "disgust"}
POSITIVE_EMOTIONS = {"радость", "happiness", "happy", "joy"}


class EmotionFusion:
    """Объединяет текстовую и голосовую эмоции."""

    TEXT_WEIGHT = 0.4
    VOICE_WEIGHT = 0.6

    def fuse(
        self,
        text_emotion: EmotionResult,
        voice_emotion: VoiceEmotionResult | None = None,
    ) -> FusedEmotionResult:

        # Если нет голосовой эмоции, то используем только текст
        if voice_emotion is None or voice_emotion.confidence < 0.3:
            return FusedEmotionResult(
                primary_emotion=text_emotion.primary_emotion,
                sentiment=text_emotion.sentiment,
                intensity=text_emotion.intensity,
                confidence=text_emotion.confidence,
                needs_support=text_emotion.needs_support,
                source="text",
                text_emotion=text_emotion.primary_emotion,
                voice_emotion="не определено",
                emotional_dissonance=False,
            )

        # Определяем диссонанс (текст позитивный, но голос негативный)
        text_is_positive = text_emotion.primary_emotion in POSITIVE_EMOTIONS
        voice_is_negative = (
            voice_emotion.primary_emotion in NEGATIVE_EMOTIONS
            and voice_emotion.confidence > 0.4
        )
        emotional_dissonance = text_is_positive and voice_is_negative

        # Если эмоции совпадают - высокая уверенность
        if text_emotion.primary_emotion == voice_emotion.primary_emotion:
            primary = text_emotion.primary_emotion
            confidence = min(
                text_emotion.confidence * self.TEXT_WEIGHT
                + voice_emotion.confidence * self.VOICE_WEIGHT,
                1.0,
            )
            source = "fused"

        # Если голос уверен сильнее текста, то доверяем голосу
        elif voice_emotion.confidence > text_emotion.confidence + 0.2:
            primary = voice_emotion.primary_emotion
            confidence = voice_emotion.confidence
            source = "voice"

        # Иначе доверяем тексту
        else:
            primary = text_emotion.primary_emotion
            confidence = text_emotion.confidence
            source = "text"

        # При диссонансе голос важнее, переключаемся на голосовую эмоцию
        if emotional_dissonance:
            primary = voice_emotion.primary_emotion
            source = "voice"

        # Определяем тональность
        if primary in NEGATIVE_EMOTIONS:
            sentiment = "negative"
        elif primary in POSITIVE_EMOTIONS:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        intensity = min(
            text_emotion.intensity * self.TEXT_WEIGHT
            + voice_emotion.confidence * self.VOICE_WEIGHT,
            1.0,
        )
        needs_support = sentiment == "negative" and intensity > 0.5

        return FusedEmotionResult(
            primary_emotion=primary,
            sentiment=sentiment,
            intensity=intensity,
            confidence=confidence,
            needs_support=needs_support,
            source=source,
            text_emotion=text_emotion.primary_emotion,
            voice_emotion=voice_emotion.primary_emotion,
            emotional_dissonance=emotional_dissonance,
        )
    