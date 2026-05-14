from __future__ import annotations

from transformers import pipeline

from . import EmotionResult


class EmotionAnalyzer:
    """Анализ эмоций текста через ruBERT."""

    NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "disgust", "грусть", "злость", "страх", "отвращение"}
    POSITIVE_EMOTIONS = {"joy", "happiness", "радость"}

    def __init__(self) -> None:
        self.emotion_model = pipeline(
            "text-classification",
            model="Aniemore/rubert-tiny2-russian-emotion-detection",
            return_all_scores=True,
        )

    def analyze(self, text: str) -> EmotionResult:
        if not text or not text.strip():
            return EmotionResult(
                primary_emotion="neutral",
                sentiment="neutral",
                intensity=0.0,
                confidence=0.0,
                needs_support=False,
                raw_scores={},
            )

        try:
            result = self.emotion_model(text)
            # Модель может вернуть список списков или список словарей
            if isinstance(result[0], list):
                emotion_scores = result[0]
            else:
                emotion_scores = result
        except Exception:
            emotion_scores = [{"label": "neutral", "score": 1.0}]

        try:
            best = max(emotion_scores, key=lambda x: x["score"])
            primary_emotion = best["label"]
            confidence = float(best["score"])
        except (TypeError, KeyError):
            primary_emotion = "neutral"
            confidence = 1.0
        
        if confidence < 0.5:
            primary_emotion = "neutral"
            sentiment = "neutral"
            needs_support = False

        if primary_emotion in self.NEGATIVE_EMOTIONS:
            sentiment = "negative"
        elif primary_emotion in self.POSITIVE_EMOTIONS:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        intensity = min(max(confidence, 0.0), 1.0)
        needs_support = sentiment == "negative" and intensity > 0.6

        raw_scores = {
            "emotions": {s["label"]: float(s["score"]) for s in emotion_scores}
            if emotion_scores and isinstance(emotion_scores[0], dict)
            else {}
        }

        return EmotionResult(
            primary_emotion=primary_emotion,
            sentiment=sentiment,
            intensity=intensity,
            confidence=confidence,
            needs_support=needs_support,
            raw_scores=raw_scores,
        )
    