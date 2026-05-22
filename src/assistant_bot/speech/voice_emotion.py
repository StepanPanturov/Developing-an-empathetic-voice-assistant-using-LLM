from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from transformers import pipeline
import subprocess


@dataclass
class VoiceEmotionResult:
    primary_emotion: str
    confidence: float
    raw_scores: dict[str, float]


EMOTION_MAP = {
    "ang": "злость",
    "hap": "радость",
    "neu": "нейтральное",
    "sad": "грусть",
    "angry": "злость",
    "happy": "радость",
    "neutral": "нейтральное",
    "sad": "грусть",
}


class VoiceEmotionAnalyzer:
    """Анализ эмоций из голоса через wav2vec2."""

    def __init__(self) -> None:
        self.model = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device="cpu",
        )
        self.sample_rate = 16000

    def analyze(self, audio_path: str | Path) -> VoiceEmotionResult:
        if not audio_path or not Path(audio_path).exists():
            return VoiceEmotionResult(
                primary_emotion="нейтральное",
                confidence=0.0,
                raw_scores={},
            )

        try:
            import subprocess
            import soundfile as sf
            import numpy as np

            # Конвертируем ogg в wav если нужно
            audio_path = Path(audio_path)
            if audio_path.suffix.lower() == ".ogg":
                wav_path = audio_path.with_suffix(".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(audio_path), str(wav_path)],
                    capture_output=True,
                )
                audio_path = wav_path

            # Загружаем аудио через soundfile
            audio_array, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)

            # Конвертируем стерео в моно
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)

            # Ресемплируем если нужно
            if sr != self.sample_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array, orig_sr=sr, target_sr=self.sample_rate
                )

            results = self.model(audio_array, top_k=5)

            raw_scores = {r["label"]: float(r["score"]) for r in results}
            best = max(results, key=lambda x: x["score"])
            primary_emotion = EMOTION_MAP.get(
                best["label"].lower(), best["label"].lower()
            )
            confidence = float(best["score"])

        except Exception as e:
            print(f"Voice emotion analysis error: {e}")
            return VoiceEmotionResult(
                primary_emotion="нейтральное",
                confidence=0.0,
                raw_scores={},
            )

        return VoiceEmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            raw_scores=raw_scores,
        )
    