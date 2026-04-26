from __future__ import annotations

from pathlib import Path

import torch


class TTSService:
    """Russian TTS via Silero model loaded from local torch hub cache."""

    def __init__(self, speaker: str = "xenia") -> None:
        self.speaker = speaker
        self._model, self._sample_rate = self._load_model()

    @staticmethod
    def _load_model():
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        model.to(torch.device("cpu"))
        return model, 48000

    def synthesize_speech(self, text: str, emotion: str = "neutral", output_path: str | Path = "output.wav") -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_wav(text=text, speaker=self.speaker, sample_rate=self._sample_rate, audio_path=str(out))
        return out
