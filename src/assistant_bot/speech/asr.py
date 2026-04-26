from pathlib import Path

from faster_whisper import WhisperModel


class ASRService:
    """Offline speech-to-text via faster-whisper."""

    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def speech_to_text(self, audio_path: str | Path) -> str:
        segments, _ = self._model.transcribe(str(audio_path), language="ru")
        return " ".join(segment.text.strip() for segment in segments).strip()
