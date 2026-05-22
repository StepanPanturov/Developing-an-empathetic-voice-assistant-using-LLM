from __future__ import annotations

import asyncio
import re
from pathlib import Path

import edge_tts


def _clean_text(text: str) -> str:
    """Убирает эмодзи и лишние символы из текста."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EdgeTTSService:
    """Russian TTS via Microsoft Edge TTS."""

    VOICES = {
        "svetlana": "ru-RU-SvetlanaNeural",
        "dmitry": "ru-RU-DmitryNeural",
        "dariya": "ru-RU-DariyaNeural",
    }

    def __init__(self, speaker: str = "svetlana") -> None:
        self.speaker = self.VOICES.get(speaker, self.VOICES["svetlana"])

    def synthesize_speech(
        self,
        text: str,
        emotion: str = "neutral",
        output_path: str | Path = "output.wav",
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        clean = _clean_text(text)
        
        if not clean:
            clean = "Понял вас."
        
        print(f"TTS text: {clean[:50]}")
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self._synthesize(clean, str(out)))
            future.result()
        
        return out

    async def _synthesize(self, text: str, output_path: str) -> None:
        communicate = edge_tts.Communicate(text, self.speaker, proxy=None)
        await communicate.save(output_path)
