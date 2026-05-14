from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from assistant_bot.config import AppSettings
from assistant_bot.main import build_pipeline


SAMPLE_RATE = 16000
CHANNELS = 1


def record_audio(duration: int = 5) -> Path:
    """Записывает голос с микрофона."""
    print(f"🎤 Говорите... ({duration} секунд)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("✅ Запись завершена")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return Path(tmp.name)


def play_audio(audio_path: Path) -> None:
    """Воспроизводит аудиофайл."""
    data, samplerate = sf.read(str(audio_path))
    sd.play(data, samplerate)
    sd.wait()


async def main() -> None:
    print("Загрузка моделей...")
    settings = AppSettings()
    pipeline = build_pipeline(settings)

    print("\n" + "="*50)
    print("Эмпатичный голосовой ассистент")
    print("Нажми Enter для записи голоса")
    print("Введи 'текст' для текстового режима")
    print("Введи 'выход' для завершения")
    print("="*50 + "\n")

    while True:
        cmd = input(">>> ").strip().lower()

        if cmd == "выход":
            print("До свидания!")
            break

        elif cmd == "текст":
            text = input("Вы: ").strip()
            if not text:
                continue
            print("⏳ Обрабатываю...")
            answer, audio_path = await pipeline.handle_text("demo_user", text)
            print(f"\nАссистент: {answer}\n")
            play_audio(audio_path)

        else:
            # Голосовой режим — Enter для записи
            try:
                duration = int(input("Длительность записи в секундах (по умолчанию 5): ").strip() or "5")
            except ValueError:
                duration = 5

            audio_path = record_audio(duration=duration)
            print("⏳ Обрабатываю...")
            answer, response_audio = await pipeline.handle_audio("demo_user", audio_path)
            print(f"\nАссистент: {answer}\n")
            play_audio(response_audio)


if __name__ == "__main__":
    asyncio.run(main())
    