# Maya — Эмпатичный голосовой ассистент

Выпускная квалификационная работа по теме **«Разработка эмпатичного голосового ассистента с использованием больших языковых моделей»**

**НИУ ВШЭ** | Факультет информатики, математики и компьютерных наук | Программная инженерия

**Студент:** Пантуров Степан Андреевич 

**Научный руководитель:** Савченко Людмила Васильевна, старший научный сотрудник, Лаборатория теоретических основ моделей ИИ

---

## О проекте

Maya — Telegram-бот, который умеет определять эмоциональное состояние пользователя из текста и голоса и адаптировать стиль ответов под эмоциональный контекст. В отличие от обычных ассистентов, Maya анализирует не только что говорит пользователь, но и как он это говорит.

**Telegram:** [@the_maya_ai_bot](https://t.me/the_maya_ai_bot)

---

## Ключевые возможности

-  **Голосовой ввод** — распознавание речи через faster-whisper (offline)
-  **Голосовые ответы** — синтез речи через Silero TTS (offline)
-  **Анализ эмоций из текста** — модель ruBERT, 7 классов эмоций на русском языке
-  **Анализ интонации голоса** — модель wav2vec2, определение эмоции из аудио
-  **Fusion модуль** — объединение текстовой и голосовой эмоций, определение эмоционального диссонанса
-  **Движок эмпатии** — 4 персоны ассистента с динамическим промптом
-  **Профиль пользователя** — персонализация по имени и интересам
-  **A/B тест** — флаг `EMPATHY_ENABLED` для сравнения режимов
-  **LLM с fallback** — GigaChat (основной) → Ollama (резервный)

---

## Архитектура

```
Голос/Текст (Telegram)
        │
        ├── ASR (faster-whisper) ──────────── Текст
        │
        ├── VoiceEmotionAnalyzer (wav2vec2) ── Голосовая эмоция
        │
        ├── EmotionAnalyzer (ruBERT) ────────── Текстовая эмоция
        │
        ├── EmotionFusion ───────────────────── Объединённая эмоция
        │                                       + диссонанс
        ├── EmpathyEngine ───────────────────── Системный промпт
        │
        ├── LLMRouter ──────────────────────── GigaChat / Ollama
        │
        └── TTS (Silero) ────────────────────── Голосовой ответ
```

---

## Стек технологий

| Компонент | Технология |
|---|---|
| Язык | Python 3.10+ |
| Telegram Bot | aiogram 3.x |
| STT | faster-whisper (small, offline) |
| TTS | Silero TTS (offline) |
| Анализ эмоций текста | ruBERT (Aniemore) |
| Анализ интонации | wav2vec2 (superb-er) |
| LLM основной | GigaChat API (Сбер) |
| LLM резервный | Ollama (llama3.1:8b, offline) |
| Память диалога | JSON (расширяемо до Redis) |

---

## Установка и запуск

### Требования

- Python 3.10+
- [Ollama](https://ollama.com) с моделью `llama3.1:8b`
- [ffmpeg](https://ffmpeg.org) (для обработки аудио)
- Токен Telegram бота ([@BotFather](https://t.me/BotFather))
- API ключ GigaChat (опционально, [developers.sber.ru](https://developers.sber.ru))

### 1.Клонируйте репозиторий

```bash
git clone https://github.com/StepanPanturov/Developing-an-empathetic-voice-assistant-using-LLM.git
cd Developing-an-empathetic-voice-assistant-using-LLM
```

### 2.Создайте виртуальное окружение

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3.Установите зависимости

```bash
pip install fasttext-wheel
pip install dostoevsky --no-deps
pip install -e ".[dev]"
```

### 4. Настройте переменные окружения

Скопируй `.env.example` в `.env` и заполни:

```bash
cp .env.example .env
```

```dotenv
TELEGRAM_TOKEN=твой_токен_от_BotFather

# LLM
GIGACHAT_CREDENTIALS=твой_base64_ключ
GIGACHAT_SCOPE=GIGACHAT_API_PERS
DEFAULT_LLM_PROVIDER=gigachat
FALLBACK_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434

# A/B тест
EMPATHY_ENABLED=true
```

### 5.Запустите Ollama

```bash
ollama serve
ollama pull llama3.1:8b
```

### 6.Запустите бота

```bash
python -m assistant_bot.main
```

### Консольное демо (без Telegram)

```bash
python demo.py
```

---

## Команды бота

| Команда | Описание |
|---|---|
| `/start` | Приветствие и список команд |
| `/help` | Справка по работе бота |
| `/profile` | Настроить имя и интересы |
| `/settings` | Выбрать персону ассистента |
| `/emotion` | Показать последнюю определённую эмоцию |
| `/history` | История диалога |
| `/clear` | Очистить историю |

---

## Персоны ассистента

| Персона | Описание |
|---|---|
| 🤝 Поддерживающий | Тёплый, мягкий — признаёт чувства, предлагает помощь |
| 💼 Деловой | Нейтральный, чёткий — по делу без лишних эмоций |
| 😄 Позитивный | Энергичный, с юмором — разделяет радость |
| 🧘 Спокойный | Философский, умиротворяющий — помогает найти покой |

---

## A/B тестирование

Для сравнения режимов с эмпатией и без:

```dotenv
# Режим A (с эмпатией, по умолчанию)
EMPATHY_ENABLED=true

# Режим B (без эмпатии)
EMPATHY_ENABLED=false
```

---

## Структура проекта

```
src/assistant_bot/
├── bot.py              # Telegram хэндлеры и команды
├── pipeline.py         # Главный оркестратор
├── main.py             # Точка входа
├── config.py           # Настройки из .env
├── emotion/
│   ├── analyzer.py     # Анализ эмоций текста (ruBERT)
│   └── fusion.py       # Объединение текстовой и голосовой эмоций
├── empathy/
│   ├── engine.py       # Движок эмпатии
│   ├── prompt_builder.py # Сборка системного промпта
│   └── persona.py      # Персоны ассистента
├── llm/
│   ├── router.py       # LLM роутер с fallback
│   ├── gigachat_provider.py
│   ├── ollama_provider.py
│   └── yandex_provider.py  # Готов к подключению
├── memory/
│   ├── storage.py      # JSON хранилище (текущее)
|   ├── short_term.py   # Redis (подготовлено)
|   ├── long_term.py    # PostgreSQL (подготовлено)
|   └── summarizer.py   # Сжатие истории через LLM
|
└── speech/
    ├── asr.py          # STT через faster-whisper
    ├── tts.py          # TTS через Silero
    └── voice_emotion.py # Анализ интонации (wav2vec2)
```

---

## Направления развития

- Подключение Redis и PostgreSQL для масштабируемой памяти диалогов
- Проведение A/B эксперимента с группой пользователей
- Интеграция YandexGPT как дополнительного LLM-провайдера
- Расширение анализа интонации с учётом русскоязычных моделей
