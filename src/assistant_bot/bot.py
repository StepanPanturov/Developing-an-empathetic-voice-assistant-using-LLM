from __future__ import annotations

import traceback
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import FSInputFile, Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from .config import AppSettings
from .pipeline import AssistantPipeline


class ProfileForm(StatesGroup):
    waiting_for_name = State()
    waiting_for_interests = State()


PERSONA_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🤝 Поддерживающий"), KeyboardButton(text="💼 Деловой")],
        [KeyboardButton(text="😄 Позитивный"), KeyboardButton(text="🧘 Спокойный")],
    ],
    resize_keyboard=True,
    one_time_keyboard=True,
)

PERSONA_MAP = {
    "🤝 Поддерживающий": "supportive",
    "💼 Деловой": "professional",
    "😄 Позитивный": "cheerful",
    "🧘 Спокойный": "calm",
}


def build_dispatcher(bot: Bot, pipeline: AssistantPipeline) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start_handler(message: Message) -> None:
        await message.answer(
            "Привет! Я эмпатичный голосовой ассистент.\n\n"
            "Отправь текст или голосовое сообщение — отвечу с пониманием.\n\n"
            "Команды:\n"
            "/help — справка\n"
            "/profile — настроить профиль\n"
            "/settings — выбрать стиль общения\n"
            "/emotion — моя текущая эмоция\n"
            "/history — история диалога\n"
            "/clear — очистить историю"
        )

    @dp.message(Command("help"))
    async def help_handler(message: Message) -> None:
        await message.answer(
            "Как я работаю:\n\n"
            "1. Ты отправляешь текст или голосовое\n"
            "2. Я определяю твоё эмоциональное состояние\n"
            "3. Подбираю стиль ответа под твои эмоции\n"
            "4. Отвечаю текстом и голосом\n\n"
            "Команды:\n"
            "/help — справка\n"
            "/profile — настроить профиль\n"
            "/settings — выбрать стиль общения\n"
            "/emotion — моя текущая эмоция\n"
            "/history — история диалога\n"
            "/clear — очистить историю"
        )

    @dp.message(Command("settings"))
    async def settings_handler(message: Message) -> None:
        await message.answer(
            "Выбери стиль общения:",
            reply_markup=PERSONA_KEYBOARD,
        )

    @dp.message(Command("profile"))
    async def profile_handler(message: Message, state: FSMContext) -> None:
        await message.answer("Как тебя зовут? Напиши своё имя.")
        await state.set_state(ProfileForm.waiting_for_name)

    @dp.message(ProfileForm.waiting_for_name)
    async def profile_name_handler(message: Message, state: FSMContext) -> None:
        name = message.text.strip()
        await state.update_data(name=name)
        await message.answer(
            f"Приятно познакомиться, {name}! 😊\n\n"
            "Расскажи о своих интересах — через запятую.\n"
            "Например: музыка, спорт, технологии"
        )
        await state.set_state(ProfileForm.waiting_for_interests)

    @dp.message(ProfileForm.waiting_for_interests)
    async def profile_interests_handler(message: Message, state: FSMContext) -> None:
        interests = [i.strip() for i in message.text.split(",")]
        data = await state.get_data()
        name = data.get("name", "пользователь")

        user_id = str(message.from_user.id)
        pipeline.user_profiles[user_id] = {
            "name": name,
            "interests": interests,
            "persona": pipeline.user_personas.get(user_id, "supportive"),
        }

        await state.clear()
        await message.answer(
            f"Профиль сохранён! ✅\n\n"
            f"Имя: {name}\n"
            f"Интересы: {', '.join(interests)}\n\n"
            "Теперь я буду обращаться к тебе по имени и учитывать твои интересы."
        )

    @dp.message(Command("emotion"))
    async def emotion_handler(message: Message) -> None:
        user_id = str(message.from_user.id)
        emotion = pipeline.last_emotions.get(user_id)
        if emotion:
            await message.answer(
                f"Последняя определённая эмоция:\n\n"
                f"Эмоция: {emotion.primary_emotion}\n"
                f"Тональность: {emotion.sentiment}\n"
                f"Интенсивность: {emotion.intensity:.2f}\n"
                f"Нужна поддержка: {'Да' if emotion.needs_support else 'Нет'}"
            )
        else:
            await message.answer("Эмоция ещё не определена — отправь сообщение.")

    @dp.message(Command("history"))
    async def history_handler(message: Message) -> None:
        user_id = str(message.from_user.id)
        history = pipeline.store.get_context(user_id, max_messages=10)
        if history:
            await message.answer(f"История диалога:\n\n{history}")
        else:
            await message.answer("История пуста.")

    @dp.message(Command("clear"))
    async def clear_handler(message: Message) -> None:
        user_id = str(message.from_user.id)
        pipeline.store._write({})
        await message.answer("История диалога очищена.")

    @dp.message(F.text.in_(PERSONA_MAP.keys()))
    async def persona_handler(message: Message) -> None:
        user_id = str(message.from_user.id)
        persona = PERSONA_MAP[message.text]
        pipeline.user_personas[user_id] = persona
        await message.answer(f"Стиль общения изменён на: {message.text}")

    @dp.message(F.text)
    async def text_handler(message: Message) -> None:
        await message.bot.send_chat_action(message.chat.id, "typing")
        user_id = str(message.from_user.id)
        try:
            reply_text, reply_audio = await pipeline.handle_text(user_id, message.text)
            await message.answer(reply_text)
            await message.answer_voice(voice=FSInputFile(reply_audio))
        except Exception as e:
            traceback.print_exc()
            await message.answer(f"Ошибка: {e}")

    @dp.message(F.voice)
    async def voice_handler(message: Message) -> None:
        await message.bot.send_chat_action(message.chat.id, "typing")
        user_id = str(message.from_user.id)
        voice_path = Path("data/inbox") / f"{message.voice.file_id}.ogg"
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_info = await bot.get_file(message.voice.file_id)
            await bot.download_file(file_info.file_path, destination=voice_path)
            reply_text, reply_audio = await pipeline.handle_audio(user_id, voice_path)
            await message.answer(reply_text)
            await message.answer_voice(voice=FSInputFile(reply_audio))
        except Exception as e:
            traceback.print_exc()
            await message.answer(f"Ошибка: {e}")

    return dp


async def run_bot(settings: AppSettings, pipeline: AssistantPipeline) -> None:
    if not settings.telegram_token:
        raise ValueError("Set TELEGRAM_TOKEN in env or .env")

    bot = Bot(token=settings.telegram_token)
    dp = build_dispatcher(bot, pipeline)
    await dp.start_polling(bot)
