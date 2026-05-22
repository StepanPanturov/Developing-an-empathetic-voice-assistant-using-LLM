from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings loaded from env variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Telegram
    telegram_token: str = Field(default="", description="Telegram bot token")

    # Speech
    whisper_model: str = Field(default="small", description="faster-whisper model size")
    whisper_device: str = Field(default="cpu", description="cpu/cuda")
    tts_speaker: str = Field(default="xenia", description="Silero TTS speaker")
    tts_engine: str = Field(default="edge", description="TTS engine: silero | edge")
    tts_edge_speaker: str = Field(default="svetlana", description="Edge TTS speaker: svetlana | dmitry | dariya")

    # LLM
    default_llm_provider: str = Field(default="gigachat", description="gigachat | yandex | ollama")
    fallback_llm_provider: str = Field(default="ollama", description="Fallback provider")
    ollama_model: str = Field(default="llama3.1:8b-instruct", description="Local model in Ollama")
    ollama_host: str = Field(default="http://localhost:11434", description="Local Ollama URL")

    # GigaChat
    gigachat_credentials: str = Field(default="", description="GigaChat base64 credentials")
    gigachat_scope: str = Field(default="GIGACHAT_API_PERS", description="GigaChat scope")

    # YandexGPT
    yandex_api_key: str = Field(default="", description="Yandex API key")
    yandex_folder_id: str = Field(default="", description="Yandex folder ID")

    # Memory
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    dialog_window_size: int = Field(default=20, description="Short-term memory window")
    dialog_ttl_hours: int = Field(default=24, description="Dialog TTL in hours")
    database_url: str = Field(default="postgresql+asyncpg://user:password@localhost:5432/empathic_bot", description="PostgreSQL URL")
    history_path: str = Field(default="data/history.json", description="Path for JSON dialogue history")

    # Empathy
    default_persona: str = Field(default="supportive", description="Default persona")
    emotion_support_threshold: float = Field(default=0.6, description="Support threshold")
    empathy_enabled: bool = Field(default=True, description="Enable empathy mode for A/B testing")

    # App
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # VPN
    proxy_url: str = Field(default="", description="Proxy URL for Telegram connection")
