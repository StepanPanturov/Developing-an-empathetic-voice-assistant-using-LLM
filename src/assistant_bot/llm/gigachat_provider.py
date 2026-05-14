from __future__ import annotations

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole


class GigaChatProvider:
    """GigaChat API провайдер через официальный SDK."""

    def __init__(self, credentials: str, scope: str = "GIGACHAT_API_PERS") -> None:
        self.credentials = credentials
        self.scope = scope

    async def generate(self, messages: list[dict]) -> str:
        gigachat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                gigachat_messages.append(
                    Messages(role=MessagesRole.SYSTEM, content=msg["content"])
                )
            elif msg["role"] == "user":
                gigachat_messages.append(
                    Messages(role=MessagesRole.USER, content=msg["content"])
                )
            elif msg["role"] == "assistant":
                gigachat_messages.append(
                    Messages(role=MessagesRole.ASSISTANT, content=msg["content"])
                )

        with GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            verify_ssl_certs=False,
        ) as giga:
            response = giga.chat(Chat(messages=gigachat_messages))
            return response.choices[0].message.content.strip()
        