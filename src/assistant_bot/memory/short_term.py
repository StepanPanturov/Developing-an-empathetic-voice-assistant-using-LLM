from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis


class ShortTermMemory:
    """Краткосрочная память через Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", window_size: int = 20, ttl_hours: int = 24) -> None:
        self.redis_url = redis_url
        self.window_size = window_size
        self.ttl_seconds = ttl_hours * 3600
        self.redis = None

    async def _get_redis(self) -> redis.Redis:
        if self.redis is None:
            self.redis = redis.from_url(self.redis_url)
        return self.redis

    async def add_message(self, user_id: str, role: str, content: str, emotion: dict[str, Any] | None = None) -> None:
        r = await self._get_redis()
        key = f"dialog:{user_id}"
        message = {
            "role": role,
            "content": content,
            "emotion": emotion or {},
            "timestamp": json.dumps(None),  # Можно добавить datetime
        }
        await r.lpush(key, json.dumps(message))
        await r.ltrim(key, 0, self.window_size - 1)
        await r.expire(key, self.ttl_seconds)

    async def get_history(self, user_id: str) -> list[dict[str, Any]]:
        r = await self._get_redis()
        key = f"dialog:{user_id}"
        messages = await r.lrange(key, 0, -1)
        return [json.loads(msg) for msg in reversed(messages)]

    async def clear(self, user_id: str) -> None:
        r = await self._get_redis()
        key = f"dialog:{user_id}"
        await r.delete(key)
        