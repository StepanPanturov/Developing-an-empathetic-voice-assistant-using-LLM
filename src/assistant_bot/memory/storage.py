from __future__ import annotations

import json
from pathlib import Path


class HistoryStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict[str, list[dict[str, str]]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, list[dict[str, str]]]) -> None:
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, user_id: str, role: str, content: str) -> None:
        data = self._read()
        data.setdefault(user_id, []).append({"role": role, "content": content})
        self._write(data)

    def get_context(self, user_id: str, max_messages: int = 8) -> str:
        data = self._read()
        messages = data.get(user_id, [])[-max_messages:]
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)
