from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EmotionResult:
    primary_emotion: str
    sentiment: str
    intensity: float
    confidence: float
    needs_support: bool
    raw_scores: dict[str, Any]