"""Result / usage value objects returned by the gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens

    @property
    def billable_input(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)


@dataclass(slots=True)
class LLMResult:
    content: str
    parsed: Any | None
    model: str
    effort: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
