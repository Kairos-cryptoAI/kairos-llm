"""Map a logical reasoning effort to a concrete provider + model.

The defaults implement the architecture's **DeepSeek-first + GPT escalation**
split: cheap DeepSeek models carry the routine flow (Text Scouts on Flash,
Aggregator-Normal on Pro), while GPT-5.5 is reserved for ``high`` and ``xhigh``
(conflict resolution, macro strategy) where the cost of error is highest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kairos_core.enums import ReasoningEffort


class Provider(StrEnum):
    """LLM provider behind a model choice."""

    DEEPSEEK = "deepseek"
    OPENAI = "openai"


@dataclass(frozen=True)
class ModelChoice:
    model: str
    provider: Provider
    # reasoning.effort value passed to the provider, or ``None`` for a
    # non-thinking call (DeepSeek Flash/Pro must NOT receive this parameter).
    provider_effort: str | None = None

    @property
    def send_reasoning_effort(self) -> bool:
        return self.provider_effort is not None


# DeepSeek-first + GPT escalation (see the updated architecture document).
DEFAULT_MAP: dict[ReasoningEffort, ModelChoice] = {
    ReasoningEffort.LOW: ModelChoice("deepseek-v4-flash", Provider.DEEPSEEK),  # non-thinking
    ReasoningEffort.MEDIUM: ModelChoice("deepseek-v4-pro", Provider.DEEPSEEK),  # non-thinking
    ReasoningEffort.HIGH: ModelChoice("gpt-5.5", Provider.OPENAI, "high"),
    ReasoningEffort.XHIGH: ModelChoice("gpt-5.5", Provider.OPENAI, "xhigh"),
}


class ModelRouter:
    def __init__(self, mapping: dict[ReasoningEffort, ModelChoice] | None = None) -> None:
        self._map = mapping or dict(DEFAULT_MAP)

    def choose(self, effort: ReasoningEffort) -> ModelChoice:
        return self._map[effort]

    def override(
        self, effort: ReasoningEffort, model: str, provider: Provider, provider_effort: str | None = None
    ) -> None:
        self._map[effort] = ModelChoice(model, provider, provider_effort)
