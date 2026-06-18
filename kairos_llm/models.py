"""Map a logical reasoning effort to a concrete model + provider effort.

The defaults implement the spec's cost-optimised split: cheap models carry the
routine flow (Text Scouts / Aggregator-Normal), the flagship is reserved for
``high`` and ``xhigh`` (conflict resolution, macro strategy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from kairos_core.enums import ReasoningEffort


@dataclass(frozen=True)
class ModelChoice:
    model: str
    provider_effort: str  # value passed to the provider's reasoning_effort


# xhigh is a Kairos label; real providers cap at "high", so we map it there but
# keep the logical effort for routing/accounting.
DEFAULT_MAP: Dict[ReasoningEffort, ModelChoice] = {
    ReasoningEffort.LOW: ModelChoice("gpt-5.5-mini", "low"),
    ReasoningEffort.MEDIUM: ModelChoice("gpt-5.5-mini", "medium"),
    ReasoningEffort.HIGH: ModelChoice("gpt-5.5", "high"),
    ReasoningEffort.XHIGH: ModelChoice("gpt-5.5", "high"),
}


class ModelRouter:
    def __init__(self, mapping: Dict[ReasoningEffort, ModelChoice] | None = None) -> None:
        self._map = mapping or dict(DEFAULT_MAP)

    def choose(self, effort: ReasoningEffort) -> ModelChoice:
        return self._map[effort]

    def override(self, effort: ReasoningEffort, model: str, provider_effort: str) -> None:
        self._map[effort] = ModelChoice(model, provider_effort)
