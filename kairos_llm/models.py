"""Resolve Kairos workloads to concrete provider models and reasoning tiers.

Explicit workload routes are authoritative for production callers.  The effort
map remains available for callers compiled against the original API and mirrors
the same four default routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kairos_core.enums import ReasoningEffort


class Provider(StrEnum):
    """LLM provider behind a model choice."""

    DEEPSEEK = "deepseek"
    OPENAI = "openai"


class LLMWorkload(StrEnum):
    """Stable analytical roles whose quality/cost route is architecture-owned."""

    TEXT_SCOUTS = "text_scouts"
    AGGREGATOR_NORMAL = "aggregator_normal"
    AGGREGATOR_CONFLICT = "aggregator_conflict"
    MACRO_STRATEGIST = "macro_strategist"


@dataclass(frozen=True)
class ModelChoice:
    model: str
    provider: Provider
    # OpenAI reasoning.effort value. DeepSeek Text Scouts calls disable
    # thinking explicitly in the provider adapter instead of passing this.
    provider_effort: str | None = None

    @property
    def send_reasoning_effort(self) -> bool:
        return self.provider_effort is not None


@dataclass(frozen=True)
class ModelRoute:
    """A resolved model choice plus its effective logical effort/workload."""

    choice: ModelChoice
    effort: ReasoningEffort
    workload: LLMWorkload | None = None
    max_output_tokens: int = 8_192

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")


DEFAULT_WORKLOAD_ROUTES: dict[LLMWorkload, ModelRoute] = {
    LLMWorkload.TEXT_SCOUTS: ModelRoute(
        ModelChoice("deepseek-v4-flash", Provider.DEEPSEEK),
        ReasoningEffort.LOW,
        LLMWorkload.TEXT_SCOUTS,
        1_024,
    ),
    LLMWorkload.AGGREGATOR_NORMAL: ModelRoute(
        ModelChoice("gpt-5.6-luna", Provider.OPENAI, "medium"),
        ReasoningEffort.MEDIUM,
        LLMWorkload.AGGREGATOR_NORMAL,
        2_048,
    ),
    LLMWorkload.AGGREGATOR_CONFLICT: ModelRoute(
        ModelChoice("gpt-5.6-terra", Provider.OPENAI, "high"),
        ReasoningEffort.HIGH,
        LLMWorkload.AGGREGATOR_CONFLICT,
        4_096,
    ),
    LLMWorkload.MACRO_STRATEGIST: ModelRoute(
        ModelChoice("gpt-5.6-sol", Provider.OPENAI, "xhigh"),
        ReasoningEffort.XHIGH,
        LLMWorkload.MACRO_STRATEGIST,
        8_192,
    ),
}

# Backward-compatible effort-only routes. New service code should pass a
# workload so unrelated callers cannot silently share a route merely because
# they requested the same amount of reasoning.
DEFAULT_MAP: dict[ReasoningEffort, ModelChoice] = {
    route.effort: route.choice for route in DEFAULT_WORKLOAD_ROUTES.values()
}


class ModelRouter:
    def __init__(
        self,
        mapping: dict[ReasoningEffort, ModelChoice] | None = None,
        workload_mapping: dict[LLMWorkload, ModelRoute] | None = None,
    ) -> None:
        self._map = dict(DEFAULT_MAP) if mapping is None else dict(mapping)
        self._workload_map = (
            dict(DEFAULT_WORKLOAD_ROUTES) if workload_mapping is None else dict(workload_mapping)
        )

    def choose(
        self,
        effort: ReasoningEffort,
        *,
        workload: LLMWorkload | None = None,
    ) -> ModelChoice:
        """Choose a model while preserving the original effort-only API."""
        if workload is not None:
            return self._workload_map[workload].choice
        return self._map[effort]

    def resolve(
        self,
        effort: ReasoningEffort | None = None,
        *,
        workload: LLMWorkload | None = None,
    ) -> ModelRoute:
        """Resolve an explicit workload or fall back to the legacy effort map."""
        if workload is not None:
            return self._workload_map[workload]
        if effort is None:
            raise ValueError("either workload or effort is required")
        return ModelRoute(choice=self._map[effort], effort=effort)

    def override(
        self,
        effort: ReasoningEffort,
        model: str,
        provider: Provider,
        provider_effort: str | None = None,
    ) -> None:
        """Override a legacy effort fallback without altering workload routes."""
        self._map[effort] = ModelChoice(model, provider, provider_effort)

    def override_workload(
        self,
        workload: LLMWorkload,
        model: str,
        provider: Provider,
        effort: ReasoningEffort,
        provider_effort: str | None = None,
        max_output_tokens: int = 8_192,
    ) -> None:
        """Override one explicit workload without coupling it to other roles."""
        self._workload_map[workload] = ModelRoute(
            choice=ModelChoice(model, provider, provider_effort),
            effort=effort,
            workload=workload,
            max_output_tokens=max_output_tokens,
        )
