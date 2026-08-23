"""Fail-closed, provider-wide spend reservations for production LLM calls."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol
from uuid import uuid4

from kairos_core.enums import ReasoningEffort
from pydantic import BaseModel

from .errors import LLMBudgetError, LLMServerError
from .gateway import LLMGateway
from .models import LLMWorkload, Provider
from .pricing import PriceTable
from .schemas import LLMResult, TokenUsage

MICROUSD_PER_USD = 1_000_000
INPUT_TOKEN_OVERHEAD = 1_024
MINIMUM_INPUT_TOKEN_RESERVATION = 4_096
REGISTERED_PROVIDER_BUDGETS_MICROUSD: dict[Provider, int] = {
    Provider.DEEPSEEK: 4_500_000,
    Provider.OPENAI: 45_000_000,
}


class LLMUsageBudget(Protocol):
    """Persistence boundary implemented by the runtime repository package."""

    async def reserve(
        self,
        *,
        provider: str,
        reservation_id: str,
        reserved_microusd: int,
        monthly_budget_microusd: int,
    ) -> None: ...

    async def commit(
        self,
        *,
        provider: str,
        reservation_id: str,
        actual_microusd: int,
    ) -> None: ...


class DenyLLMUsageBudget:
    """Block paid calls when no durable budget backend is available."""

    async def reserve(self, **_kwargs: Any) -> None:
        raise LLMBudgetError("paid LLM calls require a durable usage budget")

    async def commit(self, **_kwargs: Any) -> None:  # pragma: no cover - reserve always fails
        raise LLMBudgetError("paid LLM calls require a durable usage budget")


class BudgetedLLMGateway:
    """Reserve a worst-case allowance, call once, then commit rounded-up actual cost.

    A failed, cancelled, or ambiguously completed call deliberately leaves its
    reservation outstanding. This can reduce remaining capacity, but it cannot
    silently spend the same monthly dollars twice.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        budget: LLMUsageBudget,
        *,
        monthly_budgets_microusd: Mapping[Provider, int] | None = None,
        prices: PriceTable | None = None,
    ) -> None:
        if gateway.settings.max_retries != 0:
            raise ValueError("budgeted LLM gateway requires max_retries=0")
        configured = dict(
            REGISTERED_PROVIDER_BUDGETS_MICROUSD
            if monthly_budgets_microusd is None
            else monthly_budgets_microusd
        )
        if set(configured) != set(REGISTERED_PROVIDER_BUDGETS_MICROUSD):
            raise ValueError("monthly LLM budgets must contain the exact provider set")
        for provider, amount in configured.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                raise ValueError("monthly LLM budgets must be positive integer microdollars")
            if amount > REGISTERED_PROVIDER_BUDGETS_MICROUSD[provider]:
                raise ValueError(f"{provider.value} budget exceeds the registered ceiling")
        self.gateway = gateway
        self.budget = budget
        self.monthly_budgets_microusd = configured
        self.prices = prices or PriceTable()

    @property
    def settings(self):
        return self.gateway.settings

    @property
    def router(self):
        return self.gateway.router

    @property
    def accountant(self):
        return self.gateway.accountant

    @property
    def _on_health(self):
        """Compatibility view for existing service wiring diagnostics."""
        return self.gateway._on_health

    async def complete(
        self,
        *,
        system: str,
        user: str,
        effort: ReasoningEffort | None = None,
        workload: LLMWorkload | None = None,
        schema: type[BaseModel] | None = None,
    ) -> LLMResult:
        if workload is None:
            raise LLMBudgetError("budgeted production calls require an explicit workload")
        route = self.router.resolve(effort, workload=workload)
        input_ceiling = self._input_token_ceiling(system, user, schema)
        output_ceiling = min(self.settings.max_output_tokens, route.max_output_tokens)
        reserved_microusd = self._microusd(
            self.prices.cost(
                route.choice.model,
                TokenUsage(input_tokens=input_ceiling, output_tokens=output_ceiling),
            )
        )
        reservation_id = f"{provider_identity(route.choice.provider)}:{uuid4().hex}"
        await self.budget.reserve(
            provider=route.choice.provider.value,
            reservation_id=reservation_id,
            reserved_microusd=reserved_microusd,
            monthly_budget_microusd=self.monthly_budgets_microusd[route.choice.provider],
        )
        result = await self.gateway.complete(
            system=system,
            user=user,
            effort=effort,
            workload=workload,
            schema=schema,
        )
        if result.usage.input_tokens > input_ceiling or result.usage.output_tokens > output_ceiling:
            raise LLMServerError("provider usage exceeded the durable reservation envelope")
        actual_microusd = self._microusd(result.cost_usd)
        if actual_microusd > reserved_microusd:
            raise LLMServerError("accounted LLM cost exceeded the durable reservation")
        await self.budget.commit(
            provider=route.choice.provider.value,
            reservation_id=reservation_id,
            actual_microusd=actual_microusd,
        )
        return replace(result, budget_reservation_id=reservation_id)

    async def close(self) -> None:
        await self.gateway.close()

    @staticmethod
    def _input_token_ceiling(
        system: str,
        user: str,
        schema: type[BaseModel] | None,
    ) -> int:
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError("LLM prompts must be strings")
        schema_bytes = 0
        if schema is not None:
            encoded = json.dumps(
                schema.model_json_schema(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            schema_bytes = len(encoded)
        prompt_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8"))
        return max(
            MINIMUM_INPUT_TOKEN_RESERVATION,
            prompt_bytes + schema_bytes + INPUT_TOKEN_OVERHEAD,
        )

    @staticmethod
    def _microusd(cost_usd: float) -> int:
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("LLM cost must be finite and non-negative")
        return math.ceil(cost_usd * MICROUSD_PER_USD)


def provider_identity(provider: Provider) -> str:
    """Stable reservation domain kept separate from service-specific IDs."""
    return f"kairos-llm-v1:{provider.value}"
