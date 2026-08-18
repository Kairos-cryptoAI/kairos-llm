from types import SimpleNamespace

import pytest
from kairos_core.enums import ReasoningEffort

from kairos_llm.budget import (
    REGISTERED_PROVIDER_BUDGETS_MICROUSD,
    BudgetedLLMGateway,
    DenyLLMUsageBudget,
)
from kairos_llm.errors import LLMBudgetError
from kairos_llm.models import DEFAULT_WORKLOAD_ROUTES, LLMWorkload, ModelRouter, Provider
from kairos_llm.pricing import CostAccountant, PriceTable
from kairos_llm.schemas import LLMResult, TokenUsage


class _Budget:
    def __init__(self, error=None):
        self.error = error
        self.reservations = []
        self.commits = []

    async def reserve(self, **kwargs):
        self.reservations.append(kwargs)
        if self.error is not None:
            raise self.error

    async def commit(self, **kwargs):
        self.commits.append(kwargs)


class _Gateway:
    def __init__(self, *, retries=0, error=None):
        self.settings = SimpleNamespace(max_retries=retries, max_output_tokens=8_192)
        self.router = ModelRouter()
        self.accountant = CostAccountant()
        self.error = error
        self.calls = []
        self.closed = False

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        workload = kwargs["workload"]
        route = DEFAULT_WORKLOAD_ROUTES[workload]
        usage = TokenUsage(input_tokens=1_500, cached_input_tokens=750, output_tokens=300)
        return LLMResult(
            content='{"ok":true}',
            parsed={"ok": True},
            model=route.choice.model,
            effort=route.effort.value,
            usage=usage,
            cost_usd=PriceTable().cost(route.choice.model, usage),
            latency_s=0.5,
            workload=workload.value,
            resolved_model=route.choice.model,
        )

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_reserves_before_call_and_commits_rounded_actual_cost():
    budget = _Budget()
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, budget)

    result = await gateway.complete(
        system="return json",
        user="{}",
        workload=LLMWorkload.TEXT_SCOUTS,
    )

    assert result.workload == LLMWorkload.TEXT_SCOUTS.value
    assert len(underlying.calls) == 1
    reservation = budget.reservations[0]
    assert reservation["provider"] == Provider.DEEPSEEK.value
    assert reservation["monthly_budget_microusd"] == 4_500_000
    assert reservation["reserved_microusd"] == 861
    assert reservation["reservation_id"].startswith("kairos-llm-v1:deepseek:")
    assert budget.commits == [
        {
            "provider": Provider.DEEPSEEK.value,
            "reservation_id": reservation["reservation_id"],
            "actual_microusd": 192,
        }
    ]


@pytest.mark.asyncio
async def test_budget_denial_happens_before_provider_call():
    budget = _Budget(LLMBudgetError("monthly cap"))
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, budget)

    with pytest.raises(LLMBudgetError, match="monthly cap"):
        await gateway.complete(system="json", user="{}", workload=LLMWorkload.MACRO_STRATEGIST)

    assert underlying.calls == []
    assert budget.commits == []


@pytest.mark.asyncio
async def test_ambiguous_provider_failure_preserves_reservation():
    budget = _Budget()
    underlying = _Gateway(error=RuntimeError("ambiguous provider failure"))
    gateway = BudgetedLLMGateway(underlying, budget)

    with pytest.raises(RuntimeError, match="ambiguous"):
        await gateway.complete(system="json", user="{}", workload=LLMWorkload.AGGREGATOR_NORMAL)

    assert len(budget.reservations) == 1
    assert budget.commits == []


@pytest.mark.asyncio
async def test_deny_backend_and_legacy_effort_are_fail_closed():
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, DenyLLMUsageBudget())

    with pytest.raises(LLMBudgetError, match="explicit workload"):
        await gateway.complete(system="json", user="{}", effort=ReasoningEffort.LOW)
    with pytest.raises(LLMBudgetError, match="durable usage budget"):
        await gateway.complete(system="json", user="{}", workload=LLMWorkload.TEXT_SCOUTS)
    assert underlying.calls == []


def test_registered_caps_and_single_attempt_are_not_configurable_upward():
    with pytest.raises(ValueError, match="max_retries=0"):
        BudgetedLLMGateway(_Gateway(retries=1), _Budget())
    with pytest.raises(ValueError, match="registered ceiling"):
        BudgetedLLMGateway(
            _Gateway(),
            _Budget(),
            monthly_budgets_microusd={
                **REGISTERED_PROVIDER_BUDGETS_MICROUSD,
                Provider.OPENAI: 45_000_001,
            },
        )
    with pytest.raises(ValueError, match="exact provider set"):
        BudgetedLLMGateway(
            _Gateway(),
            _Budget(),
            monthly_budgets_microusd={Provider.OPENAI: 1},
        )


@pytest.mark.asyncio
async def test_close_is_delegated():
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, _Budget())
    await gateway.close()
    assert underlying.closed is True
