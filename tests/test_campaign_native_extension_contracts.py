"""Isolated contract tests for explicit native campaign extension hooks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
from kairos_core import canonical_sha256
from kairos_core.config import DEFAULT_TRADING_SYMBOLS
from kairos_persistence.research_campaign import ResearchCampaignRepository, ResearchReviewOutputV1
from kairos_persistence.source_state import QUALIFICATION_CAMPAIGN_ID, SourceStateRepository
from kairos_persistence.usage_budget import CampaignLLMUsageBudget
from kairos_strategy.campaign import CausalStrategyEvaluator
from pydantic import BaseModel

from kairos_llm.budget import BudgetedLLMGateway
from kairos_llm.campaign import CAMPAIGN_SCHEDULER_SHA256, AdaptiveCampaignScheduler
from kairos_llm.causal_campaign import (
    CausalAdaptiveCampaignScheduler,
    CausalReviewContextProjector,
    CausalRouterContextPolicyV1,
    causal_campaign_scheduler_sha256,
)
from kairos_llm.models import LLMWorkload, ModelRouter
from kairos_llm.proposals import LLMProposalOutputV1
from kairos_llm.research import (
    RESEARCH_INPUT_FEATURE_SHA256,
    ResearchEvidenceError,
    ResearchPromptArtifactV1,
)


class ExtensionReviewOutput(BaseModel):
    """Distinct test schema; production receipt mapping remains subclass-owned."""

    contract_version: Literal["native-extension-review.v1"] = "native-extension-review.v1"
    action: Literal["ALLOW", "VETO", "DEFER"]
    rationale: str
    evidence_ids: tuple[str, ...]


class _ReviewSchemaScheduler(AdaptiveCampaignScheduler):
    def _review_output_schema(self) -> type[BaseModel]:
        return ExtensionReviewOutput


class _Repository:
    def __init__(self, campaign_sha256: str = CAMPAIGN_SCHEDULER_SHA256):
        self.campaign_sha256 = campaign_sha256

    async def load_campaign(self, _campaign_id):
        return (
            SimpleNamespace(scheduler_sha256=self.campaign_sha256),
            SimpleNamespace(evaluator_sha256="e" * 64, strategy_id="fixture", strategy_revision="1"),
            SimpleNamespace(arms=()),
        )


class _Evaluator:
    artifact_sha256 = "e" * 64
    strategy_id = "fixture"
    strategy_revision = "1"


def _review_validation_target(scheduler_type, review_schema):
    prompt = ResearchPromptArtifactV1(
        system="Fixture system",
        user_prefix="Fixture data:",
        workload=LLMWorkload.AGGREGATOR_NORMAL,
        reasoning_effort="medium",
        provider_effort="medium",
        max_output_tokens=512,
    )
    route = ModelRouter().resolve(workload=prompt.workload)

    def arm(arm_id, schema):
        return SimpleNamespace(
            arm_id=arm_id,
            prompt_sha256=prompt.prompt_sha256,
            schema_sha256=canonical_sha256(schema.model_json_schema()),
            input_feature_sha256=RESEARCH_INPUT_FEATURE_SHA256,
            provider=route.choice.provider.value,
            model=route.choice.model,
        )

    protocol = SimpleNamespace(
        arms=(
            SimpleNamespace(arm_id="strategy-only"),
            arm("strategy-review", review_schema),
            arm("llm-proposal-research", LLMProposalOutputV1),
        )
    )
    scheduler = object.__new__(scheduler_type)
    scheduler.repository = _Repository()
    scheduler.gateway = SimpleNamespace(router=ModelRouter(), settings=SimpleNamespace(max_output_tokens=512))
    scheduler.evaluator = _Evaluator()
    scheduler.review_prompt = prompt
    scheduler.proposal_prompt = prompt
    return scheduler, protocol


@pytest.mark.asyncio
async def test_default_review_schema_is_exact_and_is_the_preregistered_schema():
    schema = AdaptiveCampaignScheduler._review_output_schema(object.__new__(AdaptiveCampaignScheduler))
    assert schema is ResearchReviewOutputV1
    scheduler, protocol = _review_validation_target(AdaptiveCampaignScheduler, ResearchReviewOutputV1)
    scheduler.repository = _Repository()
    scheduler.repository.load_campaign = lambda _campaign_id: _loaded_campaign(
        scheduler, protocol, ResearchReviewOutputV1
    )
    plan, schedule, loaded_protocol = await scheduler._validate("fixture-campaign")
    assert plan.scheduler_sha256 == scheduler.scheduler_sha256
    assert schedule.strategy_id == scheduler.evaluator.strategy_id
    assert loaded_protocol is protocol


@pytest.mark.asyncio
async def test_subclass_review_schema_must_match_exact_preregistered_digest():
    scheduler, protocol = _review_validation_target(_ReviewSchemaScheduler, ExtensionReviewOutput)
    scheduler.repository.load_campaign = lambda _campaign_id: _loaded_campaign(
        scheduler, protocol, ExtensionReviewOutput
    )
    _plan, _schedule, loaded_protocol = await scheduler._validate("fixture-campaign")
    assert loaded_protocol is protocol

    mismatched, bad_protocol = _review_validation_target(_ReviewSchemaScheduler, ResearchReviewOutputV1)
    mismatched.repository.load_campaign = lambda _campaign_id: _loaded_campaign(
        mismatched, bad_protocol, ResearchReviewOutputV1
    )
    with pytest.raises(ResearchEvidenceError, match="route/prompt/schema differs"):
        await mismatched._validate("fixture-campaign")


async def _loaded_campaign(scheduler, protocol, schema):
    route = scheduler.gateway.router.resolve(workload=scheduler.review_prompt.workload)
    arms = list(protocol.arms)
    arms[1] = SimpleNamespace(
        arm_id="strategy-review",
        prompt_sha256=scheduler.review_prompt.prompt_sha256,
        schema_sha256=canonical_sha256(schema.model_json_schema()),
        input_feature_sha256=scheduler._input_feature_sha256("strategy-review"),
        provider=route.choice.provider.value,
        model=route.choice.model,
    )
    proposal_route = scheduler.gateway.router.resolve(workload=scheduler.proposal_prompt.workload)
    arms[2] = SimpleNamespace(
        arm_id="llm-proposal-research",
        prompt_sha256=scheduler.proposal_prompt.prompt_sha256,
        schema_sha256=canonical_sha256(LLMProposalOutputV1.model_json_schema()),
        input_feature_sha256=scheduler._input_feature_sha256("llm-proposal-research"),
        provider=proposal_route.choice.provider.value,
        model=proposal_route.choice.model,
    )
    protocol.arms = tuple(arms)
    return (
        SimpleNamespace(scheduler_sha256=scheduler.scheduler_sha256),
        SimpleNamespace(evaluator_sha256="e" * 64, strategy_id="fixture", strategy_revision="1"),
        protocol,
    )


def _projector():
    return CausalReviewContextProjector(
        CausalRouterContextPolicyV1(
            sentiment_ttl_s=60,
            deadband=0.1,
            minimum_confidence=0.2,
            maximum_evidence=8,
            trading_symbols=tuple(DEFAULT_TRADING_SYMBOLS),
        )
    )


def _causal_constructor_args(projector, repository_hash):
    repository = object.__new__(ResearchCampaignRepository)
    repository._causal_scheduler_sha256 = repository_hash

    source_state = object.__new__(SourceStateRepository)
    source_state.campaign_id = QUALIFICATION_CAMPAIGN_ID
    budget = object.__new__(CampaignLLMUsageBudget)
    budget.repository = source_state

    gateway = object.__new__(BudgetedLLMGateway)
    gateway.gateway = SimpleNamespace(settings=SimpleNamespace(max_retries=0))
    gateway.budget = budget

    evaluator = object.__new__(CausalStrategyEvaluator)
    prompt = ResearchPromptArtifactV1(
        system="Fixture system",
        user_prefix="Fixture data:",
        workload=LLMWorkload.AGGREGATOR_NORMAL,
        reasoning_effort="medium",
        provider_effort="medium",
        max_output_tokens=512,
    )
    return dict(
        projector=projector,
        repository=repository,
        gateway=gateway,
        evaluator=evaluator,
        review_prompt=prompt,
        proposal_prompt=prompt,
    )


def test_causal_scheduler_artifact_hook_is_used_and_revalidated_without_io():
    projector = _projector()
    default = object.__new__(CausalAdaptiveCampaignScheduler)
    assert default._scheduler_artifact(projector) == causal_campaign_scheduler_sha256(projector)

    class _SourceBoundScheduler(CausalAdaptiveCampaignScheduler):
        calls = []
        artifact = "a" * 64

        def _scheduler_artifact(self, supplied_projector):
            self.calls.append(supplied_projector)
            return self.artifact

    args = _causal_constructor_args(projector, "a" * 64)
    scheduler = _SourceBoundScheduler(**args)
    assert scheduler.scheduler_sha256 == "a" * 64
    assert scheduler.calls == [projector, projector]
    assert not hasattr(args["repository"], "_database")

    _SourceBoundScheduler.artifact = "b" * 64
    with pytest.raises(ResearchEvidenceError, match="exact frozen causal opt-in"):
        scheduler._assert_causal()
    assert scheduler.calls == [projector, projector, projector]


def test_default_causal_scheduler_still_rejects_old_or_unmatched_campaign_identity():
    projector = _projector()
    with pytest.raises(ResearchEvidenceError, match="exact frozen causal opt-in"):
        CausalAdaptiveCampaignScheduler(**_causal_constructor_args(projector, "f" * 64))
