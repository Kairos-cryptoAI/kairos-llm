"""Explicit causal A/B hookup; never a service, alpha freeze or trading route.

The caller supplies an already enrolled SIM repository, a genuinely frozen
strategy evaluator, immutable history resolver (through that evaluator),
review/proposal prompts and the existing shared budget. Constructing this
adapter does not create a campaign, read credentials, subscribe or call a model.
Router classification is advisory context at the later observation cutoff; it
cannot backdate a CandidateRoute or change the preregistered provider/model.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

from kairos_core import canonical_sha256
from kairos_core.config import DEFAULT_TRADING_SYMBOLS
from kairos_core.contracts.base import datetime_from_unix_ms
from kairos_persistence.campaign_inputs import (
    CAMPAIGN_INPUT_BRIDGE_SHA256,
    decode_campaign_macro,
    decode_campaign_news,
)
from kairos_persistence.causal_campaign import CausalBaselineResultV1
from kairos_persistence.source_state import QUALIFICATION_CAMPAIGN_ID, SourceStateRepository
from kairos_persistence.usage_budget import CampaignLLMUsageBudget
from kairos_router.aggregation import SignalWindow
from kairos_router.candidate import classify_candidate
from kairos_strategy.campaign import CausalStrategyEvaluator

from .campaign import AdaptiveCampaignScheduler
from .research import RESEARCH_INPUT_FEATURE_SHA256, ResearchEvidenceError


def _module_sha256(name: str) -> str:
    path = import_module(name).__file__
    if path is None:
        raise ResearchEvidenceError("campaign source artifact is unavailable")
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalRouterContextPolicyV1:
    """All aggregation parameters are explicit, finite, independently frozen."""

    sentiment_ttl_s: float
    deadband: float
    minimum_confidence: float
    maximum_evidence: int
    trading_symbols: tuple[str, ...]

    def __post_init__(self):
        for value in (self.sentiment_ttl_s, self.deadband, self.minimum_confidence):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("router research policy requires finite explicit numbers")
        if (
            not 0 < self.sentiment_ttl_s <= 3600
            or not 0 <= self.deadband <= 1
            or not 0 <= self.minimum_confidence <= 1
            or type(self.maximum_evidence) is not int
            or not 1 <= self.maximum_evidence <= 64
            or type(self.trading_symbols) is not tuple
            or self.trading_symbols != tuple(DEFAULT_TRADING_SYMBOLS)
        ):
            raise ValueError("router research policy exceeds its bounded aggregation scope")


class CausalReviewContextProjector:
    """Reuse the real Router's pure aggregation/classification without dispatch."""

    def __init__(self, policy: CausalRouterContextPolicyV1):
        if type(policy) is not CausalRouterContextPolicyV1:
            raise TypeError("research Router policy must be explicitly typed")
        self._policy = policy
        self.artifact_sha256 = self._artifact()

    def _artifact(self) -> str:
        return canonical_sha256(
            {
                "contract_version": "causal-router-context.v1",
                "policy": asdict(self._policy),
                "sources": {
                    name: _module_sha256(name)
                    for name in (
                        "kairos_router.aggregation",
                        "kairos_router.candidate",
                        "kairos_router.conflict",
                        "kairos_llm.causal_campaign",
                    )
                },
                "input_bridge_sha256": CAMPAIGN_INPUT_BRIDGE_SHA256,
                "authority": "advisory-only-no-route-or-model-selection",
                "clock": "captured-before-context-cutoff-not-intent-close",
            }
        )

    def assert_frozen(self):
        if self._artifact() != self.artifact_sha256:
            raise ResearchEvidenceError("research Router artifact or policy changed")

    def validate_sources(self, window, sources):
        self.assert_frozen()
        if type(sources) is not tuple or not 3 <= len(sources) <= 16:
            raise ResearchEvidenceError("Router context requires the bounded exact captured source tuple")
        scope = {
            (source.campaign_id, source.schedule_digest, source.candidate_protocol_digest)
            for source in sources
        }
        if len(scope) != 1 or len({source.receipt_sha256 for source in sources}) != len(sources):
            raise ResearchEvidenceError("Router context cannot mix source scopes or repeat a receipt")
        news, macro = [], []
        identities = {}
        for source in sources:
            if (
                source.receipt_sha256 != canonical_sha256(source.identity_payload())
                or source.content_sha256 != canonical_sha256(source.content)
                or source.sample_id != window.sample_id
                or source.source_as_of_ts_ms > window.market_as_of_ts_ms
                or source.observed_at_ts_ms > window.market_as_of_ts_ms
            ):
                raise ResearchEvidenceError("Router context source differs from its captured cutoff")
            if source.source_kind == "NEWS":
                message = decode_campaign_news(source.content)
                news.append(message)
            elif source.source_kind == "MACRO":
                message = decode_campaign_macro(source.content)
                macro.append(message)
            else:
                continue
            delta = message.produced_at.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
            event_ms = delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000
            if event_ms != source.source_as_of_ts_ms or message.message_id != source.reference:
                raise ResearchEvidenceError("producer identity or clock differs from stored source")
            identity = (message.source, message.message_id)
            digest = canonical_sha256(message.model_dump(mode="json"))
            if identity in identities and identities[identity] != digest:
                raise ResearchEvidenceError("changed duplicate producer identity cannot reach Router")
            identities[identity] = digest
        if not news or not macro:
            raise ResearchEvidenceError("causal context requires captured Text and Macro producer messages")
        return news, macro

    def project(self, evaluation, sources, window):
        news, macro = self.validate_sources(window, sources)
        if (
            evaluation.sample_id != window.sample_id
            or evaluation.evidence_as_of_ts_ms != window.market_as_of_ts_ms
            or evaluation.source_receipt_sha256s != tuple(sorted(source.receipt_sha256 for source in sources))
            or any(
                (source.campaign_id, source.schedule_digest, source.candidate_protocol_digest)
                != (evaluation.campaign_id, evaluation.schedule_digest, evaluation.candidate_protocol_digest)
                for source in sources
            )
        ):
            raise ResearchEvidenceError("Router projection differs from its independently saved evaluation")
        if evaluation.intent is None:
            raise ResearchEvidenceError("NO_INTENT cannot fabricate a routed review candidate")
        policy = self._policy
        signals = SignalWindow(
            sentiment_ttl_s=policy.sentiment_ttl_s,
            deadband=policy.deadband,
            min_confidence=policy.minimum_confidence,
            max_evidence=policy.maximum_evidence,
        )
        for message in news:
            signals.add_sentiment(
                message.topic.strip().upper()
                if message.topic.strip().upper() in policy.trading_symbols
                else "*",
                message_id=message.message_id,
                produced_at=message.produced_at,
                sentiment=message.sentiment,
                impact=message.impact,
                confidence=message.confidence,
            )
        as_of = datetime_from_unix_ms(window.market_as_of_ts_ms)
        aggregate = signals.aggregate(window.symbol, as_of=as_of)
        if not aggregate.has_relevant_evidence:
            aggregate = signals.aggregate("*", as_of=as_of)
        context = {
            "contract_version": "causal-router-context.v1",
            "projector_sha256": self.artifact_sha256,
            "context_cutoff_ts_ms": window.market_as_of_ts_ms,
            "original_intent_id": evaluation.intent.intent_id,
            "original_intent_clock_ts_ms": evaluation.intent.decision_ts_ms,
            "tier": classify_candidate(evaluation.intent.side, aggregate.bias).value,
            "text": {
                "bias": aggregate.bias.value,
                "score": aggregate.score,
                "confidence": aggregate.confidence,
                "message_ids": list(aggregate.sentiment_ids),
                "has_relevant_evidence": aggregate.has_relevant_evidence,
            },
            "macro": [message.model_dump(mode="json") for message in macro],
            "source_receipt_sha256s": sorted(source.receipt_sha256 for source in sources),
            "model_route": "exact-preregistered-no-dynamic-escalation",
            "authority": "SIM_RESEARCH_ONLY",
        }
        return {**context, "context_sha256": canonical_sha256(context)}


def causal_campaign_scheduler_sha256(projector: CausalReviewContextProjector) -> str:
    projector.assert_frozen()
    return canonical_sha256(
        {
            "contract_version": "adaptive-causal-campaign-scheduler.v1",
            "projector_sha256": projector.artifact_sha256,
            "input_bridge_sha256": CAMPAIGN_INPUT_BRIDGE_SHA256,
            "sources": {
                name: _module_sha256(name)
                for name in (
                    "kairos_llm.campaign",
                    "kairos_llm.causal_campaign",
                    "kairos_llm.research",
                    "kairos_llm.budget",
                    "kairos_persistence.campaign_inputs",
                    "kairos_persistence.causal_campaign",
                    "kairos_persistence.research_campaign",
                    "kairos_persistence.usage_budget",
                )
            },
            "evaluation": "real-original-intent-separate-anchor-context-db-clocks",
            "pairing": "independent-causal-pair-not-core-v1-sample-or-seal",
            "budget": "existing-shared-adopted-ledger-no-reset",
            "authority": "SIM_RESEARCH_ONLY-no-campaign-registration-or-trading",
        }
    )


class CausalAdaptiveCampaignScheduler(AdaptiveCampaignScheduler):
    """Explicit opt-in path; legacy v2 scheduler and frozen campaigns stay separate."""

    def __init__(self, *, projector: CausalReviewContextProjector, **kwargs):
        self.projector = projector
        self.scheduler_sha256 = causal_campaign_scheduler_sha256(projector)
        super().__init__(**kwargs)
        if type(self.evaluator) is not CausalStrategyEvaluator:
            raise ResearchEvidenceError("causal campaign requires the actual registered strategy adapter")
        self._assert_causal()

    def _assert_causal(self):
        if (
            self.repository.causal_scheduler_sha256 != self.scheduler_sha256
            or causal_campaign_scheduler_sha256(self.projector) != self.scheduler_sha256
        ):
            raise ResearchEvidenceError("repository/scheduler lacks this exact frozen causal opt-in")
        budget = self.gateway.budget
        if (
            type(budget) is not CampaignLLMUsageBudget
            or not isinstance(budget.repository, SourceStateRepository)
            or budget.repository.campaign_id != QUALIFICATION_CAMPAIGN_ID
        ):
            raise ResearchEvidenceError("causal campaign requires the existing shared durable usage budget")

    async def _validate(self, campaign_id):
        self._assert_causal()
        plan, schedule, protocol = await super()._validate(campaign_id)
        if any(arm.provider != "openai" for arm in protocol.arms[1:]):
            raise ResearchEvidenceError("causal research retains the approved OpenAI-only provider choice")
        return plan, schedule, protocol

    async def reconcile(self, campaign_id):
        self._assert_causal()
        return await super().reconcile(campaign_id)

    def _input_feature_sha256(self, arm_id):
        if arm_id == "strategy-review":
            return self.projector.artifact_sha256
        return RESEARCH_INPUT_FEATURE_SHA256

    async def _evaluate_window(self, *, window, sources):
        self.projector.validate_sources(window, sources)
        result = await self.evaluator.evaluate(window=window, sources=sources)
        if type(result) is not CausalBaselineResultV1:
            raise ResearchEvidenceError("causal evaluator must return its distinct typed result")
        self._assert_causal()
        return result

    async def _record_evaluation(self, claim, bundle, intent):
        self._assert_causal()
        return await self.repository.record_causal_evaluation(claim=claim, bundle=bundle, result=intent)

    async def _evaluation_is_late(self, evaluation, window):
        actual = await self.repository.evaluation_recorded_at(
            campaign_id=evaluation.campaign_id,
            sample_id=evaluation.sample_id,
            evaluation_receipt_sha256=str(evaluation.receipt_sha256),
        )
        return max(actual, evaluation.evaluated_at_ts_ms) > window.paired_at_ts_ms

    async def _replay_proposal_sample(self, coordinator, observation, **kwargs):
        self._assert_causal()
        return await self.repository.record_causal_pair(
            campaign_id=kwargs["schedule"].campaign_id,
            sample_id=kwargs["sample_id"],
            evaluation_receipt_sha256=kwargs["evaluation_receipt_sha256"],
            attempt_id=observation.start.attempt_id,
        )

    def _review_input_metadata(self, evaluation, sources, window):
        self._assert_causal()
        return {
            "contract_version": "adaptive-causal-review-input.v1",
            "router_context": self.projector.project(evaluation, sources, window),
        }
