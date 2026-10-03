"""Explicit SIM research calls and crash-safe receipt replay; no trading authority.

Importing this opt-in module requires the ``qualification`` persistence extra.
It is deliberately not imported by the default gateway/package. START is an
admission fence, not proof that a remote provider received a request. A durable
START without a terminal stays unresolved and never causes an automatic retry.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast

from kairos_core import (
    AdaptiveCandidateProtocolV1,
    EvidenceReferenceV1,
    LLMCallFailureV1,
    LLMProposalAdaptiveCandidateArmV1,
    ResearchDecisionSampleV1,
    ResearchObservationScheduleV1,
    canonical_sha256,
)
from kairos_core.research_pairing import ScheduledResearchSampleV1, build_research_decision_sample
from kairos_persistence.causal_campaign import CampaignEvaluationReceipt
from kairos_persistence.research_campaign import ResearchCampaignRepository
from kairos_persistence.research_evidence import (
    ResearchLLMAttemptStartV1,
    ResearchLLMAttemptTerminalV1,
    ResearchSourceReceiptV1,
    ResearchStrategyEvaluationReceiptV1,
)
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .budget import BudgetedLLMGateway, LLMUsageBudget
from .errors import LLMBadOutput, LLMServerError, LLMTimeout
from .gateway import LLMGateway
from .models import LLMWorkload
from .proposals import (
    LLMProposalContext,
    LLMProposalOutputV1,
    build_llm_proposal_completion_receipt,
    build_preregistered_adaptive_llm_proposal,
    proposal_evidence_id,
)
from .schemas import LLMResult

# The immutable identity of this versioned, identity-only input projection.
# It does not claim to compute indicators or to qualify a feature pipeline.
RESEARCH_INPUT_FEATURE_SHA256 = canonical_sha256(
    {
        "contract_version": "research-source-input.v1",
        "renderer": "canonical-json.v1",
        "projection": "verified-source-content-and-causal-reference.v1",
        "ordering": "source-receipt-sha256",
    }
)


class ResearchPromptArtifactV1(BaseModel):
    """Frozen static instructions; sample content is rendered separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["research-proposal-prompt.v1"] = "research-proposal-prompt.v1"
    system: StrictStr = Field(min_length=1, max_length=16_384)
    user_prefix: StrictStr = Field(min_length=1, max_length=16_384)
    workload: LLMWorkload
    reasoning_effort: StrictStr = Field(min_length=1, max_length=32)
    provider_effort: StrictStr | None
    max_output_tokens: StrictInt = Field(ge=128, le=8_192)

    @property
    def prompt_sha256(self) -> str:
        return canonical_sha256(self)


class ResearchEvidenceJournal(Protocol):
    """Narrow, independently verified SIM persistence interface."""

    async def find_attempt(
        self, attempt_id: str
    ) -> tuple[ResearchLLMAttemptStartV1, ResearchLLMAttemptTerminalV1 | None] | None: ...

    async def start_attempt(self, attempt: ResearchLLMAttemptStartV1) -> bool: ...

    async def finish_attempt(self, terminal: ResearchLLMAttemptTerminalV1) -> bool: ...

    async def load_source(self, receipt_sha256: str) -> ResearchSourceReceiptV1: ...

    async def load_evaluation(self, receipt_sha256: str) -> CampaignEvaluationReceipt: ...

    async def record_verified_sample(self, sample: ResearchDecisionSampleV1) -> bool: ...


class ResearchEvidenceError(RuntimeError):
    """Sanitized failure: no provider text, prompt, credential, or DB detail."""


@dataclass(frozen=True)
class ResearchAttemptObservation:
    start: ResearchLLMAttemptStartV1
    terminal: ResearchLLMAttemptTerminalV1 | None

    @property
    def unresolved(self) -> bool:
        return self.terminal is None or self.terminal.terminal_status == "UNRESOLVED"


class _AttemptAlreadyStarted(ResearchEvidenceError):
    pass


class _AttemptBudget:
    """Remap the existing gateway's UUID into one durable research identity."""

    def __init__(
        self,
        budget: LLMUsageBudget,
        journal: ResearchEvidenceJournal,
        make_start: Callable[[int], ResearchLLMAttemptStartV1],
        clock: Callable[[], int],
    ) -> None:
        self.budget, self.journal, self.make_start, self.clock = budget, journal, make_start, clock
        self.start: ResearchLLMAttemptStartV1 | None = None

    async def reserve(
        self, *, provider: str, reservation_id: str, reserved_microusd: int, monthly_budget_microusd: int
    ) -> None:
        # Validate the causal admission window before consuming any budget.
        start = self.make_start(self.clock())
        await self.budget.reserve(
            provider=provider,
            reservation_id=start.budget_reservation_id,
            reserved_microusd=reserved_microusd,
            monthly_budget_microusd=monthly_budget_microusd,
        )
        # A reservation is retained if this insert is ambiguous. There is no
        # provider dispatch before the successful START transaction returns.
        if not await self.journal.start_attempt(start):
            raise _AttemptAlreadyStarted("research attempt already admitted; replay, do not retry")
        self.start = start

    async def commit(self, *, provider: str, reservation_id: str, actual_microusd: int) -> None:
        if self.start is None:
            raise ResearchEvidenceError("research budget commit has no admitted attempt")
        await self.budget.commit(
            provider=provider,
            reservation_id=self.start.budget_reservation_id,
            actual_microusd=actual_microusd,
        )


class _ObservedGateway:
    """Capture observed response/error clocks without altering production wiring."""

    def __init__(self, gateway: LLMGateway, clock: Callable[[], int]) -> None:
        self.gateway, self.clock = gateway, clock
        self.settings, self.router = gateway.settings, gateway.router
        self.response: LLMResult | None = None
        self.observed_at_ts_ms: int | None = None

    async def complete(self, **kwargs) -> LLMResult:
        try:
            self.response = await self.gateway.complete(**kwargs)
            return self.response
        finally:
            self.observed_at_ts_ms = self.clock()


class ResearchProposalCoordinator:
    """One explicit proposal-arm observation using the existing shared budget.

    The caller must independently preregister/enroll schedule and protocol in
    SIM persistence. This class neither enrolls a campaign nor schedules paid
    work. It cannot produce orders, risk decisions, alpha, or performance data.
    """

    def __init__(
        self,
        gateway: BudgetedLLMGateway,
        journal: ResearchEvidenceJournal,
        *,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.gateway, self.journal = gateway, journal
        self.clock = clock or (lambda: time.time_ns() // 1_000_000)

    async def observe(
        self,
        *,
        schedule: ResearchObservationScheduleV1,
        protocol: AdaptiveCandidateProtocolV1,
        sample_id: str,
        prompt: ResearchPromptArtifactV1,
        source_receipt_sha256s: tuple[str, ...],
        workload: LLMWorkload,
    ) -> ResearchAttemptObservation:
        context, user, attempt_id = await self._prepare(
            schedule, protocol, sample_id, prompt, source_receipt_sha256s, workload
        )
        existing = await self.journal.find_attempt(attempt_id)
        if existing is not None:
            return ResearchAttemptObservation(*existing)
        arm = _proposal_arm(protocol)

        def make_start(started_at: int) -> ResearchLLMAttemptStartV1:
            return ResearchLLMAttemptStartV1(
                attempt_id=attempt_id,
                campaign_id=context.campaign_id,
                arm_id="llm-proposal-research",
                sample_id=sample_id,
                schedule_digest=schedule.schedule_digest,
                candidate_protocol_digest=protocol.protocol_digest,
                arm_protocol_digest=protocol.arm_digest(context.arm_id),
                symbol=context.symbol,
                timeframe=context.timeframe,
                market_as_of_ts_ms=context.market_as_of_ts_ms,
                market_snapshot_sha256=context.market_snapshot_sha256,
                sample_deadline_ts_ms=context.expires_at_ts_ms,
                provider=arm.provider,
                requested_model=arm.model,
                prompt_sha256=prompt.prompt_sha256,
                budget_reservation_id=attempt_id,
                attempt_started_at_ts_ms=started_at,
            )

        budget = _AttemptBudget(self.gateway.budget, self.journal, make_start, self.clock)
        observed = _ObservedGateway(self.gateway.gateway, self.clock)
        gateway = BudgetedLLMGateway(
            cast(LLMGateway, observed),
            budget,
            prices=self.gateway.prices,
            monthly_budgets_microusd=self.gateway.monthly_budgets_microusd,
        )
        try:
            result = await gateway.complete(
                system=prompt.system, user=user, workload=workload, schema=LLMProposalOutputV1
            )
        except _AttemptAlreadyStarted:
            existing = await self.journal.find_attempt(attempt_id)
            if existing is None:
                raise ResearchEvidenceError("research admission conflict requires reconciliation") from None
            return ResearchAttemptObservation(*existing)
        except BaseException as exc:
            if budget.start is None:
                # Missing/ambiguous budget/START is not a model outcome.
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise ResearchEvidenceError("research admission unavailable; no provider dispatch") from None
            terminal = self._failure_terminal(budget.start, observed, exc)
            if isinstance(exc, asyncio.CancelledError):
                try:
                    await self._finish(terminal)
                finally:
                    raise exc
            await self._finish(terminal)
            if not isinstance(exc, Exception):
                raise
            return ResearchAttemptObservation(budget.start, terminal)

        if budget.start is None or observed.observed_at_ts_ms is None:
            raise ResearchEvidenceError("research completion lacks trusted admission or observation")
        result = replace(
            result,
            budget_reservation_id=attempt_id,
            attempt_started_at_ts_ms=budget.start.attempt_started_at_ts_ms,
            response_observed_at_ts_ms=observed.observed_at_ts_ms,
        )
        try:
            proposal = build_preregistered_adaptive_llm_proposal(
                schedule=schedule,
                protocol=protocol,
                context=context,
                result=result,
                input_feature_sha256=RESEARCH_INPUT_FEATURE_SHA256,
            )
            completion = build_llm_proposal_completion_receipt(
                context=context,
                result=result,
                proposal=proposal,
                sample_deadline_ts_ms=context.expires_at_ts_ms,
            )
            terminal = ResearchLLMAttemptTerminalV1(
                attempt_id=attempt_id,
                start_receipt_sha256=budget.start.receipt_sha256,
                terminal_status="COMPLETED",
                observed_at_ts_ms=observed.observed_at_ts_ms,
                proposal=proposal,
                completion=completion,
            )
        except (LLMBadOutput, ValueError, TypeError):
            # The provider was called, but no honest admissible completion can
            # be built. Do not fabricate NOT_CALLED or a transport failure.
            terminal = self._unresolved(budget.start, observed.observed_at_ts_ms)
        await self._finish(terminal)
        return ResearchAttemptObservation(budget.start, terminal)

    async def replay_sample(
        self,
        *,
        schedule: ResearchObservationScheduleV1,
        protocol: AdaptiveCandidateProtocolV1,
        sample_id: str,
        prompt: ResearchPromptArtifactV1,
        source_receipt_sha256s: tuple[str, ...],
        workload: LLMWorkload,
        evaluation_receipt_sha256: str,
    ) -> ResearchDecisionSampleV1:
        """Resolve real stored evidence, reconstruct, then independently verify.

        No provider/budget call occurs. Missing/unfinished/late receipts fail
        closed; a stored START can never become NOT_CALLED on replay.
        """
        context, _, attempt_id = await self._prepare(
            schedule, protocol, sample_id, prompt, source_receipt_sha256s, workload
        )
        evaluation = await self.journal.load_evaluation(evaluation_receipt_sha256)
        if type(evaluation) is not ResearchStrategyEvaluationReceiptV1:
            raise ResearchEvidenceError("Core V1 replay cannot adopt a different evaluation clock family")
        window = next(item for item in schedule.windows if item.sample_id == sample_id)
        if (
            evaluation.receipt_sha256 != evaluation_receipt_sha256
            or evaluation.campaign_id != context.campaign_id
            or evaluation.sample_id != sample_id
            or evaluation.schedule_digest != schedule.schedule_digest
            or evaluation.candidate_protocol_digest != protocol.protocol_digest
            or evaluation.strategy_id != schedule.strategy_id
            or evaluation.strategy_revision != schedule.strategy_revision
            or evaluation.symbol != context.symbol
            or evaluation.timeframe != context.timeframe
            or evaluation.market_snapshot_sha256 != context.market_snapshot_sha256
            or evaluation.evaluator_sha256 != schedule.evaluator_sha256
            or evaluation.evaluated_at_ts_ms > window.paired_at_ts_ms
        ):
            raise ResearchEvidenceError("stored evaluation differs from the frozen sample")
        existing = await self.journal.find_attempt(attempt_id)
        observation = ResearchAttemptObservation(*existing) if existing is not None else None
        if observation is not None and observation.unresolved:
            raise ResearchEvidenceError("unresolved research attempt cannot be paired")
        terminal = observation.terminal if observation is not None else None
        sample = build_research_decision_sample(
            sample=ScheduledResearchSampleV1(
                campaign_id=context.campaign_id,
                arm_id=context.arm_id,
                sample_id=sample_id,
                symbol=context.symbol,
                timeframe=context.timeframe,
                market_as_of_ts_ms=context.market_as_of_ts_ms,
                market_snapshot_sha256=context.market_snapshot_sha256,
                strategy_id=schedule.strategy_id,
                strategy_revision=schedule.strategy_revision,
                paired_at_ts_ms=window.paired_at_ts_ms,
                sample_deadline_ts_ms=context.expires_at_ts_ms,
            ),
            strategy_evaluation=evaluation.as_evidence(context.arm_id),
            strategy_intent=evaluation.intent,
            llm_proposal=terminal.proposal if terminal is not None else None,
            llm_completion=terminal.completion if terminal is not None else None,
            llm_failure=terminal.failure if terminal is not None else None,
            llm_was_called=observation is not None,
        )
        # The core pairing helper is arm-agnostic. Bind the independently
        # verified frozen protocol before the strict journal admits a sample,
        # rederiving its identity rather than mutating an already-hashed value.
        sample = ResearchDecisionSampleV1.model_validate(
            {
                **sample.to_payload(),
                "sample_record_id": None,
                "arm_protocol_digest": protocol.arm_digest(context.arm_id),
            }
        )
        await self.journal.record_verified_sample(sample)
        return sample

    async def _prepare(self, schedule, protocol, sample_id, prompt, source_hashes, workload):
        arm = _proposal_arm(protocol)
        if (
            schedule.campaign_id != protocol.campaign_id
            or schedule.schedule_digest != protocol.schedule_digest
        ):
            raise ResearchEvidenceError("research schedule and protocol differ")
        if (
            prompt.prompt_sha256 != arm.prompt_sha256
            or RESEARCH_INPUT_FEATURE_SHA256 != arm.input_feature_sha256
            or canonical_sha256(LLMProposalOutputV1.model_json_schema()) != arm.schema_sha256
        ):
            raise ResearchEvidenceError("research prompt, renderer, or schema differs from the frozen arm")
        route = self.gateway.router.resolve(workload=workload)
        if (
            route.choice.provider.value != arm.provider
            or route.choice.model != arm.model
            or workload != prompt.workload
            or route.effort.value != prompt.reasoning_effort
            or route.choice.provider_effort != prompt.provider_effort
            or min(route.max_output_tokens, self.gateway.settings.max_output_tokens)
            != prompt.max_output_tokens
        ):
            raise ResearchEvidenceError("research route differs from the frozen arm")
        window = next((item for item in schedule.windows if item.sample_id == sample_id), None)
        if window is None:
            raise ResearchEvidenceError("research sample is not preregistered")
        if not source_hashes or len(source_hashes) > 128 or len(set(source_hashes)) != len(source_hashes):
            raise ResearchEvidenceError("research source receipt set must be bounded, distinct, and nonempty")
        sources = [
            ResearchSourceReceiptV1.model_validate(
                (await self.journal.load_source(digest)).model_dump(mode="json")
            )
            for digest in sorted(source_hashes)
        ]
        for digest, source in zip(sorted(source_hashes), sources, strict=True):
            if (
                source.receipt_sha256 != digest
                or source.campaign_id != schedule.campaign_id
                or source.sample_id != sample_id
                or source.schedule_digest != schedule.schedule_digest
                or source.candidate_protocol_digest != protocol.protocol_digest
                or source.observed_at_ts_ms > window.market_as_of_ts_ms
                or source.source_as_of_ts_ms > window.market_as_of_ts_ms
            ):
                raise ResearchEvidenceError("stored source differs from the sample or causal cutoff")
        markets = [source for source in sources if source.source_kind == "MARKET_SNAPSHOT"]
        # Only an explicitly constructed, independently preregistered new
        # campaign repository can resolve delayed-but-causal captured bars.
        # Generic/legacy SIM journals retain their exact timestamp semantics.
        if isinstance(self.journal, ResearchCampaignRepository):
            await self.journal.resolve_causal_sources(
                campaign_id=schedule.campaign_id,
                sample_id=sample_id,
                source_receipt_sha256s=tuple(sorted(source_hashes)),
            )
            market_clock_matches = (
                len(markets) == 1 and markets[0].source_as_of_ts_ms < window.market_as_of_ts_ms
            )
        else:
            market_clock_matches = (
                len(markets) == 1 and markets[0].source_as_of_ts_ms == window.market_as_of_ts_ms
            )
        if (
            len(markets) != 1
            or not market_clock_matches
            or (
                window.market_snapshot_sha256 is not None
                and markets[0].content_sha256 != window.market_snapshot_sha256
            )
        ):
            raise ResearchEvidenceError("research sample requires its exact stored market snapshot")
        evidence = tuple(
            EvidenceReferenceV1(
                kind=source.source_kind,
                reference=source.reference,
                content_sha256=source.content_sha256,
                observed_at_ms=source.observed_at_ts_ms,
            )
            for source in sources
        )
        context = LLMProposalContext(
            campaign_id=schedule.campaign_id,
            arm_id="llm-proposal-research",
            sample_id=sample_id,
            symbol=window.symbol,
            timeframe=window.timeframe,
            market_as_of_ts_ms=window.market_as_of_ts_ms,
            expires_at_ts_ms=window.sample_deadline_ts_ms,
            market_snapshot_sha256=markets[0].content_sha256,
            prompt_sha256=prompt.prompt_sha256,
            evidence=evidence,
        )
        payload = {
            "contract_version": "research-source-input.v1",
            "campaign_id": context.campaign_id,
            "sample_id": sample_id,
            "symbol": context.symbol,
            "timeframe": context.timeframe,
            "market_as_of_ts_ms": context.market_as_of_ts_ms,
            "sources": [
                {
                    "receipt_sha256": source.receipt_sha256,
                    "source_kind": source.source_kind,
                    "source_name": source.source_name,
                    "evidence_id": proposal_evidence_id(reference),
                    "evidence": reference.model_dump(mode="json"),
                    "content": source.content,
                }
                for source, reference in zip(sources, evidence, strict=True)
            ],
        }
        user = (
            prompt.user_prefix
            + "\n"
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        )
        attempt_id = (
            "kairos-research:"
            + arm.provider
            + ":"
            + canonical_sha256(
                {
                    "campaign_id": context.campaign_id,
                    "sample_id": sample_id,
                    "arm_id": context.arm_id,
                    "schedule_digest": schedule.schedule_digest,
                    "protocol_digest": protocol.protocol_digest,
                    "source_receipt_sha256s": sorted(source_hashes),
                    "prompt_sha256": prompt.prompt_sha256,
                }
            )
        )
        return context, user, attempt_id

    async def _finish(self, terminal: ResearchLLMAttemptTerminalV1) -> None:
        try:
            await self.journal.finish_attempt(terminal)
        except Exception:
            raise ResearchEvidenceError(
                "research terminal persistence ambiguous; reconcile without retry"
            ) from None

    @staticmethod
    def _unresolved(start: ResearchLLMAttemptStartV1, observed_at: int) -> ResearchLLMAttemptTerminalV1:
        return ResearchLLMAttemptTerminalV1(
            attempt_id=start.attempt_id,
            start_receipt_sha256=start.receipt_sha256,
            terminal_status="UNRESOLVED",
            observed_at_ts_ms=observed_at,
        )

    def _failure_terminal(self, start, observed, exc):
        observed_at = observed.observed_at_ts_ms if observed.observed_at_ts_ms is not None else self.clock()
        failure_class = _failure_class(exc) if observed.response is None else None
        if failure_class is None:
            return self._unresolved(start, observed_at)
        failure = LLMCallFailureV1(
            campaign_id=start.campaign_id,
            arm_id=start.arm_id,
            sample_id=start.sample_id,
            symbol=start.symbol,
            timeframe=start.timeframe,
            market_as_of_ts_ms=start.market_as_of_ts_ms,
            market_snapshot_sha256=start.market_snapshot_sha256,
            sample_deadline_ts_ms=start.sample_deadline_ts_ms,
            attempt_id=start.attempt_id,
            provider=start.provider,
            requested_model=start.requested_model,
            prompt_sha256=start.prompt_sha256,
            budget_reservation_id=start.budget_reservation_id,
            attempt_started_at_ts_ms=start.attempt_started_at_ts_ms,
            failure_observed_at_ts_ms=observed_at,
            failure_class=failure_class,
        )
        return ResearchLLMAttemptTerminalV1(
            attempt_id=start.attempt_id,
            start_receipt_sha256=start.receipt_sha256,
            terminal_status="FAILED",
            observed_at_ts_ms=observed_at,
            failure=failure,
        )


def _proposal_arm(protocol: AdaptiveCandidateProtocolV1) -> LLMProposalAdaptiveCandidateArmV1:
    arm = next((item for item in protocol.arms if item.arm_id == "llm-proposal-research"), None)
    if not isinstance(arm, LLMProposalAdaptiveCandidateArmV1):
        raise ResearchEvidenceError("research protocol has no frozen proposal arm")
    return arm


def _failure_class(exc: BaseException) -> str | None:
    # No exception message, response body, prompt, or secret is persisted.
    if isinstance(exc, asyncio.CancelledError):
        return "CANCELLED"
    if isinstance(exc, (LLMTimeout, TimeoutError)):
        return "TIMEOUT"
    if isinstance(exc, APIConnectionError) or isinstance(exc.__cause__, APIConnectionError):
        return "TRANSPORT_ERROR"
    if isinstance(exc, APIStatusError):
        return "PROVIDER_RATE_LIMITED" if exc.status_code == 429 else "PROVIDER_ERROR"
    if isinstance(exc, LLMServerError):
        return "PROVIDER_ERROR"
    # LLMBadOutput without the original response digest, budget/DB faults and
    # unknown exceptions cannot honestly satisfy the immutable failure schema.
    return None
