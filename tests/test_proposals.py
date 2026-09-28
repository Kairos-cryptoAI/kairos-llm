from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from kairos_core import (
    AdaptiveCandidateProtocolV1,
    CandidateReviewV1,
    EvidenceReferenceV1,
    LLMProposalAction,
    LLMProposalAdaptiveCandidateArmV1,
    LLMTradeProposalV1,
    ResearchObservationScheduleV1,
    ResearchObservationWindowV1,
    RiskTradeDecisionV1,
    StrategyIntentV1,
    StrategyOnlyAdaptiveCandidateArmV1,
    StrategyReviewAdaptiveCandidateArmV1,
    canonical_sha256,
)
from kairos_core.topics import Topics
from pydantic import ValidationError

from kairos_llm import (
    LLMBadOutput,
    LLMProposalContext,
    LLMProposalOutputV1,
    LLMResult,
    TokenUsage,
    build_and_publish_llm_trade_proposal,
    build_llm_proposal_completion_receipt,
    build_llm_trade_proposal,
    build_preregistered_adaptive_llm_proposal,
    proposal_evidence_id,
)


def _evidence() -> EvidenceReferenceV1:
    return EvidenceReferenceV1(
        kind="closed_bar",
        reference="BTCUSDT:1m:2026-09-22T12:00:00Z",
        content_sha256="a" * 64,
        observed_at_ms=1_790_064_000_000,
    )


def _context(evidence: tuple[EvidenceReferenceV1, ...] | None = None) -> LLMProposalContext:
    return LLMProposalContext(
        campaign_id="adaptive-shadow-v1",
        arm_id="llm-proposal",
        sample_id="sample-0001",
        symbol="BTCUSDT",
        timeframe="1m",
        market_as_of_ts_ms=1_790_064_060_000,
        expires_at_ts_ms=1_790_064_120_000,
        market_snapshot_sha256="b" * 64,
        prompt_sha256="c" * 64,
        evidence=(evidence if evidence is not None else (_evidence(),)),
    )


def _result(
    *,
    action: LLMProposalAction = LLMProposalAction.LONG_BIAS,
    evidence_ids: tuple[str, ...] | None = None,
    extra: dict[str, object] | None = None,
) -> LLMResult:
    ids = evidence_ids if evidence_ids is not None else (proposal_evidence_id(_evidence()),)
    payload: dict[str, object] = {
        "contract_version": "kairos-llm-proposal-output.v1",
        "action": action.value,
        "rationale": "  Trend and volatility align.  ",
        "evidence_ids": ids,
    }
    if extra:
        payload.update(extra)
    content = json.dumps(payload, separators=(",", ":"))
    parsed = None
    try:
        parsed = LLMProposalOutputV1.model_validate_json(content)
    except ValidationError:
        # Adversarial provider outputs intentionally have no successful parsed object.
        pass
    return LLMResult(
        content=content,
        parsed=parsed,
        model="gpt-5.6-luna",
        effort="medium",
        usage=TokenUsage(input_tokens=120, output_tokens=24),
        cost_usd=0.00125,
        latency_s=0.1251,
        workload="adaptive_proposal",
        provider="openai",
        request_id="req-123",
        budget_reservation_id="reservation-456",
        resolved_model="gpt-5.6-luna-2026-08-07",
        system_fingerprint="fp-789",
    )


def _adaptive_schedule(*, evaluator_sha256: str = "1" * 64) -> ResearchObservationScheduleV1:
    context = _context()
    return ResearchObservationScheduleV1(
        campaign_id=context.campaign_id,
        strategy_id="adaptive-strategy",
        strategy_revision="candidate-set-v1",
        source_set_sha256="0" * 64,
        evaluator_sha256=evaluator_sha256,
        windows=(
            ResearchObservationWindowV1(
                sample_id=context.sample_id,
                symbol=context.symbol,
                timeframe=context.timeframe,
                market_as_of_ts_ms=context.market_as_of_ts_ms,
                market_snapshot_sha256=context.market_snapshot_sha256,
                paired_at_ts_ms=context.market_as_of_ts_ms + 100,
                sample_deadline_ts_ms=context.expires_at_ts_ms,
            ),
        ),
    )


def _adaptive_protocol(schedule: ResearchObservationScheduleV1) -> AdaptiveCandidateProtocolV1:
    common = {
        "candidate_revision": "v1",
        "artifact_sha256": "a" * 64,
        "input_feature_sha256": "b" * 64,
        "decision_mapping_sha256": "c" * 64,
        "hypothetical_exit_sha256": "d" * 64,
        "cost_model_sha256": "e" * 64,
    }
    return AdaptiveCandidateProtocolV1(
        campaign_id=schedule.campaign_id,
        schedule_digest=schedule.schedule_digest,
        arms=(
            StrategyOnlyAdaptiveCandidateArmV1(candidate_id="strategy-v1", **common),
            StrategyReviewAdaptiveCandidateArmV1(
                candidate_id="review-v1",
                **common,
                provider="openai",
                model="gpt-6-sol",
                prompt_sha256="c" * 64,
                schema_sha256="1" * 64,
            ),
            LLMProposalAdaptiveCandidateArmV1(
                candidate_id="proposal-v1",
                **common,
                provider="openai",
                model="gpt-6-sol",
                prompt_sha256="c" * 64,
                schema_sha256=canonical_sha256(LLMProposalOutputV1.model_json_schema()),
            ),
        ),
    )


class _RecordingProposalBus:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def publish(self, topic: str, message: LLMTradeProposalV1) -> str:
        self.calls.append((topic, message))
        return message.message_id


class _UnknownOutcomeProposalBus:
    def __init__(self) -> None:
        self.attempts = 0

    async def publish(self, topic: str, message: LLMTradeProposalV1) -> str:
        self.attempts += 1
        raise TimeoutError("publication outcome is unknown")


class _MismatchedProposalBus:
    def __init__(self) -> None:
        self.attempts = 0

    async def publish(self, topic: str, message: LLMTradeProposalV1) -> str:
        self.attempts += 1
        return "different-message-id"


def test_adapter_builds_advisory_contract_from_trusted_context_and_gateway_metadata():
    result = _result()
    proposal = build_llm_trade_proposal(context=_context(), result=result)

    assert type(proposal) is LLMTradeProposalV1
    assert proposal.source == "kairos-llm-proposal-adapter"
    assert proposal.action is LLMProposalAction.LONG_BIAS
    assert proposal.rationale == "Trend and volatility align."
    assert proposal.evidence == (_evidence(),)
    assert proposal.model_provenance.provider == "openai"
    assert proposal.model_provenance.requested_model == "gpt-5.6-luna"
    assert proposal.model_provenance.resolved_model == "gpt-5.6-luna-2026-08-07"
    assert proposal.model_provenance.request_id == "req-123"
    assert proposal.model_provenance.budget_reservation_id == "reservation-456"
    assert proposal.model_provenance.prompt_sha256 == "c" * 64
    assert proposal.model_provenance.response_sha256 == hashlib.sha256(result.content.encode()).hexdigest()
    assert proposal.model_provenance.latency_ms == 126
    assert proposal.model_provenance.cost_usd == pytest.approx(0.00125)
    assert proposal.proposal_id == hashlib.sha256(proposal.canonical_proposal_bytes()).hexdigest()

    # The proposal's runtime type is not accepted as an executable strategy/review/risk contract.
    for execution_type in (StrategyIntentV1, CandidateReviewV1, RiskTradeDecisionV1):
        with pytest.raises(ValidationError):
            execution_type.model_validate(proposal.model_dump(mode="python"))


def test_adaptive_adapter_binds_proposal_to_preregistered_protocol():
    context = replace(_context(), arm_id="llm-proposal-research")
    schedule = _adaptive_schedule()
    protocol = _adaptive_protocol(schedule)

    proposal = build_preregistered_adaptive_llm_proposal(
        schedule=schedule,
        protocol=protocol,
        context=context,
        result=replace(_result(), model="gpt-6-sol", resolved_model="gpt-6-sol"),
        input_feature_sha256="b" * 64,
    )

    assert proposal.campaign_id == protocol.campaign_id
    assert proposal.arm_id == "llm-proposal-research"
    assert type(proposal) is LLMTradeProposalV1
    for execution_type in (StrategyIntentV1, CandidateReviewV1, RiskTradeDecisionV1):
        with pytest.raises(ValidationError):
            execution_type.model_validate(proposal.model_dump(mode="python"))


def test_adaptive_adapter_rejects_schedule_digest_drift():
    context = replace(_context(), arm_id="llm-proposal-research")
    preregistered_schedule = _adaptive_schedule()
    protocol = _adaptive_protocol(preregistered_schedule)
    changed_schedule = _adaptive_schedule(evaluator_sha256="2" * 64)

    with pytest.raises(LLMBadOutput, match="schedule differs"):
        build_preregistered_adaptive_llm_proposal(
            schedule=changed_schedule,
            protocol=protocol,
            context=context,
            result=replace(_result(), model="gpt-6-sol", resolved_model="gpt-6-sol"),
            input_feature_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("context_changes", "result_changes", "input_feature_sha256", "protocol_changes", "error"),
    [
        ({"campaign_id": "other-campaign"}, {}, "b" * 64, {}, "campaign differs"),
        ({"arm_id": "llm-proposal"}, {}, "b" * 64, {}, "frozen llm-proposal-research arm"),
        ({"sample_id": "sample-0002"}, {}, "b" * 64, {}, "not present"),
        ({"timeframe": "5m"}, {}, "b" * 64, {}, "input window differs"),
        ({}, {"model": "gpt-6-luna"}, "b" * 64, {}, "gateway route differs"),
        ({}, {"resolved_model": "gpt-6-sol-2026-09-01"}, "b" * 64, {}, "gateway route differs"),
        ({}, {"provider": "deepseek"}, "b" * 64, {}, "gateway route differs"),
        ({"prompt_sha256": "9" * 64}, {}, "b" * 64, {}, "prompt differs"),
        ({}, {}, "8" * 64, {}, "input features differ"),
        (
            {},
            {},
            "b" * 64,
            {"schema_sha256": "7" * 64},
            "output schema differs",
        ),
    ],
)
def test_adaptive_adapter_rejects_protocol_or_request_drift(
    context_changes: dict[str, str],
    result_changes: dict[str, str],
    input_feature_sha256: str,
    protocol_changes: dict[str, str],
    error: str,
):
    context_values = {"arm_id": "llm-proposal-research", **context_changes}
    context = replace(_context(), **context_values)
    schedule = _adaptive_schedule()
    protocol = _adaptive_protocol(schedule)
    if protocol_changes:
        arms = list(protocol.arms)
        arms[2] = arms[2].model_copy(update=protocol_changes)
        protocol = AdaptiveCandidateProtocolV1(
            campaign_id=protocol.campaign_id,
            schedule_digest=protocol.schedule_digest,
            arms=tuple(arms),  # type: ignore[arg-type]
        )
    result_values = {"model": "gpt-6-sol", "resolved_model": "gpt-6-sol", **result_changes}

    with pytest.raises(LLMBadOutput, match=error):
        build_preregistered_adaptive_llm_proposal(
            schedule=schedule,
            protocol=protocol,
            context=context,
            result=replace(_result(), **result_values),
            input_feature_sha256=input_feature_sha256,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", 1),
        ("stop_price", 100),
        ("venue", "EVEDEX"),
        ("order_side", "BUY"),
        ("model_provenance", {"provider": "fake"}),
        ("campaign_id", "model-controlled"),
    ],
)
def test_model_cannot_supply_execution_fields_or_trusted_envelope(field: str, value: object):
    result = _result(extra={field: value})

    with pytest.raises(LLMBadOutput):
        build_llm_trade_proposal(context=_context(), result=result)


def test_adapter_rejects_unprovided_or_duplicate_evidence():
    unknown_id = "d" * 64
    with pytest.raises(LLMBadOutput, match="not present"):
        build_llm_trade_proposal(context=_context(), result=_result(evidence_ids=(unknown_id,)))

    evidence = _evidence()
    with pytest.raises(LLMBadOutput, match="duplicate"):
        build_llm_trade_proposal(
            context=_context((evidence, evidence)),
            result=_result(),
        )


@pytest.mark.parametrize(
    "field",
    ["provider", "model", "resolved_model", "request_id", "budget_reservation_id"],
)
def test_adapter_fails_closed_without_gateway_provenance(field: str):
    result = replace(_result(), **{field: None})

    with pytest.raises(LLMBadOutput, match="trusted"):
        build_llm_trade_proposal(context=_context(), result=result)


def test_adapter_rejects_mismatch_between_raw_response_and_gateway_parsed_payload():
    result = _result()
    result.parsed = LLMProposalOutputV1(
        contract_version="kairos-llm-proposal-output.v1",
        action=LLMProposalAction.SHORT_BIAS,
        rationale="different",
        evidence_ids=(proposal_evidence_id(_evidence()),),
    )

    with pytest.raises(LLMBadOutput, match="differs"):
        build_llm_trade_proposal(context=_context(), result=result)


def test_non_directional_defer_does_not_require_cited_evidence():
    proposal = build_llm_trade_proposal(
        context=_context(evidence=()),
        result=_result(action=LLMProposalAction.DEFER, evidence_ids=()),
    )

    assert proposal.action is LLMProposalAction.DEFER
    assert proposal.evidence == ()


def test_model_can_report_cited_move_without_claiming_a_direction():
    proposal = build_llm_trade_proposal(
        context=_context(), result=_result(action=LLMProposalAction.VOLATILITY_ALERT)
    )

    assert proposal.action is LLMProposalAction.VOLATILITY_ALERT
    assert proposal.evidence == (_evidence(),)
    assert not hasattr(proposal, "order_id")
    for execution_type in (StrategyIntentV1, CandidateReviewV1, RiskTradeDecisionV1):
        with pytest.raises(ValidationError):
            execution_type.model_validate(proposal.to_payload())

    with pytest.raises(LLMBadOutput):
        build_llm_trade_proposal(
            context=_context(evidence=()),
            result=_result(action=LLMProposalAction.VOLATILITY_ALERT, evidence_ids=()),
        )


def test_completion_receipt_requires_gateway_timestamps_and_exact_response() -> None:
    context = _context()
    raw_result = _result()
    proposal = build_llm_trade_proposal(context=context, result=raw_result)
    with pytest.raises(LLMBadOutput, match="trusted budgeted-gateway timestamps"):
        build_llm_proposal_completion_receipt(
            context=context,
            result=raw_result,
            proposal=proposal,
            sample_deadline_ts_ms=context.market_as_of_ts_ms + 10_000,
        )

    stamped = replace(
        raw_result,
        attempt_started_at_ts_ms=context.market_as_of_ts_ms + 100,
        response_observed_at_ts_ms=context.market_as_of_ts_ms + 300,
    )
    receipt = build_llm_proposal_completion_receipt(
        context=context,
        result=stamped,
        proposal=proposal,
        sample_deadline_ts_ms=context.market_as_of_ts_ms + 10_000,
    )
    assert receipt.proposal_id == proposal.proposal_id
    assert receipt.model_provenance == proposal.model_provenance
    assert receipt.response_observed_at_ts_ms == context.market_as_of_ts_ms + 300
    assert not hasattr(receipt, "order_id")

    altered = replace(stamped, request_id="different-provider-request")
    with pytest.raises(LLMBadOutput, match="differs"):
        build_llm_proposal_completion_receipt(
            context=context,
            result=altered,
            proposal=proposal,
            sample_deadline_ts_ms=context.market_as_of_ts_ms + 10_000,
        )


@pytest.mark.asyncio
async def test_publish_helper_uses_only_the_research_topic_and_returns_the_proposal():
    bus = _RecordingProposalBus()
    result = _result()

    proposal = await build_and_publish_llm_trade_proposal(
        context=_context(),
        result=result,
        bus=bus,
    )

    assert type(proposal) is LLMTradeProposalV1
    assert bus.calls == [(Topics.LLM_TRADE_PROPOSAL, proposal)]
    assert proposal.message_id == proposal.proposal_id
    assert (
        proposal.model_provenance.response_sha256
        == hashlib.sha256(result.content.encode("utf-8")).hexdigest()
    )


@pytest.mark.asyncio
async def test_invalid_provider_output_is_rejected_before_publishing():
    bus = _RecordingProposalBus()

    with pytest.raises(LLMBadOutput):
        await build_and_publish_llm_trade_proposal(
            context=_context(),
            result=_result(extra={"quantity": 1}),
            bus=bus,
        )

    assert bus.calls == []


@pytest.mark.asyncio
async def test_uncertain_publication_is_not_automatically_retried():
    bus = _UnknownOutcomeProposalBus()

    with pytest.raises(TimeoutError, match="outcome is unknown"):
        await build_and_publish_llm_trade_proposal(
            context=_context(),
            result=_result(),
            bus=bus,
        )

    assert bus.attempts == 1


@pytest.mark.asyncio
async def test_publication_requires_the_stable_proposal_message_id():
    bus = _MismatchedProposalBus()

    with pytest.raises(RuntimeError, match="reconcile before retrying"):
        await build_and_publish_llm_trade_proposal(
            context=_context(),
            result=_result(),
            bus=bus,
        )

    assert bus.attempts == 1
