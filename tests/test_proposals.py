from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from kairos_core import (
    CandidateReviewV1,
    EvidenceReferenceV1,
    LLMProposalAction,
    LLMTradeProposalV1,
    RiskTradeDecisionV1,
    StrategyIntentV1,
)
from pydantic import ValidationError

from kairos_llm import (
    LLMBadOutput,
    LLMProposalContext,
    LLMProposalOutputV1,
    LLMResult,
    TokenUsage,
    build_llm_trade_proposal,
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
