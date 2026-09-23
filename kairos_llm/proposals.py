"""Strict provider output and trusted adapter for advisory LLM proposals.

No function in this module makes a model call or converts a proposal into a
strategy, risk, or execution contract. Provider output supplies only an
advisory action, short rationale, and references to caller-supplied evidence.
Scope and provenance come from trusted context and the completed budgeted
gateway result. Publication is a separate, explicit opt-in to one research
topic through an injected durable message bus.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from kairos_core import (
    EvidenceReferenceV1,
    LLMProposalAction,
    LLMProposalModelProvenanceV1,
    LLMTradeProposalV1,
    canonical_sha256,
)
from kairos_core.topics import Topics
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import LLMBadOutput
from .schemas import LLMResult

_EvidenceId = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class LLMProposalOutputV1(BaseModel):
    """Closed-world JSON returned by a model; it contains no provenance/envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["kairos-llm-proposal-output.v1"]
    action: LLMProposalAction
    rationale: StrictStr = Field(..., min_length=1, max_length=1_024)
    evidence_ids: tuple[_EvidenceId, ...] = Field(..., max_length=32)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("rationale must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_evidence(self) -> LLMProposalOutputV1:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids cannot contain duplicates")
        directional = (LLMProposalAction.LONG_BIAS, LLMProposalAction.SHORT_BIAS)
        if self.action in directional and not self.evidence_ids:
            raise ValueError("directional proposals require cited evidence_ids")
        return self


@dataclass(frozen=True, slots=True)
class LLMProposalContext:
    """Trusted campaign and market-input metadata supplied outside model JSON."""

    campaign_id: str
    arm_id: str
    sample_id: str
    symbol: str
    timeframe: str
    market_as_of_ts_ms: int
    expires_at_ts_ms: int
    market_snapshot_sha256: str
    prompt_sha256: str
    evidence: tuple[EvidenceReferenceV1, ...]


class LLMProposalMessageBus(Protocol):
    """Narrow interface for a durable, research-only proposal publisher."""

    async def publish(self, topic: str, message: LLMTradeProposalV1) -> str: ...


def proposal_evidence_id(evidence: EvidenceReferenceV1) -> str:
    """Return the stable ID the prompt can expose for one trusted evidence item."""

    return canonical_sha256(evidence)


def build_llm_trade_proposal(
    *,
    context: LLMProposalContext,
    result: LLMResult,
) -> LLMTradeProposalV1:
    """Validate an exact gateway response and attach trusted scope/provenance.

    Missing provider request identity, model resolution, or durable budget
    reservation fails closed. Unknown or unprovided evidence references are
    rejected. The only return type is the non-executable core proposal contract.
    """

    output = _validated_output(result)
    evidence_by_id: dict[str, EvidenceReferenceV1] = {}
    for evidence in context.evidence:
        if not isinstance(evidence, EvidenceReferenceV1):
            raise LLMBadOutput("proposal context contains an invalid evidence object")
        identifier = proposal_evidence_id(evidence)
        if identifier in evidence_by_id:
            raise LLMBadOutput("proposal context contains duplicate evidence objects")
        evidence_by_id[identifier] = evidence

    missing_ids = set(output.evidence_ids) - evidence_by_id.keys()
    if missing_ids:
        raise LLMBadOutput("proposal cited evidence not present in the trusted input context")
    selected_evidence = tuple(evidence_by_id[identifier] for identifier in output.evidence_ids)

    provider = _required_metadata(result.provider, "provider")
    requested_model = _required_metadata(result.model, "requested model")
    resolved_model = _required_metadata(result.resolved_model, "resolved model")
    request_id = _required_metadata(result.request_id, "provider request ID")
    reservation_id = _required_metadata(result.budget_reservation_id, "durable budget reservation")
    if not isinstance(result.content, str) or not result.content:
        raise LLMBadOutput("proposal response content is missing")
    if isinstance(result.latency_s, bool) or not math.isfinite(result.latency_s) or result.latency_s < 0:
        raise LLMBadOutput("proposal response latency is invalid")
    if isinstance(result.cost_usd, bool) or not math.isfinite(result.cost_usd) or result.cost_usd < 0:
        raise LLMBadOutput("proposal response cost is invalid")

    provenance = LLMProposalModelProvenanceV1(
        provider=provider,
        requested_model=requested_model,
        resolved_model=resolved_model,
        system_fingerprint=result.system_fingerprint,
        reasoning_effort=_optional_metadata(result.effort),
        request_id=request_id,
        prompt_sha256=context.prompt_sha256,
        response_sha256=hashlib.sha256(result.content.encode("utf-8")).hexdigest(),
        budget_reservation_id=reservation_id,
        latency_ms=math.ceil(result.latency_s * 1_000),
        cost_usd=result.cost_usd,
    )
    return LLMTradeProposalV1(
        campaign_id=context.campaign_id,
        arm_id=context.arm_id,
        sample_id=context.sample_id,
        symbol=context.symbol,
        timeframe=context.timeframe,
        market_as_of_ts_ms=context.market_as_of_ts_ms,
        expires_at_ts_ms=context.expires_at_ts_ms,
        market_snapshot_sha256=context.market_snapshot_sha256,
        action=output.action,
        rationale=output.rationale,
        evidence=selected_evidence,
        model_provenance=provenance,
    )


async def build_and_publish_llm_trade_proposal(
    *,
    context: LLMProposalContext,
    result: LLMResult,
    bus: LLMProposalMessageBus,
) -> LLMTradeProposalV1:
    """Publish one completed proposal to the isolated research topic.

    This function makes no provider call and has no strategy, risk, or execution
    conversion. It publishes exactly once and deliberately does not retry: if
    the outcome is uncertain, the caller must reconcile the stable proposal ID
    before deciding whether another attempt is safe.
    """

    proposal = build_llm_trade_proposal(context=context, result=result)
    published_message_id = await bus.publish(Topics.LLM_TRADE_PROPOSAL, proposal)
    if published_message_id != proposal.message_id:
        raise RuntimeError("proposal publisher returned a different message ID; reconcile before retrying")
    return proposal


def _validated_output(result: LLMResult) -> LLMProposalOutputV1:
    if not isinstance(result.content, str) or not result.content:
        raise LLMBadOutput("proposal response content is missing")
    try:
        output = LLMProposalOutputV1.model_validate_json(result.content)
        if isinstance(result.parsed, LLMProposalOutputV1):
            parsed = result.parsed
        elif result.parsed is not None:
            parsed = LLMProposalOutputV1.model_validate(result.parsed)
        else:
            raise LLMBadOutput("gateway did not return a validated proposal payload")
    except ValidationError as exc:
        raise LLMBadOutput("proposal response failed strict schema validation") from exc
    if output != parsed:
        raise LLMBadOutput("gateway parsed proposal differs from its exact response content")
    return output


def _required_metadata(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LLMBadOutput(f"proposal is missing trusted {label} metadata")
    return value


def _optional_metadata(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_metadata(value, "optional")
