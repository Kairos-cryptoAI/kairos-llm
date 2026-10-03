"""Offline SIM journal coordination: no live provider, database, or trade calls."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from kairos_core import (
    AdaptiveCandidateProtocolV1,
    LLMProposalAdaptiveCandidateArmV1,
    ResearchObservationScheduleV1,
    ResearchObservationWindowV1,
    StrategyOnlyAdaptiveCandidateArmV1,
    StrategyReviewAdaptiveCandidateArmV1,
    canonical_sha256,
)
from kairos_persistence.research_evidence import (
    ResearchSourceReceiptV1,
    ResearchStrategyEvaluationReceiptV1,
)

from kairos_llm.budget import BudgetedLLMGateway
from kairos_llm.errors import LLMBadOutput, LLMBudgetError, LLMTimeout
from kairos_llm.models import LLMWorkload, ModelRouter
from kairos_llm.pricing import PriceTable
from kairos_llm.proposals import LLMProposalOutputV1
from kairos_llm.research import (
    RESEARCH_INPUT_FEATURE_SHA256,
    ResearchEvidenceError,
    ResearchPromptArtifactV1,
    ResearchProposalCoordinator,
)
from kairos_llm.schemas import LLMResult, TokenUsage

MARKET = 1_790_064_000_000


class _Budget:
    def __init__(self, events):
        self.events, self.reservations, self.commits = events, [], []
        self.error = None

    async def reserve(self, **kwargs):
        self.events.append("reserve")
        if self.error:
            raise self.error
        self.reservations.append(kwargs)

    async def commit(self, **kwargs):
        self.events.append("commit")
        if self.error:
            raise self.error
        self.commits.append(kwargs)


class _Journal:
    def __init__(self, events, sources, evaluation, protocol):
        self.events = events
        self.sources = {item.receipt_sha256: item for item in sources}
        self.evaluation = evaluation
        self.protocol = protocol
        self.attempts, self.samples = {}, []
        self.start_error, self.finish_error = None, None
        self.hide_existing_once = False

    async def load_source(self, digest):
        self.events.append("load-source")
        return self.sources[digest]

    async def load_evaluation(self, digest):
        self.events.append("load-evaluation")
        if digest != self.evaluation.receipt_sha256:
            raise ResearchEvidenceError("unknown evaluation")
        return ResearchStrategyEvaluationReceiptV1.model_validate(self.evaluation.model_dump(mode="json"))

    async def find_attempt(self, attempt_id):
        if self.hide_existing_once:
            self.hide_existing_once = False
            return None
        return self.attempts.get(attempt_id)

    async def start_attempt(self, start):
        self.events.append("start")
        if self.start_error:
            raise self.start_error
        if start.attempt_id in self.attempts:
            return False
        if any(
            (item.campaign_id, item.arm_id, item.sample_id)
            == (start.campaign_id, start.arm_id, start.sample_id)
            for item, _ in self.attempts.values()
        ):
            raise ResearchEvidenceError("sample already admitted under another identity")
        self.attempts[start.attempt_id] = (start, None)
        return True

    async def finish_attempt(self, terminal):
        self.events.append("finish")
        if self.finish_error:
            raise self.finish_error
        start, old = self.attempts[terminal.attempt_id]
        if old is not None:
            assert old == terminal
            return False
        self.attempts[terminal.attempt_id] = (start, terminal)
        return True

    async def record_verified_sample(self, sample):
        if sample.arm_protocol_digest != self.protocol.arm_digest(sample.arm_id):
            raise ResearchEvidenceError("sample differs from the strict journal's frozen arm")
        if sample.sample_record_id != canonical_sha256(sample.identity_payload()):
            raise ResearchEvidenceError("sample identity was not rederived after arm binding")
        self.events.append("record-verified")
        self.samples.append(sample)
        return True


class _Gateway:
    def __init__(self, events):
        self.events, self.calls = events, []
        self.settings = SimpleNamespace(max_retries=0, max_output_tokens=2_048)
        self.router = ModelRouter()
        self.error = None
        self.bad_provenance = False

    async def complete(self, **kwargs):
        self.events.append("provider")
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        payload = json.loads(kwargs["user"].split("\n", 1)[1])
        route = self.router.resolve(workload=kwargs["workload"])
        content = json.dumps(
            {
                "contract_version": "kairos-llm-proposal-output.v1",
                "action": "LONG_BIAS",
                "rationale": "Saved causal market evidence supports a hypothesis, not an order.",
                "evidence_ids": [payload["sources"][0]["evidence_id"]],
            }
        )
        usage = TokenUsage(input_tokens=150, output_tokens=30)
        return LLMResult(
            content=content,
            parsed=LLMProposalOutputV1.model_validate_json(content),
            model=route.choice.model,
            effort=route.effort.value,
            usage=usage,
            cost_usd=PriceTable().cost(route.choice.model, usage),
            latency_s=0.1,
            workload=kwargs["workload"].value,
            provider=route.choice.provider.value,
            request_id="req-test",
            resolved_model="other-model" if self.bad_provenance else route.choice.model,
        )


@pytest.fixture
def setup():
    events = []
    prompt = ResearchPromptArtifactV1(
        system="Use supplied evidence only. Return strict JSON.",
        user_prefix="Data:",
        workload=LLMWorkload.AGGREGATOR_NORMAL,
        reasoning_effort="medium",
        provider_effort="medium",
        max_output_tokens=2_048,
    )
    snapshot = {"close": 62_000, "symbol": "BTCUSDT", "timeframe": "1m"}
    schedule = ResearchObservationScheduleV1(
        campaign_id="new-adaptive-test",
        strategy_id="baseline",
        strategy_revision="v1",
        source_set_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        windows=(
            ResearchObservationWindowV1(
                sample_id="sample-1",
                symbol="BTCUSDT",
                timeframe="1m",
                market_as_of_ts_ms=MARKET,
                market_snapshot_sha256=canonical_sha256(snapshot),
                paired_at_ts_ms=MARKET + 500,
                sample_deadline_ts_ms=MARKET + 1_000,
            ),
        ),
    )
    common = {
        "candidate_revision": "v1",
        "artifact_sha256": "c" * 64,
        "input_feature_sha256": RESEARCH_INPUT_FEATURE_SHA256,
        "decision_mapping_sha256": "d" * 64,
        "hypothetical_exit_sha256": "e" * 64,
        "cost_model_sha256": "f" * 64,
    }
    llm = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "prompt_sha256": prompt.prompt_sha256,
        "schema_sha256": canonical_sha256(LLMProposalOutputV1.model_json_schema()),
    }
    protocol = AdaptiveCandidateProtocolV1(
        campaign_id=schedule.campaign_id,
        schedule_digest=schedule.schedule_digest,
        arms=(
            StrategyOnlyAdaptiveCandidateArmV1(candidate_id="baseline", **common),
            StrategyReviewAdaptiveCandidateArmV1(candidate_id="review", **common, **llm),
            LLMProposalAdaptiveCandidateArmV1(candidate_id="proposal", **common, **llm),
        ),
    )
    source = ResearchSourceReceiptV1(
        campaign_id=schedule.campaign_id,
        sample_id="sample-1",
        schedule_digest=schedule.schedule_digest,
        candidate_protocol_digest=protocol.protocol_digest,
        source_kind="MARKET_SNAPSHOT",
        source_name="saved-bars",
        reference="BTCUSDT:1m:sample-1",
        source_as_of_ts_ms=MARKET,
        observed_at_ts_ms=MARKET,
        content=snapshot,
    )
    evaluation = ResearchStrategyEvaluationReceiptV1(
        campaign_id=schedule.campaign_id,
        sample_id="sample-1",
        schedule_digest=schedule.schedule_digest,
        candidate_protocol_digest=protocol.protocol_digest,
        strategy_id=schedule.strategy_id,
        strategy_revision=schedule.strategy_revision,
        symbol="BTCUSDT",
        timeframe="1m",
        evidence_as_of_ts_ms=MARKET,
        evaluated_at_ts_ms=MARKET + 10,
        market_snapshot_sha256=source.content_sha256,
        evaluator_sha256=schedule.evaluator_sha256,
        source_receipt_sha256s=(source.receipt_sha256,),
    )
    journal = _Journal(events, (source,), evaluation, protocol)
    budget, underlying = _Budget(events), _Gateway(events)
    times = iter((MARKET + 100, MARKET + 200))
    coordinator = ResearchProposalCoordinator(
        BudgetedLLMGateway(underlying, budget), journal, clock=lambda: next(times)
    )
    request = {
        "schedule": schedule,
        "protocol": protocol,
        "sample_id": "sample-1",
        "prompt": prompt,
        "source_receipt_sha256s": (source.receipt_sha256,),
        "workload": LLMWorkload.AGGREGATOR_NORMAL,
    }
    return SimpleNamespace(
        coordinator=coordinator,
        request=request,
        journal=journal,
        source=source,
        budget=budget,
        underlying=underlying,
        events=events,
    )


async def test_one_durable_call_and_terminal_replay_without_another_reservation(setup):
    result = await setup.coordinator.observe(**setup.request)
    assert result.terminal.terminal_status == "COMPLETED"
    assert result.start.attempt_id == setup.budget.reservations[0]["reservation_id"]
    assert setup.budget.commits[0]["reservation_id"] == result.start.attempt_id
    assert result.start.attempt_started_at_ts_ms == MARKET + 100
    assert result.terminal.completion.response_observed_at_ts_ms == MARKET + 200
    assert setup.events == ["load-source", "reserve", "start", "provider", "commit", "finish"]
    second = await setup.coordinator.observe(**setup.request)
    assert second == result
    assert len(setup.budget.reservations) == len(setup.underlying.calls) == 1
    sample = await setup.coordinator.replay_sample(
        **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
    )
    assert sample.strategy_outcome == "NO_INTENT"
    assert sample.llm_outcome == "LONG_BIAS"
    assert sample.llm_proposal_id == result.terminal.proposal.proposal_id
    assert setup.journal.samples == [sample]
    assert len(setup.underlying.calls) == 1


@pytest.mark.parametrize("invalid_digest", [None, "0" * 64])
async def test_verified_replay_binds_frozen_arm_and_rederives_identity_before_strict_journal(
    setup, invalid_digest
):
    observation = await setup.coordinator.observe(**setup.request)
    sample = await setup.coordinator.replay_sample(
        **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
    )
    assert sample.arm_protocol_digest == setup.request["protocol"].arm_digest("llm-proposal-research")
    assert sample.sample_record_id == canonical_sha256(sample.identity_payload())
    assert sample.llm_proposal_id == observation.terminal.proposal.proposal_id
    assert await setup.coordinator.observe(**setup.request) == observation
    assert (
        await setup.coordinator.replay_sample(
            **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
        )
        == sample
    )
    assert len(setup.budget.reservations) == len(setup.budget.commits) == len(setup.underlying.calls) == 1
    invalid = type(sample).model_validate(
        {**sample.to_payload(), "sample_record_id": None, "arm_protocol_digest": invalid_digest}
    )
    with pytest.raises(ResearchEvidenceError, match="frozen arm"):
        await setup.journal.record_verified_sample(invalid)
    assert setup.journal.samples == [sample, sample]


@pytest.mark.parametrize(
    ("error", "status", "failure_class"),
    [
        (LLMTimeout("never persist this secret"), "FAILED", "TIMEOUT"),
        (LLMBadOutput("raw response unavailable"), "UNRESOLVED", None),
        (RuntimeError("unknown fault"), "UNRESOLVED", None),
    ],
)
async def test_failed_or_ambiguous_observation_retains_reservation_without_retry(
    setup, error, status, failure_class
):
    setup.underlying.error = error
    result = await setup.coordinator.observe(**setup.request)
    assert result.terminal.terminal_status == status
    assert (result.terminal.failure.failure_class if result.terminal.failure else None) == failure_class
    assert setup.budget.commits == []
    assert str(error) not in result.terminal.model_dump_json()
    assert await setup.coordinator.observe(**setup.request) == result
    assert len(setup.underlying.calls) == len(setup.budget.reservations) == 1
    if status == "UNRESOLVED":
        with pytest.raises(ResearchEvidenceError, match="unresolved"):
            await setup.coordinator.replay_sample(
                **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
            )
    else:
        sample = await setup.coordinator.replay_sample(
            **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
        )
        assert sample.llm_outcome == "CALL_FAILED"


async def test_cancel_keeps_observed_receipt_and_original_cancellation(setup):
    setup.underlying.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await setup.coordinator.observe(**setup.request)
    start, terminal = next(iter(setup.journal.attempts.values()))
    assert terminal.failure.failure_class == "CANCELLED"
    assert terminal.failure.failure_observed_at_ts_ms == MARKET + 200
    assert setup.budget.commits == []
    assert (await setup.coordinator.observe(**setup.request)).start == start
    assert len(setup.underlying.calls) == 1


async def test_terminal_write_failure_leaves_unresolved_start_and_never_reissues(setup):
    setup.journal.finish_error = RuntimeError("private db detail")
    with pytest.raises(ResearchEvidenceError, match="ambiguous"):
        await setup.coordinator.observe(**setup.request)
    result = await setup.coordinator.observe(**setup.request)
    assert result.unresolved and result.terminal is None
    assert len(setup.underlying.calls) == len(setup.budget.reservations) == 1
    with pytest.raises(ResearchEvidenceError, match="unresolved"):
        await setup.coordinator.replay_sample(
            **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
        )


async def test_cancellation_not_replaced_by_ambiguous_terminal_write(setup):
    setup.underlying.error = asyncio.CancelledError()
    setup.journal.finish_error = RuntimeError("db unavailable")
    with pytest.raises(asyncio.CancelledError):
        await setup.coordinator.observe(**setup.request)
    assert next(iter(setup.journal.attempts.values()))[1] is None
    assert setup.budget.commits == []


@pytest.mark.parametrize("where", ["budget", "start"])
async def test_admission_failure_never_dispatches_or_fabricates_model_outcome(setup, where):
    if where == "budget":
        setup.budget.error = LLMBudgetError("cap")
    else:
        setup.journal.start_error = RuntimeError("not enrolled")
    with pytest.raises(ResearchEvidenceError, match="no provider dispatch"):
        await setup.coordinator.observe(**setup.request)
    assert setup.underlying.calls == []
    assert setup.journal.attempts == {}


async def test_exact_duplicate_admission_race_replays_and_does_not_dispatch(setup):
    result = await setup.coordinator.observe(**setup.request)
    setup.journal.hide_existing_once = True
    setup.coordinator.clock = lambda: MARKET + 100
    assert await setup.coordinator.observe(**setup.request) == result
    assert len(setup.underlying.calls) == 1


async def test_bad_resolved_model_never_becomes_completed_proposal(setup):
    setup.underlying.bad_provenance = True
    result = await setup.coordinator.observe(**setup.request)
    assert result.terminal.terminal_status == "UNRESOLVED"
    assert result.terminal.proposal is None
    assert len(setup.underlying.calls) == 1


async def test_expired_window_blocks_before_budget_or_provider(setup):
    setup.coordinator.clock = lambda: MARKET + 1_000
    with pytest.raises(ResearchEvidenceError, match="admission"):
        await setup.coordinator.observe(**setup.request)
    assert setup.budget.reservations == setup.underlying.calls == []


@pytest.mark.parametrize("fault", ["late-source", "mutated-content", "prompt", "route", "unknown-source"])
async def test_real_source_or_frozen_configuration_conflict_blocks_before_spending(setup, fault):
    if fault == "late-source":
        late = setup.source.model_copy(update={"observed_at_ts_ms": MARKET + 1, "receipt_sha256": None})
        late = ResearchSourceReceiptV1.model_validate(late.model_dump(mode="json"))
        setup.journal.sources[late.receipt_sha256] = late
        setup.request["source_receipt_sha256s"] = (late.receipt_sha256,)
    elif fault == "mutated-content":
        setup.source.content["close"] = 1
    elif fault == "prompt":
        setup.request["prompt"] = setup.request["prompt"].model_copy(
            update={"system": "different instructions"}
        )
    elif fault == "route":
        setup.request["workload"] = LLMWorkload.MACRO_STRATEGIST
    else:
        setup.request["source_receipt_sha256s"] = ("0" * 64,)
    with pytest.raises((ResearchEvidenceError, ValueError, KeyError)):
        await setup.coordinator.observe(**setup.request)
    assert setup.budget.reservations == setup.underlying.calls == []


async def test_late_failure_is_retained_but_cannot_backdate_pair(setup):
    setup.underlying.error = LLMTimeout("late")
    times = iter((MARKET + 100, MARKET + 1_001))
    setup.coordinator.clock = lambda: next(times)
    result = await setup.coordinator.observe(**setup.request)
    assert result.terminal.failure.is_late
    with pytest.raises(ValueError):
        await setup.coordinator.replay_sample(
            **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
        )
    assert setup.journal.samples == []


async def test_replay_resolves_saved_evaluation_without_model_call_and_rejects_unknown_hash(setup):
    with pytest.raises(ResearchEvidenceError, match="unknown evaluation"):
        await setup.coordinator.replay_sample(**setup.request, evaluation_receipt_sha256="0" * 64)
    sample = await setup.coordinator.replay_sample(
        **setup.request, evaluation_receipt_sha256=setup.journal.evaluation.receipt_sha256
    )
    assert sample.llm_outcome == "NOT_CALLED"
    assert setup.budget.reservations == setup.underlying.calls == []


async def test_valid_source_returned_under_a_different_lookup_hash_cannot_be_spent(setup):
    setup.journal.sources["0" * 64] = setup.source
    setup.request["source_receipt_sha256s"] = ("0" * 64,)
    with pytest.raises(ResearchEvidenceError, match="stored source"):
        await setup.coordinator.observe(**setup.request)
    assert setup.budget.reservations == setup.underlying.calls == []


async def test_valid_evaluation_returned_under_a_different_lookup_hash_cannot_be_paired(setup, monkeypatch):
    async def misbound_lookup(digest):
        return setup.journal.evaluation

    monkeypatch.setattr(setup.journal, "load_evaluation", misbound_lookup)
    with pytest.raises(ResearchEvidenceError, match="stored evaluation"):
        await setup.coordinator.replay_sample(**setup.request, evaluation_receipt_sha256="0" * 64)
    assert setup.journal.samples == setup.budget.reservations == setup.underlying.calls == []
