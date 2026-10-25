import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from kairos_llm.budget import BudgetedLLMGateway
from kairos_llm.call_evidence import (
    CreateOnlyDirectorySink,
    EvidenceConflictError,
    build_call_completion_record,
)
from kairos_llm.models import DEFAULT_WORKLOAD_ROUTES, LLMWorkload, ModelRouter
from kairos_llm.pricing import CostAccountant, PriceTable
from kairos_llm.schemas import LLMResult, TokenUsage


class _Output(BaseModel):
    ok: bool


class _Budget:
    def __init__(self):
        self.reservations = []
        self.commits = []

    async def reserve(self, **values):
        self.reservations.append(values)

    async def commit(self, **values):
        self.commits.append(values)


class _Gateway:
    def __init__(self, *, cached=False):
        self.settings = SimpleNamespace(max_retries=0, max_output_tokens=8_192)
        self.router = ModelRouter()
        self.accountant = CostAccountant()
        self.cached = cached
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        route = DEFAULT_WORKLOAD_ROUTES[kwargs["workload"]]
        usage = TokenUsage(input_tokens=1_500, output_tokens=300)
        return LLMResult(
            content='{"ok":true}',
            parsed={"ok": True},
            model=route.choice.model,
            effort=route.effort.value,
            usage=usage,
            cost_usd=PriceTable().cost(route.choice.model, usage),
            workload=kwargs["workload"].value,
            provider=route.choice.provider.value,
            request_id="provider-request-123",
            resolved_model="resolved-model",
            cached=self.cached,
        )


class _CollectSink:
    def __init__(self, budget):
        self.budget = budget
        self.records = {}

    def create_only(self, call_id, phase, record):
        assert self.budget.reservations  # Attempt evidence follows a durable hold.
        if phase == "completed":
            assert self.budget.commits  # Completion follows the durable spend outcome.
        key = (call_id, phase)
        if key in self.records:
            if self.records[key] != record:
                raise EvidenceConflictError("conflict")
            return
        self.records[key] = record


@pytest.mark.asyncio
async def test_gateway_records_exact_application_call_after_committed_budget():
    budget = _Budget()
    sink = _CollectSink(budget)
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, budget, evidence_sink=sink)
    system = "system exact Ω"
    user = 'user exact {"x": 1}'

    result = await gateway.complete(
        system=system, user=user, workload=LLMWorkload.TEXT_SCOUTS, schema=_Output
    )

    assert len(sink.records) == 2
    (call_id, phase), raw = next((key, value) for key, value in sink.records.items() if key[1] == "started")
    assert len(call_id) == 32
    assert phase == "started"
    attempt = json.loads(raw)
    assert attempt["capture_scope"] == "application-boundary-not-provider-http"
    assert attempt["phase"] == "ATTEMPT_STARTED"
    assert attempt["provider_call"]["attempt_started_at_ms"] == result.attempt_started_at_ts_ms
    assert base64.b64decode(attempt["request"]["system_utf8"]["base64"]) == system.encode()
    assert base64.b64decode(attempt["request"]["user_utf8"]["base64"]) == user.encode()
    schema_bytes = json.dumps(
        _Output.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert base64.b64decode(attempt["request"]["schema_canonical_json_utf8"]["base64"]) == schema_bytes
    _completion_key, completion_raw = next(
        (key, value) for key, value in sink.records.items() if key[1] == "completed"
    )
    completion = json.loads(completion_raw)
    assert completion["phase"] == "COMPLETED"
    assert completion["attempt_record_sha256"] == hashlib.sha256(raw).hexdigest()
    assert completion["provider_call"] == {
        "attempt_started_at_ms": result.attempt_started_at_ts_ms,
        "cached": False,
        "cache_scope": "PROVIDER_INPUT_TOKENS_NOT_RESPONSE_REUSE",
        "model": result.model,
        "provider": result.provider,
        "request_id": "provider-request-123",
        "resolved_model": "resolved-model",
        "response_observed_at_ms": result.response_observed_at_ts_ms,
    }
    assert base64.b64decode(completion["response"]["content_utf8"]["base64"]) == b'{"ok":true}'
    assert completion["budget"] == {
        "actual_microusd": budget.commits[0]["actual_microusd"],
        "commit_observed_at_ms": completion["budget"]["commit_observed_at_ms"],
        "reservation_id": budget.reservations[0]["reservation_id"],
        "reserved_microusd": budget.reservations[0]["reserved_microusd"],
        "status": "COMMITTED",
    }
    assert result.budget_reservation_id == completion["budget"]["reservation_id"]
    assert not {"authorization", "api_key", "headers", "environment"}.intersection(attempt)
    assert not {"authorization", "api_key", "headers", "environment"}.intersection(completion)


@pytest.mark.asyncio
async def test_sink_failure_fails_call_without_undoing_committed_cost():
    class BrokenSink:
        def create_only(self, _call_id, phase, _record):
            assert phase == "started"
            raise OSError("disk unavailable")

    budget = _Budget()
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, budget, evidence_sink=BrokenSink())
    with pytest.raises(OSError, match="disk unavailable"):
        await gateway.complete(system="system", user="prompt", workload=LLMWorkload.TEXT_SCOUTS)
    assert len(budget.reservations) == 1
    assert len(budget.commits) == 0
    assert underlying.calls == []


@pytest.mark.asyncio
async def test_ambiguous_provider_failure_retains_started_record_and_open_hold():
    class FailingGateway(_Gateway):
        async def complete(self, **_kwargs):
            raise TimeoutError("provider completion unknown")

    budget = _Budget()
    sink = _CollectSink(budget)
    gateway = BudgetedLLMGateway(FailingGateway(), budget, evidence_sink=sink)
    with pytest.raises(TimeoutError, match="unknown"):
        await gateway.complete(system="system", user="prompt", workload=LLMWorkload.TEXT_SCOUTS)
    assert len(budget.reservations) == 1
    assert budget.commits == []
    assert len(sink.records) == 1
    attempt = json.loads(next(iter(sink.records.values())))
    assert attempt["phase"] == "ATTEMPT_STARTED"
    assert attempt["budget"]["status"] == "RESERVED"


@pytest.mark.asyncio
async def test_async_sink_cannot_skip_durable_attempt_acknowledgement():
    class AsyncSink:
        async def create_only(self, _call_id, _phase, _record):
            raise AssertionError("this coroutine must not be mistaken for persisted evidence")

    budget = _Budget()
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(underlying, budget, evidence_sink=AsyncSink())
    with pytest.raises(ValueError, match="synchronously acknowledge"):
        await gateway.complete(system="system", user="prompt", workload=LLMWorkload.TEXT_SCOUTS)
    assert budget.reservations and not budget.commits
    assert underlying.calls == []


@pytest.mark.asyncio
async def test_provider_input_cache_flag_does_not_discard_a_successful_paid_reply():
    budget = _Budget()
    sink = _CollectSink(budget)
    gateway = BudgetedLLMGateway(_Gateway(cached=True), budget, evidence_sink=sink)
    result = await gateway.complete(system="system", user="prompt", workload=LLMWorkload.TEXT_SCOUTS)
    assert budget.commits
    assert len(sink.records) == 2
    completion = json.loads(next(raw for (_id, phase), raw in sink.records.items() if phase == "completed"))
    assert completion["provider_call"]["cached"] is True
    assert completion["provider_call"]["cache_scope"] == "PROVIDER_INPUT_TOKENS_NOT_RESPONSE_REUSE"
    assert result.cached is True


@pytest.mark.parametrize("changed_field", ["call_id", "provider", "model", "reservation", "amount", "clock"])
@pytest.mark.asyncio
async def test_completion_refuses_cross_call_or_rebound_attempt(changed_field):
    budget = _Budget()
    sink = _CollectSink(budget)
    gateway = BudgetedLLMGateway(_Gateway(), budget, evidence_sink=sink)
    result = await gateway.complete(system="system", user="prompt", workload=LLMWorkload.TEXT_SCOUTS)
    (call_id, _phase), raw = next((key, value) for key, value in sink.records.items() if key[1] == "started")
    attempt = json.loads(raw)
    if changed_field == "call_id":
        attempt["call_id"] = "f" * 32
    elif changed_field in {"provider", "model"}:
        attempt["provider_call"][changed_field] = "different-value"
    elif changed_field == "reservation":
        attempt["budget"]["reservation_id"] = "other-hold"
    elif changed_field == "amount":
        attempt["budget"]["reserved_microusd"] += 1
    else:
        attempt["provider_call"]["attempt_started_at_ms"] += 1
    with pytest.raises(ValueError, match="exact original attempt"):
        build_call_completion_record(
            call_id=call_id,
            attempt_record=json.dumps(attempt).encode(),
            result=result,
            reserved_microusd=budget.reservations[0]["reserved_microusd"],
            actual_microusd=budget.commits[0]["actual_microusd"],
            budget_commit_observed_at_ms=result.response_observed_at_ts_ms + 1,
        )


def test_directory_sink_is_create_only_and_exact_replay_does_not_refresh_bytes(tmp_path: Path):
    sink = CreateOnlyDirectorySink(tmp_path / "evidence")
    call_id = "a" * 32
    body = b'{"fixed":1}'
    sink.create_only(call_id, "started", body)
    evidence_path = tmp_path / "evidence" / f"{call_id}.started.json"
    original_mtime = evidence_path.stat().st_mtime_ns
    sink.create_only(call_id, "started", body)
    assert evidence_path.read_bytes() == body
    assert evidence_path.stat().st_mtime_ns == original_mtime
    with pytest.raises(EvidenceConflictError, match="conflicts"):
        sink.create_only(call_id, "started", b'{"fixed":2}')
    assert evidence_path.read_bytes() == body
    if os.name != "nt":
        assert (tmp_path / "evidence").stat().st_mode & 0o777 == 0o700
        assert evidence_path.stat().st_mode & 0o777 == 0o600


def test_directory_sink_requires_a_new_capture_directory(tmp_path: Path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        CreateOnlyDirectorySink(target)


def test_directory_sink_detects_replacement_before_writing(tmp_path: Path):
    target = tmp_path / "evidence"
    sink = CreateOnlyDirectorySink(target)
    target.rename(tmp_path / "original")
    target.mkdir()
    with pytest.raises(ValueError, match="identity changed"):
        sink.create_only("b" * 32, "started", b"record")


@pytest.mark.asyncio
async def test_input_size_is_rejected_before_reserving_or_calling_provider(tmp_path: Path):
    budget = _Budget()
    underlying = _Gateway()
    gateway = BudgetedLLMGateway(
        underlying, budget, evidence_sink=CreateOnlyDirectorySink(tmp_path / "evidence")
    )
    with pytest.raises(ValueError, match="bounded preflight size"):
        await gateway.complete(system="x" * 2_900_001, user="", workload=LLMWorkload.TEXT_SCOUTS)
    assert budget.reservations == []
    assert underlying.calls == []
