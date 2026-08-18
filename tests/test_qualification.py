import json
from datetime import UTC, datetime

import pytest

from kairos_llm.models import DEFAULT_WORKLOAD_ROUTES, LLMWorkload, Provider
from kairos_llm.qualification import (
    LLMQualificationReport,
    ProbePayload,
    ProbeStatus,
    QuotaObservation,
    _quota_headers,
    _read_secret,
    _write_report,
    qualify_llms,
)
from kairos_llm.schemas import LLMResult, TokenUsage

NOW = datetime(2026, 8, 18, tzinfo=UTC)


def _result(workload: LLMWorkload, *, cost: float = 0.001) -> LLMResult:
    route = DEFAULT_WORKLOAD_ROUTES[workload]
    return LLMResult(
        content='{"protocol":"KAIROS_LLM_PROBE_V1","arithmetic":42,"decision":"NO_TRADE"}',
        parsed=ProbePayload(
            protocol="KAIROS_LLM_PROBE_V1",
            arithmetic=42,
            decision="NO_TRADE",
        ),
        model=route.choice.model,
        effort=route.effort.value,
        usage=TokenUsage(input_tokens=40, cached_input_tokens=10, output_tokens=20),
        cost_usd=cost,
        latency_s=0.5,
        workload=workload.value,
        resolved_model=f"{route.choice.model}-resolved",
        system_fingerprint="fingerprint",
    )


async def _quota(provider, _key):
    return QuotaObservation(
        provider=provider.value,
        status=ProbeStatus.PASS,
        http_status=200,
        latency_ms=10,
        headers={"x-ratelimit-remaining-requests": "100"},
        detail="ok",
    )


@pytest.mark.asyncio
async def test_all_workloads_pass_exact_contract_latency_usage_and_cost():
    async def runner(workload):
        return _result(workload)

    report = await qualify_llms(
        samples_per_workload=2,
        available_keys={Provider.OPENAI: "openai", Provider.DEEPSEEK: "deepseek"},
        runner=runner,
        quota_probe=_quota,
        now=NOW,
    )

    assert report.status is ProbeStatus.PASS
    assert report.live_orders_allowed is False
    assert len(report.calls) == 8
    assert all(item.availability == 1 and item.quality_rate == 1 for item in report.workloads)
    assert sum(item.estimated_cost_usd for item in report.workloads) == pytest.approx(0.008)


@pytest.mark.asyncio
async def test_missing_keys_make_no_billable_calls_and_block_all_workloads():
    called = False

    async def runner(_workload):
        nonlocal called
        called = True
        raise AssertionError("runner must not be called")

    async def quota(_provider, _key):
        raise AssertionError("quota probe must not be called")

    report = await qualify_llms(
        samples_per_workload=2,
        available_keys={Provider.OPENAI: None, Provider.DEEPSEEK: None},
        runner=runner,
        quota_probe=quota,
        now=NOW,
    )

    assert called is False
    assert report.status is ProbeStatus.BLOCKED
    assert all(item.status is ProbeStatus.BLOCKED for item in report.calls)
    assert all(item.estimated_cost_usd == 0 for item in report.calls)


@pytest.mark.asyncio
async def test_bad_result_fails_and_redacts_key_from_error():
    async def runner(workload):
        result = _result(workload)
        result.model = "secret-openai-key"
        return result

    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={Provider.OPENAI: "secret-openai-key", Provider.DEEPSEEK: "deepseek"},
        runner=runner,
        quota_probe=_quota,
        now=NOW,
    )

    assert report.status is ProbeStatus.FAIL
    assert "secret-openai-key" not in json.dumps(report.to_dict())


@pytest.mark.asyncio
async def test_total_cost_gate_is_fail_closed():
    async def runner(workload):
        return _result(workload, cost=0.1)

    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={Provider.OPENAI: "openai", Provider.DEEPSEEK: "deepseek"},
        runner=runner,
        quota_probe=_quota,
        now=NOW,
    )

    assert report.status is ProbeStatus.FAIL
    assert all("total_cost_above_threshold" in item.reasons for item in report.workloads)


def test_only_quota_headers_are_persisted():
    assert _quota_headers(
        {
            "Authorization": "secret",
            "Set-Cookie": "secret",
            "X-RateLimit-Remaining": "7",
            "Retry-After": "2",
        }
    ) == {"x-ratelimit-remaining": "7", "retry-after": "2"}


def test_secret_files_and_atomic_report_writer(tmp_path):
    secret = tmp_path / "key"
    secret.write_text(" secret-value\n", encoding="utf-8")
    assert _read_secret(secret, "provider") == "secret-value"

    report = LLMQualificationReport(
        schema_version=1,
        generated_at=NOW.isoformat(),
        samples_per_workload=1,
        thresholds={},
        quotas=(),
        calls=(),
        workloads=(),
    )
    destination = tmp_path / "report.json"
    _write_report(destination, report, overwrite=False)
    assert json.loads(destination.read_text(encoding="utf-8"))["live_orders_allowed"] is False
    with pytest.raises(FileExistsError):
        _write_report(destination, report, overwrite=False)
