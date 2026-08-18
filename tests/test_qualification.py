import json
from datetime import UTC, datetime

import pytest

from kairos_llm.models import DEFAULT_WORKLOAD_ROUTES, LLMWorkload, Provider
from kairos_llm.qualification import (
    QUALIFICATION_SYSTEM_PROMPT,
    QUALIFICATION_USER_PROMPT,
    LLMQualificationReport,
    ProbePayload,
    ProbeStatus,
    QuotaObservation,
    _quota_headers,
    _read_secret,
    _selected_workloads,
    _write_report,
    planned_cost_ceiling_usd,
    qualify_live_llms,
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
async def test_targeted_workload_calls_only_its_provider_and_route():
    called_workloads = []
    probed_providers = []

    async def runner(workload):
        called_workloads.append(workload)
        return _result(workload)

    async def quota(provider, key):
        probed_providers.append(provider)
        return await _quota(provider, key)

    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={Provider.OPENAI: "openai", Provider.DEEPSEEK: "deepseek"},
        runner=runner,
        quota_probe=quota,
        workloads=(LLMWorkload.TEXT_SCOUTS,),
        now=NOW,
    )

    assert called_workloads == [LLMWorkload.TEXT_SCOUTS]
    assert probed_providers == [Provider.DEEPSEEK]
    assert [item.workload for item in report.calls] == [LLMWorkload.TEXT_SCOUTS.value]
    assert [item.workload for item in report.workloads] == [LLMWorkload.TEXT_SCOUTS.value]


def test_planned_cost_ceiling_is_route_specific_and_rejects_bad_selection():
    deepseek_only = planned_cost_ceiling_usd(
        workloads=(LLMWorkload.TEXT_SCOUTS,),
        samples_per_workload=1,
    )
    all_routes = planned_cost_ceiling_usd(workloads=None, samples_per_workload=1)

    assert deepseek_only == pytest.approx(0.00032256)
    assert all_routes == pytest.approx(0.02059776)
    assert deepseek_only < all_routes
    with pytest.raises(ValueError, match="duplicates"):
        _selected_workloads((LLMWorkload.TEXT_SCOUTS, LLMWorkload.TEXT_SCOUTS))
    with pytest.raises(ValueError, match="at least one"):
        _selected_workloads(())


def test_deepseek_qualification_prompts_explicitly_request_json():
    assert "json" in QUALIFICATION_SYSTEM_PROMPT.casefold()
    assert "json" in QUALIFICATION_USER_PROMPT.casefold()


@pytest.mark.asyncio
async def test_live_qualification_refuses_over_budget_before_network():
    with pytest.raises(ValueError, match="exceeds"):
        await qualify_live_llms(
            openai_api_key="not-used",
            deepseek_api_key=None,
            samples_per_workload=1,
            workloads=(LLMWorkload.MACRO_STRATEGIST,),
            maximum_planned_cost_usd=0.001,
        )


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
