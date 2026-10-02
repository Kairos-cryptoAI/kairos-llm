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
    _deepseek_balance_status,
    _parser,
    _positive_openai_rate_limit,
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


def _result(
    workload: LLMWorkload,
    *,
    cost: float = 0.001,
    rate_limit_headers: dict[str, str] | None = None,
) -> LLMResult:
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
        rate_limit_headers=rate_limit_headers or {},
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
    assert probed_providers == [Provider.OPENAI]
    assert [item.workload for item in report.calls] == [LLMWorkload.TEXT_SCOUTS.value]
    assert [item.workload for item in report.workloads] == [LLMWorkload.TEXT_SCOUTS.value]


@pytest.mark.asyncio
async def test_inference_headers_override_unusable_model_list_quota_probe():
    async def runner(workload):
        return _result(
            workload,
            rate_limit_headers={
                "x-ratelimit-remaining-requests": "499",
                "Authorization": "synthetic-secret-never-persisted",
            },
        )

    async def forbidden_model_list(provider, _key):
        return QuotaObservation(
            provider=provider.value,
            status=ProbeStatus.FAIL,
            http_status=403,
            latency_ms=1,
            headers={},
            detail="model-list unavailable",
        )

    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={Provider.OPENAI: "openai", Provider.DEEPSEEK: None},
        runner=runner,
        quota_probe=forbidden_model_list,
        workloads=(LLMWorkload.AGGREGATOR_NORMAL,),
        now=NOW,
    )

    assert report.quotas[0].status is ProbeStatus.PASS
    assert report.quotas[0].headers == {"x-ratelimit-remaining-requests": "499"}
    assert "synthetic-secret-never-persisted" not in json.dumps(report.to_dict())


def test_planned_cost_ceiling_is_route_specific_and_rejects_bad_selection():
    text_scouts_only = planned_cost_ceiling_usd(
        workloads=(LLMWorkload.TEXT_SCOUTS,),
        samples_per_workload=1,
    )
    all_routes = planned_cost_ceiling_usd(workloads=None, samples_per_workload=1)

    assert text_scouts_only == pytest.approx(0.000576)
    assert all_routes == pytest.approx(0.024192)
    assert text_scouts_only < all_routes
    with pytest.raises(ValueError, match="duplicates"):
        _selected_workloads((LLMWorkload.TEXT_SCOUTS, LLMWorkload.TEXT_SCOUTS))
    with pytest.raises(ValueError, match="at least one"):
        _selected_workloads(())


def test_deepseek_qualification_prompts_explicitly_request_json():
    assert "json" in QUALIFICATION_SYSTEM_PROMPT.casefold()
    assert "json" in QUALIFICATION_USER_PROMPT.casefold()


@pytest.mark.parametrize("value", [True, False, 127, 4097, 2048.0, "2048"])
def test_output_allowance_rejects_out_of_bounds_or_untyped_values(value):
    with pytest.raises(ValueError, match="128 through 4096"):
        planned_cost_ceiling_usd(workloads=None, samples_per_workload=1, max_output_tokens=value)


def test_output_allowance_cost_preserves_workload_specific_caps():
    from kairos_llm.pricing import PriceTable

    allowance = 2048
    expected = sum(
        PriceTable().reservation_cost(
            route.choice.model,
            TokenUsage(input_tokens=4096, output_tokens=min(allowance, route.max_output_tokens)),
        )
        for route in DEFAULT_WORKLOAD_ROUTES.values()
    )
    actual = planned_cost_ceiling_usd(workloads=None, samples_per_workload=1, max_output_tokens=allowance)
    assert actual == pytest.approx(expected)
    assert actual > planned_cost_ceiling_usd(workloads=None, samples_per_workload=1)
    assert actual < 0.1


@pytest.mark.asyncio
async def test_output_allowance_above_planned_budget_rejects_before_any_gateway(monkeypatch):
    import kairos_llm.qualification as qualification

    def forbidden_gateway(_settings):
        raise AssertionError("no network-capable gateway before cost admission")

    monkeypatch.setattr(qualification, "LLMGateway", forbidden_gateway)
    with pytest.raises(ValueError, match="exceeds"):
        await qualify_live_llms(
            openai_api_key="synthetic-not-dispatched",
            deepseek_api_key=None,
            samples_per_workload=1,
            max_output_tokens=2048,
            maximum_planned_cost_usd=0.05,
        )


@pytest.mark.asyncio
async def test_bounded_output_allowance_reaches_settings_reservation_and_report(monkeypatch):
    import kairos_llm.qualification as qualification
    from tests.test_budget import _Budget, _Gateway

    budget = _Budget()
    underlying = _Gateway()
    captured = {}

    def gateway(settings):
        underlying.settings = settings
        return underlying

    async def fake_qualify(**kwargs):
        captured.update(kwargs)
        return await kwargs["runner"](LLMWorkload.TEXT_SCOUTS)

    monkeypatch.setattr(qualification, "LLMGateway", gateway)
    monkeypatch.setattr(qualification, "qualify_llms", fake_qualify)
    await qualify_live_llms(
        openai_api_key="synthetic-not-dispatched",
        deepseek_api_key=None,
        samples_per_workload=1,
        workloads=(LLMWorkload.TEXT_SCOUTS,),
        usage_budget=budget,
        max_output_tokens=2048,
        maximum_planned_cost_usd=0.1,
    )
    assert underlying.settings.max_output_tokens == 2048
    assert captured["max_output_tokens"] == 2048
    assert budget.reservations[0]["reserved_microusd"] == round(captured["planned_cost_ceiling"] * 1_000_000)
    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={},
        runner=underlying.complete,
        quota_probe=_quota,
        max_output_tokens=2048,
    )
    assert report.to_dict()["max_output_tokens"] == 2048


def test_cli_requires_explicit_campaign_database_target():
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["--output", "qualification.json"])
    assert exc.value.code == 2
    parsed = _parser().parse_args(["--expected-database-name", "kairos", "--output", "qualification.json"])
    assert parsed.expected_database_name == "kairos"


def test_probe_payload_rejects_extra_provider_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ProbePayload.model_validate_json(
            '{"protocol":"KAIROS_LLM_PROBE_V1","arithmetic":42,'
            '"decision":"NO_TRADE","unexpected":"ignored-before"}'
        )


@pytest.mark.asyncio
async def test_raw_provider_response_must_match_parsed_exact_contract():
    async def runner(workload):
        result = _result(workload)
        result.content = (
            '{"protocol":"KAIROS_LLM_PROBE_V1","arithmetic":42,'
            '"decision":"NO_TRADE","unexpected":"ignored-before"}'
        )
        return result

    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={Provider.OPENAI: "synthetic-not-dispatched"},
        runner=runner,
        quota_probe=_quota,
        workloads=(LLMWorkload.AGGREGATOR_NORMAL,),
        now=NOW,
    )

    assert report.status is ProbeStatus.FAIL
    assert report.calls[0].status is ProbeStatus.FAIL
    assert "exact qualification schema" in report.calls[0].detail


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
async def test_live_qualification_refuses_missing_shared_campaign_budget_before_network():
    with pytest.raises(ValueError, match="shared durable campaign"):
        await qualify_live_llms(
            openai_api_key="not-used",
            deepseek_api_key=None,
            samples_per_workload=1,
        )


@pytest.mark.asyncio
async def test_live_probe_reserves_and_accounts_through_shared_budget(monkeypatch):
    import kairos_llm.qualification as qualification
    from tests.test_budget import _Budget, _Gateway

    underlying = _Gateway()
    budget = _Budget()
    monkeypatch.setattr(qualification, "LLMGateway", lambda _settings: underlying)

    async def fake_qualify(**kwargs):
        return await kwargs["runner"](LLMWorkload.TEXT_SCOUTS)

    monkeypatch.setattr(qualification, "qualify_llms", fake_qualify)
    result = await qualify_live_llms(
        deepseek_api_key=None,
        openai_api_key="synthetic-openai-not-dispatched",
        samples_per_workload=1,
        workloads=(LLMWorkload.TEXT_SCOUTS,),
        usage_budget=budget,
    )
    assert result.budget_reservation_id is not None
    assert len(budget.reservations) == len(budget.commits) == len(underlying.calls) == 1
    assert underlying.closed


@pytest.mark.asyncio
async def test_qualification_pins_official_endpoints_despite_environment(monkeypatch):
    import kairos_llm.qualification as qualification
    from tests.test_budget import _Budget, _Gateway

    captured_settings = []
    underlying = _Gateway()
    monkeypatch.setenv("KAIROS_OPENAI_BASE_URL", "https://untrusted.example/v1")
    monkeypatch.setenv("KAIROS_DEEPSEEK_BASE_URL", "https://untrusted.example")

    def fake_gateway(settings):
        captured_settings.append(settings)
        return underlying

    async def fake_qualify(**kwargs):
        return await kwargs["runner"](LLMWorkload.TEXT_SCOUTS)

    monkeypatch.setattr(qualification, "LLMGateway", fake_gateway)
    monkeypatch.setattr(qualification, "qualify_llms", fake_qualify)
    await qualify_live_llms(
        openai_api_key="synthetic-openai",
        deepseek_api_key=None,
        samples_per_workload=1,
        workloads=(LLMWorkload.TEXT_SCOUTS,),
        usage_budget=_Budget(),
    )

    assert captured_settings[0].openai_base_url == "https://api.openai.com/v1"
    assert captured_settings[0].deepseek_base_url == "https://api.deepseek.com"
    assert underlying.closed


@pytest.mark.asyncio
async def test_live_probe_budget_failure_cannot_dispatch_provider(monkeypatch):
    import kairos_llm.qualification as qualification
    from tests.test_budget import _Budget, _Gateway

    underlying = _Gateway()
    monkeypatch.setattr(qualification, "LLMGateway", lambda _settings: underlying)

    async def fake_qualify(**kwargs):
        return await kwargs["runner"](LLMWorkload.TEXT_SCOUTS)

    monkeypatch.setattr(qualification, "qualify_llms", fake_qualify)
    with pytest.raises(RuntimeError, match="campaign exhausted"):
        await qualify_live_llms(
            deepseek_api_key=None,
            openai_api_key="synthetic-openai-not-dispatched",
            samples_per_workload=1,
            workloads=(LLMWorkload.TEXT_SCOUTS,),
            usage_budget=_Budget(RuntimeError("campaign exhausted")),
        )
    assert underlying.calls == []
    assert underlying.closed


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


def test_quota_evidence_needs_positive_remaining_capacity_or_deepseek_balance():
    assert _positive_openai_rate_limit({"x-ratelimit-remaining-requests": "2"})
    assert not _positive_openai_rate_limit({"x-ratelimit-limit-requests": "60"})
    assert not _positive_openai_rate_limit({"retry-after": "2"})
    assert not _positive_openai_rate_limit({"x-ratelimit-remaining-requests": "0"})
    assert not _positive_openai_rate_limit(
        {"x-ratelimit-remaining-requests": "2", "x-ratelimit-remaining-tokens": "0"}
    )
    assert _deepseek_balance_status({"is_available": True}) is ProbeStatus.PASS
    assert _deepseek_balance_status({"is_available": False}) is ProbeStatus.FAIL
    assert _deepseek_balance_status({"is_available": "true"}) is ProbeStatus.BLOCKED


@pytest.mark.asyncio
async def test_retry_after_alone_cannot_promote_quota_probe():
    async def runner(workload):
        return _result(workload, rate_limit_headers={"retry-after": "10"})

    async def failed_model_list(provider, _key):
        return QuotaObservation(provider.value, ProbeStatus.FAIL, 403, 1, {}, "unavailable")

    report = await qualify_llms(
        samples_per_workload=1,
        available_keys={Provider.OPENAI: "synthetic-not-dispatched"},
        runner=runner,
        quota_probe=failed_model_list,
        workloads=(LLMWorkload.AGGREGATOR_NORMAL,),
        now=NOW,
    )

    assert report.calls[0].status is ProbeStatus.PASS
    assert report.quotas[0].status is ProbeStatus.BLOCKED
    assert report.status is ProbeStatus.BLOCKED


@pytest.mark.asyncio
async def test_resolved_backend_change_within_probe_fails_closed():
    count = 0

    async def runner(workload):
        nonlocal count
        count += 1
        result = _result(workload)
        result.resolved_model = f"backend-{count}"
        return result

    report = await qualify_llms(
        samples_per_workload=2,
        available_keys={Provider.OPENAI: "synthetic-not-dispatched"},
        runner=runner,
        quota_probe=_quota,
        workloads=(LLMWorkload.AGGREGATOR_NORMAL,),
        now=NOW,
    )

    assert report.workloads[0].status is ProbeStatus.FAIL
    assert "resolved_model_changed_within_probe" in report.workloads[0].reasons


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
