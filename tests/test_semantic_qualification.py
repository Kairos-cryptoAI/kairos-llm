"""Fixed golden corpus tests use fake clients, never paid inference."""

import json
from dataclasses import replace

import pytest

from kairos_llm.models import DEFAULT_WORKLOAD_ROUTES, LLMWorkload
from kairos_llm.schemas import LLMResult, TokenUsage
from kairos_llm.semantic_qualification import (
    SEMANTIC_CORPUS,
    SEMANTIC_SYSTEM_PROMPT,
    ShadowPolicyOutputV1,
    _parser,
    main,
    qualify_live_semantic_llms,
    qualify_semantic_llms,
    semantic_plan,
)


def _result(workload, case):
    route = DEFAULT_WORKLOAD_ROUTES[workload]
    payload = {
        "protocol": "KAIROS_SHADOW_POLICY_V1",
        "authority": "SHADOW_POLICY_PROBE_ONLY",
        "decision": case.expected_decision,
        "reason_code": case.expected_reason_code,
    }
    return LLMResult(
        content=json.dumps(payload),
        parsed=ShadowPolicyOutputV1.model_validate(payload),
        model=route.choice.model,
        effort=route.effort.value,
        workload=workload.value,
        provider=route.choice.provider.value,
        request_id="req-fake",
        resolved_model=route.choice.model,
        budget_reservation_id=f"reservation-{workload.value}-{case.case_id}",
        attempt_started_at_ts_ms=100,
        response_observed_at_ts_ms=200,
        usage=TokenUsage(input_tokens=150, output_tokens=40),
        cost_usd=0.0001,
        latency_s=0.1,
    )


def test_fixed_corpus_and_exact_schema_cost_are_preregistered_before_calls():
    plan = semantic_plan()
    assert len(plan["cases"]) == len(plan["routes"]) == 4
    assert len(plan["reservations"]) == 16
    assert plan["planned_cost_ceiling_usd"] == 0.256
    assert plan == semantic_plan()
    assert plan["trading_authority"] is False
    assert "expected_decision" in plan["cases"][0]
    assert len(plan["plan_sha256"]) == len(plan["corpus_sha256"]) == len(plan["schema_sha256"]) == 64


@pytest.mark.parametrize("bound", [0, -1, float("nan"), float("inf"), 0.401, 0.1])
def test_bad_or_inadequate_cost_ceiling_blocks_before_any_provider(bound):
    with pytest.raises(ValueError):
        semantic_plan(maximum_planned_cost_usd=bound)


async def test_all_four_routes_receive_identical_case_inputs_once_and_pass_exact_gold():
    calls, checkpoints = [], []

    async def runner(workload, system, user, schema):
        calls.append((workload, system, user, schema))
        case = next(item for item in SEMANTIC_CORPUS if item.input_json == user)
        return _result(workload, case)

    report = await qualify_semantic_llms(runner=runner, plan=semantic_plan(), checkpoint=checkpoints.append)
    assert report.status == "PASS"
    assert len(calls) == len(checkpoints) == len(report.observations) == 16
    for workload in LLMWorkload:
        matching = [item for item in calls if item[0] == workload]
        assert [item[2] for item in matching] == [case.input_json for case in SEMANTIC_CORPUS]
        assert all(item[1] == SEMANTIC_SYSTEM_PROMPT and item[3] is ShadowPolicyOutputV1 for item in matching)
    payload = report.to_dict()
    assert payload["production_quality_qualified"] is False
    assert payload["live_orders_allowed"] is False
    assert payload["alpha_ready"] is False
    assert payload["observed_cost_usd"] == pytest.approx(0.0016)
    assert all(item.response_sha256 and item.budget_reservation_id for item in report.observations)
    assert all("content" not in item for item in payload["observations"])


@pytest.mark.parametrize(
    "fault", ["wrong-label", "extra-field", "parsed-disagrees", "no-reservation", "bad-route"]
)
async def test_bad_policy_schema_or_identity_fails_without_posthoc_gold_changes(fault):
    calls = []

    async def runner(workload, system, user, schema):
        calls.append((workload, user))
        case = next(item for item in SEMANTIC_CORPUS if item.input_json == user)
        result = _result(workload, case)
        payload = json.loads(result.content)
        if fault == "wrong-label":
            payload["decision"] = "ALLOW" if payload["decision"] != "ALLOW" else "VETO"
            result = replace(
                result, content=json.dumps(payload), parsed=ShadowPolicyOutputV1.model_validate(payload)
            )
        elif fault == "extra-field":
            payload["execute_order"] = True
            result = replace(result, content=json.dumps(payload))
        elif fault == "parsed-disagrees":
            payload["reason_code"] = "STALE_INPUT"
            result = replace(result, content=json.dumps(payload))
        elif fault == "no-reservation":
            result = replace(result, budget_reservation_id=None)
        else:
            result = replace(result, model="other-model")
        return result

    plan = semantic_plan()
    report = await qualify_semantic_llms(runner=runner, plan=plan)
    assert report.status == "FAIL"
    assert len(calls) == 16  # one attempt per independently preregistered case, no retry
    assert report.plan == plan == semantic_plan()
    assert any(item.status == "FAIL" for item in report.observations)


async def test_backend_change_within_route_fails_and_is_retained():
    calls = {}

    async def runner(workload, system, user, schema):
        calls[workload] = calls.get(workload, 0) + 1
        case = next(item for item in SEMANTIC_CORPUS if item.input_json == user)
        result = _result(workload, case)
        return replace(result, resolved_model=f"resolved-{calls[workload]}")

    report = await qualify_semantic_llms(runner=runner, plan=semantic_plan())
    assert report.status == "FAIL"
    assert sum(item.status == "PASS" for item in report.observations) == 4


async def test_provider_failure_never_persists_exception_text_or_invents_zero_cost():
    calls = []

    async def runner(*args):
        calls.append(args)
        raise RuntimeError("PRIVATE_PROVIDER_RESPONSE_AND_KEY")

    report = await qualify_semantic_llms(runner=runner, plan=semantic_plan())
    assert report.status == "FAIL" and len(calls) == 16
    assert report.to_dict()["failed_call_cost_unknown"] is True
    assert all(item.cost_usd is None for item in report.observations)
    assert "PRIVATE_PROVIDER_RESPONSE_AND_KEY" not in json.dumps(report.to_dict())
    assert {item.failure_class for item in report.observations} == {"RuntimeError"}


async def test_tampered_plan_is_rejected_before_runner():
    async def forbidden(*args):
        raise AssertionError("must not call")

    plan = semantic_plan()
    plan["cases"][0]["expected_decision"] = "VETO"
    with pytest.raises(ValueError, match="identity"):
        await qualify_semantic_llms(runner=forbidden, plan=plan)


async def test_live_admission_requires_existing_budget_and_pins_endpoints_before_calls(monkeypatch):
    import kairos_llm.semantic_qualification as qualification

    settings = []

    class Gateway:
        def __init__(self, config):
            settings.append(config)

    class Budgeted(qualification.BudgetedLLMGateway):
        def __init__(self, gateway, budget):
            pass

        async def close(self):
            pass

    async def fake_qualify(**kwargs):
        return kwargs["plan"]

    monkeypatch.setenv("KAIROS_OPENAI_BASE_URL", "http://untrusted.invalid")
    monkeypatch.setenv("KAIROS_DEEPSEEK_BASE_URL", "http://untrusted.invalid")
    monkeypatch.setattr(qualification, "LLMGateway", Gateway)
    monkeypatch.setattr(qualification, "BudgetedLLMGateway", Budgeted)
    monkeypatch.setattr(qualification, "qualify_semantic_llms", fake_qualify)
    with pytest.raises(ValueError, match="existing adopted"):
        await qualify_live_semantic_llms(openai_api_key="test-only", usage_budget=None, plan=semantic_plan())
    assert settings == []
    await qualify_live_semantic_llms(openai_api_key="test-only", usage_budget=object(), plan=semantic_plan())
    assert settings[0].openai_base_url == "https://api.openai.com/v1"
    assert settings[0].deepseek_base_url == "https://api.deepseek.com"
    assert settings[0].deepseek_api_key is None
    assert settings[0].max_retries == 0 and settings[0].max_output_tokens == 2_048


def test_cli_requires_named_target_and_expected_plan_and_refuses_before_key_read(monkeypatch, tmp_path):
    import kairos_llm.semantic_qualification as qualification

    with pytest.raises(SystemExit):
        _parser().parse_args(["--output", "result.json"])

    def forbidden_read(*args):
        raise AssertionError("key read before admission")

    monkeypatch.setattr(qualification, "_read_secret", forbidden_read)
    argv = ["--expected-database-name", "kairos", "--output", str(tmp_path / "result.json")]
    with pytest.raises(ValueError, match="identity"):
        main([*argv, "--expected-plan-sha256", "0" * 64])
    output = tmp_path / "result.json"
    output.touch()
    with pytest.raises(FileExistsError):
        main([*argv, "--expected-plan-sha256", semantic_plan()["plan_sha256"]])
