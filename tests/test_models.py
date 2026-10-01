import pytest
from kairos_core.enums import ReasoningEffort

from kairos_llm.models import (
    DEFAULT_WORKLOAD_ROUTES,
    LLMWorkload,
    ModelChoice,
    ModelRoute,
    ModelRouter,
    Provider,
)


@pytest.mark.parametrize(
    ("workload", "model", "provider", "effort", "max_output_tokens"),
    [
        (
            LLMWorkload.TEXT_SCOUTS,
            "gpt-6-luna",
            Provider.OPENAI,
            ReasoningEffort.LOW,
            1_024,
        ),
        (
            LLMWorkload.AGGREGATOR_NORMAL,
            "gpt-6-luna",
            Provider.OPENAI,
            ReasoningEffort.MEDIUM,
            2_048,
        ),
        (
            LLMWorkload.AGGREGATOR_CONFLICT,
            "gpt-6.1-sol",
            Provider.OPENAI,
            ReasoningEffort.HIGH,
            4_096,
        ),
        (
            LLMWorkload.MACRO_STRATEGIST,
            "gpt-6.1-sol",
            Provider.OPENAI,
            ReasoningEffort.XHIGH,
            8_192,
        ),
    ],
)
def test_explicit_workload_routes(workload, model, provider, effort, max_output_tokens):
    route = ModelRouter().resolve(workload=workload)

    assert route.choice.model == model
    assert route.choice.provider is provider
    assert route.effort is effort
    assert route.workload is workload
    assert route.max_output_tokens == max_output_tokens


def test_provider_reasoning_modes_match_roles():
    router = ModelRouter()

    text = router.resolve(workload=LLMWorkload.TEXT_SCOUTS).choice
    normal = router.resolve(workload=LLMWorkload.AGGREGATOR_NORMAL).choice
    conflict = router.resolve(workload=LLMWorkload.AGGREGATOR_CONFLICT).choice
    macro = router.resolve(workload=LLMWorkload.MACRO_STRATEGIST).choice

    assert text.provider_effort == "low"
    assert normal.provider_effort == "medium"
    assert conflict.provider_effort == "high"
    assert macro.provider_effort == "xhigh"


def test_effort_only_routing_remains_backward_compatible():
    router = ModelRouter()

    assert router.choose(ReasoningEffort.LOW).model == "gpt-6-luna"
    assert router.choose(ReasoningEffort.MEDIUM).model == "gpt-6-luna"
    assert router.choose(ReasoningEffort.HIGH).model == "gpt-6.1-sol"
    assert router.choose(ReasoningEffort.XHIGH).model == "gpt-6.1-sol"
    assert str(Provider.DEEPSEEK) == "deepseek"


def test_all_default_workload_routes_are_openai_only():
    assert all(route.choice.provider is Provider.OPENAI for route in DEFAULT_WORKLOAD_ROUTES.values())


def test_workload_route_is_independent_of_legacy_effort_override():
    router = ModelRouter()
    router.override(ReasoningEffort.MEDIUM, "legacy-override", Provider.OPENAI)

    assert router.choose(ReasoningEffort.MEDIUM).model == "legacy-override"
    assert router.choose(ReasoningEffort.MEDIUM, workload=LLMWorkload.AGGREGATOR_NORMAL).model == "gpt-6-luna"


def test_non_openai_provider_requires_explicit_opt_in():
    router = ModelRouter()

    with pytest.raises(ValueError, match="not enabled"):
        router.override(ReasoningEffort.MEDIUM, "deepseek-flash", Provider.DEEPSEEK)

    router = ModelRouter(allowed_providers={Provider.OPENAI, Provider.DEEPSEEK})
    router.override(ReasoningEffort.MEDIUM, "deepseek-flash", Provider.DEEPSEEK)
    assert router.choose(ReasoningEffort.MEDIUM).provider is Provider.DEEPSEEK


def test_non_openai_provider_routes_require_explicit_opt_in_at_construction():
    routes = dict(DEFAULT_WORKLOAD_ROUTES)
    routes[LLMWorkload.TEXT_SCOUTS] = ModelRoute(
        ModelChoice("deepseek-flash", Provider.DEEPSEEK),
        ReasoningEffort.LOW,
        LLMWorkload.TEXT_SCOUTS,
    )

    with pytest.raises(ValueError, match="not enabled"):
        ModelRouter(workload_mapping=routes)


def test_route_requires_workload_or_effort():
    with pytest.raises(ValueError, match="workload or effort"):
        ModelRouter().resolve()


def test_router_requires_at_least_one_allowed_provider():
    with pytest.raises(ValueError, match="must not be empty"):
        ModelRouter(allowed_providers=set())


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_route_output_limit_is_strict(value):
    with pytest.raises(ValueError, match="positive integer"):
        ModelRoute(ModelChoice("model", Provider.OPENAI), ReasoningEffort.HIGH, max_output_tokens=value)
