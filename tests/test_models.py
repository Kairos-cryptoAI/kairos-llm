import pytest
from kairos_core.enums import ReasoningEffort

from kairos_llm.models import LLMWorkload, ModelChoice, ModelRoute, ModelRouter, Provider


@pytest.mark.parametrize(
    ("workload", "model", "provider", "effort", "max_output_tokens"),
    [
        (
            LLMWorkload.TEXT_SCOUTS,
            "deepseek-v4-flash",
            Provider.DEEPSEEK,
            ReasoningEffort.LOW,
            1_024,
        ),
        (
            LLMWorkload.AGGREGATOR_NORMAL,
            "gpt-5.6-luna",
            Provider.OPENAI,
            ReasoningEffort.MEDIUM,
            2_048,
        ),
        (
            LLMWorkload.AGGREGATOR_CONFLICT,
            "gpt-5.6-terra",
            Provider.OPENAI,
            ReasoningEffort.HIGH,
            4_096,
        ),
        (
            LLMWorkload.MACRO_STRATEGIST,
            "gpt-5.6-sol",
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

    assert text.send_reasoning_effort is False
    assert normal.provider_effort == "medium"
    assert conflict.provider_effort == "high"
    assert macro.provider_effort == "xhigh"


def test_effort_only_routing_remains_backward_compatible():
    router = ModelRouter()

    assert router.choose(ReasoningEffort.LOW).model == "deepseek-v4-flash"
    assert router.choose(ReasoningEffort.MEDIUM).model == "gpt-5.6-luna"
    assert router.choose(ReasoningEffort.HIGH).model == "gpt-5.6-terra"
    assert router.choose(ReasoningEffort.XHIGH).model == "gpt-5.6-sol"
    assert str(Provider.DEEPSEEK) == "deepseek"


def test_workload_route_is_independent_of_legacy_effort_override():
    router = ModelRouter()
    router.override(ReasoningEffort.MEDIUM, "legacy-override", Provider.DEEPSEEK)

    assert router.choose(ReasoningEffort.MEDIUM).model == "legacy-override"
    assert (
        router.choose(ReasoningEffort.MEDIUM, workload=LLMWorkload.AGGREGATOR_NORMAL).model == "gpt-5.6-luna"
    )


def test_route_requires_workload_or_effort():
    with pytest.raises(ValueError, match="workload or effort"):
        ModelRouter().resolve()


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_route_output_limit_is_strict(value):
    with pytest.raises(ValueError, match="positive integer"):
        ModelRoute(ModelChoice("model", Provider.OPENAI), ReasoningEffort.HIGH, max_output_tokens=value)
