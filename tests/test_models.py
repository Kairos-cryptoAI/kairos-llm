from kairos_core.enums import ReasoningEffort

from kairos_llm.models import ModelRouter, Provider


def test_routine_flow_uses_deepseek():
    r = ModelRouter()
    assert r.choose(ReasoningEffort.LOW).model == "deepseek-v4-flash"
    assert r.choose(ReasoningEffort.MEDIUM).model == "deepseek-v4-pro"
    assert r.choose(ReasoningEffort.LOW).provider is Provider.DEEPSEEK
    assert r.choose(ReasoningEffort.MEDIUM).provider is Provider.DEEPSEEK
    assert str(Provider.DEEPSEEK) == "deepseek"


def test_deepseek_routine_is_non_thinking():
    # Flash / Pro must run without a reasoning.effort parameter.
    r = ModelRouter()
    assert r.choose(ReasoningEffort.LOW).send_reasoning_effort is False
    assert r.choose(ReasoningEffort.MEDIUM).send_reasoning_effort is False


def test_escalation_uses_gpt55():
    r = ModelRouter()
    assert r.choose(ReasoningEffort.HIGH).model == "gpt-5.5"
    assert r.choose(ReasoningEffort.XHIGH).model == "gpt-5.5"
    assert r.choose(ReasoningEffort.HIGH).provider is Provider.OPENAI


def test_effort_values_match_doc():
    r = ModelRouter()
    assert r.choose(ReasoningEffort.HIGH).provider_effort == "high"
    assert r.choose(ReasoningEffort.XHIGH).provider_effort == "xhigh"
