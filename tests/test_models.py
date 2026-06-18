from kairos_core.enums import ReasoningEffort
from kairos_llm.models import ModelRouter


def test_routine_flow_uses_mini():
    r = ModelRouter()
    assert "mini" in r.choose(ReasoningEffort.LOW).model
    assert "mini" in r.choose(ReasoningEffort.MEDIUM).model


def test_high_effort_uses_flagship():
    r = ModelRouter()
    assert r.choose(ReasoningEffort.HIGH).model == "gpt-5.5"
    assert r.choose(ReasoningEffort.XHIGH).model == "gpt-5.5"


def test_xhigh_maps_to_provider_high():
    r = ModelRouter()
    assert r.choose(ReasoningEffort.XHIGH).provider_effort == "high"
