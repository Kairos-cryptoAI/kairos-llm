"""Gateway parsing/accounting with a fake OpenAI client (no network)."""
import asyncio
from types import SimpleNamespace

import pytest

from kairos_core.enums import ReasoningEffort
from kairos_llm.gateway import LLMGateway
from kairos_llm.errors import LLMBadOutput


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    async def create(self, **kwargs):
        msg = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(prompt_tokens=1500, completion_tokens=300,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=750))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class _FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def test_parses_json_and_accounts_cost():
    gw = LLMGateway(client=_FakeClient('{"sentiment": 0.85, "impact": "bullish"}'))
    res = asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.LOW))
    assert res.parsed["sentiment"] == 0.85
    assert res.usage.cached_input_tokens == 750
    assert res.cost_usd > 0
    assert gw.accountant.calls == 1


def test_bad_json_raises():
    gw = LLMGateway(client=_FakeClient("not json"))
    with pytest.raises(LLMBadOutput):
        asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.LOW))
