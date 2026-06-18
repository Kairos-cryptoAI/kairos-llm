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


class _RecordingCompletions:
    def __init__(self, content, calls):
        self._content = content
        self._calls = calls

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        msg = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class _RecordingClient:
    def __init__(self, content, calls):
        self.chat = SimpleNamespace(completions=_RecordingCompletions(content, calls))


def test_deepseek_call_omits_reasoning_effort():
    calls = []
    gw = LLMGateway(client=_RecordingClient('{"ok": true}', calls))
    asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.LOW))   # -> deepseek-v4-flash
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert "reasoning_effort" not in calls[0]


def test_gpt_escalation_sends_reasoning_effort():
    calls = []
    gw = LLMGateway(client=_RecordingClient('{"ok": true}', calls))
    asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.HIGH))  # -> gpt-5.5 high
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["reasoning_effort"] == "high"


from kairos_llm.config import LLMSettings
from kairos_llm.errors import LLMServerError, LLMTimeout


class _FailingClient:
    def __init__(self, status=None, timeout=False):
        self._status = status
        self._timeout = timeout
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        if self._timeout:
            raise asyncio.TimeoutError()
        err = RuntimeError("boom")
        err.status_code = self._status
        raise err


def test_health_hook_fires_on_success():
    events = []
    gw = LLMGateway(client=_FakeClient('{"x": 1}'), on_health=lambda *a: events.append(a))
    asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.LOW))  # deepseek-v4-flash
    assert events == [("deepseek-v4-flash", "deepseek", True, "ok", events[0][4])]
    assert events[0][2] is True and events[0][3] == "ok"


def test_health_hook_fires_on_5xx():
    events = []
    gw = LLMGateway(settings=LLMSettings(max_retries=0), client=_FailingClient(status=503),
                    on_health=lambda *a: events.append(a))
    with pytest.raises(LLMServerError):
        asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.HIGH))  # gpt-5.5
    model, provider, ok, kind, _ = events[-1]
    assert (model, provider, ok, kind) == ("gpt-5.5", "openai", False, "5xx")


def test_health_hook_fires_on_timeout():
    events = []
    gw = LLMGateway(settings=LLMSettings(max_retries=0), client=_FailingClient(timeout=True),
                    on_health=lambda *a: events.append(a))
    with pytest.raises(LLMTimeout):
        asyncio.run(gw.complete(system="s", user="u", effort=ReasoningEffort.MEDIUM))  # deepseek-v4-pro
    assert events[-1][2] is False and events[-1][3] == "timeout"
