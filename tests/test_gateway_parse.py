"""Provider routing, parsing, accounting, and failure semantics."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from kairos_core.enums import ReasoningEffort
from pydantic import BaseModel

from kairos_llm.config import LLMSettings
from kairos_llm.errors import LLMBadOutput, LLMServerError, LLMTimeout
from kairos_llm.gateway import LLMGateway


class SentimentOutput(BaseModel):
    sentiment: float
    impact: str


class _FakeCompletions:
    def __init__(self, content, calls, error=None):
        self._content = content
        self._calls = calls
        self._error = error

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(
            prompt_tokens=1_500,
            completion_tokens=300,
            prompt_tokens_details=SimpleNamespace(cached_tokens=750),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeResponses:
    def __init__(self, content, calls, error=None, status="completed"):
        self._content = content
        self._calls = calls
        self._error = error
        self._status = status

    async def parse(self, **kwargs):
        self._calls.append(kwargs)
        if self._error:
            raise self._error
        data = json.loads(self._content)
        parsed = kwargs["text_format"].model_validate(data)
        usage = SimpleNamespace(
            input_tokens=1_000,
            output_tokens=200,
            input_tokens_details=SimpleNamespace(cached_tokens=500),
        )
        return SimpleNamespace(
            output_parsed=parsed,
            output_text=self._content,
            status=self._status,
            usage=usage,
        )


class _FakeClient:
    def __init__(self, content, *, error=None, status="completed"):
        self.chat_calls = []
        self.response_calls = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, self.chat_calls, error))
        self.responses = _FakeResponses(content, self.response_calls, error, status)


def test_deepseek_parses_json_and_accounts_cost():
    client = _FakeClient('{"sentiment": 0.85, "impact": "bullish"}')
    gateway = LLMGateway(client=client)

    result = asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    assert result.parsed["sentiment"] == 0.85
    assert result.usage.cached_input_tokens == 750
    assert result.cost_usd > 0
    assert gateway.accountant.calls == 1


def test_openai_uses_responses_structured_outputs():
    client = _FakeClient('{"sentiment": 0.85, "impact": "bullish"}')
    gateway = LLMGateway(client=client)

    result = asyncio.run(
        gateway.complete(
            system="s",
            user="u",
            effort=ReasoningEffort.HIGH,
            schema=SentimentOutput,
        )
    )

    assert result.parsed == SentimentOutput(sentiment=0.85, impact="bullish")
    assert result.usage.cached_input_tokens == 500
    assert client.response_calls[0]["model"] == "gpt-5.6-sol"
    assert client.response_calls[0]["reasoning"] == {"effort": "high"}
    assert client.response_calls[0]["store"] is False
    assert client.chat_calls == []


def test_bad_json_raises_without_recording_success():
    events = []
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=0),
        client=_FakeClient("not json"),
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMBadOutput):
        asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    assert gateway.accountant.calls == 0
    assert events[-1][2:4] == (False, "bad_output")


def test_deepseek_explicitly_disables_default_thinking():
    client = _FakeClient('{"ok": true}')
    gateway = LLMGateway(client=client)

    asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    call = client.chat_calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in call


def test_health_hook_fires_after_validated_success():
    events = []
    gateway = LLMGateway(client=_FakeClient('{"x": 1}'), on_health=lambda *args: events.append(args))

    asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    assert events[0][:4] == ("deepseek-v4-flash", "deepseek", True, "ok")


def test_health_sink_failure_does_not_repeat_a_paid_call():
    client = _FakeClient('{"x": 1}')

    def broken_health_sink(*_args):
        raise RuntimeError("telemetry unavailable")

    gateway = LLMGateway(client=client, on_health=broken_health_sink)
    result = asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    assert result.parsed == {"x": 1}
    assert len(client.chat_calls) == 1


class _ProviderFailure(RuntimeError):
    def __init__(self, *, status=None):
        super().__init__("boom")
        self.status_code = status


def test_health_hook_fires_on_5xx():
    events = []
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=0),
        client=_FakeClient("{}", error=_ProviderFailure(status=503)),
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMServerError):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.HIGH))

    assert events[-1][:4] == ("gpt-5.6-sol", "openai", False, "5xx")


def test_health_hook_fires_on_timeout():
    events = []
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=0),
        client=_FakeClient("{}", error=TimeoutError()),
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMTimeout):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.MEDIUM))

    assert events[-1][2:4] == (False, "timeout")
