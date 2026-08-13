"""Provider routing, parsing, accounting, and failure semantics."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from kairos_core.enums import ReasoningEffort
from openai import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError
from pydantic import BaseModel

from kairos_llm.config import LLMSettings
from kairos_llm.errors import LLMBadOutput, LLMServerError, LLMTimeout
from kairos_llm.gateway import LLMGateway
from kairos_llm.models import LLMWorkload


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
        if isinstance(self._error, list) and self._error:
            raise self._error.pop(0)
        if self._error is not None and not isinstance(self._error, list):
            raise self._error
        message = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(
            prompt_tokens=1_500,
            completion_tokens=300,
            prompt_tokens_details=SimpleNamespace(cached_tokens=750),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
            model="deepseek-v4-flash-0731",
            system_fingerprint="fp_deepseek_0731",
        )


class _FakeResponses:
    def __init__(self, content, calls, error=None, status="completed"):
        self._content = content
        self._calls = calls
        self._error = error
        self._status = status

    async def parse(self, **kwargs):
        self._calls.append(kwargs)
        if isinstance(self._error, list) and self._error:
            raise self._error.pop(0)
        if self._error is not None and not isinstance(self._error, list):
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
            model=kwargs["model"],
            system_fingerprint="fp_openai_test",
        )


class _FakeClient:
    def __init__(self, content, *, error=None, status="completed"):
        self.chat_calls = []
        self.response_calls = []
        chat_error = list(error) if isinstance(error, list) else error
        response_error = list(error) if isinstance(error, list) else error
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, self.chat_calls, chat_error))
        self.responses = _FakeResponses(content, self.response_calls, response_error, status)


def test_deepseek_parses_json_and_accounts_cost():
    client = _FakeClient('{"sentiment": 0.85, "impact": "bullish"}')
    gateway = LLMGateway(client=client)

    result = asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    assert result.parsed["sentiment"] == 0.85
    assert result.usage.cached_input_tokens == 750
    assert result.cost_usd > 0
    assert gateway.accountant.calls == 1
    assert result.model == "deepseek-v4-flash"
    assert result.resolved_model == "deepseek-v4-flash-0731"
    assert result.system_fingerprint == "fp_deepseek_0731"


def test_openai_workload_uses_responses_structured_outputs():
    client = _FakeClient('{"sentiment": 0.85, "impact": "bullish"}')
    gateway = LLMGateway(client=client)

    result = asyncio.run(
        gateway.complete(
            system="s",
            user="u",
            workload=LLMWorkload.AGGREGATOR_CONFLICT,
            schema=SentimentOutput,
        )
    )

    assert result.parsed == SentimentOutput(sentiment=0.85, impact="bullish")
    assert result.usage.cached_input_tokens == 500
    assert client.response_calls[0]["model"] == "gpt-5.6-terra"
    assert client.response_calls[0]["reasoning"] == {"effort": "high"}
    assert client.response_calls[0]["store"] is False
    assert client.chat_calls == []
    assert result.effort == "high"
    assert result.workload == "aggregator_conflict"
    assert result.resolved_model == "gpt-5.6-terra"
    assert result.system_fingerprint == "fp_openai_test"


def test_bad_json_raises_without_recording_success():
    events = []
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=3),
        client=_FakeClient("not json"),
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMBadOutput):
        asyncio.run(gateway.complete(system="return json", user="u", effort=ReasoningEffort.LOW))

    assert gateway.accountant.calls == 0
    assert len(gateway._injected_client.chat_calls) == 1
    assert events[-1][2:4] == (False, "bad_output")


def test_deepseek_explicitly_disables_default_thinking():
    client = _FakeClient('{"ok": true}')
    gateway = LLMGateway(client=client)

    result = asyncio.run(gateway.complete(system="return json", user="u", workload=LLMWorkload.TEXT_SCOUTS))

    call = client.chat_calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in call
    assert result.workload == "text_scouts"


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
        client=_FakeClient("{}", error=_sdk_status_error(APIStatusError, 503)),
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMServerError):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.HIGH))

    assert events[-1][:4] == ("gpt-5.6-terra", "openai", False, "5xx")


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


def _sdk_status_error(error_type, status_code):
    request = httpx.Request("POST", "https://provider.invalid/v1/responses")
    response = httpx.Response(status_code, request=request)
    return error_type("permanent provider error", response=response, body={})


@pytest.mark.parametrize(
    "error",
    [
        _sdk_status_error(BadRequestError, 400),
        _sdk_status_error(AuthenticationError, 401),
        _sdk_status_error(APIStatusError, 403),
        _sdk_status_error(APIStatusError, 422),
    ],
    ids=["bad-request", "authentication", "forbidden", "unprocessable"],
)
def test_permanent_http_4xx_fail_fast(error):
    events = []
    client = _FakeClient("{}", error=error)
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=3),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(type(error)):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.HIGH))

    assert len(client.response_calls) == 1
    assert events[-1][2:4] == (False, "http_4xx")


def test_invalid_openai_structured_output_fails_fast():
    events = []
    client = _FakeClient('{"sentiment": "not-a-number", "impact": "bullish"}')
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=3),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMBadOutput):
        asyncio.run(
            gateway.complete(
                system="s",
                user="u",
                effort=ReasoningEffort.HIGH,
                schema=SentimentOutput,
            )
        )

    assert len(client.response_calls) == 1
    assert gateway.accountant.calls == 0
    assert events[-1][2:4] == (False, "bad_output")


def test_programming_error_fails_fast():
    events = []
    client = _FakeClient("{}", error=ValueError("client adapter bug"))
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=3),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(ValueError, match="client adapter bug"):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.LOW))

    assert len(client.chat_calls) == 1
    assert events[-1][2:4] == (False, "error")


def test_status_shaped_programming_error_fails_fast():
    events = []
    error = _ProviderFailure(status=503)
    client = _FakeClient("{}", error=error)
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=3),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(_ProviderFailure):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.LOW))

    assert len(client.chat_calls) == 1
    assert events[-1][2:4] == (False, "error")


@pytest.mark.parametrize(
    "error",
    [
        _sdk_status_error(APIStatusError, 408),
        _sdk_status_error(APIStatusError, 409),
        _sdk_status_error(APIStatusError, 429),
        _sdk_status_error(APIStatusError, 503),
    ],
    ids=["request-timeout", "lock-conflict", "rate-limit", "server-error"],
)
def test_transient_http_failures_retry_then_succeed(monkeypatch, error):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    events = []
    client = _FakeClient('{"ok": true}', error=[error])
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=2),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    result = asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.LOW))

    assert result.parsed == {"ok": True}
    assert len(client.chat_calls) == 2
    assert sleeps == [0.5]
    assert len(events) == 1
    assert events[0][2:4] == (True, "ok")


def test_network_failure_retries_then_succeeds(monkeypatch):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    client = _FakeClient('{"ok": true}', error=[APIConnectionError(request=request)])
    gateway = LLMGateway(settings=LLMSettings(max_retries=2), client=client)

    result = asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.LOW))

    assert result.parsed == {"ok": True}
    assert len(client.chat_calls) == 2
    assert sleeps == [0.5]


def test_transient_failure_exhausts_retry_budget_and_emits_final_health(monkeypatch):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    events = []
    client = _FakeClient("{}", error=_sdk_status_error(APIStatusError, 503))
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=2),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMServerError):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.HIGH))

    assert len(client.response_calls) == 3
    assert sleeps == [0.5, 1.0]
    assert len(events) == 1
    assert events[0][2:4] == (False, "5xx")


def test_connection_failure_exhausts_budget_as_breaker_outage(monkeypatch):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    events = []
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    client = _FakeClient("{}", error=APIConnectionError(request=request))
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=2),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(LLMServerError):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.LOW))

    assert len(client.chat_calls) == 3
    assert sleeps == [0.5, 1.0]
    assert len(events) == 1
    assert events[0][2:4] == (False, "connection")


def test_nonstandard_sdk_status_does_not_retry(monkeypatch):
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    events = []
    error = _sdk_status_error(APIStatusError, 600)
    client = _FakeClient("{}", error=error)
    gateway = LLMGateway(
        settings=LLMSettings(max_retries=3),
        client=client,
        on_health=lambda *args: events.append(args),
    )

    with pytest.raises(APIStatusError):
        asyncio.run(gateway.complete(system="s", user="u", effort=ReasoningEffort.HIGH))

    assert len(client.response_calls) == 1
    assert sleeps == []
    assert events[-1][2:4] == (False, "http_error")
