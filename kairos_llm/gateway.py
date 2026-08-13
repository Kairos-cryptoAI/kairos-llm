"""Provider-aware async gateway for all Kairos model calls.

OpenAI uses the Responses API with SDK-native Pydantic parsing. DeepSeek keeps
its compatible Chat Completions endpoint, but thinking mode is disabled
explicitly for the low/medium routine tiers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kairos_core.enums import ReasoningEffort
from kairos_core.logging import get_logger
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import BaseModel, RootModel, ValidationError

from .config import LLMSettings
from .errors import LLMBadOutput, LLMServerError, LLMTimeout
from .models import LLMWorkload, ModelChoice, ModelRoute, ModelRouter, Provider
from .pricing import CostAccountant
from .schemas import LLMResult, TokenUsage

log = get_logger(__name__)


class _JsonObject(RootModel[dict[str, Any]]):
    """Fallback schema when a caller only needs an arbitrary JSON object."""


@dataclass(frozen=True, slots=True)
class _Failure:
    error: Exception
    health_kind: str
    retryable: bool


class LLMGateway:
    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        router: ModelRouter | None = None,
        accountant: CostAccountant | None = None,
        client: Any | None = None,
        on_health: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings or LLMSettings()
        self.router = router or ModelRouter()
        self.accountant = accountant or CostAccountant()
        self._injected_client = client
        self._clients: dict[Provider, Any] = {}
        self._on_health = on_health

    def _client_for(self, provider: Provider) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if provider not in self._clients:
            api_key: str | None
            base_url: str | None
            if provider is Provider.DEEPSEEK:
                api_key, base_url = self.settings.deepseek_api_key, self.settings.deepseek_base_url
            else:
                api_key, base_url = self.settings.openai_api_key, self.settings.openai_base_url
            self._clients[provider] = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.settings.request_timeout_s,
                max_retries=0,
            )
        return self._clients[provider]

    async def complete(
        self,
        *,
        system: str,
        user: str,
        effort: ReasoningEffort | None = None,
        workload: LLMWorkload | None = None,
        schema: type[BaseModel] | None = None,
    ) -> LLMResult:
        route = self.router.resolve(effort, workload=workload)
        choice = route.choice
        client = self._client_for(choice.provider)

        max_retries = max(0, self.settings.max_retries)
        for attempt in range(max_retries + 1):
            started = time.monotonic()
            try:
                async with asyncio.timeout(self.settings.request_timeout_s):
                    if choice.provider is Provider.OPENAI:
                        response = await self._complete_openai(client, choice, system, user, schema)
                        result = self._finish_openai(response, route)
                    else:
                        response = await self._complete_deepseek(client, choice, system, user)
                        result = self._finish_deepseek(response, route, schema)
                result.latency_s = time.monotonic() - started
                await self._emit_health_safely(choice, ok=True, kind="ok", latency_s=result.latency_s)
                return result
            except Exception as exc:
                failure = self._classify_failure(exc)
                latency = time.monotonic() - started
                if failure.retryable and attempt < max_retries:
                    # The SDK's own retries are disabled. Only the explicit transient
                    # allow-list below may cause another potentially billable request.
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue

                await self._emit_health_safely(
                    choice,
                    ok=False,
                    kind=failure.health_kind,
                    latency_s=latency,
                )
                if failure.error is exc:
                    raise
                raise failure.error from exc

        raise LLMServerError("model request loop exited unexpectedly")  # pragma: no cover

    @classmethod
    def _classify_failure(cls, exc: Exception) -> _Failure:
        """Classify failures conservatively so permanent errors never spend retry budget."""
        if isinstance(exc, (TimeoutError, APITimeoutError)):
            return _Failure(
                LLMTimeout(str(exc) or "model request timed out"),
                health_kind="timeout",
                retryable=True,
            )
        if isinstance(exc, APIConnectionError):
            return _Failure(
                LLMServerError(str(exc) or "provider connection failed"),
                # Keep provider-wide connectivity separate from a slow model.
                # Risk aggregates this signal across Luna/Terra/Sol.
                health_kind="connection",
                retryable=True,
            )
        if isinstance(exc, LLMBadOutput):
            return _Failure(exc, health_kind="bad_output", retryable=False)
        if isinstance(exc, (APIResponseValidationError, ValidationError, json.JSONDecodeError)):
            return _Failure(LLMBadOutput(str(exc)), health_kind="bad_output", retryable=False)

        if isinstance(exc, APIStatusError):
            return cls._classify_http_failure(exc, cls._status_code(exc))
        return _Failure(exc, health_kind="error", retryable=False)

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        raw_status = getattr(exc, "status_code", None)
        if raw_status is None:
            raw_status = getattr(exc, "status", None)
        try:
            return int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _classify_http_failure(exc: Exception, status: int | None) -> _Failure:
        if status == 408:
            return _Failure(
                LLMTimeout(str(exc) or "provider request timed out"),
                health_kind="timeout",
                retryable=True,
            )
        if status == 409:
            return _Failure(exc, health_kind="conflict", retryable=True)
        if status == 429:
            return _Failure(exc, health_kind="rate_limit", retryable=True)
        if status is not None and 500 <= status < 600:
            return _Failure(
                LLMServerError(str(exc) or f"provider returned HTTP {status}"),
                health_kind="5xx",
                retryable=True,
            )
        if status is not None and 400 <= status < 500:
            return _Failure(exc, health_kind="http_4xx", retryable=False)
        return _Failure(exc, health_kind="http_error", retryable=False)

    async def _complete_openai(
        self,
        client: Any,
        choice: ModelChoice,
        system: str,
        user: str,
        schema: type[BaseModel] | None,
    ) -> Any:
        return await client.responses.parse(
            model=choice.model,
            instructions=system,
            input=user,
            text_format=schema or _JsonObject,
            reasoning={"effort": choice.provider_effort},
            max_output_tokens=self.settings.max_output_tokens,
            store=False,
        )

    async def _complete_deepseek(self, client: Any, choice: ModelChoice, system: str, user: str) -> Any:
        return await client.chat.completions.create(
            model=choice.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=self.settings.max_output_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def _finish_openai(self, response: Any, route: ModelRoute) -> LLMResult:
        status = getattr(response, "status", "completed")
        if status != "completed":
            raise LLMBadOutput(f"OpenAI response did not complete: {status}")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMBadOutput("OpenAI response contained no parsed output")
        parsed_value = parsed.root if isinstance(parsed, _JsonObject) else parsed
        content = getattr(response, "output_text", "") or self._content_from(parsed_value)
        return self._result(content, parsed_value, response, route)

    def _finish_deepseek(
        self,
        response: Any,
        route: ModelRoute,
        schema: type[BaseModel] | None,
    ) -> LLMResult:
        content = response.choices[0].message.content or ""
        if not content:
            raise LLMBadOutput("DeepSeek response contained no content")
        try:
            data = json.loads(content)
            parsed = schema.model_validate(data) if schema else data
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMBadOutput(str(exc)) from exc
        return self._result(content, parsed, response, route)

    def _result(
        self,
        content: str,
        parsed: Any,
        response: Any,
        route: ModelRoute,
    ) -> LLMResult:
        choice = route.choice
        usage = self._usage_from(response)
        cost = self.accountant.record(choice.model, usage)
        resolved_model = self._optional_string(getattr(response, "model", None))
        system_fingerprint = self._optional_string(getattr(response, "system_fingerprint", None))
        log.info(
            "llm.response",
            provider=choice.provider.value,
            requested_model=choice.model,
            resolved_model=resolved_model,
            system_fingerprint=system_fingerprint,
            workload=route.workload.value if route.workload else None,
        )
        return LLMResult(
            content=content,
            parsed=parsed,
            model=choice.model,
            effort=route.effort.value,
            usage=usage,
            cost_usd=cost,
            cached=usage.cached_input_tokens > 0,
            workload=route.workload.value if route.workload else None,
            resolved_model=resolved_model,
            system_fingerprint=system_fingerprint,
        )

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        return rendered or None

    @staticmethod
    def _content_from(parsed: Any) -> str:
        if isinstance(parsed, BaseModel):
            return parsed.model_dump_json()
        return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)

    async def _emit_health_safely(
        self, choice: ModelChoice, *, ok: bool, kind: str, latency_s: float
    ) -> None:
        """Health telemetry is best-effort and never changes inference semantics."""
        if self._on_health is None:
            return
        try:
            result = self._on_health(choice.model, choice.provider.value, ok, kind, latency_s)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - logging is the fallback sink
            log.warning("llm.health_sink_failed", error=str(exc), model=choice.model)

    @staticmethod
    def _usage_from(response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsage()
        details = getattr(usage, "input_tokens_details", None) or getattr(
            usage, "prompt_tokens_details", None
        )
        cached = getattr(details, "cached_tokens", 0) if details else 0
        return TokenUsage(
            input_tokens=int(getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0),
            cached_input_tokens=int(cached or 0),
            output_tokens=int(
                getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
            ),
        )

    async def close(self) -> None:
        """Close provider clients created by this gateway."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
