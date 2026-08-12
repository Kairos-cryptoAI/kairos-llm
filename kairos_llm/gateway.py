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
from typing import Any

from kairos_core.enums import ReasoningEffort
from kairos_core.logging import get_logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, RootModel, ValidationError

from .config import LLMSettings
from .errors import LLMBadOutput, LLMServerError, LLMTimeout
from .models import ModelChoice, ModelRouter, Provider
from .pricing import CostAccountant
from .schemas import LLMResult, TokenUsage

log = get_logger(__name__)


class _JsonObject(RootModel[dict[str, Any]]):
    """Fallback schema when a caller only needs an arbitrary JSON object."""


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
        effort: ReasoningEffort,
        schema: type[BaseModel] | None = None,
    ) -> LLMResult:
        choice = self.router.choose(effort)
        client = self._client_for(choice.provider)
        last_exc: Exception | None = None
        last_kind = "error"
        last_latency = 0.0

        for attempt in range(self.settings.max_retries + 1):
            started = time.monotonic()
            try:
                async with asyncio.timeout(self.settings.request_timeout_s):
                    if choice.provider is Provider.OPENAI:
                        response = await self._complete_openai(client, choice, system, user, schema)
                        result = self._finish_openai(response, choice, effort)
                    else:
                        response = await self._complete_deepseek(client, choice, system, user)
                        result = self._finish_deepseek(response, choice, effort, schema)
                result.latency_s = time.monotonic() - started
                await self._emit_health_safely(choice, ok=True, kind="ok", latency_s=result.latency_s)
                return result
            except (TimeoutError, APITimeoutError) as exc:
                last_exc = LLMTimeout(str(exc) or "model request timed out")
                last_kind = "timeout"
            except APIConnectionError as exc:
                last_exc = LLMServerError(str(exc))
                last_kind = "connection"
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = LLMServerError(str(exc))
                    last_kind = "5xx"
                else:
                    last_exc = exc
                    last_kind = "http_4xx"
            except LLMBadOutput as exc:
                last_exc = exc
                last_kind = "bad_output"
            except Exception as exc:  # provider-compatible clients may use custom errors
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                if status and int(status) >= 500:
                    last_exc = LLMServerError(str(exc))
                    last_kind = "5xx"
                else:
                    last_exc = exc
                    last_kind = "error"

            last_latency = time.monotonic() - started
            if attempt < self.settings.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        await self._emit_health_safely(choice, ok=False, kind=last_kind, latency_s=last_latency)
        raise last_exc if last_exc else LLMServerError("unknown model error")

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

    def _finish_openai(self, response: Any, choice: ModelChoice, effort: ReasoningEffort) -> LLMResult:
        status = getattr(response, "status", "completed")
        if status != "completed":
            raise LLMBadOutput(f"OpenAI response did not complete: {status}")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMBadOutput("OpenAI response contained no parsed output")
        parsed_value = parsed.root if isinstance(parsed, _JsonObject) else parsed
        content = getattr(response, "output_text", "") or self._content_from(parsed_value)
        return self._result(content, parsed_value, response, choice, effort)

    def _finish_deepseek(
        self,
        response: Any,
        choice: ModelChoice,
        effort: ReasoningEffort,
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
        return self._result(content, parsed, response, choice, effort)

    def _result(
        self,
        content: str,
        parsed: Any,
        response: Any,
        choice: ModelChoice,
        effort: ReasoningEffort,
    ) -> LLMResult:
        usage = self._usage_from(response)
        cost = self.accountant.record(choice.model, usage)
        return LLMResult(
            content=content,
            parsed=parsed,
            model=choice.model,
            effort=effort.value,
            usage=usage,
            cost_usd=cost,
            cached=usage.cached_input_tokens > 0,
        )

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
