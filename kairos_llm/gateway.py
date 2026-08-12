"""The async gateway every layer uses to call an LLM.

Responsibilities:
  * pick the provider + model from the reasoning effort,
  * enforce a hard timeout and a small retry budget,
  * parse + (optionally) validate JSON output,
  * account for token spend,
  * raise typed errors so the Risk Manager breaker can react to 5xx/timeouts.

DeepSeek-first: routine calls go to DeepSeek (Flash/Pro) as *non-thinking*
requests — the ``reasoning_effort`` parameter is omitted for them and only sent
for the GPT-5.5 escalation tier.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from typing import Any

from kairos_core.enums import ReasoningEffort
from pydantic import BaseModel, ValidationError

from .config import LLMSettings
from .errors import LLMBadOutput, LLMServerError, LLMTimeout
from .models import ModelRouter, Provider
from .pricing import CostAccountant
from .schemas import LLMResult, TokenUsage


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
        self._injected_client = client  # tests inject one client used for every provider
        self._clients: dict[Provider, Any] = {}  # lazily created, one per provider
        self._on_health = on_health  # callback(model, provider, ok, kind, latency_s)

    def _client_for(self, provider: Provider):
        if self._injected_client is not None:
            return self._injected_client
        if provider not in self._clients:
            from openai import AsyncOpenAI  # imported lazily so tests need no key

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
                max_retries=0,  # we own the retry loop
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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            t0 = time.monotonic()
            try:
                kwargs: dict[str, Any] = dict(
                    model=choice.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                # Only the GPT-5.5 escalation tier is a thinking model; DeepSeek
                # Flash/Pro run non-thinking and must NOT receive this parameter.
                if choice.send_reasoning_effort:
                    kwargs["reasoning_effort"] = choice.provider_effort
                resp = await client.chat.completions.create(**kwargs)
                latency = time.monotonic() - t0
                await self._emit_health(choice, ok=True, kind="ok", latency_s=latency)
                return self._finish(resp, choice, effort, latency, schema)
            except TimeoutError as exc:  # pragma: no cover - network
                last_exc = LLMTimeout(str(exc))
            except Exception as exc:  # pragma: no cover - network
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                last_exc = LLMServerError(str(exc)) if status and int(status) >= 500 else exc
            await asyncio.sleep(0.5 * (attempt + 1))
        await self._emit_health(choice, ok=False, kind=self._failure_kind(last_exc), latency_s=0.0)
        raise last_exc if last_exc else LLMServerError("unknown error")

    def _finish(self, resp, choice, effort, latency, schema) -> LLMResult:
        content = resp.choices[0].message.content or "{}"
        usage = self._usage_from(resp)
        cost = self.accountant.record(choice.model, usage)
        parsed: Any = None
        try:
            data = json.loads(content)
            parsed = schema.model_validate(data) if schema else data
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMBadOutput(str(exc)) from exc
        return LLMResult(
            content=content,
            parsed=parsed,
            model=choice.model,
            effort=effort.value,
            usage=usage,
            cost_usd=cost,
            latency_s=latency,
            cached=usage.cached_input_tokens > 0,
        )

    async def _emit_health(self, choice, *, ok: bool, kind: str, latency_s: float) -> None:
        """Notify the optional health sink; supports sync or async callbacks."""
        if self._on_health is None:
            return
        result = self._on_health(choice.model, choice.provider.value, ok, kind, latency_s)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _failure_kind(exc) -> str:
        if isinstance(exc, LLMTimeout):
            return "timeout"
        if isinstance(exc, LLMServerError):
            return "5xx"
        return "error"

    @staticmethod
    def _usage_from(resp) -> TokenUsage:
        u = getattr(resp, "usage", None)
        if not u:
            return TokenUsage()
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        return TokenUsage(
            input_tokens=getattr(u, "prompt_tokens", 0),
            cached_input_tokens=cached or 0,
            output_tokens=getattr(u, "completion_tokens", 0),
        )
