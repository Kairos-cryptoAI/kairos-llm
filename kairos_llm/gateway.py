"""The async gateway every layer uses to call an LLM.

Responsibilities:
  * pick the model from the reasoning effort,
  * enforce a hard timeout and a small retry budget,
  * parse + (optionally) validate JSON output,
  * account for token spend,
  * raise typed errors so the Risk Manager breaker can react to 5xx/timeouts.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, List, Optional, Type

from pydantic import BaseModel, ValidationError

from kairos_core.enums import ReasoningEffort

from .config import LLMSettings
from .errors import LLMBadOutput, LLMServerError, LLMTimeout
from .models import ModelRouter
from .pricing import CostAccountant
from .schemas import LLMResult, TokenUsage


class LLMGateway:
    def __init__(self, settings: LLMSettings | None = None, *, router: ModelRouter | None = None,
                 accountant: CostAccountant | None = None, client: Any | None = None) -> None:
        self.settings = settings or LLMSettings()
        self.router = router or ModelRouter()
        self.accountant = accountant or CostAccountant()
        self._client = client  # injectable for tests; lazily created otherwise

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # imported lazily so tests need no key

            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                timeout=self.settings.request_timeout_s,
                max_retries=0,  # we own the retry loop
            )
        return self._client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        effort: ReasoningEffort,
        schema: Optional[Type[BaseModel]] = None,
    ) -> LLMResult:
        choice = self.router.choose(effort)
        client = self._ensure_client()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        last_exc: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(
                    model=choice.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    reasoning_effort=choice.provider_effort,
                )
                latency = time.monotonic() - t0
                return self._finish(resp, choice, effort, latency, schema)
            except asyncio.TimeoutError as exc:  # pragma: no cover - network
                last_exc = LLMTimeout(str(exc))
            except Exception as exc:  # pragma: no cover - network
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                last_exc = LLMServerError(str(exc)) if status and int(status) >= 500 else exc
            await asyncio.sleep(0.5 * (attempt + 1))
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
            raise LLMBadOutput(str(exc))
        return LLMResult(content=content, parsed=parsed, model=choice.model,
                         effort=effort.value, usage=usage, cost_usd=cost,
                         latency_s=latency, cached=usage.cached_input_tokens > 0)

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
