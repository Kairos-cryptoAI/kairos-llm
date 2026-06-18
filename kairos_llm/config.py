"""LLM gateway configuration."""
from __future__ import annotations

from typing import Optional

from kairos_core.config import CoreSettings


class LLMSettings(CoreSettings):
    service_name: str = "kairos-llm"

    openai_api_key: Optional[str] = None          # KAIROS_OPENAI_API_KEY (or OPENAI_API_KEY)
    openai_base_url: Optional[str] = None          # for OpenAI-compatible gateways
    request_timeout_s: float = 20.0
    max_retries: int = 2
    # Server-side prompt caching is automatic; this just flags that our system
    # prompts are stable enough to benefit (used for accounting/estimates).
    assume_cached_system_prompt: bool = True
