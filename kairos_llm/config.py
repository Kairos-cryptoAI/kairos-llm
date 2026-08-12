"""LLM gateway configuration."""

from __future__ import annotations

from kairos_core.config import CoreSettings


class LLMSettings(CoreSettings):
    service_name: str = "kairos-llm"

    # --- OpenAI: GPT-5.6 Sol escalation tier (conflict resolution + macro strategy) ---
    openai_api_key: str | None = None  # KAIROS_OPENAI_API_KEY (or OPENAI_API_KEY)
    openai_base_url: str | None = None  # for OpenAI-compatible gateways

    # --- DeepSeek: routine tier (Text Scouts Flash + Aggregator-Normal Pro) ---
    # DeepSeek exposes an OpenAI-compatible API, so the same AsyncOpenAI client
    # works once pointed at this base URL.
    deepseek_api_key: str | None = None  # KAIROS_DEEPSEEK_API_KEY
    deepseek_base_url: str = "https://api.deepseek.com"

    request_timeout_s: float = 20.0
    max_retries: int = 2
    max_output_tokens: int = 8_192
    # Server-side prompt caching is automatic; this just flags that our system
    # prompts are stable enough to benefit (used for accounting/estimates).
    assume_cached_system_prompt: bool = True
