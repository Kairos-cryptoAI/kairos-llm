"""Kairos — the LLM gateway.

A single choke-point for every model call in the system. It maps a logical
``ReasoningEffort`` to a concrete model, accounts for token spend (including
cached input), retries transient errors and surfaces health signals so the Risk
Manager's circuit breaker can detach the LLM when the API misbehaves.

No layer talks to OpenAI directly — they all go through :class:`LLMGateway`.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .schemas import LLMResult, TokenUsage
from .pricing import PriceTable, CostAccountant
from .models import ModelRouter, ModelChoice, Provider
from .errors import LLMError, LLMTimeout, LLMServerError, LLMBadOutput
from .gateway import LLMGateway
from .config import LLMSettings

__all__ = [
    "LLMResult", "TokenUsage", "PriceTable", "CostAccountant", "ModelRouter",
    "ModelChoice", "Provider",
    "LLMError", "LLMTimeout", "LLMServerError", "LLMBadOutput", "LLMGateway",
    "LLMSettings", "__version__",
]
