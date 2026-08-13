"""Kairos — the LLM gateway.

A single choke-point for every model call in the system. It maps a logical
``ReasoningEffort`` to a concrete model, accounts for token spend (including
cached input), retries transient errors and surfaces health signals so the Risk
Manager's circuit breaker can detach the LLM when the API misbehaves.

No layer talks to OpenAI directly — they all go through :class:`LLMGateway`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import LLMSettings
from .errors import LLMBadOutput, LLMError, LLMServerError, LLMTimeout
from .gateway import LLMGateway
from .models import ModelChoice, ModelRouter, Provider
from .pricing import CostAccountant, PriceTable
from .schemas import LLMResult, TokenUsage

__all__ = [
    "LLMResult",
    "TokenUsage",
    "PriceTable",
    "CostAccountant",
    "ModelRouter",
    "ModelChoice",
    "Provider",
    "LLMError",
    "LLMTimeout",
    "LLMServerError",
    "LLMBadOutput",
    "LLMGateway",
    "LLMSettings",
    "__version__",
]
