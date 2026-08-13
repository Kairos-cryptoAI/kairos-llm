"""Kairos — the LLM gateway.

A single choke-point for every model call in the system. It maps explicit
analytical workloads (with an effort-only compatibility path) to concrete
models, accounts for token spend, retries transient errors and surfaces health
signals so the Risk Manager can detach an unhealthy provider.

No layer talks to OpenAI directly — they all go through :class:`LLMGateway`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import LLMSettings
from .errors import LLMBadOutput, LLMError, LLMServerError, LLMTimeout
from .gateway import LLMGateway
from .models import LLMWorkload, ModelChoice, ModelRoute, ModelRouter, Provider
from .pricing import CostAccountant, PriceTable
from .schemas import LLMResult, TokenUsage

__all__ = [
    "LLMResult",
    "TokenUsage",
    "PriceTable",
    "CostAccountant",
    "ModelRouter",
    "ModelChoice",
    "ModelRoute",
    "LLMWorkload",
    "Provider",
    "LLMError",
    "LLMTimeout",
    "LLMServerError",
    "LLMBadOutput",
    "LLMGateway",
    "LLMSettings",
    "__version__",
]
