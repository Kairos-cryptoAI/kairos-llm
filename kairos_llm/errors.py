"""Typed errors. Server/timeout errors are what trip the circuit breaker."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all gateway errors."""


class LLMTimeout(LLMError):
    """The model did not respond within the configured timeout."""


class LLMServerError(LLMError):
    """The provider returned a 5xx (e.g. 502) — counts toward the breaker."""


class LLMBadOutput(LLMError):
    """The model returned content that failed schema validation."""
