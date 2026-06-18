"""Token pricing and running cost accounting.

Default prices follow the GPT-5.5 tariff from the system spec:
  * input              $5.00  / 1M tokens
  * cached input       $0.50  / 1M tokens
  * output (incl. reasoning) $30.00 / 1M tokens
Override per-model via :class:`PriceTable`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .schemas import TokenUsage


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


DEFAULT_PRICE = ModelPrice(input_per_m=5.0, cached_input_per_m=0.5, output_per_m=30.0)
# Cheap models used for the routine flow in the cost-optimised configuration.
MINI_PRICE = ModelPrice(input_per_m=0.15, cached_input_per_m=0.075, output_per_m=0.60)


class PriceTable:
    def __init__(self, prices: Dict[str, ModelPrice] | None = None, default: ModelPrice = DEFAULT_PRICE) -> None:
        self._prices = prices or {}
        self._default = default

    def for_model(self, model: str) -> ModelPrice:
        return self._prices.get(model, self._default)

    def cost(self, model: str, usage: TokenUsage) -> float:
        p = self.for_model(model)
        return (
            usage.billable_input / 1e6 * p.input_per_m
            + usage.cached_input_tokens / 1e6 * p.cached_input_per_m
            + usage.output_tokens / 1e6 * p.output_per_m
        )


@dataclass
class CostAccountant:
    """Tracks cumulative spend, broken down per model — handy for the budget alerts."""

    table: PriceTable = field(default_factory=PriceTable)
    total_usd: float = 0.0
    per_model: Dict[str, float] = field(default_factory=dict)
    calls: int = 0

    def record(self, model: str, usage: TokenUsage) -> float:
        c = self.table.cost(model, usage)
        self.total_usd += c
        self.per_model[model] = self.per_model.get(model, 0.0) + c
        self.calls += 1
        return c
