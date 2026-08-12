"""Token pricing and running cost accounting.

Prices follow the **DeepSeek-first + GPT escalation** standard list prices from
the updated architecture document (per 1M tokens):

  * DeepSeek-V4-Flash : $0.14 in  / $0.28 out   (Text Scouts)
  * DeepSeek-V4-Pro   : $0.435 in / $0.87 out   (Aggregator-Normal)
  * GPT-5.5           : $5.00 in  / $30.00 out, $0.50 cached  (escalation)

The monthly base budget is computed without batch/priority and without cache
hits, so DeepSeek cached input is priced the same as fresh input here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import TokenUsage


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


# GPT-5.5 escalation tier (also the fallback price for unknown models).
GPT55_PRICE = ModelPrice(input_per_m=5.0, cached_input_per_m=0.5, output_per_m=30.0)
DEFAULT_PRICE = GPT55_PRICE
# DeepSeek routine tier (no cache discount assumed in the base budget).
DEEPSEEK_FLASH_PRICE = ModelPrice(input_per_m=0.14, cached_input_per_m=0.14, output_per_m=0.28)
DEEPSEEK_PRO_PRICE = ModelPrice(input_per_m=0.435, cached_input_per_m=0.435, output_per_m=0.87)

DEFAULT_PRICES: dict[str, ModelPrice] = {
    "deepseek-v4-flash": DEEPSEEK_FLASH_PRICE,
    "deepseek-v4-pro": DEEPSEEK_PRO_PRICE,
    "gpt-5.5": GPT55_PRICE,
}


class PriceTable:
    def __init__(
        self, prices: dict[str, ModelPrice] | None = None, default: ModelPrice = DEFAULT_PRICE
    ) -> None:
        self._prices = dict(DEFAULT_PRICES) if prices is None else prices
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
    per_model: dict[str, float] = field(default_factory=dict)
    calls: int = 0

    def record(self, model: str, usage: TokenUsage) -> float:
        c = self.table.cost(model, usage)
        self.total_usd += c
        self.per_model[model] = self.per_model.get(model, 0.0) + c
        self.calls += 1
        return c
