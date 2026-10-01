"""Token pricing and running cost accounting.

Prices follow the workload-aware routing architecture (per 1M tokens).
Current DeepSeek Flash is recorded at its peak rate so reservations and spend
alerts remain conservative regardless of dispatch time:

  * DeepSeek-V4.1-Flash: $0.30 in / $0.006 cached / $1.20 out (peak)
  * DeepSeek-V4-Pro   : $1.32 in / $0.044 cached / $3.96 out (peak)
  * GPT-6 Luna        : $0.10 in / $0.01 cached / $0.125 cache write / $0.50 out
  * GPT-6 Sol         : $2.00 in / $0.20 cached / $2.50 cache write / $10.00 out
  * GPT-6.1 Sol       : $2.00 in / $0.10 cached / $2.50 cache write / $10.00 out
  * GPT-5.6 Luna      : $0.20 in / $0.02 cached / $1.20 out
  * GPT-5.6 Terra     : $2.00 in / $0.20 cached / $12.00 out
  * GPT-5.6 Sol       : $4.00 in / $0.40 cached / $20.00 out

The monthly base budget is computed without batch/priority and assumes zero
cache hits, so every input token in that estimate uses the cache-miss rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import TokenUsage


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float
    cache_write_per_m: float | None = None

    @property
    def effective_cache_write_per_m(self) -> float:
        return self.input_per_m if self.cache_write_per_m is None else self.cache_write_per_m


GPT56_LUNA_PRICE = ModelPrice(input_per_m=0.2, cached_input_per_m=0.02, output_per_m=1.2)
GPT56_TERRA_PRICE = ModelPrice(input_per_m=2.0, cached_input_per_m=0.2, output_per_m=12.0)
GPT56_SOL_PRICE = ModelPrice(input_per_m=4.0, cached_input_per_m=0.4, output_per_m=20.0)
GPT6_LUNA_PRICE = ModelPrice(
    input_per_m=0.10,
    cached_input_per_m=0.01,
    output_per_m=0.50,
    cache_write_per_m=0.125,
)
GPT6_SOL_PRICE = ModelPrice(
    input_per_m=2.00,
    cached_input_per_m=0.20,
    output_per_m=10.00,
    cache_write_per_m=2.50,
)
GPT61_SOL_PRICE = ModelPrice(
    input_per_m=2.00,
    cached_input_per_m=0.10,
    output_per_m=10.00,
    cache_write_per_m=2.50,
)
# Preserve the public constant and conservative unknown-model fallback.
GPT56_PRICE = GPT56_SOL_PRICE
DEFAULT_PRICE = GPT56_SOL_PRICE
DEEPSEEK_FLASH_PRICE = ModelPrice(input_per_m=0.30, cached_input_per_m=0.006, output_per_m=1.20)
DEEPSEEK_PRO_PRICE = ModelPrice(input_per_m=1.32, cached_input_per_m=0.044, output_per_m=3.96)

DEFAULT_PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": DEEPSEEK_FLASH_PRICE,
    # The legacy alias now resolves to V4.1 Flash and is billed at its rate.
    "deepseek-v4-flash": DEEPSEEK_FLASH_PRICE,
    "deepseek-v4-pro": DEEPSEEK_PRO_PRICE,
    "gpt-6-luna": GPT6_LUNA_PRICE,
    "gpt-6-sol": GPT6_SOL_PRICE,
    "gpt-6.1-sol": GPT61_SOL_PRICE,
    "gpt-5.6-luna": GPT56_LUNA_PRICE,
    "gpt-5.6-terra": GPT56_TERRA_PRICE,
    "gpt-5.6": GPT56_SOL_PRICE,
    "gpt-5.6-sol": GPT56_SOL_PRICE,
}


class PriceTable:
    def __init__(
        self, prices: dict[str, ModelPrice] | None = None, default: ModelPrice = DEFAULT_PRICE
    ) -> None:
        self._prices = dict(DEFAULT_PRICES) if prices is None else prices
        self._default = default

    def for_model(self, model: str) -> ModelPrice:
        return self._prices.get(model, self._default)

    def has_model(self, model: str) -> bool:
        return model in self._prices

    @staticmethod
    def _validate_usage(usage: TokenUsage) -> None:
        if (
            usage.input_tokens < 0
            or usage.cached_input_tokens < 0
            or usage.cache_write_tokens < 0
            or usage.output_tokens < 0
            or usage.cached_input_tokens + usage.cache_write_tokens > usage.input_tokens
        ):
            raise ValueError("invalid token usage breakdown")

    def cost(self, model: str, usage: TokenUsage) -> float:
        self._validate_usage(usage)
        p = self.for_model(model)
        return (
            (usage.billable_input - usage.cache_write_tokens) / 1e6 * p.input_per_m
            + usage.cached_input_tokens / 1e6 * p.cached_input_per_m
            + usage.cache_write_tokens / 1e6 * p.effective_cache_write_per_m
            + usage.output_tokens / 1e6 * p.output_per_m
        )

    def reservation_cost(self, model: str, usage: TokenUsage) -> float:
        """Reserve as if every input token incurred the highest input rate."""
        self._validate_usage(usage)
        p = self.for_model(model)
        return (
            usage.input_tokens / 1e6 * max(p.input_per_m, p.effective_cache_write_per_m)
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
