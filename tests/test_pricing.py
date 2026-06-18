from kairos_llm.pricing import PriceTable, CostAccountant, DEFAULT_PRICE
from kairos_llm.schemas import TokenUsage


def test_default_gpt55_cost_matches_spec():
    pt = PriceTable()
    # 1M input, 0 cached, 1M output -> $5 + $30
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert round(pt.cost("gpt-5.5", usage), 2) == 35.00


def test_cached_input_is_cheaper():
    pt = PriceTable()
    full = pt.cost("gpt-5.5", TokenUsage(input_tokens=1_000_000))
    cached = pt.cost("gpt-5.5", TokenUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000))
    assert cached < full
    assert round(cached, 2) == 0.50


def test_accountant_accumulates():
    acc = CostAccountant()
    acc.record("gpt-5.5", TokenUsage(input_tokens=3000, output_tokens=800))
    acc.record("gpt-5.5-mini", TokenUsage(input_tokens=1500, output_tokens=300))
    assert acc.calls == 2
    assert round(acc.total_usd, 6) == round(sum(acc.per_model.values()), 6)
