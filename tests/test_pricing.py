from kairos_llm.pricing import CostAccountant, PriceTable
from kairos_llm.schemas import TokenUsage


def test_default_gpt56_cost_matches_spec():
    pt = PriceTable()
    # 1M input, 0 cached, 1M output -> $5 + $30
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert round(pt.cost("gpt-5.6-sol", usage), 2) == 35.00


def test_cached_input_is_cheaper():
    pt = PriceTable()
    full = pt.cost("gpt-5.6-sol", TokenUsage(input_tokens=1_000_000))
    cached = pt.cost("gpt-5.6-sol", TokenUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000))
    assert cached < full
    assert round(cached, 2) == 0.50


def test_deepseek_tier_prices():
    pt = PriceTable()
    assert round(pt.cost("deepseek-v4-flash", TokenUsage(input_tokens=1_000_000)), 3) == 0.140
    assert round(pt.cost("deepseek-v4-flash", TokenUsage(output_tokens=1_000_000)), 3) == 0.280
    assert round(pt.cost("deepseek-v4-pro", TokenUsage(input_tokens=1_000_000)), 3) == 0.435
    assert round(pt.cost("deepseek-v4-pro", TokenUsage(output_tokens=1_000_000)), 3) == 0.870
    assert (
        round(
            pt.cost(
                "deepseek-v4-flash",
                TokenUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000),
            ),
            4,
        )
        == 0.0028
    )


def test_monthly_api_budget_matches_doc():
    """Reproduce the $138.62 monthly API budget from the architecture document."""
    pt = PriceTable()
    flash = 14_400 * pt.cost("deepseek-v4-flash", TokenUsage(input_tokens=1_500, output_tokens=300))
    pro = 7_340 * pt.cost("deepseek-v4-pro", TokenUsage(input_tokens=3_000, output_tokens=800))
    conflict = 1_300 * pt.cost("gpt-5.6-sol", TokenUsage(input_tokens=3_000, output_tokens=2_200))
    macro = 60 * pt.cost("gpt-5.6-sol", TokenUsage(input_tokens=15_000, output_tokens=5_500))
    assert round(flash + pro + conflict + macro, 2) == 138.62


def test_accountant_accumulates():
    acc = CostAccountant()
    acc.record("gpt-5.6-sol", TokenUsage(input_tokens=3000, output_tokens=800))
    acc.record("deepseek-v4-flash", TokenUsage(input_tokens=1500, output_tokens=300))
    assert acc.calls == 2
    assert round(acc.total_usd, 6) == round(sum(acc.per_model.values()), 6)
