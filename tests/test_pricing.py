import pytest

from kairos_llm.pricing import CostAccountant, PriceTable
from kairos_llm.schemas import TokenUsage


@pytest.mark.parametrize(
    ("model", "input_cost", "cached_cost", "output_cost"),
    [
        ("gpt-5.6-luna", 0.20, 0.02, 1.20),
        ("gpt-5.6-terra", 2.00, 0.20, 12.00),
        ("gpt-5.6-sol", 4.00, 0.40, 20.00),
    ],
)
def test_gpt56_tier_prices(model, input_cost, cached_cost, output_cost):
    table = PriceTable()

    assert table.cost(model, TokenUsage(input_tokens=1_000_000)) == input_cost
    assert table.cost(model, TokenUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000)) == cached_cost
    assert table.cost(model, TokenUsage(output_tokens=1_000_000)) == output_cost


def test_deepseek_tier_prices_use_conservative_peak_rates():
    table = PriceTable()
    assert table.cost("deepseek-v4-flash", TokenUsage(input_tokens=1_000_000)) == 0.440
    assert table.cost("deepseek-v4-flash", TokenUsage(output_tokens=1_000_000)) == 1.320
    assert table.cost("deepseek-v4-pro", TokenUsage(input_tokens=1_000_000)) == 1.320
    assert table.cost("deepseek-v4-pro", TokenUsage(output_tokens=1_000_000)) == 3.960
    assert (
        table.cost(
            "deepseek-v4-flash",
            TokenUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000),
        )
        == 0.014
    )


def test_monthly_role_route_scenario():
    """Reproduce the conservative peak-price monthly scenario from the README."""
    table = PriceTable()
    text = 14_400 * table.cost("deepseek-v4-flash", TokenUsage(input_tokens=1_500, output_tokens=300))
    normal = 7_340 * table.cost("gpt-5.6-luna", TokenUsage(input_tokens=3_000, output_tokens=800))
    conflict = 1_300 * table.cost("gpt-5.6-terra", TokenUsage(input_tokens=3_000, output_tokens=2_200))
    macro = 60 * table.cost("gpt-5.6-sol", TokenUsage(input_tokens=15_000, output_tokens=5_500))
    assert round(text + normal + conflict + macro, 2) == 78.98


def test_accountant_accumulates():
    accountant = CostAccountant()
    accountant.record("gpt-5.6-terra", TokenUsage(input_tokens=3_000, output_tokens=800))
    accountant.record("deepseek-v4-flash", TokenUsage(input_tokens=1_500, output_tokens=300))
    assert accountant.calls == 2
    assert round(accountant.total_usd, 6) == round(sum(accountant.per_model.values()), 6)
