"""Tests for pricing registry."""

from stateloom.pricing.registry import PricingRegistry


def test_known_model_pricing():
    reg = PricingRegistry()
    price = reg.get_price("gpt-4o")
    assert price is not None
    assert price.input_per_token > 0
    assert price.output_per_token > 0


def test_cost_calculation():
    reg = PricingRegistry()
    cost = reg.calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert cost > 0
    # gpt-4o: $2.50/M input, $10/M output
    expected = (1000 * 0.0000025) + (500 * 0.00001)
    assert abs(cost - expected) < 0.0001


def test_unknown_model_returns_zero():
    reg = PricingRegistry()
    cost = reg.calculate_cost("totally-unknown-model", 100, 50)
    assert cost == 0.0


def test_alias_resolution():
    reg = PricingRegistry()
    price = reg.get_price("gpt-4o-2024-11-20")
    assert price is not None
    # Should resolve to same as gpt-4o
    base_price = reg.get_price("gpt-4o")
    assert price == base_price


def test_anthropic_pricing():
    reg = PricingRegistry()
    cost = reg.calculate_cost("claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=500)
    assert cost > 0


def test_custom_model_registration():
    reg = PricingRegistry()
    reg.register("my-custom-model", input_per_token=0.001, output_per_token=0.002)
    cost = reg.calculate_cost("my-custom-model", 100, 50)
    assert cost == (100 * 0.001) + (50 * 0.002)


def test_gemini_pricing():
    reg = PricingRegistry()
    # gemini-2.0-flash: $0.10/M input, $0.40/M output
    cost = reg.calculate_cost("gemini-2.0-flash", prompt_tokens=1000, completion_tokens=500)
    expected = (1000 * 0.0000001) + (500 * 0.0000004)
    assert abs(cost - expected) < 1e-10

    # gemini-1.5-pro: $1.25/M input, $5.00/M output
    cost = reg.calculate_cost("gemini-1.5-pro", prompt_tokens=1000, completion_tokens=500)
    expected = (1000 * 0.00000125) + (500 * 0.000005)
    assert abs(cost - expected) < 1e-10

    # gemini-2.0-flash-lite is free
    cost = reg.calculate_cost("gemini-2.0-flash-lite", prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.0


def test_gemini_alias_resolution():
    reg = PricingRegistry()
    # Versioned alias should resolve to base model
    price = reg.get_price("gemini-2.0-flash-001")
    base_price = reg.get_price("gemini-2.0-flash")
    assert price is not None
    assert price == base_price

    price = reg.get_price("gemini-1.5-pro-002")
    base_price = reg.get_price("gemini-1.5-pro")
    assert price is not None
    assert price == base_price

    price = reg.get_price("gemini-1.5-flash-8b-001")
    base_price = reg.get_price("gemini-1.5-flash-8b")
    assert price is not None
    assert price == base_price
