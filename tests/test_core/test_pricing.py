"""Tests for pricing registry."""

from stateloom.pricing.registry import ModelPrice, PriceTier, PricingRegistry


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


# --- Tiered pricing tests ---


def test_gemini_25_pro_tiered_pricing():
    """Gemini 2.5 Pro charges 2x input above 200K tokens."""
    reg = PricingRegistry()
    # Below threshold — uses base rates ($1.25/M input, $10/M output)
    cost_low = reg.calculate_cost("gemini-2.5-pro", prompt_tokens=100000, completion_tokens=1000)
    expected_low = (100000 * 0.00000125) + (1000 * 0.00001)
    assert abs(cost_low - expected_low) < 1e-10

    # Above threshold — uses tier rates ($2.50/M input, $10/M output)
    cost_high = reg.calculate_cost("gemini-2.5-pro", prompt_tokens=250000, completion_tokens=1000)
    expected_high = (250000 * 0.0000025) + (1000 * 0.00001)
    assert abs(cost_high - expected_high) < 1e-10

    # The high-context call should cost more
    assert cost_high > cost_low


def test_gemini_15_pro_tiered_pricing():
    """Gemini 1.5 Pro charges 2x above 128K tokens."""
    reg = PricingRegistry()
    cost_low = reg.calculate_cost("gemini-1.5-pro", prompt_tokens=50000, completion_tokens=1000)
    expected_low = (50000 * 0.00000125) + (1000 * 0.000005)
    assert abs(cost_low - expected_low) < 1e-10

    cost_high = reg.calculate_cost("gemini-1.5-pro", prompt_tokens=200000, completion_tokens=1000)
    expected_high = (200000 * 0.0000025) + (1000 * 0.00001)
    assert abs(cost_high - expected_high) < 1e-10

    assert cost_high > cost_low


def test_gemini_15_flash_tiered_pricing():
    """Gemini 1.5 Flash charges 2x above 128K tokens."""
    reg = PricingRegistry()
    cost_low = reg.calculate_cost("gemini-1.5-flash", prompt_tokens=50000, completion_tokens=1000)
    expected_low = (50000 * 0.0000000375) + (1000 * 0.00000015)
    assert abs(cost_low - expected_low) < 1e-10

    cost_high = reg.calculate_cost("gemini-1.5-flash", prompt_tokens=200000, completion_tokens=1000)
    expected_high = (200000 * 0.000000075) + (1000 * 0.0000003)
    assert abs(cost_high - expected_high) < 1e-10

    assert cost_high > cost_low


def test_tiered_pricing_exactly_at_threshold():
    """At exactly the threshold, base rates apply (tier is strictly above)."""
    reg = PricingRegistry()
    # Exactly 200K tokens — should use base rate (not tier)
    cost = reg.calculate_cost("gemini-2.5-pro", prompt_tokens=200000, completion_tokens=1000)
    expected = (200000 * 0.00000125) + (1000 * 0.00001)
    assert abs(cost - expected) < 1e-10


def test_tiered_pricing_zero_tokens():
    """Zero tokens should cost zero regardless of tiers."""
    reg = PricingRegistry()
    cost = reg.calculate_cost("gemini-2.5-pro", prompt_tokens=0, completion_tokens=0)
    assert cost == 0.0


def test_non_tiered_model_unaffected():
    """Models without tiers should work exactly as before."""
    reg = PricingRegistry()
    price = reg.get_price("gpt-4o")
    assert price is not None
    assert price.tiers == ()
    # Cost should be the same regardless of prompt size
    cost_small = reg.calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    cost_large = reg.calculate_cost("gpt-4o", prompt_tokens=500000, completion_tokens=500)
    expected_small = (1000 * 0.0000025) + (500 * 0.00001)
    expected_large = (500000 * 0.0000025) + (500 * 0.00001)
    assert abs(cost_small - expected_small) < 1e-10
    assert abs(cost_large - expected_large) < 1e-10


def test_register_with_custom_tiers():
    """register() accepts optional tiers."""
    reg = PricingRegistry()
    reg.register(
        "my-tiered-model",
        input_per_token=0.001,
        output_per_token=0.002,
        tiers=[PriceTier(above_tokens=10000, input_per_token=0.002, output_per_token=0.004)],
    )
    # Below threshold
    cost_low = reg.calculate_cost("my-tiered-model", prompt_tokens=5000, completion_tokens=100)
    assert abs(cost_low - (5000 * 0.001 + 100 * 0.002)) < 1e-10

    # Above threshold
    cost_high = reg.calculate_cost("my-tiered-model", prompt_tokens=20000, completion_tokens=100)
    assert abs(cost_high - (20000 * 0.002 + 100 * 0.004)) < 1e-10


def test_register_without_tiers_backward_compat():
    """register() without tiers still works (backward compatible)."""
    reg = PricingRegistry()
    reg.register("plain-model", input_per_token=0.001, output_per_token=0.002)
    price = reg.get_price("plain-model")
    assert price is not None
    assert price.tiers == ()
    cost = reg.calculate_cost("plain-model", prompt_tokens=100, completion_tokens=50)
    assert abs(cost - (100 * 0.001 + 50 * 0.002)) < 1e-10


def test_model_price_tiers_sorted():
    """Tiers on ModelPrice are applied in ascending order; last matching wins."""
    price = ModelPrice(
        input_per_token=0.001,
        output_per_token=0.002,
        tiers=(
            PriceTier(above_tokens=100, input_per_token=0.002, output_per_token=0.004),
            PriceTier(above_tokens=1000, input_per_token=0.003, output_per_token=0.006),
        ),
    )
    # Below first tier
    assert abs(price.calculate(50, 10) - (50 * 0.001 + 10 * 0.002)) < 1e-10
    # Above first tier, below second
    assert abs(price.calculate(500, 10) - (500 * 0.002 + 10 * 0.004)) < 1e-10
    # Above both tiers — last matching wins
    assert abs(price.calculate(5000, 10) - (5000 * 0.003 + 10 * 0.006)) < 1e-10


def test_new_models_in_registry():
    """Newly added models (o3, gpt-4.1 family, claude-haiku-4-5-20251001) are present."""
    reg = PricingRegistry()
    for model in ["o3", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "claude-haiku-4-5-20251001"]:
        price = reg.get_price(model)
        assert price is not None, f"Missing pricing for {model}"
        assert price.input_per_token >= 0
        assert price.output_per_token >= 0
