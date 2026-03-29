"""End-to-end integration tests for custom provider registration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import stateloom
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget
from stateloom.intercept.provider_registry import clear_adapters, get_adapter


class _TestProviderAdapter(BaseProviderAdapter):
    """A fake provider adapter used in integration tests."""

    @property
    def name(self) -> str:
        return "test-provider"

    @property
    def method_label(self) -> str:
        return "test.generate"

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        try:
            return (response.prompt_t, response.completion_t, response.total_t)
        except AttributeError:
            return (0, 0, 0)

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        kwargs["system_instruction"] = prompt


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stateloom.shutdown()
    clear_adapters()


class TestRegisterProvider:
    def test_register_before_init(self):
        """Adapter is available in the registry after register_provider()."""
        adapter = _TestProviderAdapter()
        stateloom.register_provider(adapter)
        assert get_adapter("test-provider") is adapter

    def test_deferred_pricing_applied_after_init(self):
        """Pricing registered before init() is flushed into the pricing registry."""
        stateloom.register_provider(
            _TestProviderAdapter(),
            pricing={"test-model": (0.000001, 0.000002)},
        )
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )
        cost = gate.pricing.calculate_cost("test-model", 1000, 500)
        # 1000 * 0.000001 + 500 * 0.000002 = 0.001 + 0.001 = 0.002
        assert abs(cost - 0.002) < 1e-9

    def test_register_after_init_pricing_immediate(self):
        """Pricing registered after init() is applied immediately."""
        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )
        stateloom.register_provider(
            _TestProviderAdapter(),
            pricing={"late-model": (0.000003, 0.000006)},
        )
        cost = gate.pricing.calculate_cost("late-model", 1000, 1000)
        # 1000 * 0.000003 + 1000 * 0.000006 = 0.003 + 0.006 = 0.009
        assert abs(cost - 0.009) < 1e-9


class TestCustomProviderInExperiment:
    def test_system_prompt_uses_adapter(self):
        """ExperimentMiddleware delegates system prompt to the custom adapter."""
        from stateloom.middleware.experiment import ExperimentMiddleware

        stateloom.register_provider(_TestProviderAdapter())
        stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )

        middleware = ExperimentMiddleware()

        # Build a context that simulates the custom provider
        from stateloom.core.config import StateLoomConfig
        from stateloom.core.session import Session
        from stateloom.middleware.base import MiddlewareContext

        session = Session(id="test-exp")
        session.metadata["experiment_variant_config"] = {
            "request_overrides": {"system_prompt": "Be concise."},
        }
        ctx = MiddlewareContext(
            session=session,
            config=StateLoomConfig(auto_patch=False, dashboard=False, store_backend="memory"),
            provider="test-provider",
            model="test-model",
            request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
        )

        middleware._apply_overrides(ctx)
        assert ctx.request_kwargs["system_instruction"] == "Be concise."


class TestCustomProviderCostTracker:
    def test_cost_tracker_uses_adapter_tokens(self):
        """CostTracker delegates token extraction to the custom adapter."""
        from stateloom.intercept.provider_registry import register_adapter

        register_adapter(_TestProviderAdapter())

        from stateloom.middleware.cost_tracker import CostTracker
        from stateloom.pricing.registry import PricingRegistry

        pricing = PricingRegistry()
        tracker = CostTracker(pricing)

        from stateloom.core.config import StateLoomConfig
        from stateloom.core.session import Session
        from stateloom.middleware.base import MiddlewareContext

        ctx = MiddlewareContext(
            session=Session(id="ct-test"),
            config=StateLoomConfig(auto_patch=False, dashboard=False, store_backend="memory"),
            provider="test-provider",
            model="test-model",
            response=SimpleNamespace(prompt_t=100, completion_t=50, total_t=150),
        )

        pt, ct = tracker._extract_tokens(ctx)
        assert pt == 100
        assert ct == 50
