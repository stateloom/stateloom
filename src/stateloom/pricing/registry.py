"""Model pricing registry for cost calculation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("stateloom.pricing")

_PRICES_FILE = Path(__file__).parent / "data" / "prices.json"


@dataclass(frozen=True)
class PriceTier:
    """Context-length pricing tier (e.g. Gemini 2x rates above 128K tokens)."""

    above_tokens: int
    input_per_token: float
    output_per_token: float


@dataclass(frozen=True)
class ModelPrice:
    """Price per token for a model, with optional context-length tiers."""

    input_per_token: float
    output_per_token: float
    tiers: tuple[PriceTier, ...] = ()

    def calculate(self, prompt_tokens: int, completion_tokens: int) -> float:
        inp, out = self.input_per_token, self.output_per_token
        for tier in self.tiers:  # sorted ascending by above_tokens
            if prompt_tokens > tier.above_tokens:
                inp, out = tier.input_per_token, tier.output_per_token
        return (prompt_tokens * inp) + (completion_tokens * out)


class PricingRegistry:
    """Lookup model pricing. Falls back to zero for unknown models."""

    def __init__(self) -> None:
        self._prices: dict[str, ModelPrice] = {}
        self._aliases: dict[str, str] = {}
        self._load_bundled()

    def _load_bundled(self) -> None:
        """Load bundled prices.json."""
        try:
            with open(_PRICES_FILE) as f:
                data = json.load(f)

            for model_id, prices in data.get("models", {}).items():
                tiers_data = prices.get("tiers", [])
                tiers = tuple(
                    PriceTier(t["above_tokens"], t["input"], t["output"])
                    for t in sorted(tiers_data, key=lambda t: t["above_tokens"])
                )
                self._prices[model_id] = ModelPrice(
                    input_per_token=prices["input"],
                    output_per_token=prices["output"],
                    tiers=tiers,
                )

            self._aliases = data.get("aliases", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load bundled pricing: {e}")

    def get_price(self, model: str) -> ModelPrice | None:
        """Get pricing for a model. Resolves aliases. Returns None if unknown."""
        # Direct lookup
        if model in self._prices:
            return self._prices[model]

        # Alias lookup
        canonical = self._aliases.get(model)
        if canonical and canonical in self._prices:
            return self._prices[canonical]

        # Fuzzy prefix match (e.g. "gpt-4o-2024-12-01" → "gpt-4o")
        for known_model in self._prices:
            if model.startswith(known_model):
                return self._prices[known_model]

        logger.debug(f"No pricing found for model: {model}")
        return None

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a call. Returns 0.0 for unknown models."""
        price = self.get_price(model)
        if price is None:
            return 0.0
        return price.calculate(prompt_tokens, completion_tokens)

    def register(
        self,
        model: str,
        input_per_token: float,
        output_per_token: float,
        tiers: list[PriceTier] | None = None,
    ) -> None:
        """Register custom pricing for a model."""
        self._prices[model] = ModelPrice(
            input_per_token=input_per_token,
            output_per_token=output_per_token,
            tiers=tuple(sorted(tiers or [], key=lambda t: t.above_tokens)),
        )
