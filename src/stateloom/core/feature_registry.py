"""Feature registry — typed, tiered feature definitions with gating."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("stateloom.core.feature_registry")


@dataclass
class Feature:
    """A registered feature with tier and enabled state."""

    name: str
    tier: str = "community"  # "community" or "enterprise"
    description: str = ""
    enabled: bool = False


class FeatureRegistry:
    """Tiered feature registry — the single source of truth for feature availability.

    Community features auto-enable on ``define()``.  Enterprise features start
    disabled and require ``provide()`` (called by EE when licensed) to become
    available.
    """

    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    # --- Core API ---

    def define(self, name: str, *, tier: str = "community", description: str = "") -> None:
        """Define a feature.  Community features auto-enable; enterprise start disabled."""
        self._features[name] = Feature(
            name=name,
            tier=tier,
            description=description,
            enabled=(tier == "community"),
        )

    def provide(self, name: str) -> None:
        """Enable an enterprise feature (called by EE when the license authorizes it)."""
        feat = self._features.get(name)
        if feat is None:
            # Unknown feature — no-op (forward-compatible with future features)
            return
        feat.enabled = True

    def is_available(self, name: str) -> bool:
        """Return True if the feature is defined AND enabled."""
        feat = self._features.get(name)
        return feat is not None and feat.enabled

    def require(self, name: str) -> None:
        """Raise ``StateLoomFeatureError`` if the feature is not available."""
        if not self.is_available(name):
            from stateloom.core.errors import StateLoomFeatureError

            raise StateLoomFeatureError(name)

    # --- Backward-compat aliases ---

    def register(self, name: str, **meta: Any) -> None:
        """Backward-compatible: define + provide in one call."""
        if name not in self._features:
            self.define(name, tier="enterprise", description=meta.get("description", ""))
        self.provide(name)

    def is_loaded(self, name: str) -> bool:
        """Backward-compatible alias for ``is_available()``."""
        return self.is_available(name)

    # --- Introspection ---

    def enterprise_feature_names(self) -> list[str]:
        """Return names of all enterprise-tier features."""
        return [f.name for f in self._features.values() if f.tier == "enterprise"]

    def status(self) -> dict[str, Any]:
        """Return status of all registered features."""
        return {
            "features": {
                name: {
                    "tier": feat.tier,
                    "enabled": feat.enabled,
                    "description": feat.description,
                }
                for name, feat in self._features.items()
            },
            "count": len(self._features),
        }
