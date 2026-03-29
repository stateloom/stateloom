"""End-to-end tests for feature gating — FastAPI dependency, middleware, and service init."""

from __future__ import annotations

import pytest

from stateloom.core.feature_registry import FeatureRegistry

# ---------------------------------------------------------------------------
# require_feature() FastAPI dependency
# ---------------------------------------------------------------------------


class TestRequireFeatureDependency:
    """Test the require_feature() FastAPI dependency."""

    def test_passes_when_feature_available(self):
        """Dependency allows request when feature is available."""
        pytest.importorskip("fastapi")
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from stateloom.auth.dependencies import require_feature

        registry = FeatureRegistry()
        registry.define("oidc", tier="enterprise")
        registry.provide("oidc")

        app = FastAPI()

        @app.get("/test", dependencies=[Depends(require_feature(registry, "oidc"))])
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_blocks_when_feature_unavailable(self):
        """Dependency returns 403 when feature is not available."""
        pytest.importorskip("fastapi")
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from stateloom.auth.dependencies import require_feature

        registry = FeatureRegistry()
        registry.define("oidc", tier="enterprise")  # not provided

        app = FastAPI()

        @app.get("/test", dependencies=[Depends(require_feature(registry, "oidc"))])
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 403
        assert "enterprise license" in resp.json()["detail"]

    def test_passes_when_registry_is_none(self):
        """Dependency passes through when registry is None (backward compat)."""
        pytest.importorskip("fastapi")
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from stateloom.auth.dependencies import require_feature

        app = FastAPI()

        @app.get("/test", dependencies=[Depends(require_feature(None, "oidc"))])
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OIDC endpoint gating
# ---------------------------------------------------------------------------


class TestOIDCGating:
    """Test that OIDC endpoints are gated behind the 'oidc' feature."""

    @pytest.fixture
    def _setup(self):
        pytest.importorskip("jwt")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.auth.endpoints import create_auth_router
        from stateloom.store.memory_store import MemoryStore

        class FakeConfig:
            auth_enabled = True
            auth_jwt_algorithm = "HS256"
            auth_jwt_access_ttl = 900
            auth_jwt_refresh_ttl = 604800

        class FakeGateBlocked:
            def __init__(self):
                self.store = MemoryStore()
                self.config = FakeConfig()
                self._feature_registry = FeatureRegistry()
                self._feature_registry.define("oidc", tier="enterprise")
                # NOT provided — should block

        class FakeGateAllowed:
            def __init__(self):
                self.store = MemoryStore()
                self.config = FakeConfig()
                self._feature_registry = FeatureRegistry()
                self._feature_registry.define("oidc", tier="enterprise")
                self._feature_registry.provide("oidc")

        return FakeGateBlocked, FakeGateAllowed, FakeConfig

    def test_oidc_providers_gated(self, _setup):
        FakeGateBlocked, _, _ = _setup
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.auth.endpoints import create_auth_router

        gate = FakeGateBlocked()
        app = FastAPI()
        router = create_auth_router(gate)
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        resp = client.get("/api/v1/auth/oidc/providers")
        assert resp.status_code == 403

    def test_oidc_providers_works_when_provided(self, _setup):
        _, FakeGateAllowed, _ = _setup
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.auth.endpoints import create_auth_router

        gate = FakeGateAllowed()
        app = FastAPI()
        router = create_auth_router(gate)
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        resp = client.get("/api/v1/auth/oidc/providers")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Semantic cache gating
# ---------------------------------------------------------------------------


class TestSemanticCacheGating:
    def test_semantic_cache_skipped_without_feature(self):
        """Semantic matcher is None when semantic_cache feature is not available."""
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(
            store_backend="memory",
            cache_semantic_enabled=True,
        )
        from stateloom.gate import Gate

        gate = Gate(config)
        # Enterprise feature not provided — semantic matcher should be None
        assert gate._semantic_matcher is None


# ---------------------------------------------------------------------------
# Guardrails local model gating
# ---------------------------------------------------------------------------


class TestGuardrailsLocalGating:
    def test_local_validator_returns_none_without_feature(self):
        """Local validator returns None when guardrails_local feature is unavailable."""
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(
            store_backend="memory",
            guardrails_enabled=True,
            guardrails_local_model_enabled=True,
        )

        registry = FeatureRegistry()
        registry.define("guardrails_local", tier="enterprise")
        # NOT provided

        from stateloom.middleware.guardrails import GuardrailMiddleware

        mw = GuardrailMiddleware(config, registry=registry)
        result = mw._get_local_validator()
        assert result is None

    def test_local_validator_allowed_when_feature_provided(self):
        """Local validator initializes when guardrails_local feature is available."""
        from unittest.mock import MagicMock, patch

        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(
            store_backend="memory",
            guardrails_enabled=True,
            guardrails_local_model_enabled=True,
        )

        registry = FeatureRegistry()
        registry.define("guardrails_local", tier="enterprise")
        registry.provide("guardrails_local")

        from stateloom.middleware.guardrails import GuardrailMiddleware

        mw = GuardrailMiddleware(config, registry=registry)

        # Mock the lazy import inside _get_local_validator
        mock_validator = MagicMock()
        mock_module = MagicMock()
        mock_module.LocalGuardrailValidator.return_value = mock_validator
        with patch.dict(
            "sys.modules",
            {"stateloom.guardrails.local_validator": mock_module},
        ):
            result = mw._get_local_validator()
            assert result == mock_validator


# ---------------------------------------------------------------------------
# Dashboard /api/features endpoint
# ---------------------------------------------------------------------------


class TestFeaturesEndpoint:
    """Test the GET /api/features dashboard endpoint."""

    def test_features_endpoint_shows_tiers(self):
        pytest.importorskip("fastapi")
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(store_backend="memory")
        from stateloom.gate import Gate

        gate = Gate(config)

        from stateloom.dashboard.api import create_api_router

        app = FastAPI()
        router = create_api_router(gate)
        app.include_router(router, prefix="/api")
        client = TestClient(app)

        resp = client.get("/api/features")
        assert resp.status_code == 200
        data = resp.json()

        # Should have features list and summary
        assert "features" in data
        assert "summary" in data
        assert data["summary"]["community"] > 0
        assert data["summary"]["enterprise"] > 0

        # Enterprise features should be disabled (no license)
        enterprise_features = [f for f in data["features"] if f["tier"] == "enterprise"]
        for feat in enterprise_features:
            assert feat["enabled"] is False
            assert "upgrade_hint" in feat

        # Community features should be enabled
        community_features = [f for f in data["features"] if f["tier"] == "community"]
        for feat in community_features:
            assert feat["enabled"] is True
            assert "upgrade_hint" not in feat

    def test_features_endpoint_shows_unlocked_enterprise(self):
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(store_backend="memory")
        from stateloom.gate import Gate

        gate = Gate(config)

        # Simulate license providing a feature
        gate._feature_registry.provide("oidc")

        from stateloom.dashboard.api import create_api_router

        app = FastAPI()
        router = create_api_router(gate)
        app.include_router(router, prefix="/api")
        client = TestClient(app)

        resp = client.get("/api/features")
        data = resp.json()

        oidc = next(f for f in data["features"] if f["name"] == "oidc")
        assert oidc["enabled"] is True
        assert "upgrade_hint" not in oidc
        assert data["summary"]["unlocked"] >= 1

    def test_consensus_advanced_defined_as_enterprise(self):
        """consensus_advanced appears in the features endpoint as enterprise."""
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(store_backend="memory")
        gate = Gate(config)

        from stateloom.dashboard.api import create_api_router

        app = FastAPI()
        router = create_api_router(gate)
        app.include_router(router, prefix="/api")
        client = TestClient(app)

        resp = client.get("/api/features")
        data = resp.json()

        ca = next(
            (f for f in data["features"] if f["name"] == "consensus_advanced"),
            None,
        )
        assert ca is not None, "consensus_advanced not in features list"
        assert ca["tier"] == "enterprise"
        assert ca["enabled"] is False  # No license
        assert "upgrade_hint" in ca
