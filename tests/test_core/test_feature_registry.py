"""Tests for the FeatureRegistry — tiered feature definitions and gating."""

import pytest

from stateloom.core.errors import StateLoomFeatureError
from stateloom.core.feature_registry import FeatureRegistry


class TestDefine:
    def test_community_auto_enables(self):
        r = FeatureRegistry()
        r.define("cache", tier="community", description="Caching")
        assert r.is_available("cache") is True

    def test_enterprise_starts_disabled(self):
        r = FeatureRegistry()
        r.define("oidc", tier="enterprise", description="SSO")
        assert r.is_available("oidc") is False


class TestProvide:
    def test_provide_enables_enterprise(self):
        r = FeatureRegistry()
        r.define("oidc", tier="enterprise")
        r.provide("oidc")
        assert r.is_available("oidc") is True

    def test_provide_unknown_feature_is_noop(self):
        r = FeatureRegistry()
        r.provide("nonexistent")  # should not raise
        assert r.is_available("nonexistent") is False


class TestRequire:
    def test_require_raises_when_unavailable(self):
        r = FeatureRegistry()
        r.define("oidc", tier="enterprise")
        with pytest.raises(StateLoomFeatureError) as exc_info:
            r.require("oidc")
        assert "oidc" in str(exc_info.value)
        assert exc_info.value.feature == "oidc"

    def test_require_passes_when_available(self):
        r = FeatureRegistry()
        r.define("cache", tier="community")
        r.require("cache")  # should not raise

    def test_require_raises_for_undefined(self):
        r = FeatureRegistry()
        with pytest.raises(StateLoomFeatureError):
            r.require("nonexistent")


class TestBackwardCompat:
    def test_is_loaded_alias(self):
        r = FeatureRegistry()
        r.define("cache", tier="community")
        assert r.is_loaded("cache") is True
        assert r.is_loaded("nonexistent") is False

    def test_register_backward_compat(self):
        """Old-style register() should define + provide."""
        r = FeatureRegistry()
        r.register("compliance")
        assert r.is_available("compliance") is True
        # Verify it was defined as enterprise (default for register())
        status = r.status()
        assert status["features"]["compliance"]["tier"] == "enterprise"


class TestIntrospection:
    def test_enterprise_feature_names(self):
        r = FeatureRegistry()
        r.define("cache", tier="community")
        r.define("oidc", tier="enterprise")
        r.define("compliance", tier="enterprise")
        names = r.enterprise_feature_names()
        assert sorted(names) == ["compliance", "oidc"]

    def test_status_includes_tier_and_enabled(self):
        r = FeatureRegistry()
        r.define("cache", tier="community", description="Caching")
        r.define("oidc", tier="enterprise", description="SSO")
        status = r.status()
        assert status["count"] == 2
        assert status["features"]["cache"]["tier"] == "community"
        assert status["features"]["cache"]["enabled"] is True
        assert status["features"]["cache"]["description"] == "Caching"
        assert status["features"]["oidc"]["tier"] == "enterprise"
        assert status["features"]["oidc"]["enabled"] is False
        assert status["features"]["oidc"]["description"] == "SSO"
