"""Tests for compliance profile presets and resolution."""

from __future__ import annotations

import pytest

from stateloom.compliance.profiles import (
    PROFILE_PRESETS,
    ccpa_profile,
    gdpr_profile,
    hipaa_profile,
    resolve_profile,
)
from stateloom.core.config import ComplianceProfile


class TestGDPRProfile:
    def test_standard(self):
        p = gdpr_profile()
        assert p.standard == "gdpr"

    def test_region(self):
        p = gdpr_profile()
        assert p.region == "eu"

    def test_session_ttl(self):
        p = gdpr_profile()
        assert p.session_ttl_days == 30

    def test_blocks_local_routing(self):
        p = gdpr_profile()
        assert p.block_local_routing is True

    def test_blocks_shadow(self):
        p = gdpr_profile()
        assert p.block_shadow is True

    def test_has_pii_rules(self):
        p = gdpr_profile()
        assert len(p.pii_rules) > 0
        patterns = [r.pattern for r in p.pii_rules]
        assert "vat_id" in patterns
        assert "iban" in patterns


class TestHIPAAProfile:
    def test_standard(self):
        p = hipaa_profile()
        assert p.standard == "hipaa"

    def test_zero_retention(self):
        p = hipaa_profile()
        assert p.zero_retention_logs is True

    def test_no_cache(self):
        p = hipaa_profile()
        assert p.cache_ttl_seconds == 0

    def test_blocks_local(self):
        p = hipaa_profile()
        assert p.block_local_routing is True

    def test_has_phi_rules(self):
        p = hipaa_profile()
        patterns = [r.pattern for r in p.pii_rules]
        assert "medical_record_number" in patterns
        assert "npi" in patterns


class TestCCPAProfile:
    def test_standard(self):
        p = ccpa_profile()
        assert p.standard == "ccpa"

    def test_session_ttl(self):
        p = ccpa_profile()
        assert p.session_ttl_days == 90

    def test_has_california_dl(self):
        p = ccpa_profile()
        patterns = [r.pattern for r in p.pii_rules]
        assert "california_dl" in patterns
        assert "credit_card" in patterns


class TestResolveProfile:
    def test_resolve_gdpr_string(self):
        p = resolve_profile("gdpr")
        assert p.standard == "gdpr"
        assert p.region == "eu"

    def test_resolve_hipaa_string(self):
        p = resolve_profile("hipaa")
        assert p.standard == "hipaa"
        assert p.zero_retention_logs is True

    def test_resolve_ccpa_string(self):
        p = resolve_profile("ccpa")
        assert p.standard == "ccpa"

    def test_resolve_passthrough(self):
        """ComplianceProfile instance is returned as-is."""
        custom = ComplianceProfile(standard="custom", region="apac")
        result = resolve_profile(custom)
        assert result is custom

    def test_resolve_unknown(self):
        """Unknown string creates a profile with that standard name."""
        p = resolve_profile("pci-dss")
        assert p.standard == "pci-dss"

    def test_presets_dict(self):
        assert "gdpr" in PROFILE_PRESETS
        assert "hipaa" in PROFILE_PRESETS
        assert "ccpa" in PROFILE_PRESETS


class TestBackwardCompat:
    def test_no_profile_zero_behavior(self):
        """No compliance profile = default ComplianceProfile with standard='none'."""
        p = ComplianceProfile()
        assert p.standard == "none"
        assert p.region == "global"
        assert p.session_ttl_days == 0
        assert p.block_local_routing is False
        assert p.block_shadow is False
        assert p.zero_retention_logs is False
        assert p.pii_rules == []
