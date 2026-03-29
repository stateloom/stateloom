"""Tests for compliance legal rule mappings."""

from __future__ import annotations

from stateloom.compliance.legal_rules import LEGAL_RULES, get_legal_rule


class TestGetLegalRule:
    def test_gdpr_pii_blocked(self):
        rule = get_legal_rule("gdpr", "pii_blocked")
        assert "GDPR-Art-32" in rule

    def test_gdpr_data_purged(self):
        rule = get_legal_rule("gdpr", "data_purged")
        assert "Art-17" in rule

    def test_hipaa_zero_retention(self):
        rule = get_legal_rule("hipaa", "zero_retention")
        assert "HIPAA" in rule

    def test_ccpa_data_purged(self):
        rule = get_legal_rule("ccpa", "data_purged")
        assert "CCPA-1798.105" in rule

    def test_unknown_standard(self):
        rule = get_legal_rule("unknown", "pii_blocked")
        assert rule == ""

    def test_unknown_action(self):
        rule = get_legal_rule("gdpr", "nonexistent")
        assert rule == ""
