"""Tests for PIIScannerMiddleware with org-level PII rule inheritance."""

from __future__ import annotations

import asyncio

import pytest

from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.errors import StateLoomPIIBlockedError
from stateloom.core.session import Session
from stateloom.core.types import FailureAction, PIIMode
from stateloom.middleware.pii_scanner import PIIScannerMiddleware, _pii_mode_severity


class TestPIIModeSeverity:
    def test_audit_lowest(self):
        assert _pii_mode_severity(PIIMode.AUDIT) == 0

    def test_redact_middle(self):
        assert _pii_mode_severity(PIIMode.REDACT) == 1

    def test_block_highest(self):
        assert _pii_mode_severity(PIIMode.BLOCK) == 2

    def test_strictest_wins(self):
        assert _pii_mode_severity(PIIMode.BLOCK) > _pii_mode_severity(PIIMode.REDACT)
        assert _pii_mode_severity(PIIMode.REDACT) > _pii_mode_severity(PIIMode.AUDIT)


class TestPIIOrgRuleInheritance:
    def test_org_block_overrides_session_audit(self):
        """Org BLOCK rule overrides session AUDIT for same pattern."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_default_mode=PIIMode.AUDIT,
        )

        def org_rules(org_id):
            if org_id == "org-1":
                return [PIIRule(pattern="email", mode=PIIMode.BLOCK)]
            return []

        scanner = PIIScannerMiddleware(config, org_rules_fn=org_rules)
        mode = scanner._get_mode("email", org_id="org-1")
        assert mode == PIIMode.BLOCK

    def test_org_redact_overrides_session_audit(self):
        """Org REDACT rule overrides session AUDIT."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_default_mode=PIIMode.AUDIT,
        )

        def org_rules(org_id):
            return [PIIRule(pattern="ssn", mode=PIIMode.REDACT)]

        scanner = PIIScannerMiddleware(config, org_rules_fn=org_rules)
        mode = scanner._get_mode("ssn", org_id="org-1")
        assert mode == PIIMode.REDACT

    def test_session_block_not_downgraded_by_org_audit(self):
        """Strictest wins — session BLOCK should not be downgraded by org AUDIT."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="credit_card",
                    mode=PIIMode.BLOCK,
                    on_middleware_failure=FailureAction.BLOCK,
                ),
            ],
        )

        def org_rules(org_id):
            return [PIIRule(pattern="credit_card", mode=PIIMode.AUDIT)]

        scanner = PIIScannerMiddleware(config, org_rules_fn=org_rules)
        mode = scanner._get_mode("credit_card", org_id="org-1")
        assert mode == PIIMode.BLOCK

    def test_no_org_id_uses_session_rules(self):
        """When no org_id, only session rules apply."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )

        def org_rules(org_id):
            return [PIIRule(pattern="email", mode=PIIMode.BLOCK)]

        scanner = PIIScannerMiddleware(config, org_rules_fn=org_rules)
        mode = scanner._get_mode("email", org_id="")
        assert mode == PIIMode.REDACT

    def test_no_org_rules_fn(self):
        """When org_rules_fn is None, session rules used alone."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_default_mode=PIIMode.AUDIT,
        )
        scanner = PIIScannerMiddleware(config)
        mode = scanner._get_mode("email", org_id="org-1")
        assert mode == PIIMode.AUDIT

    def test_org_rule_group_pattern_match(self):
        """Org rule with group pattern (e.g., 'api_key') matches specific patterns."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_default_mode=PIIMode.AUDIT,
        )

        def org_rules(org_id):
            return [PIIRule(pattern="api_key", mode=PIIMode.BLOCK)]

        scanner = PIIScannerMiddleware(config, org_rules_fn=org_rules)
        mode = scanner._get_mode("api_key_openai", org_id="org-1")
        assert mode == PIIMode.BLOCK
