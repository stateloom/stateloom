"""Integration tests for compliance feature in Gate."""

from __future__ import annotations

import pytest

from stateloom.core.config import ComplianceProfile, StateLoomConfig
from stateloom.gate import Gate


class TestCreateOrgWithCompliance:
    def test_create_org_gdpr_string(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="EU Corp", compliance_profile="gdpr")
        assert org.compliance_profile is not None
        assert org.compliance_profile.standard == "gdpr"
        assert org.compliance_profile.region == "eu"

    def test_create_org_hipaa_string(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="Hospital", compliance_profile="hipaa")
        assert org.compliance_profile.standard == "hipaa"
        assert org.compliance_profile.zero_retention_logs is True

    def test_create_org_custom_profile(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        custom = ComplianceProfile(standard="custom", region="apac", session_ttl_days=7)
        org = gate.create_organization(name="APAC Corp", compliance_profile=custom)
        assert org.compliance_profile.standard == "custom"
        assert org.compliance_profile.region == "apac"

    def test_create_org_no_profile(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="No Compliance")
        assert org.compliance_profile is None


class TestCreateTeamWithCompliance:
    def test_create_team_gdpr(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="EU Corp")
        team = gate.create_team(org.id, name="Team A", compliance_profile="gdpr")
        assert team.compliance_profile is not None
        assert team.compliance_profile.standard == "gdpr"


class TestComplianceProfileResolution:
    def test_team_profile_wins_over_org(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="Corp", compliance_profile="gdpr")
        team = gate.create_team(org.id, name="US Team", compliance_profile="ccpa")
        profile = gate._get_compliance_profile(org.id, team.id)
        assert profile is not None
        assert profile.standard == "ccpa"

    def test_org_profile_when_no_team_profile(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="Corp", compliance_profile="hipaa")
        team = gate.create_team(org.id, name="Team B")
        profile = gate._get_compliance_profile(org.id, team.id)
        assert profile is not None
        assert profile.standard == "hipaa"

    def test_global_fallback(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
            compliance_profile=ComplianceProfile(standard="gdpr"),
        )
        gate = Gate(config)
        profile = gate._get_compliance_profile("", "")
        assert profile is not None
        assert profile.standard == "gdpr"

    def test_no_profile_anywhere(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        profile = gate._get_compliance_profile("", "")
        assert profile is None


class TestPIIRuleMerging:
    def test_org_pii_rules_include_compliance(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        from stateloom.core.config import PIIRule

        org = gate.create_organization(
            name="Corp",
            pii_rules=[PIIRule(pattern="email", mode="audit")],
            compliance_profile="gdpr",
        )
        rules = gate._get_org_pii_rules(org.id)
        patterns = [r.pattern for r in rules]
        # Should include both org-level and compliance-level rules
        assert "email" in patterns
        assert "vat_id" in patterns
        assert "iban" in patterns


class TestComplianceCleanup:
    def test_cleanup_expired_sessions(self):
        from datetime import datetime, timedelta, timezone

        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        org = gate.create_organization(name="Corp", compliance_profile="gdpr")

        # Create a session that ended 45 days ago (past GDPR 30-day TTL)
        from stateloom.core.session import Session
        from stateloom.core.types import SessionStatus

        old_session = Session(
            id="old-session",
            org_id=org.id,
            ended_at=datetime.now(timezone.utc) - timedelta(days=45),
            status=SessionStatus.COMPLETED,
        )
        gate.store.save_session(old_session)

        # Create a recent session
        recent_session = Session(
            id="recent-session",
            org_id=org.id,
            ended_at=datetime.now(timezone.utc) - timedelta(days=5),
            status=SessionStatus.COMPLETED,
        )
        gate.store.save_session(recent_session)

        purged = gate.compliance_cleanup()
        assert purged == 1  # Only the old session should be purged
        assert gate.store.get_session("old-session") is None
        assert gate.store.get_session("recent-session") is not None

    def test_cleanup_no_compliance_orgs(self):
        config = StateLoomConfig(
            console_output=False,
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
        )
        gate = Gate(config)
        gate.create_organization(name="No Compliance")
        purged = gate.compliance_cleanup()
        assert purged == 0
