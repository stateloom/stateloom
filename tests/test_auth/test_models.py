"""Tests for auth models (User, UserTeamRole)."""

from datetime import datetime, timezone

from stateloom.auth.models import User, UserTeamRole, _new_user_id, _new_utr_id
from stateloom.core.types import Role


def test_user_default_fields():
    user = User(email="test@example.com")
    assert user.id.startswith("usr-")
    assert len(user.id) == 16  # "usr-" + 12 hex chars
    assert user.email == "test@example.com"
    assert user.display_name == ""
    assert user.password_hash == ""
    assert user.email_verified is False
    assert user.org_id == ""
    assert user.org_role is None
    assert user.oidc_provider_id == ""
    assert user.oidc_subject == ""
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)
    assert user.last_login is None
    assert user.metadata == {}


def test_user_with_all_fields():
    now = datetime.now(timezone.utc)
    user = User(
        id="usr-abc123def456",
        email="admin@corp.com",
        display_name="Admin",
        password_hash="$argon2id$...",
        email_verified=True,
        org_id="org-123",
        org_role=Role.ORG_ADMIN,
        oidc_provider_id="oidc-google",
        oidc_subject="sub-12345",
        is_active=True,
        created_at=now,
        last_login=now,
        metadata={"source": "oidc"},
    )
    assert user.org_role == Role.ORG_ADMIN
    assert user.email_verified is True
    assert user.metadata["source"] == "oidc"


def test_user_model_validate():
    d = {
        "id": "usr-test123test",
        "email": "val@test.com",
        "org_role": "org_admin",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    user = User.model_validate(d)
    assert user.org_role == Role.ORG_ADMIN
    assert user.email == "val@test.com"


def test_user_extra_fields_ignored():
    user = User(email="test@test.com", unknown_field="ignored")
    assert not hasattr(user, "unknown_field")


def test_user_team_role_defaults():
    role = UserTeamRole(user_id="usr-123", team_id="team-456", role=Role.TEAM_EDITOR)
    assert role.id.startswith("utr-")
    assert len(role.id) == 16
    assert role.user_id == "usr-123"
    assert role.team_id == "team-456"
    assert role.role == Role.TEAM_EDITOR
    assert isinstance(role.granted_at, datetime)
    assert role.granted_by == ""


def test_user_team_role_with_granter():
    role = UserTeamRole(
        user_id="usr-111",
        team_id="team-222",
        role=Role.TEAM_ADMIN,
        granted_by="usr-000",
    )
    assert role.granted_by == "usr-000"


def test_new_user_id_format():
    uid = _new_user_id()
    assert uid.startswith("usr-")
    assert len(uid) == 16


def test_new_utr_id_format():
    uid = _new_utr_id()
    assert uid.startswith("utr-")
    assert len(uid) == 16


def test_role_enum_values():
    assert Role.ORG_ADMIN == "org_admin"
    assert Role.TEAM_ADMIN == "team_admin"
    assert Role.TEAM_EDITOR == "team_editor"
    assert Role.TEAM_VIEWER == "team_viewer"
    assert Role.ORG_AUDITOR == "org_auditor"
