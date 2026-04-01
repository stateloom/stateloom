"""Store protocol — the persistence interface for StateLoom."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from stateloom.core.event import Event
from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session
from stateloom.experiment.models import Experiment, ExperimentAssignment, SessionFeedback

if TYPE_CHECKING:
    from stateloom.agent.models import Agent, AgentVersion
    from stateloom.auth.models import User, UserTeamRole
    from stateloom.auth.oidc_models import OIDCProvider
    from stateloom.core.job import Job
    from stateloom.proxy.virtual_key import VirtualKey


@runtime_checkable
class Store(Protocol):
    """Protocol for event/session persistence backends."""

    def save_session(self, session: Session) -> None:
        """Persist or update a session."""
        ...

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        ...

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
        org_id: str | None = None,
        team_id: str | None = None,
        end_user: str | None = None,
    ) -> list[Session]:
        """List sessions, optionally filtered by status, org_id, team_id, or end_user."""
        ...

    def count_sessions(
        self,
        status: str | None = None,
        org_id: str | None = None,
        team_id: str | None = None,
        end_user: str | None = None,
    ) -> int:
        """Count sessions matching the given filters."""
        ...

    def save_event(self, event: Event) -> None:
        """Persist an event."""
        ...

    def get_session_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Event]:
        """Get events for a session, optionally filtered by type."""
        ...

    def get_global_stats(self) -> dict[str, Any]:
        """Get global statistics (total cost, calls, savings, etc.)."""
        ...

    def get_call_counts(self) -> dict[str, int]:
        """Count cloud vs local LLM calls."""
        ...

    def get_cost_by_model(self) -> dict[str, float]:
        """Aggregate cost per model from LLM call events."""
        ...

    def save_session_with_events(self, session: Session, events: list[Event]) -> None:
        """Atomically persist a session and its events in a single transaction."""
        ...

    def get_event_messages(self, event_id: str) -> str | None:
        """Lazy-load request_messages_json for a single event."""
        ...

    def cleanup_request_messages(self, retention_hours: int = 24) -> int:
        """Null out request_messages_json older than retention_hours."""
        ...

    def cleanup(self, retention_days: int = 30) -> int:
        """Remove events older than retention_days. Returns count deleted."""
        ...

    # --- Experiment methods ---

    def save_experiment(self, experiment: Experiment) -> None:
        """Persist or update an experiment."""
        ...

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        ...

    def list_experiments(self, status: str | None = None) -> list[Experiment]:
        """List experiments, optionally filtered by status."""
        ...

    def save_assignment(self, assignment: ExperimentAssignment) -> None:
        """Persist an experiment assignment."""
        ...

    def get_assignment(self, session_id: str) -> ExperimentAssignment | None:
        """Get the experiment assignment for a session."""
        ...

    def list_assignments(self, experiment_id: str) -> list[ExperimentAssignment]:
        """List all assignments for an experiment."""
        ...

    def save_feedback(self, feedback: SessionFeedback) -> None:
        """Persist session feedback."""
        ...

    def get_feedback(self, session_id: str) -> SessionFeedback | None:
        """Get feedback for a session."""
        ...

    def list_feedback(self, experiment_id: str | None = None) -> list[SessionFeedback]:
        """List feedback, optionally filtered by experiment."""
        ...

    def get_experiment_metrics(self, experiment_id: str) -> dict[str, Any]:
        """Get per-variant aggregated metrics for an experiment."""
        ...

    # --- Organization methods ---

    def save_organization(self, org: Organization) -> None:
        """Persist or update an organization."""
        ...

    def get_organization(self, org_id: str) -> Organization | None:
        """Get an organization by ID."""
        ...

    def list_organizations(self) -> list[Organization]:
        """List all organizations."""
        ...

    # --- Team methods ---

    def save_team(self, team: Team) -> None:
        """Persist or update a team."""
        ...

    def get_team(self, team_id: str) -> Team | None:
        """Get a team by ID."""
        ...

    def list_teams(self, org_id: str | None = None) -> list[Team]:
        """List teams, optionally filtered by org_id."""
        ...

    # --- Job methods ---

    def save_job(self, job: Job) -> None:
        """Persist or update a job."""
        ...

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        ...

    def list_jobs(
        self,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs, optionally filtered by status or session_id."""
        ...

    def delete_job(self, job_id: str) -> bool:
        """Delete a job. Returns True if it existed."""
        ...

    def get_job_stats(self) -> dict[str, Any]:
        """Get aggregate job statistics."""
        ...

    # --- Parent-child queries ---

    def list_child_sessions(
        self,
        parent_session_id: str,
        limit: int = 100,
    ) -> list[Session]:
        """List child sessions for a parent session."""
        ...

    # --- Hierarchy queries ---

    def get_org_stats(self, org_id: str) -> dict[str, Any]:
        """Get aggregated stats for an organization from its sessions."""
        ...

    def get_team_stats(self, team_id: str) -> dict[str, Any]:
        """Get aggregated stats for a team from its sessions."""
        ...

    # --- Virtual key methods (proxy) ---

    def save_virtual_key(self, vk: VirtualKey) -> None:
        """Persist a virtual key."""
        ...

    def get_virtual_key_by_hash(self, key_hash: str) -> VirtualKey | None:
        """Look up a virtual key by its SHA256 hash. Returns None if not found."""
        ...

    def get_virtual_key(self, key_id: str) -> VirtualKey | None:
        """Look up a virtual key by its ID. Returns None if not found."""
        ...

    def list_virtual_keys(self, team_id: str | None = None) -> list[VirtualKey]:
        """List virtual keys, optionally filtered by team_id."""
        ...

    def revoke_virtual_key(self, key_id: str) -> bool:
        """Revoke a virtual key. Returns True if it existed."""
        ...

    # --- Agent methods ---

    def save_agent(self, agent: Agent) -> None:
        """Persist or update an agent."""
        ...

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        ...

    def get_agent_by_slug(self, slug: str, team_id: str) -> Agent | None:
        """Get an agent by slug within a team."""
        ...

    def list_agents(
        self,
        team_id: str | None = None,
        org_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Agent]:
        """List agents, optionally filtered."""
        ...

    def save_agent_version(self, version: AgentVersion) -> None:
        """Persist an agent version."""
        ...

    def get_agent_version(self, version_id: str) -> AgentVersion | None:
        """Get an agent version by ID."""
        ...

    def list_agent_versions(self, agent_id: str, limit: int = 100) -> list[AgentVersion]:
        """List versions for an agent (newest first)."""
        ...

    def get_next_version_number(self, agent_id: str) -> int:
        """Get the next version number for an agent."""
        ...

    # --- Admin lock methods ---

    def save_admin_lock(
        self, setting: str, value: str, locked_by: str = "", reason: str = ""
    ) -> None:
        """Persist an admin lock for a config setting."""
        ...

    def get_admin_lock(self, setting: str) -> dict[str, Any] | None:
        """Get an admin lock by setting name. Returns dict or None."""
        ...

    def list_admin_locks(self) -> list[dict[str, Any]]:
        """List all admin-locked settings."""
        ...

    def delete_admin_lock(self, setting: str) -> None:
        """Remove an admin lock."""
        ...

    # --- Purge methods (compliance) ---

    def purge_session(self, session_id: str) -> int:
        """Delete a session and all its events. Returns deleted event count."""
        ...

    def purge_user_data(self, user_identifier: str) -> dict[str, int]:
        """Scan and delete all data matching a user identifier. Returns counts."""
        ...

    # --- User methods (auth) ---

    def save_user(self, user: User) -> None:
        """Persist or update a user."""
        ...

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID. Returns None if not found."""
        ...

    def get_user_by_email(self, email: str) -> User | None:
        """Get a user by email (case-insensitive). Returns None if not found."""
        ...

    def get_user_by_oidc(self, provider_id: str, subject: str) -> User | None:
        """Get a user by OIDC provider+subject pair. Returns None if not found."""
        ...

    def list_users(
        self,
        org_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        """List users, optionally filtered by org_id."""
        ...

    def delete_user(self, user_id: str) -> bool:
        """Soft-delete a user (set is_active=False). Returns True if found."""
        ...

    # --- UserTeamRole methods (auth) ---

    def save_user_team_role(self, role: UserTeamRole) -> None:
        """Persist a user-team role assignment."""
        ...

    def get_user_team_roles(self, user_id: str) -> list[UserTeamRole]:
        """Get all team role assignments for a user."""
        ...

    def get_team_members(self, team_id: str) -> list[UserTeamRole]:
        """Get all role assignments for a team."""
        ...

    def delete_user_team_role(self, role_id: str) -> bool:
        """Delete a user-team role assignment. Returns True if found."""
        ...

    # --- Refresh token methods (auth) ---

    def save_refresh_token(
        self,
        token_hash: str,
        user_id: str,
        expires_at: str,
        family_id: str = "",
    ) -> None:
        """Persist a refresh token record."""
        ...

    def get_refresh_token(self, token_hash: str) -> dict[str, Any] | None:
        """Get a refresh token by its hash. Returns dict or None."""
        ...

    def revoke_refresh_token(self, token_hash: str) -> None:
        """Revoke a single refresh token."""
        ...

    def revoke_all_refresh_tokens(self, user_id: str) -> int:
        """Revoke all refresh tokens for a user. Returns count revoked."""
        ...

    # --- OIDC provider methods ---

    def save_oidc_provider(self, provider: OIDCProvider) -> None:
        """Persist or update an OIDC provider configuration."""
        ...

    def get_oidc_provider(self, provider_id: str) -> OIDCProvider | None:
        """Get an OIDC provider by ID. Returns None if not found."""
        ...

    def get_oidc_provider_by_issuer(self, issuer_url: str) -> OIDCProvider | None:
        """Get an OIDC provider by issuer URL. Returns None if not found."""
        ...

    def list_oidc_providers(self) -> list[OIDCProvider]:
        """List all OIDC providers."""
        ...

    def delete_oidc_provider(self, provider_id: str) -> bool:
        """Delete an OIDC provider. Returns True if deleted."""
        ...

    # --- Secret methods ---

    def save_secret(self, key: str, value: str) -> None:
        """Persist a secret key-value pair."""
        ...

    def get_secret(self, key: str) -> str:
        """Get a secret by key. Returns empty string if not found."""
        ...
