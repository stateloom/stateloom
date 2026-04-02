"""In-memory store for testing and ephemeral use."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from stateloom.core.event import Event
from stateloom.core.job import Job
from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session
from stateloom.experiment.models import (
    Experiment,
    ExperimentAssignment,
    SessionFeedback,
)

if TYPE_CHECKING:
    from stateloom.agent.models import Agent, AgentVersion
    from stateloom.auth.models import User, UserTeamRole
    from stateloom.auth.oidc_models import OIDCProvider
    from stateloom.proxy.virtual_key import VirtualKey


class MemoryStore:
    """In-memory store. Data is lost on process exit."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._events: list[Event] = []
        self._experiments: dict[str, Experiment] = {}
        self._assignments: dict[str, ExperimentAssignment] = {}  # session_id -> assignment
        self._feedback: dict[str, SessionFeedback] = {}  # session_id -> feedback
        self._organizations: dict[str, Organization] = {}
        self._teams: dict[str, Team] = {}
        self._jobs: dict[str, Job] = {}
        self._secrets: dict[str, str] = {}
        self._virtual_keys: dict[str, Any] = {}  # id -> VirtualKey
        self._virtual_key_hashes: dict[str, str] = {}  # key_hash -> id
        self._agents: dict[str, Agent] = {}  # id -> Agent
        self._agent_versions: dict[str, Any] = {}  # id -> AgentVersion
        self._admin_locks: dict[str, dict[str, Any]] = {}  # setting -> lock record
        self._users: dict[str, User] = {}  # id -> User
        self._user_team_roles: dict[str, Any] = {}  # id -> UserTeamRole
        self._refresh_tokens: dict[str, dict[str, Any]] = {}  # token_hash -> record
        self._oidc_providers: dict[str, OIDCProvider] = {}  # id -> OIDCProvider
        self._lock = threading.Lock()

    def save_session(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
        org_id: str | None = None,
        team_id: str | None = None,
        end_user: str | None = None,
    ) -> list[Session]:
        with self._lock:
            sessions = list(self._sessions.values())
            if status:
                sessions = [s for s in sessions if s.status.value == status]
            if org_id is not None:
                sessions = [s for s in sessions if s.org_id == org_id]
            if team_id is not None:
                sessions = [s for s in sessions if s.team_id == team_id]
            if end_user is not None:
                sessions = [s for s in sessions if s.end_user == end_user]
            sessions.sort(key=lambda s: s.started_at, reverse=True)
            return sessions[offset : offset + limit]

    def count_sessions(
        self,
        status: str | None = None,
        org_id: str | None = None,
        team_id: str | None = None,
        end_user: str | None = None,
    ) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            if status:
                sessions = [s for s in sessions if s.status.value == status]
            if org_id is not None:
                sessions = [s for s in sessions if s.org_id == org_id]
            if team_id is not None:
                sessions = [s for s in sessions if s.team_id == team_id]
            if end_user is not None:
                sessions = [s for s in sessions if s.end_user == end_user]
            return len(sessions)

    def list_child_sessions(
        self,
        parent_session_id: str,
        limit: int = 100,
    ) -> list[Session]:
        with self._lock:
            children = [
                s for s in self._sessions.values() if s.parent_session_id == parent_session_id
            ]
            children.sort(key=lambda s: s.started_at, reverse=True)
            return children[:limit]

    def save_session_with_events(self, session: Session, events: list[Event]) -> None:
        with self._lock:
            for event in events:
                self._events.append(event)
            self._sessions[session.id] = session

    def save_event(self, event: Event) -> None:
        with self._lock:
            # Upsert: replace existing event with the same ID (matches SQLiteStore
            # INSERT OR REPLACE semantics).  This is needed because model testing
            # saves the event twice — once without similarity, once with.
            for i, existing in enumerate(self._events):
                if existing.id == event.id:
                    self._events[i] = event
                    return
            self._events.append(event)

    def get_session_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        desc: bool = False,
    ) -> list[Event]:
        with self._lock:
            if session_id:
                events = [e for e in self._events if e.session_id == session_id]
            else:
                events = list(self._events)
            if event_type:
                events = [e for e in events if e.event_type.value == event_type]
            events.sort(key=lambda e: e.timestamp, reverse=desc)
            return events[offset : offset + limit]

    def count_events(
        self,
        session_id: str = "",
        event_type: str | None = None,
    ) -> int:
        with self._lock:
            if session_id:
                events = [e for e in self._events if e.session_id == session_id]
            else:
                events = list(self._events)
            if event_type:
                events = [e for e in events if e.event_type.value == event_type]
            return len(events)

    def get_pii_stats(self) -> dict[str, Any]:
        with self._lock:
            pii_events = [e for e in self._events if e.event_type.value == "pii_detection"]
            by_type: dict[str, int] = {}
            by_action: dict[str, int] = {}
            sessions: set[str] = set()
            for e in pii_events:
                pii_type = getattr(e, "pii_type", "")
                if pii_type:
                    by_type[pii_type] = by_type.get(pii_type, 0) + 1
                action = getattr(e, "action_taken", "")
                if action:
                    by_action[action] = by_action.get(action, 0) + 1
                sessions.add(e.session_id)
            return {
                "sessions_affected": len(sessions),
                "by_type": by_type,
                "by_action": by_action,
            }

    def get_global_stats(self) -> dict[str, Any]:
        with self._lock:
            sessions = list(self._sessions.values())
            return {
                "total_sessions": len(sessions),
                "active_sessions": sum(1 for s in sessions if s.status.value == "active"),
                "total_cost": sum(s.total_cost for s in sessions),
                "total_tokens": sum(s.total_tokens for s in sessions),
                "total_calls": sum(s.call_count for s in sessions),
                "total_cache_hits": sum(s.cache_hits for s in sessions),
                "total_cache_savings": sum(s.cache_savings for s in sessions),
                "total_pii_detections": sum(s.pii_detections for s in sessions),
                "total_guardrail_detections": sum(
                    getattr(s, "guardrail_detections", 0) for s in sessions
                ),
                "total_estimated_api_cost": sum(
                    getattr(s, "estimated_api_cost", 0.0) for s in sessions
                ),
            }

    def get_call_counts(self) -> dict[str, int]:
        with self._lock:
            llm_events = [e for e in self._events if getattr(e, "event_type", None) == "llm_call"]
            local = sum(1 for e in llm_events if getattr(e, "provider", "") == "local")
            return {
                "total": len(llm_events),
                "local_calls": local,
                "cloud_calls": len(llm_events) - local,
            }

    def get_cost_by_model(self) -> dict[str, float]:
        with self._lock:
            llm_events = [e for e in self._events if getattr(e, "event_type", None) == "llm_call"]
            model_costs: dict[str, float] = {}
            for e in llm_events:
                model = getattr(e, "model", "unknown") or "unknown"
                cost = getattr(e, "cost", 0.0)
                model_costs[model] = model_costs.get(model, 0.0) + cost
            return model_costs

    def get_event_messages(self, event_id: str) -> str | None:
        """Lazy-load request_messages_json for a single event."""
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    return getattr(e, "request_messages_json", None)
        return None

    def cleanup_request_messages(self, retention_hours: int = 24) -> int:
        """Null out request_messages_json older than retention_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        count = 0
        with self._lock:
            for e in self._events:
                if getattr(e, "request_messages_json", None) is not None and e.timestamp < cutoff:
                    e.request_messages_json = None  # type: ignore[attr-defined]
                    count += 1
        return count

    def cleanup(self, retention_days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        with self._lock:
            before = len(self._events)
            self._events = [e for e in self._events if e.timestamp >= cutoff]
            # Also clean up ended sessions
            expired_ids = {
                sid for sid, s in self._sessions.items() if s.ended_at and s.ended_at < cutoff
            }
            for sid in expired_ids:
                del self._sessions[sid]
            return before - len(self._events)

    # --- Experiment methods ---

    def save_experiment(self, experiment: Experiment) -> None:
        with self._lock:
            self._experiments[experiment.id] = experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self, status: str | None = None) -> list[Experiment]:
        with self._lock:
            experiments = list(self._experiments.values())
            if status:
                experiments = [e for e in experiments if e.status.value == status]
            experiments.sort(key=lambda e: e.created_at, reverse=True)
            return experiments

    def save_assignment(self, assignment: ExperimentAssignment) -> None:
        with self._lock:
            self._assignments[assignment.session_id] = assignment

    def get_assignment(self, session_id: str) -> ExperimentAssignment | None:
        with self._lock:
            return self._assignments.get(session_id)

    def list_assignments(self, experiment_id: str) -> list[ExperimentAssignment]:
        with self._lock:
            assignments = [
                a for a in self._assignments.values() if a.experiment_id == experiment_id
            ]
            assignments.sort(key=lambda a: a.assigned_at)
            return assignments

    def save_feedback(self, feedback: SessionFeedback) -> None:
        with self._lock:
            self._feedback[feedback.session_id] = feedback

    def get_feedback(self, session_id: str) -> SessionFeedback | None:
        with self._lock:
            return self._feedback.get(session_id)

    def list_feedback(self, experiment_id: str | None = None) -> list[SessionFeedback]:
        with self._lock:
            if experiment_id:
                # Get session IDs assigned to this experiment
                exp_sessions = {
                    a.session_id
                    for a in self._assignments.values()
                    if a.experiment_id == experiment_id
                }
                feedbacks = [f for f in self._feedback.values() if f.session_id in exp_sessions]
            else:
                feedbacks = list(self._feedback.values())
            feedbacks.sort(key=lambda f: f.created_at)
            return feedbacks

    def get_experiment_metrics(self, experiment_id: str) -> dict[str, Any]:
        with self._lock:
            # Get all assignments for this experiment
            assignments = [
                a for a in self._assignments.values() if a.experiment_id == experiment_id
            ]

            # Group by variant
            variant_sessions: dict[str, list[str]] = {}
            for a in assignments:
                variant_sessions.setdefault(a.variant_name, []).append(a.session_id)

            variants: dict[str, dict[str, Any]] = {}
            for vname, session_ids in variant_sessions.items():
                sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
                feedbacks = [self._feedback[sid] for sid in session_ids if sid in self._feedback]

                # Compute cost metrics
                costs = [s.total_cost for s in sessions]
                costs_sorted = sorted(costs)

                # Compute latency from events
                latencies: list[float] = []
                for sid in session_ids:
                    for e in self._events:
                        if e.session_id == sid and e.event_type.value == "llm_call":
                            latencies.append(getattr(e, "latency_ms", 0.0))

                success = sum(1 for f in feedbacks if f.rating == "success")
                failure = sum(1 for f in feedbacks if f.rating == "failure")
                partial = sum(1 for f in feedbacks if f.rating == "partial")
                unrated = len(sessions) - len(feedbacks)
                rated = success + failure + partial
                success_rate = success / rated if rated > 0 else 0.0

                scores = [f.score for f in feedbacks if f.score is not None]
                avg_score = sum(scores) / len(scores) if scores else None

                n = len(sessions)
                variants[vname] = {
                    "session_count": n,
                    "avg_cost": sum(costs) / n if n else 0.0,
                    "total_cost": sum(costs),
                    "median_cost": _median(costs_sorted),
                    "p95_cost": _percentile(costs_sorted, 95),
                    "avg_tokens": sum(s.total_tokens for s in sessions) / n if n else 0.0,
                    "avg_prompt_tokens": (
                        sum(s.total_prompt_tokens for s in sessions) / n if n else 0.0
                    ),
                    "avg_completion_tokens": (
                        sum(s.total_completion_tokens for s in sessions) / n if n else 0.0
                    ),
                    "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                    "avg_call_count": sum(s.call_count for s in sessions) / n if n else 0.0,
                    "success_count": success,
                    "failure_count": failure,
                    "partial_count": partial,
                    "unrated_count": unrated,
                    "success_rate": success_rate,
                    "avg_score": avg_score,
                }

            return {"experiment_id": experiment_id, "variants": variants}

    # --- Organization methods ---

    def save_organization(self, org: Organization) -> None:
        with self._lock:
            self._organizations[org.id] = org

    def get_organization(self, org_id: str) -> Organization | None:
        with self._lock:
            return self._organizations.get(org_id)

    def list_organizations(self) -> list[Organization]:
        with self._lock:
            orgs = list(self._organizations.values())
            orgs.sort(key=lambda o: o.created_at, reverse=True)
            return orgs

    # --- Team methods ---

    def save_team(self, team: Team) -> None:
        with self._lock:
            self._teams[team.id] = team

    def get_team(self, team_id: str) -> Team | None:
        with self._lock:
            return self._teams.get(team_id)

    def list_teams(self, org_id: str | None = None) -> list[Team]:
        with self._lock:
            teams = list(self._teams.values())
            if org_id is not None:
                teams = [t for t in teams if t.org_id == org_id]
            teams.sort(key=lambda t: t.created_at, reverse=True)
            return teams

    # --- Virtual key methods (proxy) ---

    def save_virtual_key(self, vk: VirtualKey) -> None:
        with self._lock:
            self._virtual_keys[vk.id] = vk
            self._virtual_key_hashes[vk.key_hash] = vk.id

    def get_virtual_key_by_hash(self, key_hash: str) -> VirtualKey | None:
        with self._lock:
            vk_id = self._virtual_key_hashes.get(key_hash)
            if vk_id is None:
                return None
            return self._virtual_keys.get(vk_id)

    def get_virtual_key(self, key_id: str) -> VirtualKey | None:
        with self._lock:
            return self._virtual_keys.get(key_id)

    def list_virtual_keys(self, team_id: str | None = None) -> list[VirtualKey]:
        with self._lock:
            keys = list(self._virtual_keys.values())
            if team_id is not None:
                keys = [k for k in keys if k.team_id == team_id]
            keys.sort(key=lambda k: k.created_at, reverse=True)
            return keys

    def revoke_virtual_key(self, key_id: str) -> bool:
        with self._lock:
            vk = self._virtual_keys.get(key_id)
            if vk is None:
                return False
            if vk.revoked:
                return False
            vk.revoked = True
            return True

    # --- Admin lock methods ---

    def save_admin_lock(
        self, setting: str, value: str, locked_by: str = "", reason: str = ""
    ) -> None:
        with self._lock:
            self._admin_locks[setting] = {
                "setting": setting,
                "value": value,
                "locked_by": locked_by,
                "reason": reason,
                "locked_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_admin_lock(self, setting: str) -> dict[str, Any] | None:
        with self._lock:
            return self._admin_locks.get(setting)

    def list_admin_locks(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._admin_locks.values(), key=lambda d: d["setting"])

    def delete_admin_lock(self, setting: str) -> None:
        with self._lock:
            self._admin_locks.pop(setting, None)

    # --- Secret methods ---

    def save_secret(self, key: str, value: str) -> None:
        with self._lock:
            self._secrets[key] = value

    def get_secret(self, key: str) -> str:
        with self._lock:
            return self._secrets.get(key, "")

    def list_secrets(self) -> list[str]:
        with self._lock:
            return list(self._secrets.keys())

    def delete_secret(self, key: str) -> None:
        with self._lock:
            self._secrets.pop(key, None)

    # --- Purge methods (compliance) ---

    def purge_session(self, session_id: str) -> int:
        """Delete a session and all its events. Returns deleted event count."""
        with self._lock:
            before = len(self._events)
            self._events = [e for e in self._events if e.session_id != session_id]
            deleted = before - len(self._events)
            self._sessions.pop(session_id, None)
            self._assignments.pop(session_id, None)
            self._feedback.pop(session_id, None)
            return deleted

    def purge_user_data(self, user_identifier: str) -> dict[str, int]:
        """Scan and delete all data matching a user identifier."""
        with self._lock:
            # Find sessions where metadata contains the identifier
            matching_ids: list[str] = []
            for sid, session in self._sessions.items():
                meta_str = str(session.metadata)
                if user_identifier in meta_str:
                    matching_ids.append(sid)

            events_deleted = 0
            before = len(self._events)
            self._events = [e for e in self._events if e.session_id not in matching_ids]
            events_deleted = before - len(self._events)

            for sid in matching_ids:
                self._sessions.pop(sid, None)
                self._assignments.pop(sid, None)
                self._feedback.pop(sid, None)

            return {"sessions": len(matching_ids), "events": events_deleted}

    # --- Hierarchy queries ---

    def get_org_stats(self, org_id: str) -> dict[str, Any]:
        with self._lock:
            sessions = [s for s in self._sessions.values() if s.org_id == org_id]
            return {
                "org_id": org_id,
                "total_sessions": len(sessions),
                "active_sessions": sum(1 for s in sessions if s.status.value == "active"),
                "total_cost": sum(s.total_cost for s in sessions),
                "total_tokens": sum(s.total_tokens for s in sessions),
                "total_calls": sum(s.call_count for s in sessions),
                "total_cache_hits": sum(s.cache_hits for s in sessions),
                "total_pii_detections": sum(s.pii_detections for s in sessions),
            }

    # --- Job methods ---

    def save_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
            if status:
                jobs = [j for j in jobs if j.status.value == status]
            if session_id:
                jobs = [j for j in jobs if j.session_id == session_id]
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs[offset : offset + limit]

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def get_job_stats(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
            by_status: dict[str, int] = {}
            processing_times: list[float] = []
            for j in jobs:
                s = j.status.value
                by_status[s] = by_status.get(s, 0) + 1
                if j.started_at and j.completed_at:
                    ms = (j.completed_at - j.started_at).total_seconds() * 1000
                    processing_times.append(ms)
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
            return {
                "total": len(jobs),
                "by_status": by_status,
                "avg_processing_time_ms": avg_time,
            }

    # --- Agent methods ---

    def save_agent(self, agent: Agent) -> None:
        with self._lock:
            self._agents[agent.id] = agent

    def get_agent(self, agent_id: str) -> Agent | None:
        with self._lock:
            return self._agents.get(agent_id)

    def get_agent_by_slug(self, slug: str, team_id: str) -> Agent | None:
        with self._lock:
            for agent in self._agents.values():
                if agent.slug == slug and agent.team_id == team_id:
                    return agent
            return None

    def list_agents(
        self,
        team_id: str | None = None,
        org_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Agent]:
        with self._lock:
            agents = list(self._agents.values())
            if team_id is not None:
                agents = [a for a in agents if a.team_id == team_id]
            if org_id is not None:
                agents = [a for a in agents if a.org_id == org_id]
            if status is not None:
                status_val = status
                agents = [a for a in agents if a.status.value == status_val]
            agents.sort(key=lambda a: a.created_at, reverse=True)
            return agents[offset : offset + limit]

    def save_agent_version(self, version: AgentVersion) -> None:
        with self._lock:
            self._agent_versions[version.id] = version

    def get_agent_version(self, version_id: str) -> AgentVersion | None:
        with self._lock:
            return self._agent_versions.get(version_id)

    def list_agent_versions(self, agent_id: str, limit: int = 100) -> list[AgentVersion]:
        with self._lock:
            versions = [v for v in self._agent_versions.values() if v.agent_id == agent_id]
            versions.sort(key=lambda v: v.version_number, reverse=True)
            return versions[:limit]

    def get_next_version_number(self, agent_id: str) -> int:
        with self._lock:
            max_ver = 0
            for v in self._agent_versions.values():
                if v.agent_id == agent_id and v.version_number > max_ver:
                    max_ver = v.version_number
            return max_ver + 1

    def get_team_stats(self, team_id: str) -> dict[str, Any]:
        with self._lock:
            sessions = [s for s in self._sessions.values() if s.team_id == team_id]
            return {
                "team_id": team_id,
                "total_sessions": len(sessions),
                "active_sessions": sum(1 for s in sessions if s.status.value == "active"),
                "total_cost": sum(s.total_cost for s in sessions),
                "total_tokens": sum(s.total_tokens for s in sessions),
                "total_calls": sum(s.call_count for s in sessions),
                "total_cache_hits": sum(s.cache_hits for s in sessions),
                "total_pii_detections": sum(s.pii_detections for s in sessions),
            }

    # --- User methods (auth) ---

    def save_user(self, user: User) -> None:
        with self._lock:
            self._users[user.id] = user

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        email_lower = email.lower()
        with self._lock:
            for user in self._users.values():
                if user.email.lower() == email_lower:
                    return user
            return None

    def get_user_by_oidc(self, provider_id: str, subject: str) -> User | None:
        with self._lock:
            for user in self._users.values():
                if user.oidc_provider_id == provider_id and user.oidc_subject == subject:
                    return user
            return None

    def list_users(
        self,
        org_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        with self._lock:
            users = list(self._users.values())
            if org_id is not None:
                users = [u for u in users if u.org_id == org_id]
            users.sort(key=lambda u: u.created_at, reverse=True)
            return users[offset : offset + limit]

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                return False
            user.is_active = False
            return True

    # --- UserTeamRole methods (auth) ---

    def save_user_team_role(self, role: UserTeamRole) -> None:
        with self._lock:
            # Enforce unique (user_id, team_id)
            for existing in list(self._user_team_roles.values()):
                if existing.user_id == role.user_id and existing.team_id == role.team_id:
                    if existing.id != role.id:
                        del self._user_team_roles[existing.id]
            self._user_team_roles[role.id] = role

    def get_user_team_roles(self, user_id: str) -> list[UserTeamRole]:
        with self._lock:
            return [r for r in self._user_team_roles.values() if r.user_id == user_id]

    def get_team_members(self, team_id: str) -> list[UserTeamRole]:
        with self._lock:
            return [r for r in self._user_team_roles.values() if r.team_id == team_id]

    def delete_user_team_role(self, role_id: str) -> bool:
        with self._lock:
            return self._user_team_roles.pop(role_id, None) is not None

    # --- Refresh token methods (auth) ---

    def save_refresh_token(
        self,
        token_hash: str,
        user_id: str,
        expires_at: str,
        family_id: str = "",
    ) -> None:
        with self._lock:
            self._refresh_tokens[token_hash] = {
                "token_hash": token_hash,
                "user_id": user_id,
                "expires_at": expires_at,
                "family_id": family_id,
                "revoked": False,
            }

    def get_refresh_token(self, token_hash: str) -> dict[str, Any] | None:
        with self._lock:
            return self._refresh_tokens.get(token_hash)

    def revoke_refresh_token(self, token_hash: str) -> None:
        with self._lock:
            record = self._refresh_tokens.get(token_hash)
            if record:
                record["revoked"] = True

    def revoke_all_refresh_tokens(self, user_id: str) -> int:
        with self._lock:
            count = 0
            for record in self._refresh_tokens.values():
                if record["user_id"] == user_id and not record["revoked"]:
                    record["revoked"] = True
                    count += 1
            return count

    # --- OIDC provider methods ---

    def save_oidc_provider(self, provider: OIDCProvider) -> None:
        with self._lock:
            self._oidc_providers[provider.id] = provider

    def get_oidc_provider(self, provider_id: str) -> OIDCProvider | None:
        with self._lock:
            return self._oidc_providers.get(provider_id)

    def get_oidc_provider_by_issuer(self, issuer_url: str) -> OIDCProvider | None:
        with self._lock:
            for p in self._oidc_providers.values():
                if p.issuer_url == issuer_url:
                    return p
            return None

    def list_oidc_providers(self) -> list[OIDCProvider]:
        with self._lock:
            return list(self._oidc_providers.values())

    def delete_oidc_provider(self, provider_id: str) -> bool:
        with self._lock:
            if provider_id in self._oidc_providers:
                del self._oidc_providers[provider_id]
                return True
            return False


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    idx = int(len(values) * pct / 100)
    idx = min(idx, len(values) - 1)
    return values[idx]
