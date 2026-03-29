"""Purge engine — Right to Be Forgotten / data deletion for compliance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from stateloom.compliance.audit import compute_audit_hash
from stateloom.compliance.legal_rules import get_legal_rule
from stateloom.core.event import ComplianceAuditEvent

logger = logging.getLogger("stateloom.compliance.purge")


@dataclass
class PurgeResult:
    """Summary of a purge operation."""

    user_identifier: str
    sessions_deleted: int = 0
    events_deleted: int = 0
    cache_entries_deleted: int = 0
    jobs_deleted: int = 0
    virtual_keys_deleted: int = 0
    audit_event_id: str = ""


class PurgeEngine:
    """Purges user data from store and cache, creates audit trail."""

    def __init__(self, store: Any, cache_store: Any = None) -> None:
        self._store = store
        self._cache_store = cache_store

    def purge(
        self,
        user_identifier: str,
        standard: str = "gdpr",
        org_id: str = "",
        team_id: str = "",
        audit_salt: str = "",
    ) -> PurgeResult:
        """Purge all data matching a user identifier.

        Args:
            user_identifier: The user identifier to purge.
            standard: Compliance standard for the audit event.
            org_id: Organization ID for the audit event.
            team_id: Team ID for the audit event.
            audit_salt: Salt for the audit hash.

        Returns:
            PurgeResult with counts of deleted items.
        """
        result = PurgeResult(user_identifier=user_identifier)

        # 1. Purge from main store
        if hasattr(self._store, "purge_user_data"):
            counts = self._store.purge_user_data(user_identifier)
            result.sessions_deleted = counts.get("sessions", 0)
            result.events_deleted = counts.get("events", 0)

        # 2. Purge from cache store
        if self._cache_store is not None and hasattr(self._cache_store, "purge_by_content"):
            result.cache_entries_deleted = self._cache_store.purge_by_content(user_identifier)

        # 3. Purge job data
        if hasattr(self._store, "purge_user_jobs"):
            try:
                result.jobs_deleted = self._store.purge_user_jobs(user_identifier)
            except Exception:
                logger.debug("Failed to purge jobs", exc_info=True)

        # 4. Purge virtual keys
        if hasattr(self._store, "purge_user_virtual_keys"):
            try:
                result.virtual_keys_deleted = self._store.purge_user_virtual_keys(user_identifier)
            except Exception:
                logger.debug("Failed to purge virtual keys", exc_info=True)

        # 5. Create ComplianceAuditEvent recording the purge
        legal_rule = get_legal_rule(standard, "data_purged")
        audit_event = ComplianceAuditEvent(
            compliance_standard=standard,
            action="data_purged",
            legal_rule=legal_rule,
            justification=(
                f"Right to be forgotten: purged data for '{user_identifier}'. "
                f"Sessions: {result.sessions_deleted}, "
                f"Events: {result.events_deleted}, "
                f"Cache: {result.cache_entries_deleted}, "
                f"Jobs: {result.jobs_deleted}, "
                f"VirtualKeys: {result.virtual_keys_deleted}."
            ),
            target_type="user",
            target_id=user_identifier,
            org_id=org_id,
            team_id=team_id,
        )
        audit_event.integrity_hash = compute_audit_hash(audit_event, audit_salt)
        result.audit_event_id = audit_event.id

        # Persist the audit event
        try:
            self._store.save_event(audit_event)
        except Exception:
            logger.warning("Failed to persist purge audit event", exc_info=True)

        return result
