"""Rich error classes with help URLs."""

from __future__ import annotations

from typing import Any

DOCS_BASE = "https://docs.stateloom.io/errors"


class StateLoomError(Exception):
    """Base exception for all StateLoom errors."""

    help_url: str = f"{DOCS_BASE}/general"
    error_code: str = "STATELOOM_ERROR"

    def __init__(self, message: str, *, details: str | None = None):
        self.details = details
        full = f"[{self.error_code}] {message}"
        if details:
            full += f"\n  Details: {details}"
        full += f"\n  Docs: {self.help_url}"
        super().__init__(full)


class StateLoomPIIBlockedError(StateLoomError):
    """Raised when a PII block rule prevents an LLM call."""

    help_url = f"{DOCS_BASE}/pii-blocked"
    error_code = "PII_BLOCKED"

    def __init__(self, pii_type: str, session_id: str):
        self.pii_type = pii_type
        self.session_id = session_id
        super().__init__(
            f"LLM call blocked: PII type '{pii_type}' detected in session '{session_id}'",
            details=(
                "Next steps:\n"
                f"  1. Redact instead of block: set pii_mode='redact' in config\n"
                f"  2. Allow this PII type: exclude '{pii_type}' from pii.rules\n"
                "  3. Config file: pii_rules section in stateloom.yaml"
            ),
        )


class StateLoomBudgetError(StateLoomError):
    """Raised when a session exceeds its budget limit."""

    help_url = f"{DOCS_BASE}/budget-exceeded"
    error_code = "BUDGET_EXCEEDED"

    def __init__(self, limit: float, spent: float, session_id: str):
        self.limit = limit
        self.spent = spent
        self.session_id = session_id
        super().__init__(
            f"Budget exceeded: ${spent:.4f} spent, limit is ${limit:.4f} (session '{session_id}')",
            details=(
                "Next steps:\n"
                f"  1. Increase budget: stateloom.session(budget={limit * 2:.2f})\n"
                "  2. Switch to warnings: set budget_action='warn' in config\n"
                "  3. Dashboard: /sessions → select session → adjust budget"
            ),
        )


class StateLoomLoopError(StateLoomError):
    """Raised when a loop is detected and circuit-break is configured."""

    help_url = f"{DOCS_BASE}/loop-detected"
    error_code = "LOOP_DETECTED"

    def __init__(self, session_id: str, pattern: str, count: int):
        self.session_id = session_id
        self.pattern = pattern
        self.count = count
        super().__init__(
            f"Loop detected in session '{session_id}': pattern repeated {count} times",
            details=(
                f"Repeating pattern: {pattern[:100]}\n"
                "Next steps:\n"
                "  1. Add exit logic to your tool/agent loop\n"
                "  2. Increase threshold: loop_detection_threshold in config\n"
                "  3. Disable loop detection: loop_detection=False"
            ),
        )


class StateLoomReplayError(StateLoomError):
    """Raised for replay engine errors."""

    help_url = f"{DOCS_BASE}/replay"
    error_code = "REPLAY_ERROR"

    def __init__(self, message: str, session_id: str):
        self.session_id = session_id
        super().__init__(message, details=f"Session: {session_id}")


class StateLoomKillSwitchError(StateLoomError):
    """Raised when the global kill switch is active — all LLM traffic blocked."""

    help_url = f"{DOCS_BASE}/kill-switch"
    error_code = "KILL_SWITCH"

    def __init__(
        self,
        message: str = "Service temporarily unavailable. Please try again later.",
        *,
        model: str = "",
        provider: str = "",
    ):
        self.model = model
        self.provider = provider
        parts = ["Next steps:"]
        parts.append("  1. Check kill switch status: GET /api/v1/kill-switch")
        parts.append("  2. Disable via dashboard: /kill-switch → toggle off")
        if model:
            parts.append(f"  3. Blocked model: {model} — check kill_switch_rules in config")
        super().__init__(message, details="\n".join(parts))


class StateLoomGuardrailError(StateLoomError):
    """Raised when a guardrail blocks an LLM call (prompt injection, jailbreak, etc.)."""

    help_url = f"{DOCS_BASE}/guardrails"
    error_code = "GUARDRAIL_BLOCKED"

    def __init__(self, rule_name: str, category: str, session_id: str):
        self.rule_name = rule_name
        self.category = category
        self.session_id = session_id
        super().__init__(
            f"LLM call blocked by guardrail '{rule_name}' ({category}) in session '{session_id}'",
            details=(
                "Next steps:\n"
                f"  1. Switch to audit mode: guardrails_mode='audit'\n"
                f"  2. Disable specific rule: exclude '{rule_name}' from guardrails config\n"
                f"  3. Session: {session_id}"
            ),
        )


class StateLoomBlastRadiusError(StateLoomError):
    """Raised when a session is paused due to blast radius containment."""

    help_url = f"{DOCS_BASE}/blast-radius"
    error_code = "BLAST_RADIUS"

    def __init__(self, session_id: str, trigger: str):
        self.session_id = session_id
        self.trigger = trigger
        super().__init__(
            f"Session '{session_id}' paused: blast radius containment triggered by {trigger}",
            details=(
                "Next steps:\n"
                f"  1. Unpause: stateloom.unpause_session('{session_id}')\n"
                f"  2. API: POST /api/v1/blast-radius/unpause/{session_id}\n"
                "  3. Review failures in dashboard: /sessions → select session → events"
            ),
        )


class StateLoomRateLimitError(StateLoomError):
    """Raised when a team's rate limit is exceeded and queue is full or timed out."""

    help_url = f"{DOCS_BASE}/rate-limit"
    error_code = "RATE_LIMITED"

    def __init__(
        self,
        team_id: str,
        tps: float,
        queue_size: int,
        message: str = "Rate limit exceeded",
    ):
        self.team_id = team_id
        self.tps = tps
        self.queue_size = queue_size
        super().__init__(
            message,
            details=(
                f"Team '{team_id}' at {tps} TPS, queue depth: {queue_size}\n"
                "Next steps:\n"
                f"  1. Increase TPS: PUT /api/v1/teams/{team_id}/rate-limit\n"
                "  2. Raise priority: set rate_limit_priority on the team\n"
                "  3. Retry after a brief backoff"
            ),
        )


class StateLoomJobError(StateLoomError):
    """Raised for async job errors."""

    help_url = f"{DOCS_BASE}/async-jobs"
    error_code = "ASYNC_JOB_ERROR"

    def __init__(self, job_id: str, message: str):
        self.job_id = job_id
        super().__init__(message, details=f"Job: {job_id}")


class StateLoomRetryError(StateLoomError):
    """Raised when all retry attempts are exhausted."""

    help_url = f"{DOCS_BASE}/semantic-retry"
    error_code = "RETRY_EXHAUSTED"

    def __init__(self, attempts: int, last_error: str, session_id: str = ""):
        self.attempts = attempts
        self.last_error = last_error
        self.session_id = session_id
        super().__init__(
            f"All {attempts} retry attempts exhausted",
            details=(
                f"Last error: {last_error}\n"
                "Next steps:\n"
                f"  1. Increase retries: @stateloom.durable_task(retries={attempts + 2})\n"
                "  2. Add validation: durable_task(validate=my_validator)\n"
                "  3. Check session events for the failure pattern"
            ),
        )


class StateLoomTimeoutError(StateLoomError):
    """Raised when a session exceeds its timeout or idle timeout."""

    help_url = f"{DOCS_BASE}/session-timeout"
    error_code = "SESSION_TIMED_OUT"

    def __init__(
        self,
        session_id: str,
        timeout_type: str,
        elapsed: float,
        limit: float,
    ):
        self.session_id = session_id
        self.timeout_type = timeout_type
        self.elapsed = elapsed
        self.limit = limit
        if timeout_type == "idle_timeout":
            hint = f"  1. Extend idle timeout: stateloom.session(idle_timeout={limit * 2:.0f})"
        else:
            hint = f"  1. Extend timeout: stateloom.session(timeout={limit * 2:.0f})"
        super().__init__(
            f"Session '{session_id}' timed out: {timeout_type} "
            f"({elapsed:.1f}s elapsed, limit {limit:.1f}s)",
            details=(
                f"{hint}\n"
                "  2. Send heartbeats for long-running work: session.heartbeat()\n"
                "  3. Set timeout=None to disable"
            ),
        )


class StateLoomCancellationError(StateLoomError):
    """Raised when a session is cancelled."""

    help_url = f"{DOCS_BASE}/session-cancelled"
    error_code = "SESSION_CANCELLED"

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session '{session_id}' has been cancelled",
            details=(
                "The session was explicitly cancelled.\n"
                "Next steps:\n"
                "  1. Start a new session: stateloom.session()\n"
                f"  2. Check who cancelled: review /sessions/{session_id} in dashboard"
            ),
        )


class StateLoomSuspendedError(StateLoomError):
    """Raised when an LLM call is attempted on a suspended session."""

    help_url = f"{DOCS_BASE}/session-suspended"
    error_code = "SESSION_SUSPENDED"

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session '{session_id}' is suspended awaiting external signal",
            details=(
                "Next steps:\n"
                f"  1. Resume: stateloom.signal_session('{session_id}', payload={{...}})\n"
                f"  2. API: POST /api/v1/sessions/{session_id}/signal\n"
                f"  3. Cancel instead: stateloom.cancel_session('{session_id}')"
            ),
        )


class StateLoomCircuitBreakerError(StateLoomError):
    """Raised when a provider's circuit breaker is open."""

    help_url = f"{DOCS_BASE}/circuit-breaker"
    error_code = "CIRCUIT_BREAKER_OPEN"

    def __init__(
        self,
        provider: str,
        fallback_model: str = "",
        message: str = "",
    ):
        self.provider = provider
        self.fallback_model = fallback_model
        msg = message or f"Provider '{provider}' circuit breaker is open"
        parts = [f"Provider '{provider}' is experiencing failures."]
        parts.append("Next steps:")
        if fallback_model:
            parts.append(f"  1. Use fallback model: model='{fallback_model}'")
            parts.append("  2. Check provider status page")
            parts.append("  3. Monitor: GET /api/v1/circuit-breaker")
        else:
            parts.append("  1. Check provider status page")
            parts.append("  2. Monitor: GET /api/v1/circuit-breaker")
        super().__init__(msg, details="\n".join(parts))


class StateLoomComplianceError(StateLoomError):
    """Raised when a compliance policy blocks an LLM call."""

    help_url = f"{DOCS_BASE}/compliance"
    error_code = "COMPLIANCE_BLOCKED"

    def __init__(self, message: str, standard: str = "", action: str = ""):
        self.standard = standard
        self.action = action
        parts: list[str] = []
        if standard:
            parts.append(f"Compliance standard: {standard.upper()}")
        parts.append("Next steps:")
        parts.append("  1. Review compliance profile in config")
        parts.append("  2. Check allowed_endpoints for data residency rules")
        if standard:
            parts.append(f"  3. Reference: {standard.upper()} compliance docs")
        super().__init__(message, details="\n".join(parts))


class StateLoomConfigLockedError(StateLoomError):
    """Raised when attempting to override an admin-locked setting."""

    help_url = f"{DOCS_BASE}/config-locked"
    error_code = "CONFIG_LOCKED"

    def __init__(self, setting: str, locked_value: Any, reason: str = ""):
        self.setting = setting
        self.locked_value = locked_value
        self.reason = reason
        msg = f"Setting '{setting}' is admin-locked (value: {locked_value})"
        details = (
            reason
            or "An admin has locked this setting via the dashboard."
            " Contact your admin to unlock it."
        )
        super().__init__(msg, details=details)


class StateLoomAuthError(StateLoomError):
    """Raised for authentication failures (invalid credentials, expired tokens)."""

    help_url = f"{DOCS_BASE}/auth"
    error_code = "AUTH_ERROR"

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message,
            details=(
                "Next steps:\n"
                "  1. Check your credentials or token\n"
                "  2. Re-authenticate: POST /api/v1/auth/login\n"
                "  3. Refresh token: POST /api/v1/auth/refresh"
            ),
        )


class StateLoomPermissionError(StateLoomError):
    """Raised when a user lacks the required permission."""

    help_url = f"{DOCS_BASE}/permission-denied"
    error_code = "PERMISSION_DENIED"

    def __init__(self, permission: str = "", message: str = ""):
        self.permission = permission
        msg = message or f"Permission denied: requires '{permission}'"
        super().__init__(
            msg,
            details=(
                "Next steps:\n"
                "  1. Check your role: GET /api/v1/auth/me\n"
                "  2. Request access from your org admin\n"
                f"  3. Required permission: {permission}"
            ),
        )


class StateLoomLicenseError(StateLoomError):
    """Raised when a dev mode guardrail is violated without a license."""

    help_url = f"{DOCS_BASE}/license-required"
    error_code = "LICENSE_REQUIRED"

    def __init__(self, constraint: str, message: str = ""):
        self.constraint = constraint
        msg = message or f"License required: {constraint}"
        super().__init__(
            msg,
            details=(
                "This restriction applies in dev mode without a license.\n"
                "Next steps:\n"
                "  1. Purchase a license: sales@stateloom.io\n"
                "  2. Set STATELOOM_LICENSE_KEY in your environment\n"
                "  3. This limit does not apply with a valid license"
            ),
        )


class StateLoomFeatureError(StateLoomError):
    """Raised when an enterprise feature is used without a valid license."""

    help_url = f"{DOCS_BASE}/feature-not-licensed"
    error_code = "FEATURE_NOT_LICENSED"

    def __init__(self, feature: str, message: str | None = None) -> None:
        self.feature = feature
        super().__init__(
            message
            or (
                f"Feature '{feature}' requires an enterprise license. "
                f"Set STATELOOM_LICENSE_KEY or use STATELOOM_ENV=development."
            ),
        )


class StateLoomSecurityError(StateLoomError):
    """Raised when a security policy blocks an operation."""

    help_url = f"{DOCS_BASE}/security"
    error_code = "SECURITY_BLOCKED"

    def __init__(self, audit_event: str, detail: str, session_id: str = ""):
        self.audit_event = audit_event
        self.detail = detail
        self.session_id = session_id
        super().__init__(
            f"Operation blocked by security policy: {audit_event} ({detail})",
            details="Switch to audit mode or add to allow list to permit this operation.",
        )


class StateLoomSideEffectError(StateLoomError):
    """Raised when an outbound HTTP call is blocked during replay.

    During strict replay mode, any HTTP call not captured via @gate.tool()
    is blocked before connection to prevent unintended side effects.
    """

    help_url = f"{DOCS_BASE}/side-effect-blocked"
    error_code = "SIDE_EFFECT_BLOCKED"

    def __init__(self, host: str, session_id: str, step: int):
        self.host = host
        self.session_id = session_id
        self.step = step
        super().__init__(
            f"Side effect blocked: outbound HTTP call to '{host}' "
            f"at step {step} in session '{session_id}'",
            details=(
                "Fix options:\n"
                "  1. Decorate the function with @stateloom.tool() to capture it\n"
                "  2. Use @stateloom.tool(mutates_state=False) if it's read-only\n"
                "  3. Add the host to allow_hosts in replay config"
            ),
        )
