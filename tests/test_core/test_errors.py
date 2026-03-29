"""Tests for StateLoom error classes."""

from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomCancellationError,
    StateLoomCircuitBreakerError,
    StateLoomComplianceError,
    StateLoomError,
    StateLoomKillSwitchError,
    StateLoomLoopError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
    StateLoomReplayError,
    StateLoomRetryError,
    StateLoomSideEffectError,
    StateLoomSuspendedError,
    StateLoomTimeoutError,
)


def test_base_error():
    err = StateLoomError("Something broke")
    assert "Something broke" in str(err)
    assert "docs.stateloom.io" in str(err)
    assert err.error_code == "STATELOOM_ERROR"
    assert "[STATELOOM_ERROR]" in str(err)


def test_pii_blocked_error():
    err = StateLoomPIIBlockedError(pii_type="credit_card", session_id="s1")
    assert "credit_card" in str(err)
    assert "s1" in str(err)
    assert err.pii_type == "credit_card"
    assert err.session_id == "s1"
    assert err.error_code == "PII_BLOCKED"


def test_pii_blocked_error_next_steps():
    err = StateLoomPIIBlockedError(pii_type="ssn", session_id="s1")
    msg = str(err)
    assert "Next steps" in msg
    assert "pii_mode='redact'" in msg
    assert "'ssn'" in msg


def test_budget_error():
    err = StateLoomBudgetError(limit=5.0, spent=5.02, session_id="s1")
    assert "$5.02" in str(err)
    assert "$5.00" in str(err)
    assert err.error_code == "BUDGET_EXCEEDED"


def test_budget_error_next_steps():
    err = StateLoomBudgetError(limit=5.0, spent=5.02, session_id="s1")
    msg = str(err)
    assert "Next steps" in msg
    assert "budget=10.00" in msg
    assert "budget_action='warn'" in msg


def test_loop_error():
    err = StateLoomLoopError(session_id="s1", pattern="abc", count=3)
    assert "3 times" in str(err)
    assert err.error_code == "LOOP_DETECTED"


def test_loop_error_next_steps():
    err = StateLoomLoopError(session_id="s1", pattern="abc", count=3)
    msg = str(err)
    assert "Next steps" in msg
    assert "Repeating pattern: abc" in msg
    assert "loop_detection_threshold" in msg
    assert "loop_detection=False" in msg


def test_side_effect_error():
    err = StateLoomSideEffectError(host="evil.example.com", session_id="s1", step=5)
    assert "evil.example.com" in str(err)
    assert "s1" in str(err)
    assert "step 5" in str(err)
    assert err.host == "evil.example.com"
    assert err.session_id == "s1"
    assert err.step == 5


def test_side_effect_error_fix_options():
    err = StateLoomSideEffectError(host="api.stripe.com", session_id="s1", step=3)
    msg = str(err)
    assert "@stateloom.tool()" in msg
    assert "allow_hosts" in msg
    assert "side-effect-blocked" in msg


def test_replay_error():
    err = StateLoomReplayError(message="Replay failed", session_id="s1")
    assert err.error_code == "REPLAY_ERROR"
    assert "[REPLAY_ERROR]" in str(err)


def test_side_effect_error_code():
    err = StateLoomSideEffectError(host="example.com", session_id="s1", step=1)
    assert err.error_code == "SIDE_EFFECT_BLOCKED"


def test_side_effect_error_help_url():
    err = StateLoomSideEffectError(host="example.com", session_id="s1", step=1)
    assert "side-effect-blocked" in err.help_url


def test_kill_switch_error_default():
    err = StateLoomKillSwitchError()
    msg = str(err)
    assert "Service temporarily unavailable" in msg
    assert "Next steps" in msg
    assert "GET /api/v1/kill-switch" in msg


def test_kill_switch_error_with_context():
    err = StateLoomKillSwitchError("Blocked!", model="gpt-4o", provider="openai")
    msg = str(err)
    assert "Blocked!" in msg
    assert "Next steps" in msg
    assert "gpt-4o" in msg
    assert "kill_switch_rules" in msg
    assert err.model == "gpt-4o"
    assert err.provider == "openai"


def test_kill_switch_error_without_model():
    err = StateLoomKillSwitchError("Blocked!")
    msg = str(err)
    assert "Next steps" in msg
    # Should have 2 steps, not 3 (no model-specific hint)
    assert "Blocked model" not in msg


def test_blast_radius_error_next_steps():
    err = StateLoomBlastRadiusError(session_id="s123", trigger="consecutive_failures")
    msg = str(err)
    assert "Next steps" in msg
    assert "unpause_session('s123')" in msg
    assert "/api/v1/blast-radius/unpause/s123" in msg
    assert err.session_id == "s123"
    assert err.trigger == "consecutive_failures"


def test_rate_limit_error_next_steps():
    err = StateLoomRateLimitError(team_id="team-1", tps=10.0, queue_size=50)
    msg = str(err)
    assert "Next steps" in msg
    assert "team-1" in msg
    assert "10.0 TPS" in msg
    assert "/api/v1/teams/team-1/rate-limit" in msg
    assert "rate_limit_priority" in msg
    assert "backoff" in msg


def test_timeout_error_next_steps_timeout():
    err = StateLoomTimeoutError(
        session_id="s1",
        timeout_type="timeout",
        elapsed=65.0,
        limit=60.0,
    )
    msg = str(err)
    assert "timeout=120" in msg
    assert "heartbeat()" in msg
    assert "timeout=None" in msg


def test_timeout_error_next_steps_idle_timeout():
    err = StateLoomTimeoutError(
        session_id="s1",
        timeout_type="idle_timeout",
        elapsed=35.0,
        limit=30.0,
    )
    msg = str(err)
    assert "idle_timeout=60" in msg
    assert "heartbeat()" in msg


def test_cancellation_error_next_steps():
    err = StateLoomCancellationError(session_id="s42")
    msg = str(err)
    assert "Next steps" in msg
    assert "stateloom.session()" in msg
    assert "/sessions/s42" in msg


def test_retry_error_next_steps():
    err = StateLoomRetryError(attempts=3, last_error="Invalid JSON")
    msg = str(err)
    assert "Next steps" in msg
    assert "retries=5" in msg  # 3 + 2
    assert "Last error: Invalid JSON" in msg
    assert "validate=my_validator" in msg


def test_circuit_breaker_error_with_fallback():
    err = StateLoomCircuitBreakerError(provider="openai", fallback_model="claude-3-sonnet")
    msg = str(err)
    assert "Next steps" in msg
    assert "claude-3-sonnet" in msg
    assert "fallback model" in msg.lower()
    assert "/api/v1/circuit-breaker" in msg


def test_circuit_breaker_error_without_fallback():
    err = StateLoomCircuitBreakerError(provider="openai")
    msg = str(err)
    assert "Next steps" in msg
    assert "status page" in msg
    assert "/api/v1/circuit-breaker" in msg
    # No fallback step
    assert "fallback model" not in msg.lower()


def test_compliance_error_with_standard():
    err = StateLoomComplianceError("Data residency violation", standard="gdpr")
    msg = str(err)
    assert "Next steps" in msg
    assert "Compliance standard: GDPR" in msg
    assert "GDPR compliance docs" in msg
    assert "compliance profile" in msg


def test_compliance_error_without_standard():
    err = StateLoomComplianceError("Policy blocked")
    msg = str(err)
    assert "Next steps" in msg
    assert "compliance profile" in msg
    # No standard-specific reference
    assert "Compliance standard" not in msg


def test_suspended_error_next_steps():
    err = StateLoomSuspendedError(session_id="s99")
    msg = str(err)
    assert "Next steps" in msg
    assert "signal_session('s99'" in msg
    assert "/api/v1/sessions/s99/signal" in msg
    assert "cancel_session('s99')" in msg
