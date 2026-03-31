"""Tests for v4 config validation — on_middleware_failure requirements."""

import pytest

import stateloom
from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.errors import StateLoomError
from stateloom.core.types import BudgetAction, FailureAction, PIIMode
from stateloom.gate import Gate


def test_block_pii_rule_without_failure_action_defaults_to_block():
    """PII rule with mode=block auto-defaults on_middleware_failure to block."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        pii_enabled=True,
        pii_rules=[
            PIIRule(pattern="credit_card", mode=PIIMode.BLOCK),
        ],
    )
    gate = Gate(config)
    assert gate.config.pii_rules[0].on_middleware_failure == FailureAction.BLOCK


def test_block_pii_rule_with_failure_action_passes():
    """PII rule with mode=block and on_middleware_failure should work."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        pii_enabled=True,
        pii_rules=[
            PIIRule(
                pattern="credit_card", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.PASS
            ),
        ],
    )
    gate = Gate(config)
    assert gate.config.pii_rules[0].on_middleware_failure == FailureAction.PASS


def test_audit_pii_rule_without_failure_action_passes():
    """PII rule with mode=audit does not require on_middleware_failure."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        pii_enabled=True,
        pii_rules=[
            PIIRule(pattern="email", mode=PIIMode.AUDIT),
        ],
    )
    gate = Gate(config)
    assert len(gate.config.pii_rules) == 1


def test_budget_hard_stop_without_failure_action_defaults_to_block():
    """Budget with hard_stop auto-defaults on_middleware_failure to block."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        budget_per_session=5.0,
        budget_action=BudgetAction.HARD_STOP,
    )
    gate = Gate(config)
    assert gate.config.budget_on_middleware_failure == FailureAction.BLOCK


def test_budget_hard_stop_with_failure_action_passes():
    """Budget with hard_stop and on_middleware_failure should work."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        budget_per_session=5.0,
        budget_action=BudgetAction.HARD_STOP,
        budget_on_middleware_failure=FailureAction.BLOCK,
    )
    gate = Gate(config)
    assert gate.config.budget_on_middleware_failure == FailureAction.BLOCK


def test_budget_warn_without_failure_action_passes():
    """Budget with action=warn does not require on_middleware_failure."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        budget_per_session=5.0,
        budget_action=BudgetAction.WARN,
    )
    gate = Gate(config)
    assert gate.config.budget_action == BudgetAction.WARN


def test_no_budget_no_failure_action_passes():
    """No budget config does not require on_middleware_failure."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )
    gate = Gate(config)
    assert gate.config.budget_per_session is None


def test_multiple_block_rules_auto_default():
    """Multiple block-mode PII rules all get on_middleware_failure auto-defaulted."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        pii_enabled=True,
        pii_rules=[
            PIIRule(pattern="credit_card", mode=PIIMode.BLOCK),
            PIIRule(pattern="ssn", mode=PIIMode.BLOCK),
        ],
    )
    gate = Gate(config)
    assert gate.config.pii_rules[0].on_middleware_failure == FailureAction.BLOCK
    assert gate.config.pii_rules[1].on_middleware_failure == FailureAction.BLOCK
