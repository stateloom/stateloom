"""Tests for StateLoom configuration."""

import os
import tempfile
from pathlib import Path

import pytest

from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.types import BudgetAction, PIIMode


def test_default_config():
    config = StateLoomConfig()
    assert config.auto_patch is True
    assert config.fail_open is True
    assert config.dashboard is True
    assert config.dashboard_port == 4781
    assert config.budget_per_session is None
    assert config.pii_enabled is False
    assert config.cache_enabled is True
    assert config.store_backend == "sqlite"


def test_config_with_kwargs():
    config = StateLoomConfig(
        auto_patch=False,
        budget_per_session=10.0,
        pii_enabled=True,
        dashboard_port=9999,
    )
    assert config.auto_patch is False
    assert config.budget_per_session == 10.0
    assert config.pii_enabled is True
    assert config.dashboard_port == 9999


def test_config_from_yaml():
    yaml_content = """
auto_patch: false
dashboard:
  port: 8888
budget:
  per_session: 5.0
  action: warn
pii:
  default_mode: redact
  rules:
    - pattern: email
      mode: audit
    - pattern: credit_card
      mode: block
cache:
  enabled: true
  max_size: 500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = StateLoomConfig.from_yaml(f.name)

    os.unlink(f.name)

    assert config.auto_patch is False
    assert config.dashboard_port == 8888
    assert config.budget_per_session == 5.0
    assert config.budget_action == BudgetAction.WARN
    assert config.pii_enabled is True
    assert config.pii_default_mode == PIIMode.REDACT
    assert len(config.pii_rules) == 2
    assert config.pii_rules[0].pattern == "email"
    assert config.pii_rules[0].mode == PIIMode.AUDIT
    assert config.pii_rules[1].pattern == "credit_card"
    assert config.pii_rules[1].mode == PIIMode.BLOCK


def test_config_from_yaml_not_found():
    with pytest.raises(FileNotFoundError):
        StateLoomConfig.from_yaml("/nonexistent/path.yaml")


def test_config_env_vars(monkeypatch):
    monkeypatch.setenv("STATELOOM_DASHBOARD_PORT", "7777")
    monkeypatch.setenv("STATELOOM_LOG_LEVEL", "DEBUG")
    config = StateLoomConfig()
    assert config.dashboard_port == 7777
    assert config.log_level == "DEBUG"
