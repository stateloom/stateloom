"""Tests for the SecretVault."""

from __future__ import annotations

import os

from stateloom.security.vault import SecretVault


def test_store_and_retrieve():
    vault = SecretVault()
    vault.configure(enabled=True, keys=[])
    vault.store("MY_KEY", "secret123")
    assert vault.retrieve("MY_KEY") == "secret123"


def test_retrieve_missing():
    vault = SecretVault()
    vault.configure(enabled=True, keys=[])
    assert vault.retrieve("NONEXISTENT") is None


def test_has():
    vault = SecretVault()
    vault.configure(enabled=True, keys=[])
    assert not vault.has("K")
    vault.store("K", "v")
    assert vault.has("K")


def test_list_keys():
    vault = SecretVault()
    vault.configure(enabled=True, keys=[])
    vault.store("B_KEY", "b")
    vault.store("A_KEY", "a")
    assert vault.list_keys() == ["A_KEY", "B_KEY"]


def test_remove():
    vault = SecretVault()
    vault.configure(enabled=True, keys=[])
    vault.store("K", "v")
    assert vault.remove("K") is True
    assert vault.has("K") is False
    # Idempotent
    assert vault.remove("K") is False


def test_configure_with_scrub(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_KEY", "scrub-me")
    vault = SecretVault()
    vault.configure(enabled=True, scrub_environ=True, keys=["TEST_SECRET_KEY"])
    assert vault.retrieve("TEST_SECRET_KEY") == "scrub-me"
    assert os.environ.get("TEST_SECRET_KEY") is None


def test_configure_without_scrub(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_KEY2", "keep-me")
    vault = SecretVault()
    vault.configure(enabled=True, scrub_environ=False, keys=["TEST_SECRET_KEY2"])
    assert vault.retrieve("TEST_SECRET_KEY2") == "keep-me"
    assert os.environ.get("TEST_SECRET_KEY2") == "keep-me"


def test_restore_environ(monkeypatch):
    monkeypatch.setenv("RESTORE_KEY", "orig-value")
    vault = SecretVault()
    vault.configure(enabled=True, scrub_environ=True, keys=["RESTORE_KEY"])
    assert os.environ.get("RESTORE_KEY") is None
    restored = vault.restore_environ()
    assert "RESTORE_KEY" in restored
    assert os.environ.get("RESTORE_KEY") == "orig-value"


def test_get_status():
    vault = SecretVault()
    vault.configure(enabled=True, keys=[])
    vault.store("KEY1", "val1")
    status = vault.get_status()
    assert status["enabled"] is True
    assert status["key_count"] == 1
    assert "KEY1" in status["keys"]
    assert isinstance(status["scrubbed"], list)


def test_disabled_noop():
    vault = SecretVault()
    vault.configure(enabled=False)
    # Nothing should happen
    assert vault._enabled is False
    status = vault.get_status()
    assert status["enabled"] is False
