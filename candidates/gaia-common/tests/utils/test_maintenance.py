"""Tests for GAIA_Project-5jdw — maintenance mode TTL/staleness handling.

Maintenance mode used to have no expiry: SleepWakeManager.receive_wake_signal
unconditionally suppresses Discord wake while is_maintenance_active() is
True, and a dashboard-entered flag from 2026-07-19 was never exited,
silently blocking wake for 3+ days with nothing surfacing it. Fix: give the
flag the same TTL pattern as the VRAM tenant guard (85mb/9zrx) — a stale
flag auto-clears (fails toward not-blocking) instead of persisting forever.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from gaia_common.utils import maintenance


@pytest.fixture(autouse=True)
def _isolated_flags(tmp_path, monkeypatch):
    """Point the module at a scratch directory so tests never touch /shared."""
    monkeypatch.setattr(maintenance, "_SHARED_DIR", tmp_path)
    monkeypatch.setattr(maintenance, "_FLAG_FILE", tmp_path / "maintenance_mode.json")
    monkeypatch.setattr(maintenance, "_LEGACY_FLAG", tmp_path / "ha_maintenance")
    yield


def _write_flag(entered_at=None, ttl_seconds=None, active=True):
    data = {"active": active, "entered_by": "test", "reason": "unit-test"}
    if entered_at is not None:
        data["entered_at"] = entered_at
    if ttl_seconds is not None:
        data["ttl_seconds"] = ttl_seconds
    maintenance._SHARED_DIR.mkdir(parents=True, exist_ok=True)
    maintenance._FLAG_FILE.write_text(json.dumps(data))
    return data


def test_enter_maintenance_writes_default_ttl():
    data = maintenance.enter_maintenance(reason="testing", entered_by="pytest")
    assert data["ttl_seconds"] == maintenance._DEFAULT_TTL_SECONDS
    assert maintenance.is_maintenance_active() is True


def test_enter_maintenance_custom_ttl_is_honored():
    data = maintenance.enter_maintenance(ttl_seconds=99)
    assert data["ttl_seconds"] == 99


def test_exit_maintenance_clears_flag():
    maintenance.enter_maintenance()
    assert maintenance.is_maintenance_active() is True
    result = maintenance.exit_maintenance()
    assert result["active"] is False
    assert maintenance.is_maintenance_active() is False
    assert not maintenance._FLAG_FILE.exists()
    assert not maintenance._LEGACY_FLAG.exists()


def test_fresh_flag_within_ttl_stays_active():
    recent = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    _write_flag(entered_at=recent, ttl_seconds=3600)
    assert maintenance.is_maintenance_active() is True


def test_stale_flag_past_ttl_auto_clears():
    # This is the 5jdw scenario: entered 5 days ago, well past the 4h
    # default — must NOT keep blocking wake forever.
    old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    _write_flag(entered_at=old, ttl_seconds=maintenance._DEFAULT_TTL_SECONDS)
    assert maintenance.is_maintenance_active() is False
    # Auto-clear should also remove the flag file so subsequent checks
    # don't keep re-deriving staleness from disk.
    assert not maintenance._FLAG_FILE.exists()


def test_flag_without_ttl_field_falls_back_to_default():
    # Flags written before this fix shipped have no ttl_seconds field —
    # this is what retroactively closes 5jdw for an already-stuck flag.
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    _write_flag(entered_at=old, ttl_seconds=None)
    assert maintenance.is_maintenance_active() is False


def test_missing_entered_at_treated_as_stale():
    _write_flag(entered_at=None, ttl_seconds=3600)
    assert maintenance.is_maintenance_active() is False


def test_unparseable_entered_at_treated_as_stale():
    _write_flag(entered_at="not-a-timestamp", ttl_seconds=3600)
    assert maintenance.is_maintenance_active() is False


def test_get_maintenance_info_none_after_staleness_clear():
    old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    _write_flag(entered_at=old, ttl_seconds=maintenance._DEFAULT_TTL_SECONDS)
    assert maintenance.get_maintenance_info() is None


def test_inactive_flag_is_not_reported_stale_or_active():
    recent = datetime.now(timezone.utc).isoformat()
    _write_flag(entered_at=recent, ttl_seconds=3600, active=False)
    assert maintenance.is_maintenance_active() is False
