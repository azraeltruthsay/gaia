"""Engineering shifts and the held park (Azrael, 2026-11-19)."""
import importlib

import pytest


@pytest.fixture
def tx(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSMISSION_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("GAIA_PARK_HOLD_FILE", str(tmp_path / "hold.json"))
    from gaia_core.cognition import transmission
    importlib.reload(transmission)
    return transmission


def test_shifts_up_from_gear_1_and_waits(tx, monkeypatch):
    gears = iter(["awake", "transitioning", "focusing"])
    shifts = []
    monkeypatch.setattr(tx, "_current_gear", lambda: next(gears))
    monkeypatch.setattr(tx, "_shift", lambda target, reason: shifts.append(target) or True)
    assert tx.shift_for_engineering("s1", wait_s=30, sleep=lambda s: None)
    assert shifts == ["focusing"] and tx._load()["shift"]["why"] == "engineering"


def test_already_in_gear_2_refreshes(tx, monkeypatch):
    monkeypatch.setattr(tx, "_current_gear", lambda: "focusing")
    monkeypatch.setattr(tx, "_shift", lambda *a: pytest.fail("no shift needed"))
    assert tx.shift_for_engineering("s1")


def test_held_park_is_left_alone(tx, monkeypatch):
    tx.set_hold(True, "Azrael parked her")
    monkeypatch.setattr(tx, "_current_gear", lambda: "parked")
    monkeypatch.setattr(tx, "_shift", lambda *a: pytest.fail("held"))
    assert not tx.shift_for_engineering("s1")
    tx.set_hold(False)
    assert tx.held() is None


def test_engineering_shift_returns_after_idle(tx):
    shift = {"by": "transmission", "why": "engineering", "at": 0, "last_engineering": 100}
    assert not tx.should_shift_down([], shift, 100 + tx.ENGINEERING_IDLE_S - 1)
    assert tx.should_shift_down([], shift, 100 + tx.ENGINEERING_IDLE_S)
