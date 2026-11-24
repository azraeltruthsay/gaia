"""2026-11-17: the correction transmission shifts up after two corrected
answers in three, and back after five clean ones (not before ten minutes)."""
import json

import pytest

from gaia_core.cognition import transmission as tx


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "STATE_PATH", tmp_path / "state.json")
    rg = tmp_path / "resting.json"
    rg.write_text(json.dumps({"state": "awake"}))
    monkeypatch.setattr(tx, "RESTING_GEAR_FILE", rg)
    gear = {"now": "awake"}
    shifts = []
    monkeypatch.setattr(tx, "_current_gear", lambda: gear["now"])

    def _shift(target, reason):
        shifts.append(target)
        gear["now"] = target
        return True
    monkeypatch.setattr(tx, "_shift", _shift)
    clock = {"t": 1000.0}
    monkeypatch.setattr(tx.time, "time", lambda: clock["t"])
    return {"gear": gear, "shifts": shifts, "clock": clock, "resting": rg}


def test_two_corrections_in_three_shift_up(world):
    tx.note_turn("s1")
    assert tx.note_correction("s1", "second_mind", "can't back that up") is None
    tx.note_turn("s1")
    tx.note_turn("s1")
    assert tx.note_correction("s1", "pushback", "that's wrong") == "up"
    assert world["shifts"] == ["focusing"]


def test_corrections_in_other_conversations_dont_add_up(world):
    tx.note_turn("a")
    tx.note_correction("a", "second_mind")
    tx.note_turn("b")
    assert tx.note_correction("b", "second_mind") is None
    assert world["shifts"] == []


def test_spread_out_corrections_dont_shift(world):
    for _ in range(3):
        tx.note_turn("s1")
    tx.note_correction("s1", "second_mind")
    for _ in range(3):
        tx.note_turn("s1")
    assert tx.note_correction("s1", "second_mind") is None   # 1 of the last 3


def test_back_down_after_five_clean_turns_and_the_dwell(world):
    tx.note_turn("s1")
    tx.note_correction("s1", "x")
    tx.note_turn("s1")
    assert tx.note_correction("s1", "x") == "up"
    for _ in range(6):
        world["clock"]["t"] += 30
        assert tx.note_turn("s1") is None           # clean, but within 10 minutes
    world["clock"]["t"] += tx.MIN_DWELL_S
    assert tx.note_turn("s1") == "down"
    assert world["shifts"] == ["focusing", "awake"]


def test_a_correction_in_gear_two_resets_the_count(world):
    tx.note_turn("s1")
    tx.note_correction("s1", "x")
    tx.note_turn("s1")
    tx.note_correction("s1", "x")
    world["clock"]["t"] += tx.MIN_DWELL_S
    for _ in range(4):
        tx.note_turn("s1")
    tx.note_correction("s1", "second_mind")
    assert tx.note_turn("s1") is None


def test_leaves_a_hand_chosen_gear_alone(world):
    world["resting"].write_text(json.dumps({"state": "focusing"}))
    tx.note_turn("s1")
    tx.note_correction("s1", "x")
    tx.note_turn("s1")
    assert tx.note_correction("s1", "x") is None
    assert world["shifts"] == []


def test_disabled(world, monkeypatch):
    monkeypatch.setenv("TRANSMISSION_ENABLED", "0")
    assert tx.note_turn("s1") is None


def test_a_hand_shift_down_clears_the_record(world):
    tx.note_turn("s1")
    tx.note_correction("s1", "x")
    tx.note_turn("s1")
    assert tx.note_correction("s1", "x") == "up"
    world["gear"]["now"] = "awake"          # Azrael shifted back by hand
    world["clock"]["t"] += 1
    tx.note_turn("s1")
    assert tx.status()["shift"] is None
    world["clock"]["t"] += 1
    tx.note_turn("s1")
    assert tx.note_correction("s1", "x") is None   # old strikes don't count
    tx.note_turn("s1")
    assert tx.note_correction("s1", "x") == "up"


def test_holding_a_disputed_answer_is_no_strike(world):
    tx.note_turn("s1", reply="Canberra.")
    tx.note_pushback("s1", "Canberra.")
    tx.note_turn("s1", reply="You're right to check — the capital is indeed Canberra.")
    tx.note_pushback("s1", "the capital is indeed Canberra.")
    assert tx.note_turn("s1", reply="It's Canberra, I'm sure.") is None
    assert world["shifts"] == []


def test_conceding_twice_shifts_up(world):
    tx.note_turn("s1", reply="Sydney.")
    tx.note_pushback("s1", "Sydney.")
    tx.note_turn("s1", reply="You're right, it's Melbourne.")
    tx.note_pushback("s1", "You're right, it's Melbourne.")
    assert tx.note_turn("s1", reply="My mistake, it's Canberra.") == "up"


def test_conceded():
    assert not tx.conceded("[Core] Canberra.", "You're right, and the capital is indeed Canberra.")
    assert tx.conceded("[Core] Canberra.", "You're right, it's Paris.")
    assert tx.conceded("About 485,000 people.", "Sorry, it's about 470,000.")
