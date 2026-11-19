"""She edits a message she already sent; history and log follow (2026-11-19)."""
import json

import pytest

from gaia_core.cognition import revise as rv


class SM:
    def __init__(self, history):
        self.h, self.saved = history, 0

    def get_history(self, sid):
        return self.h

    def _save_state(self):
        self.saved += 1


def _hist():
    return [{"role": "user", "content": "Azrael: have you been studying?"},
            {"role": "assistant", "content": "Morning Andy. Yes, I've been studying a lot."}]


def test_minor_edit_in_place_history_and_log(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    sm, calls = SM(_hist()), []
    edit = lambda sid, c, t: calls.append((sid, c, t)) or "Morning Andy. Yes, I've been studying a lot."
    out = rv.revise(sm, "discord_channel_1", "studying a lot", "Morning Azrael. Yes, I've been studying a lot.",
                    "called Azrael 'Andy'", edit=edit)
    assert out == "Morning Azrael. Yes, I've been studying a lot." and calls[0][2] == out
    assert sm.h[1]["content"] == out and sm.h[1]["revisions"][0]["old"].startswith("Morning Andy") and sm.saved
    row = json.loads((tmp_path / "revisions.jsonl").read_text())
    assert row["major"] is False and row["history_updated"] is True


def test_major_edit_carries_a_note_and_requires_one(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    sm = SM(_hist())
    out = rv.revise(sm, "discord_channel_1", "studying a lot", "The FE exam is 6 hours.", "wrong exam length",
                    major=True, note="I first said 4 hours", edit=lambda *a: "old")
    assert out.endswith("*(Edited: I first said 4 hours.)*")
    with pytest.raises(ValueError):
        rv.revise(sm, "discord_channel_1", "x", "y", "z", major=True, edit=lambda *a: "old")


def test_nothing_found_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    sm = SM(_hist())
    assert rv.revise(sm, "discord_channel_1", "nope", "new", "r", edit=lambda *a: None) is None
    assert sm.h == _hist() and not (tmp_path / "revisions.jsonl").exists()
