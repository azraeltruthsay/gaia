"""9nxd: a reply that claims an action must match a tool that did it."""

import time

import pytest

from gaia_core.cognition import action_claims as AC
from gaia_core.cognition import outcome_ledger as OL


def _call(tool, action="", ok=True):
    return (time.time(), tool, action, ok)


@pytest.mark.parametrize("reply", [
    "Noted: gi-000000 — Checked Again",
    "Noted in the issue ledger. ——\nGI-000000: checked again",
    "Sure. Saved to your memory palace.",
    "The note was added successfully.",
    "I've added a note saying 'checked again' to that issue.",
    "Done, I saved it to my memory.",
    "Your reminder has been scheduled.",
    "Your note is now saved.",
    "It was recorded successfully in the ledger.",
])
def test_claims_seen(reply):
    assert AC.claims(reply)


@pytest.mark.parametrize("reply", [
    "I'll add that note once you tell me which issue.",
    "I haven't saved anything yet.",
    "The note wasn't added: I couldn't find that issue.",
    "Python was created in 1991 by Guido van Rossum.",
    "Steady, thanks. How are you doing?",
    "Noted. Anything else?",
    "Added sugar is the main culprit here.",
    "Run `git commit` and it is committed.\n```\nfile was written\n```",
    "> The note was added successfully\nThat line is from your log.",
])
def test_no_claim(reply):
    assert not AC.claims(reply)


def test_9nxd_turns_are_corrected():
    # turn 1: no tool at all; turn 2: only a memory lookup
    first = AC.check("Noted: gi-000000 — Checked Again", [])
    assert first and not first["failed"] and "didn't actually" in AC.correction(first)
    second = AC.check("The note was added successfully.", [_call("knowledge", "memory")])
    assert second and "didn't actually" in AC.correction(second)


def test_a_write_that_ran_backs_the_claim():
    assert AC.check("The note was added successfully.", [_call("issue", "note")]) is None
    assert AC.check("I've saved it.", [_call("add_document")]) is None  # told fact, code-made call


def test_failed_write_named_in_correction():
    r = AC.check("I've added the note.", [_call("issue", "note", ok=False), _call("web", "search")])
    assert r["failed"] == ["issue(note)"]
    assert "issue(note) failed" in AC.correction(r)


@pytest.mark.parametrize("tool,action,write", [
    ("issue", "note", True), ("issue", "list", False), ("knowledge", "memory", False),
    ("knowledge", "add", True), ("file", "write", True), ("file", "read", False),
    ("web", "search", False), ("introspect", "turns", False), ("palace", "store", True),
    ("palace_recall_lookup", "", False), ("add_document", "", True), ("list_files", "", False),
])
def test_is_write(tool, action, write):
    assert AC.is_write(tool, action) is write


def test_ledger_keeps_session_tools_even_when_file_off(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTCOME_LEDGER_ENABLED", "0")
    OL.record("tool_execution", "s-9nxd", "p1", tool="issue", action="note", ok=True)
    OL.record("tool_execution", "s-9nxd", "p1", tool="web", action="search", ok=False, gate="rejected")
    calls = OL.session_tools("s-9nxd")
    assert [(c[1], c[2], c[3]) for c in calls] == [("issue", "note", True)]
    assert OL.session_tools("s-9nxd", within_s=-1) == []


def test_mcp_call_noted_for_the_turn(monkeypatch):
    from gaia_core.models import turn_context
    from gaia_core.utils import mcp_client
    monkeypatch.setattr(mcp_client, "_call_jsonrpc",
                        lambda m, p, e=None, t=None: {"ok": True, "response": {"result": {"ok": False}}})
    turn_context.set_turn("s-mcp-9nxd", "t1")
    try:
        mcp_client.call_jsonrpc("add_document", {"fact": "x"})
    finally:
        turn_context.clear_turn()
    mcp_client.call_jsonrpc("add_document", {"fact": "y"})  # no turn: not noted
    assert [(c[1], c[3]) for c in OL.session_tools("s-mcp-9nxd")] == [("add_document", False)]
