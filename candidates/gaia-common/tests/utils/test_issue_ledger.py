"""a0aw.1: GAIA's issue ledger — the Record step of the self-repair loop."""

import json

from gaia_common.utils import issue_ledger as L
from gaia_common.utils.domain_tools import ACTION_TO_LEGACY
from gaia_common.utils.packet_utils import SAFE_SIDECAR_TOOLS
from gaia_common.utils.tools_registry import TOOLS

EVIDENCE = ["2026-11-08 INFO AgentCore: Tool routing triggered for intent: brainstorming"]


def _file(tmp_path, title="Haiku about a cat routed to a tool",
          symptom="A request for a haiku about a cat went through tool routing", **kw):
    kw.setdefault("evidence", EVIDENCE)
    ev = kw["evidence"]
    L.record_shown(ev.splitlines() if isinstance(ev, str) else ev, base=tmp_path)
    return L.file_issue(title, symptom, base=tmp_path, **kw)


def test_file_requires_evidence(tmp_path):
    r = L.file_issue("t", "s", [], base=tmp_path)
    assert not r["ok"] and "evidence" in r["error"]
    assert not (tmp_path / L.LEDGER).exists()


def test_file_then_list_and_get(tmp_path):
    r = _file(tmp_path, suspected_cause="'cat ' in exec_indicators", playbook="ts-misrouted-request")
    assert r["ok"] and r["id"].startswith("gi-")
    listed = L.list_issues(base=tmp_path)
    assert listed["count"] == 1 and listed["issues"][0]["bead"] is None
    full = L.get_issue(r["id"], base=tmp_path)["issue"]
    assert full["evidence"] == EVIDENCE and full["playbook"] == "ts-misrouted-request"


def test_evidence_string_split_into_lines(tmp_path):
    r = _file(tmp_path, evidence="line one\n\nline two")
    assert L.get_issue(r["id"], base=tmp_path)["issue"]["evidence"] == ["line one", "line two"]


def test_duplicate_returns_open_issue(tmp_path):
    first = _file(tmp_path)
    again = _file(tmp_path, title="Haiku about a cat routed to tools")
    assert again["duplicate"] and again["id"] == first["id"]
    assert L.list_issues(base=tmp_path)["count"] == 1


def test_unrelated_issue_is_not_duplicate(tmp_path):
    _file(tmp_path)
    other = _file(tmp_path, title="Stray period after short replies",
                  symptom="Excalibur. . — an extra dot appended to one-word answers")
    assert other["ok"] and not other.get("duplicate")


def test_closed_issue_does_not_block_refiling(tmp_path):
    first = _file(tmp_path)
    L.set_status(first["id"], "wontfix", base=tmp_path)
    again = _file(tmp_path)
    assert again["id"] != first["id"]


def test_daily_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "MAX_NEW_PER_DAY", 2)
    _file(tmp_path, title="alpha one", symptom="alpha symptom")
    _file(tmp_path, title="bravo two", symptom="bravo symptom")
    r = _file(tmp_path, title="charlie three", symptom="charlie symptom")
    assert not r["ok"] and "limit" in r["error"]


def test_note_and_status(tmp_path):
    iid = _file(tmp_path)["id"]
    assert L.note_issue(iid, "Found 'cat ' in exec_indicators", base=tmp_path)["ok"]
    assert not L.set_status(iid, "done", base=tmp_path)["ok"]
    assert L.set_status(iid, "proposed", note="candidate patch + test", base=tmp_path)["ok"]
    issue = L.get_issue(iid, base=tmp_path)["issue"]
    assert issue["status"] == "proposed" and len(issue["notes"]) == 2
    assert not L.note_issue("gi-nope", "x", base=tmp_path)["ok"]


def test_fixed_hidden_from_default_list(tmp_path):
    iid = _file(tmp_path)["id"]
    L.set_status(iid, "fixed", base=tmp_path)
    assert L.list_issues(base=tmp_path)["count"] == 0
    assert L.list_issues(include_closed=True, base=tmp_path)["count"] == 1


def test_bridge_map_joined(tmp_path):
    iid = _file(tmp_path)["id"]
    (tmp_path / L.BEADS_MAP).write_text(json.dumps(
        {"issues": {iid: {"bead": "GAIA_Project-abcd", "bead_status": "closed"}}, "ledger_offset": 1}))
    issue = L.get_issue(iid, base=tmp_path)["issue"]
    assert issue["bead"] == "GAIA_Project-abcd"
    # a bead closed on the host closes it for her too
    assert L.list_issues(base=tmp_path)["count"] == 0


def test_corrupt_line_skipped(tmp_path):
    iid = _file(tmp_path)["id"]
    with open(tmp_path / L.LEDGER, "a") as fh:
        fh.write("{not json\n")
    assert L.get_issue(iid, base=tmp_path)["ok"]


def test_evidence_must_match_what_she_was_shown(tmp_path):
    L.record_shown(["[gaia-core.log:39065] 16:27:21 web session yn31",
                    "  intent brainstorming | tool routing fired (intent brainstorming) | selector: no tool needed"],
                   base=tmp_path)
    invented = L.file_issue("generate_code sent to nano", "a generate_code call went to nano",
                            ["generate_code (0.596) was sent to nano"], base=tmp_path)
    assert not invented["ok"] and "introspect" in invented["error"]
    seen = L.file_issue("Haiku routed", "routing fired on a haiku",
                        ["tool routing fired (intent brainstorming)", "generate_code (0.596) was sent to nano"],
                        base=tmp_path)
    assert seen["ok"] and seen["evidence_verified"] == [True, False]
    assert L.get_issue(seen["id"], base=tmp_path)["issue"]["evidence_verified"] == [True, False]


def test_user_reported_issue_needs_no_shown_match(tmp_path):
    r = L.file_issue("Stray period", "Azrael saw 'Excalibur. .'", ["Excalibur. ."], source="user", base=tmp_path)
    assert r["ok"] and r["evidence_verified"] == [False]


def test_shown_lines_expire(tmp_path, monkeypatch):
    L.record_shown(["tool routing fired (intent brainstorming)"], base=tmp_path)
    assert L.verify_evidence(["tool routing fired (intent brainstorming)"], base=tmp_path) == [True]
    monkeypatch.setattr(L, "SHOWN_TTL_S", -1)
    assert L.verify_evidence(["tool routing fired (intent brainstorming)"], base=tmp_path) == [False]


def test_tool_wiring():
    assert ACTION_TO_LEGACY[("introspect", "turns")] == "introspect_turns"
    assert "introspect_turns" in TOOLS and "introspect_turns" in SAFE_SIDECAR_TOOLS
    for action, legacy in [("file", "issue_file"), ("note", "issue_note"),
                           ("status", "issue_status"), ("list", "issue_list")]:
        assert ACTION_TO_LEGACY[("issue", action)] == legacy
        assert legacy in TOOLS and legacy in SAFE_SIDECAR_TOOLS
    assert not any(a.startswith("issue") for (d, a) in ACTION_TO_LEGACY if d == "introspect")
