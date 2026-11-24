"""She reads the whole bead tracker (Azrael, 2026-11-24): asked about her beads
she answered from the empty promotion queue and said the list was empty."""
import importlib.util

import pytest
import json
import os
from pathlib import Path

from gaia_common.utils import issue_ledger as ledger
from gaia_common.utils.domain_tools import DOMAIN_TOOLS


def _tracker(tmp_path, rows):
    (tmp_path / "tracker.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


ROWS = [
    {"id": "GAIA_Project-aaaa", "title": "Old closed thing", "status": "closed", "priority": 1,
     "issue_type": "bug", "updated_at": "2026-11-01T00:00:00Z", "close_reason": "done"},
    {"id": "GAIA_Project-b1ak", "title": "Parked Core counted as an outside tenant", "status": "open",
     "priority": 1, "issue_type": "bug", "updated_at": "2026-11-20T00:00:00Z",
     "description": "the orchestrator refuses parked to awake"},
    {"id": "GAIA_Project-w8c9", "title": "AI answer site presented as research", "status": "open",
     "priority": 3, "issue_type": "bug", "updated_at": "2026-11-23T00:00:00Z"},
    {"id": "GAIA_Project-zz01", "title": "Second mind reads file names", "status": "in_progress",
     "priority": 2, "issue_type": "task", "updated_at": "2026-11-24T00:00:00Z"},
]


def test_active_beads_by_priority(tmp_path):
    out = ledger.list_beads(base=_tracker(tmp_path, ROWS))
    assert out["ok"] and [b["id"][-4:] for b in out["beads"]] == ["b1ak", "zz01", "w8c9"]
    assert out["tracker_totals"] == {"closed": 1, "open": 2, "in_progress": 1}
    assert out["exported_minutes_ago"] == 0


def test_all_and_query(tmp_path):
    base = _tracker(tmp_path, ROWS)
    assert ledger.list_beads(status="all", base=base)["matching"] == 4
    assert [b["id"] for b in ledger.list_beads(query="orchestrator parked", base=base)["beads"]] == ["GAIA_Project-b1ak"]


def test_one_bead_by_short_id(tmp_path):
    out = ledger.get_bead("b1ak", base=_tracker(tmp_path, ROWS))
    assert out["ok"] and out["bead"]["description"].startswith("the orchestrator")
    assert not ledger.get_bead("nope", base=tmp_path)["ok"]


def test_no_export_yet_says_so(tmp_path):
    out = ledger.list_beads(base=tmp_path)
    assert not out["ok"] and "exported" in out["error"]


def test_tool_names_beads():
    issue = DOMAIN_TOOLS["issue"]
    assert {"beads", "bead"} <= set(issue["actions"])
    assert "beads" in issue["description"].lower() and "not file or manage" in issue["description"]


def test_audit_reads_beads_but_never_sets_a_status():
    from gaia_core.cognition.finding_grounding import _read_only
    assert _read_only("issue", {"action": "beads"}) and _read_only("issue", {"action": "bead", "id": "b1ak"})
    assert _read_only("issue", {"action": "status"})
    assert not _read_only("issue", {"action": "status", "id": "gi-1", "status": "fixed"})


def test_bridge_drops_identity(monkeypatch):
    root = Path(os.environ.get("GAIA_REPO_ROOT", "/gaia/GAIA_Project"))
    path = root / "scripts" / "gaia_issue_bridge.py"
    if not path.exists():
        import pytest
        pytest.skip("bridge script not mounted")
    spec = importlib.util.spec_from_file_location("bridge", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    row = {"id": "GAIA_Project-x", "owner": "someone@example.com", "created_by": "a", "assignee": "b",
           "dependencies": [{"issue_id": "GAIA_Project-x", "depends_on_id": "GAIA_Project-y", "type": "blocks",
                             "created_by": "a"}],
           "comments": [{"id": "c", "author": "a", "text": "hi", "created_at": "t"}]}
    out = bridge.tracker_rows(json.dumps(row))[0]
    assert "owner" not in out and "created_by" not in out and "assignee" not in out
    assert out["dependencies"] == [{"on": "GAIA_Project-y", "type": "blocks"}]
    assert out["comments"] == [{"at": "t", "text": "hi"}]


def test_summary_says_how_many_are_shown(tmp_path):
    out = ledger.list_beads(limit=2, base=_tracker(tmp_path, ROWS))
    assert out["summary"] == ("3 beads active (open, in progress or blocked); these are the top 2 by priority. "
                              "Whole tracker: 4 beads (1 closed, 1 in progress, 2 open).")


@pytest.mark.parametrize("text,asks", [
    ("Can you see your development beads?", True),
    ("What's on the backlog?", True),
    ("Any open issues?", True),
    ("What does GAIA_Project-b87r say?", True),
    ("Show me bead b1ak", True),
    ("What P1 beads are open?", True),
    ("She strung glass beads on a necklace", False),
    ("the beads of sweat on his brow", False),
    ("I filed my taxes", False),
])
def test_tracker_questions(text, asks):
    assert ledger.asks_about_tracker(text) is asks
