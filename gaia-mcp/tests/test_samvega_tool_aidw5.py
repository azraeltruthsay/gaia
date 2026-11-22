"""aidw.5: samvega() as she calls it — her verbs, her param names; verified
never from a call."""
import asyncio
import json

import pytest

from gaia_mcp import tools


@pytest.fixture
def sdir(tmp_path, monkeypatch):
    d = tmp_path / "samvega"
    d.mkdir()
    monkeypatch.setenv("SAMVEGA_DIR", str(d))
    p = d / "samvega_20261121_211651196478.json"
    p.write_text(json.dumps({"timestamp": "2026-11-21T21:16:51", "session_id": "discord_channel_1",
                             "status": "open", "claim": "rooms cover the watchdog", "grounding": "confirmed",
                             "grounding_call": "palace(action=navigate, wing=operational)",
                             "evidence": "3 rooms: problems, preferences, milestones", "history": []}))
    from gaia_common.utils.samvega_ledger import short_id
    return short_id(p.stem)


def _call(params):
    return asyncio.run(tools.execute_limb("samvega", dict(params), None, pre_approved=False))


def test_list_shows_what_the_source_shows(sdir):
    out = _call({"action": "list"})
    f = out["findings"][0]
    assert f["id"] == sdir and f["what_went_wrong"].startswith('You said: "rooms cover the watchdog"')
    assert "palace(action=navigate" in f["what_went_wrong"]


def test_her_verbs_and_names(sdir):
    assert _call({"action": "investigate", "finding_id": sdir})["status"] == "investigating"
    assert _call({"action": "note", "id": sdir, "status": "open"})["status"] == "open"
    assert not _call({"action": "dismiss", "id": sdir})["ok"]            # needs a reason
    assert _call({"action": "dismiss", "id": sdir, "reason": "duplicate"})["status"] == "dismissed"


def test_verified_never_from_a_call(sdir):
    _call({"action": "fix", "id": sdir, "reason": "routing fixed"})
    out = _call({"action": "status", "id": sdir, "status": "verified", "as_code": True, "_by": "replay"})
    assert not out["ok"]
