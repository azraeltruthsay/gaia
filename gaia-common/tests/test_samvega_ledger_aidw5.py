"""aidw.5: Samvega findings as a tracker — ids, statuses, history."""
import json

import pytest

from gaia_common.utils import samvega_ledger as L


@pytest.fixture
def sdir(tmp_path, monkeypatch):
    d = tmp_path / "samvega"
    d.mkdir()
    monkeypatch.setenv("SAMVEGA_DIR", str(d))
    (d / "samvega_20260307_011954938002.json").write_text(json.dumps(
        {"timestamp": "2026-03-07T01:19:54", "what_went_wrong": "old", "weight": 0.4}))
    new = d / "samvega_20261121_211651196478.json"
    new.write_text(json.dumps({"timestamp": "2026-11-21T21:16:51", "session_id": "discord_channel_1484003096337191067",
                               "what_went_wrong": "invented rooms", "weight": 0.75}))
    L.stamp(new, {**L.initial_fields("confirmed", "discord_channel_1484003096337191067", "The Operational wing holds..."),
                  "claim": "Rooms there cover everything from my health monitoring"})
    return d


def test_ids_and_legacy(sdir):
    out = L.list_findings(status="all")
    assert out["counts"] == {"open": 1, "legacy": 1}
    new = [f for f in out["findings"] if f["status"] == "open"][0]
    assert new["id"] == L.short_id("samvega_20261121_211651196478") and new["where"] == "a Discord channel"
    assert L.list_findings()["findings"][0]["claim"].startswith("Rooms there")


def test_message_ref(sdir):
    f = L.show(L.short_id("samvega_20261121_211651196478"))
    ref = f["message_ref"]
    assert ref["platform"] == "discord" and ref["channel_id"] == "1484003096337191067"
    assert ref["said"].startswith("The Operational wing")
    assert L.message_ref("discord_dm_596925786208993283", "x")["user_id"] == "596925786208993283"
    assert L.message_ref("web_abc", "x")["platform"] == "web"


def test_her_moves(sdir):
    sid = L.short_id("samvega_20261121_211651196478")
    assert not L.set_status(sid, "verified")["ok"]                      # code's alone
    assert not L.set_status(sid, "dismissed")["ok"]                     # needs a reason
    assert L.set_status(sid, "investigating")["ok"]
    assert L.note(sid, "the router never sent this to the palace")["ok"]
    assert L.set_status(sid, "fixed", reason="aidw.1: she decides on every turn")["ok"]
    assert not L.set_status(sid, "open")["ok"]                          # fixed -> only investigating
    assert L.set_status(sid, "verified", by="replay", as_code=True)["ok"]
    f = L.show(sid)
    assert f["status"] == "verified"
    assert [h["status"] for h in f["history"]] == ["open", "investigating", "investigating", "fixed", "verified"]


def test_legacy_can_be_opened_and_unknown_ref(sdir):
    assert L.set_status("samvega_20260307_011954938002", "open")["ok"]
    assert not L.show("sv-00000000")["ok"]
    assert not L.set_status("sv-00000000", "fixed", reason="x")["ok"]


def test_noted_when_not_confirmed():
    assert L.initial_fields("unverified")["status"] == "noted"
    assert L.initial_fields("confirmed")["status"] == "open"


def test_tool_mapping():
    from gaia_common.utils.domain_tools import ACTION_TO_LEGACY as A
    assert A[("samvega", "list")] == "samvega_list" and A[("samvega", "status")] == "samvega_status"


def test_ledger_questions_are_not_the_spec():
    from gaia_common.utils.founding_docs import named_document
    assert named_document("Do you have any open findings in your samvega ledger?") is None
    assert named_document("What mistakes are in your samvega?") is None
    assert named_document("What do you think of your Samvega system?")[1] == "the Samvega Spec"
