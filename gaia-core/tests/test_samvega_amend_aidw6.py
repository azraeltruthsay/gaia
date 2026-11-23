"""aidw.6: a confirmed finding about a fresh message — she corrects the
message itself, checked against the evidence, with a note saying what changed."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from gaia_common.utils import samvega_ledger as L
from gaia_core.cognition import samvega_amend as A

SID = "discord_channel_1484003096337191067"
SAID = ("The Operational wing holds the nuts and bolts of how I stay myself. Rooms there cover everything from "
        "my health monitoring and watchdog systems to my immune system, self-repair protocols.")
EVIDENCE = ("The operational wing has 3 rooms, no more: problems (0 memories), preferences (0 memories), "
            "milestones (4 memories). problems: Bugs, fixes, root causes. preferences: Conventions, style, "
            "workflow. milestones: Deployments, breakthroughs, versions.")
GOOD = "Three rooms: problems (bugs, fixes, root causes), preferences (conventions, style, workflow) and milestones, which holds 4 memories."
CALL = "palace(action=navigate, wing=operational)"


@pytest.fixture
def finding(tmp_path, monkeypatch):
    d = tmp_path / "samvega"
    d.mkdir()
    monkeypatch.setenv("SAMVEGA_DIR", str(d))
    p = d / "samvega_20261121_211651196478.json"
    p.write_text(json.dumps({"timestamp": "2026-11-21T21:16:51", "session_id": SID}))
    L.stamp(p, L.initial_fields("confirmed", SID, SAID))
    return p


def _steps(*answers):
    """The rewrite, then the blind check's steps, in order (aidw.11)."""
    it = iter(answers)
    return lambda messages: next(it, "")


def _ref(p):
    return json.loads(p.read_text())["message_ref"]


def _amend(p, generate, revise, sid=SID, ref=None, **kw):
    return A.maybe_amend(p, question="What's in your Operational wing?", said=SAID, evidence=EVIDENCE,
                         grounding_call=CALL, session_id=sid, message_ref=ref or _ref(p),
                         generate=generate, revise=revise, **kw)


def test_corrects_the_message_with_a_note(finding):
    calls = []

    def revise(session_id, containing, text, reason, note):
        calls.append((session_id, containing, text, note))
        return f"{text}\n\n*(Edited: {note}.)*"

    out = _amend(finding, _steps("YES", GOOD, "Q: What rooms are in the Operational wing?\nA: problems, preferences and milestones",
                                 "1: problems, preferences, milestones (complete)"), revise)
    assert out and out.startswith(GOOD)
    sid, containing, text, note = calls[0]
    assert sid == SID and SAID.startswith(containing) and text == GOOD
    assert note == "I checked my memory palace and corrected this"
    f = L.show(finding.stem)
    assert f["status"] == "amended" and f["history"][-1]["by"] == "samvega_amend"
    assert json.loads(finding.read_text())["amended_text"] == out


def test_a_rewrite_that_is_wrong_too_is_not_used(finding):
    out = _amend(finding, _steps("YES", "The wing has twelve rooms, including the watchdog kennel and immune vault.",
                                 "Q: How many rooms does the Operational wing have?\nA: twelve",
                                 "1: 3 (complete)"),
                 lambda *a: pytest.fail("must not edit"))
    assert out is None
    f = L.show(finding.stem)
    assert f["status"] == "open" and "didn't hold up" in f["history"][-1]["note"]


def test_old_message_is_left(finding):
    old = {**_ref(finding), "sent_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}
    assert _amend(finding, lambda m: GOOD, lambda *a: pytest.fail("no"), ref=old) is None
    assert "minutes old" in L.show(finding.stem)["history"][-1]["note"]
    assert L.show(finding.stem)["status"] == "open"


# aidw.10: DMs and web chat are amended too; other sessions are left.
@pytest.mark.parametrize("sid", ["discord_dm_596925786208993283", "web_abc12345"])
def test_dm_and_web_are_amended(finding, sid):
    seen = []

    def revise(session_id, containing, text, reason, note):
        seen.append(session_id)
        return text
    out = _amend(finding, _steps("YES", GOOD, "Q: What rooms are in the Operational wing?\nA: problems, preferences and milestones",
                                 "1: problems, preferences, milestones (complete)"), revise, sid=sid)
    assert out == GOOD and seen == [sid]


def test_other_sessions_are_left(finding):
    assert _amend(finding, lambda m: GOOD, lambda *a: pytest.fail("no"), sid="cli_local") is None
    assert "can be edited" in L.show(finding.stem)["history"][-1]["note"]


def test_message_not_found(finding):
    assert _amend(finding, _steps("YES", GOOD), lambda *a: None) is None
    assert "couldn't be found" in L.show(finding.stem)["history"][-1]["note"]


def test_source_phrase_and_anchor():
    assert A.source_phrase(CALL) == "my memory palace"
    assert A.source_phrase("weird(x=1)") == "the source"
    assert A._anchor("[Prime]\n\nThe palace has six wings.\nMore") == "The palace has six wings."


def test_evidence_that_doesnt_answer_is_not_rewritten_from(finding):
    assert _amend(finding, _steps("NO"), lambda *a: pytest.fail("no edit")) is None
    assert "doesn't answer the question" in L.show(finding.stem)["history"][-1]["note"]


def test_world_state_alone_is_no_source_to_correct_from(finding):
    """2026-11-23 DM: rewritten from her clock and recent events, noted as
    "I checked the source and corrected this"."""
    out = A.maybe_amend(finding, question="Why don't we see the other perspective?", said=SAID,
                        evidence="[world state]\nClock: 21:22 UTC\nRecent Events: ...", grounding_call="",
                        session_id=SID, message_ref=_ref(finding),
                        generate=lambda m: pytest.fail("must not generate"),
                        revise=lambda *a: pytest.fail("must not edit"))
    assert out is None
    assert "nothing was looked up" in json.dumps(L.show(finding.stem))
