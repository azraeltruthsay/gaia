"""a0aw.3: replay the offending turn on gaia-core-candidate and check its
own log for the symptom — mocked urlopen + mocked log reads, no network."""

import json

from gaia_common.utils import codemind_replay as CR


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b""


def test_check_for_playbook_known_and_unknown():
    checks = CR.check_for_playbook("ts-misrouted-request")
    assert checks and "expect_absent" in checks
    assert CR.check_for_playbook("ts-not-a-real-playbook") is None
    assert CR.check_for_playbook("") is None


def test_build_packet_shape():
    packet = CR._build_packet("Write a haiku about a cat watching snow.")
    assert packet["version"] == "v0.4"
    assert packet["content"]["original_prompt"] == "Write a haiku about a cat watching snow."
    assert packet["header"]["session_id"].startswith("codemind-replay-")
    assert packet["header"]["persona"] == {"persona_id": "gaia", "role": "assistant"}
    assert packet["governance"]["security_scan"]["passed"] is True
    # unique per call, so a replay never collides with a real session
    assert CR._build_packet("x")["header"]["packet_id"] != CR._build_packet("x")["header"]["packet_id"]


def test_replay_turn_no_check_given_is_inconclusive_without_any_io(monkeypatch):
    called = []
    monkeypatch.setattr(CR.urllib.request, "urlopen", lambda *a, **k: called.append(1) or _FakeResp())
    result = CR.replay_turn("Write a haiku about a cat.")
    assert result == {"ok": False, "symptom_seen": None, "detail": "no original_prompt or no check given"}
    assert not called  # never even tries to reach the candidate


def test_replay_turn_detects_the_symptom(monkeypatch):
    sent = []
    monkeypatch.setattr(CR.urllib.request, "urlopen",
                        lambda req, timeout=None: sent.append(json.loads(req.data)) or _FakeResp())
    monkeypatch.setattr(CR.turn_review, "read_log_lines", lambda paths, since: [
        ("gaia-core-candidate.log:12", since, "INFO", "GAIA.AgentCore",
         "Tool routing triggered for intent: brainstorming")])
    result = CR.replay_turn("Write a haiku about a cat watching snow.",
                            expect_absent=[r"Tool routing triggered for intent:"])
    assert result["ok"] is True and result["symptom_seen"] is True
    assert "Tool routing triggered" in result["detail"]
    assert sent[0]["content"]["original_prompt"] == "Write a haiku about a cat watching snow."


def test_replay_turn_clean_window_confirms_gone(monkeypatch):
    monkeypatch.setattr(CR.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    monkeypatch.setattr(CR.time, "sleep", lambda s: None)
    monkeypatch.setattr(CR.turn_review, "read_log_lines", lambda paths, since: [
        ("gaia-core-candidate.log:12", since, "INFO", "GAIA.AgentCore", "Processing packet x: '...' from web")])
    result = CR.replay_turn("Write a haiku about a cat watching snow.",
                            expect_absent=[r"Tool routing triggered for intent:"],
                            poll_timeout_s=0.05)
    assert result == {"ok": True, "symptom_seen": False, "detail": "no disallowed line seen within the poll window"}


def test_replay_turn_unreachable_candidate_is_inconclusive(monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(CR.urllib.request, "urlopen", _boom)
    monkeypatch.setattr(CR.time, "sleep", lambda s: None)
    monkeypatch.setattr(CR.turn_review, "read_log_lines", lambda paths, since: [])
    result = CR.replay_turn("Write a haiku about a cat watching snow.",
                            expect_absent=[r"Tool routing triggered for intent:"],
                            poll_timeout_s=0.05)
    assert result == {"ok": False, "symptom_seen": None,
                      "detail": "candidate produced no log output in the window"}


def test_replay_turn_expect_present_satisfied(monkeypatch):
    monkeypatch.setattr(CR.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    monkeypatch.setattr(CR.turn_review, "read_log_lines", lambda paths, since: [
        ("f:1", since, "INFO", "x", "Processing packet done")])
    result = CR.replay_turn("What's 2+2?", expect_present=[r"Processing packet"], poll_timeout_s=5)
    assert result == {"ok": True, "symptom_seen": False, "detail": "expected line seen, nothing disallowed"}
