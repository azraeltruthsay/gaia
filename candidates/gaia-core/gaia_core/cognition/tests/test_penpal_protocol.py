"""fwzo: the penpal cycle must not review the back catalogue, review an
episode it never heard, or post failure text into the notebook."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gaia_core.cognition import penpal_protocol as pp

TRANSCRIPT = "Host one: so GAIA's gearbox shifts Prime onto the GPU. " * 20


def _ep(eid, created, kind="ArtifactType.AUDIO", done=True):
    return {"id": eid, "title": f"Episode {eid}", "kind": kind,
            "is_completed": done, "created_at": created}


EPISODES = [
    _ep("e3", "2026-05-23 10:00:00+00:00"),
    _ep("e1", "2026-03-01 10:00:00+00:00"),
    _ep("v1", "2026-04-01 10:00:00+00:00", kind="ArtifactType.VIDEO"),
    _ep("e2", "2026-04-03 10:00:00+00:00"),
]


class _Dir(type(Path())):
    """The penpal dir, carrying what was parked and pushed."""


@pytest.fixture
def penpal_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PENPAL_DIR", tmp_path)
    monkeypatch.setattr(pp, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(pp, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(pp, "EPISODE_DIR", tmp_path / "episodes")
    monkeypatch.setattr(pp, "LETTER_DIR", tmp_path / "letters")
    from gaia_core.cognition import approval_reply
    monkeypatch.setattr(approval_reply, "APPROVALS_DIR", tmp_path / "approvals")
    monkeypatch.setattr(pp, "_azrael_dm", lambda: "discord_dm_596925786208993283")
    pushed = []
    monkeypatch.setattr(pp, "_push", lambda sid, text: pushed.append((sid, text)) or True)
    parked = []

    def request_approval(method, params):
        parked.append((method, params))
        return {"ok": True, "action_id": f"a-{len(parked)}", "challenge": "EOKYC"}
    import gaia_core.utils.mcp_client as mc
    monkeypatch.setattr(mc, "request_approval_via_mcp", request_approval)
    d = _Dir(tmp_path)
    d.pushed, d.parked = pushed, parked
    return d


class FakeMCP:
    def __init__(self, artifacts, transcription="", sources=()):
        self.artifacts = artifacts
        self.transcription = transcription
        self.sources = list(sources)
        self.calls = []

    def __call__(self, method, params):
        self.calls.append(method)
        if method == "notebooklm_list_artifacts":
            return {"artifacts": self.artifacts}
        if method == "notebooklm_download_audio":
            return {"ok": True, "transcription": self.transcription,
                    "transcribed": bool(self.transcription)}
        if method == "notebooklm_list_sources":
            return {"sources": [{"title": t} for t in self.sources]}
        raise AssertionError(method)   # nothing is posted directly: it needs approval


def _prime_reply(text):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": text}}]}).encode()
    return Resp()


def _state(d):
    return json.loads((d / "state.json").read_text())


def _write_state(d, **kw):
    (d / "state.json").write_text(json.dumps(kw))


def test_first_run_seeds_to_newest_and_reviews_nothing(penpal_dir):
    mcp = FakeMCP(EPISODES)
    with patch.object(pp, "_mcp_call", mcp):
        result = pp.run_penpal_cycle()
    assert result == {"new_episodes": 0}
    assert not penpal_dir.parked
    st = _state(penpal_dir)
    assert st["last_reviewed_id"] == "e3"
    assert st["last_reviewed_at"].startswith("2026-05-23")


def test_only_audio_newer_than_last_review(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-03-15 00:00:00+00:00")
    with patch.object(pp, "_mcp_call", FakeMCP(EPISODES)):
        assert [a["id"] for a in pp.check_for_new_episodes()] == ["e2", "e3"]


def test_legacy_id_state_is_resolved_by_date(penpal_dir):
    _write_state(penpal_dir, last_reviewed_id="e2", reviews=[])
    with patch.object(pp, "_mcp_call", FakeMCP(EPISODES)):
        assert [a["id"] for a in pp.check_for_new_episodes()] == ["e3"]


def test_vanished_last_id_reseeds_instead_of_stalling(penpal_dir):
    _write_state(penpal_dir, last_reviewed_id="gone", reviews=[])
    with patch.object(pp, "_mcp_call", FakeMCP(EPISODES)):
        assert pp.check_for_new_episodes() == []
    assert _state(penpal_dir)["last_reviewed_id"] == "e3"


def test_no_transcript_parks_episode_without_posting(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00")
    mcp = FakeMCP(EPISODES, transcription="")
    with patch.object(pp, "_mcp_call", mcp), patch.object(pp, "urlopen") as prime:
        result = pp.run_penpal_cycle()
        assert result["reviewed"] == 0 and result["awaiting_transcript"] == 1
        prime.assert_not_called()
        # Next cycle: still waiting, and the audio is not downloaded again.
        pp.run_penpal_cycle()
    assert not penpal_dir.parked
    assert mcp.calls.count("notebooklm_download_audio") == 1
    st = _state(penpal_dir)
    assert "e3" in st["awaiting_transcript"]
    assert st["last_reviewed_at"].startswith("2026-05-01")


def test_dropped_transcript_gets_reviewed_and_unparked(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00",
                 awaiting_transcript={"e3": {"title": "Episode e3"}})
    (penpal_dir / "transcripts").mkdir()
    (penpal_dir / "transcripts" / "e3.txt").write_text(TRANSCRIPT)
    mcp = FakeMCP(EPISODES, sources=["GAIA's letter: reply to Episode e3"])
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("<think>hm</think>Dear hosts, ...")):
        result = pp.run_penpal_cycle()
    assert result["reviewed"] == 1 and result["awaiting_transcript"] == 0
    assert "notebooklm_download_audio" not in mcp.calls
    st = _state(penpal_dir)
    assert st["last_reviewed_id"] == "e3"
    assert st["reviews"][0]["response_preview"] == "Dear hosts, ..."


def test_relay_transcription_is_used_and_saved(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00")
    mcp = FakeMCP(EPISODES, transcription=TRANSCRIPT, sources=["GAIA's letter: reply to Episode e3"])
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("Dear hosts")):
        assert pp.run_penpal_cycle()["reviewed"] == 1
    assert (penpal_dir / "transcripts" / "e3.txt").read_text() == TRANSCRIPT.strip()


@pytest.mark.parametrize("prime", [RuntimeError("Prime down"), "", "<think>only thinking</think>"])
def test_prime_failure_posts_nothing_and_retries(penpal_dir, prime):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00")
    (penpal_dir / "transcripts").mkdir()
    (penpal_dir / "transcripts" / "e3.txt").write_text(TRANSCRIPT)
    mcp = FakeMCP(EPISODES)
    side = {"side_effect": prime} if isinstance(prime, Exception) else {"return_value": _prime_reply(prime)}
    with patch.object(pp, "_mcp_call", mcp), patch.object(pp, "urlopen", **side):
        assert pp.run_penpal_cycle()["reviewed"] == 0
    assert not penpal_dir.parked and not penpal_dir.pushed
    assert _state(penpal_dir)["last_reviewed_at"].startswith("2026-05-01")


def _ready(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00")
    (penpal_dir / "transcripts").mkdir()
    (penpal_dir / "transcripts" / "e3.txt").write_text(TRANSCRIPT)


def test_letter_waits_for_azraels_approval_as_a_source(penpal_dir):
    """The hosts read sources, not notes; adding one needs Azrael's code."""
    _ready(penpal_dir)
    mcp = FakeMCP(EPISODES)
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("Dear hosts")) as llm:
        assert pp.run_penpal_cycle()["reviewed"] == 0
        assert pp.run_penpal_cycle()["awaiting_approval"] == 1    # same window: no second ask
        assert llm.call_count == 1                                  # the letter is kept, not rewritten
    (method, params), = penpal_dir.parked
    assert method == "notebooklm_add_text_source"
    assert params["title"] == "GAIA's letter: reply to Episode e3" and params["content"] == "Dear hosts"
    (sid, text), = penpal_dir.pushed
    assert sid == "discord_dm_596925786208993283" and "CYKOE" in text
    from gaia_core.cognition import approval_reply
    assert approval_reply._parked(sid)["action_id"] == "a-1"
    assert (penpal_dir / "letters" / "e3.txt").read_text() == "Dear hosts"
    assert _state(penpal_dir)["last_reviewed_at"].startswith("2026-05-01")


def test_asks_again_after_the_window_then_stops(penpal_dir):
    _ready(penpal_dir)
    mcp = FakeMCP(EPISODES)
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("Dear hosts")):
        for _ in range(5):
            pp.run_penpal_cycle()
            st = _state(penpal_dir)
            st["awaiting_approval"]["e3"]["at"] -= pp.APPROVAL_WINDOW_S + 1
            (penpal_dir / "state.json").write_text(json.dumps(st))
    assert len(penpal_dir.parked) == pp.MAX_ASKS
    assert _state(penpal_dir)["awaiting_approval"]["e3"]["gave_up"] is True


def test_once_the_letter_is_in_the_notebook_the_episode_is_reviewed(penpal_dir):
    _ready(penpal_dir)
    mcp = FakeMCP(EPISODES)
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("Dear hosts")):
        pp.run_penpal_cycle()
        mcp.sources.append("GAIA's letter: reply to Episode e3")    # Azrael approved
        assert pp.run_penpal_cycle()["reviewed"] == 1
    st = _state(penpal_dir)
    assert st["last_reviewed_id"] == "e3" and st["awaiting_approval"] == {}


def test_a_failed_park_leaves_the_episode_unreviewed(penpal_dir, monkeypatch):
    _ready(penpal_dir)
    import gaia_core.utils.mcp_client as mc
    monkeypatch.setattr(mc, "request_approval_via_mcp", lambda m, p: {"ok": False, "error": "mcp down"})
    mcp = FakeMCP(EPISODES)
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("Dear hosts")):
        assert pp.run_penpal_cycle()["reviewed"] == 0
    assert not penpal_dir.pushed
    assert not _state(penpal_dir).get("awaiting_approval")


def test_mcp_error_response_raises():
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"jsonrpc": "2.0", "id": 1,
                               "error": {"code": -32000, "message": "Storage file not found"}}).encode()
    with patch.object(pp, "urlopen", return_value=Resp()):
        with pytest.raises(RuntimeError, match="Storage file not found"):
            pp._mcp_call("notebooklm_list_artifacts", {})
