"""fwzo: the penpal cycle must not review the back catalogue, review an
episode it never heard, or post failure text into the notebook."""

import json
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


@pytest.fixture
def penpal_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PENPAL_DIR", tmp_path)
    monkeypatch.setattr(pp, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(pp, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(pp, "EPISODE_DIR", tmp_path / "episodes")
    return tmp_path


class FakeMCP:
    def __init__(self, artifacts, transcription="", note_error=None):
        self.artifacts = artifacts
        self.transcription = transcription
        self.note_error = note_error
        self.calls = []

    def __call__(self, method, params):
        self.calls.append(method)
        if method == "notebooklm_list_artifacts":
            return {"artifacts": self.artifacts}
        if method == "notebooklm_download_audio":
            return {"ok": True, "transcription": self.transcription,
                    "transcribed": bool(self.transcription)}
        if method == "notebooklm_create_note":
            if self.note_error:
                raise RuntimeError(self.note_error)
            return {"ok": True}
        raise AssertionError(method)

    def notes(self):
        return self.calls.count("notebooklm_create_note")


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
    assert mcp.notes() == 0
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
    assert mcp.notes() == 0
    assert mcp.calls.count("notebooklm_download_audio") == 1
    st = _state(penpal_dir)
    assert "e3" in st["awaiting_transcript"]
    assert st["last_reviewed_at"].startswith("2026-05-01")


def test_dropped_transcript_gets_reviewed_and_unparked(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00",
                 awaiting_transcript={"e3": {"title": "Episode e3"}})
    (penpal_dir / "transcripts").mkdir()
    (penpal_dir / "transcripts" / "e3.txt").write_text(TRANSCRIPT)
    mcp = FakeMCP(EPISODES)
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
    mcp = FakeMCP(EPISODES, transcription=TRANSCRIPT)
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
    assert mcp.notes() == 0
    assert _state(penpal_dir)["last_reviewed_at"].startswith("2026-05-01")


def test_note_failure_leaves_episode_unreviewed(penpal_dir):
    _write_state(penpal_dir, last_reviewed_at="2026-05-01 00:00:00+00:00")
    (penpal_dir / "transcripts").mkdir()
    (penpal_dir / "transcripts" / "e3.txt").write_text(TRANSCRIPT)
    mcp = FakeMCP(EPISODES, note_error="auth expired")
    with patch.object(pp, "_mcp_call", mcp), \
            patch.object(pp, "urlopen", return_value=_prime_reply("Dear hosts")):
        assert pp.run_penpal_cycle()["reviewed"] == 0
    st = _state(penpal_dir)
    assert st["last_reviewed_at"].startswith("2026-05-01") and not st.get("reviews")


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
