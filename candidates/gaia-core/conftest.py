"""Tests run the same in production and candidate containers: the candidate
sets GAIA_IS_CANDIDATE=1 (gx27), which makes core skip every /gpu/wake and
presence call, so tests of those paths would pass or fail by container."""
import pytest


@pytest.fixture(autouse=True)
def _not_a_candidate(monkeypatch):
    monkeypatch.delenv("GAIA_IS_CANDIDATE", raising=False)


@pytest.fixture(autouse=True)
def _cwd_is_scratch(tmp_path, monkeypatch):
    """A MagicMock config used as a path ("…/MagicMock/mock.LOGS_DIR/…") is a
    relative path: run from scratch, it lands in tmp, not in the repo (65k
    leftover files, found 2026-11-24 by om8n's tree walk)."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _shared_is_scratch(tmp_path, monkeypatch):
    """Tests never write to the live /shared. The doctor's code audit runs this
    suite inside the running gaia-core: CodeMind cycle tests without their own
    issue dir read her real ledger, aimed a stub patch at the real candidate
    second_mind.py (stopped only by the diff-safety check), noted on her real
    issue gi-2fbe7e and used one of its tries (2026-11-25); a kept fix would
    have gone to /shared/doctor/restart_requests. A test that needs one of
    these sets it again itself."""
    shared = tmp_path / "_shared"
    monkeypatch.setenv("SHARED_DIR", str(shared))
    monkeypatch.setenv("GAIA_ISSUES_DIR", str(shared / "issues"))
    monkeypatch.setenv("CODEMIND_ATTEMPTS_DIR", str(shared / "codemind" / "attempts"))
    monkeypatch.setenv("CODEMIND_CHANGELOG_PATH", str(shared / "codemind" / "changelog.jsonl"))
