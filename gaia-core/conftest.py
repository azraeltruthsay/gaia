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
