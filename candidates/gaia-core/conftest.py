"""Tests run the same in production and candidate containers: the candidate
sets GAIA_IS_CANDIDATE=1 (gx27), which makes core skip every /gpu/wake and
presence call, so tests of those paths would pass or fail by container."""
import pytest


@pytest.fixture(autouse=True)
def _not_a_candidate(monkeypatch):
    monkeypatch.delenv("GAIA_IS_CANDIDATE", raising=False)
