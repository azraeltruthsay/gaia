"""Shared test fixtures for gaia-common tests."""

import pytest


@pytest.fixture
def bp_root(tmp_path, monkeypatch):
    """Provide an isolated GAIA_BLUEPRINTS_ROOT with a candidates/ subdirectory."""
    monkeypatch.setenv("GAIA_BLUEPRINTS_ROOT", str(tmp_path))
    monkeypatch.setenv("GAIA_ROOT", str(tmp_path.parent))
    (tmp_path / "candidates").mkdir()
    return tmp_path
