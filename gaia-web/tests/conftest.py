"""Shared fixtures for gaia-web tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    We import inside the fixture to avoid import-time side effects
    (Discord bot startup, etc.) by setting safe env vars first.
    """
    import os
    os.environ.setdefault("ENABLE_DISCORD", "0")
    os.environ.setdefault("DISCORD_BOT_TOKEN", "")
    os.environ.setdefault("CORE_ENDPOINT", "http://localhost:9999")

    from gaia_web.main import app
    with TestClient(app) as c:
        yield c
