"""Tests for the /cognition/checkpoint endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked cognitive system."""
    import gaia_core.main as main_mod

    # Mock the cognitive system components
    mock_config = MagicMock()
    mock_config.SHARED_DIR = "/tmp/test_shared"
    mock_model_pool = MagicMock()
    mock_session_manager = MagicMock()

    mock_ai_manager = MagicMock()
    mock_ai_manager.config = mock_config
    mock_ai_manager.model_pool = mock_model_pool
    mock_ai_manager.session_manager = mock_session_manager

    main_mod._ai_manager = mock_ai_manager
    main_mod._agent_core = MagicMock()

    yield TestClient(main_mod.app, raise_server_exceptions=False)

    # Cleanup
    main_mod._ai_manager = None
    main_mod._agent_core = None


def test_checkpoint_endpoint_returns_200(client):
    """POST /cognition/checkpoint should return 200 with results."""
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = lambda self: "/tmp/test_shared/sleep_state/prime.md"

    with (
        patch(
            "gaia_core.cognition.prime_checkpoint.PrimeCheckpointManager",
            autospec=True,
        ) as MockPCM,
        patch(
            "gaia_core.cognition.lite_journal.LiteJournal",
            autospec=True,
        ) as MockLJ,
    ):
        mock_pcm_inst = MockPCM.return_value
        mock_pcm_inst.create_checkpoint.return_value = mock_path
        mock_lj_inst = MockLJ.return_value
        mock_lj_inst.write_entry.return_value = "Test journal entry"

        resp = client.post("/cognition/checkpoint")

    assert resp.status_code == 200
    data = resp.json()
    assert data["prime"]["status"] == "ok"
    assert data["lite"]["status"] == "ok"


def test_checkpoint_endpoint_returns_503_when_not_initialized():
    """POST /cognition/checkpoint returns 503 when system not initialized."""
    import gaia_core.main as main_mod

    old_ai = main_mod._ai_manager
    main_mod._ai_manager = None

    try:
        test_client = TestClient(main_mod.app, raise_server_exceptions=False)
        resp = test_client.post("/cognition/checkpoint")
        assert resp.status_code == 503
    finally:
        main_mod._ai_manager = old_ai


def test_checkpoint_endpoint_handles_prime_failure(client):
    """If prime checkpoint fails, lite should still be attempted."""
    with (
        patch(
            "gaia_core.cognition.prime_checkpoint.PrimeCheckpointManager",
            autospec=True,
        ) as MockPCM,
        patch(
            "gaia_core.cognition.lite_journal.LiteJournal",
            autospec=True,
        ) as MockLJ,
    ):
        mock_pcm_inst = MockPCM.return_value
        mock_pcm_inst.rotate_checkpoints.side_effect = RuntimeError("disk full")

        mock_lj_inst = MockLJ.return_value
        mock_lj_inst.write_entry.return_value = "Entry ok"

        resp = client.post("/cognition/checkpoint")

    assert resp.status_code == 200
    data = resp.json()
    assert data["prime"]["status"] == "error"
    assert data["lite"]["status"] == "ok"


def test_checkpoint_endpoint_lite_skipped_no_model(client):
    """If Lite model unavailable, lite entry is skipped gracefully."""
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = lambda self: "/tmp/prime.md"

    with (
        patch(
            "gaia_core.cognition.prime_checkpoint.PrimeCheckpointManager",
            autospec=True,
        ) as MockPCM,
        patch(
            "gaia_core.cognition.lite_journal.LiteJournal",
            autospec=True,
        ) as MockLJ,
    ):
        mock_pcm_inst = MockPCM.return_value
        mock_pcm_inst.create_checkpoint.return_value = mock_path

        mock_lj_inst = MockLJ.return_value
        mock_lj_inst.write_entry.return_value = None  # No model available

        resp = client.post("/cognition/checkpoint")

    assert resp.status_code == 200
    data = resp.json()
    assert data["prime"]["status"] == "ok"
    assert data["lite"]["status"] == "skipped"
