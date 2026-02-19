"""Tests for terminal endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gaia_web.main import app

client = TestClient(app)


def _mock_container(name, status="running", image_tag="localhost:5000/test:local"):
    """Create a mock Docker container."""
    c = MagicMock()
    c.name = name
    c.status = status
    c.image.tags = [image_tag]
    return c


class TestContainerListing:
    @patch("gaia_web.routes.terminal.docker")
    def test_list_gaia_containers(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.containers.list.return_value = [
            _mock_container("gaia-core", "running"),
            _mock_container("gaia-web", "running"),
            _mock_container("redis", "running"),  # non-gaia, should be excluded
        ]

        resp = client.get("/api/terminal/containers")
        assert resp.status_code == 200
        data = resp.json()
        names = [c["name"] for c in data]
        assert "gaia-core" in names
        assert "gaia-web" in names
        assert "redis" not in names

    @patch("gaia_web.routes.terminal.docker")
    def test_list_containers_includes_status(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.containers.list.return_value = [
            _mock_container("gaia-core", "running"),
            _mock_container("gaia-prime", "exited"),
        ]

        resp = client.get("/api/terminal/containers")
        assert resp.status_code == 200
        data = resp.json()
        statuses = {c["name"]: c["status"] for c in data}
        assert statuses["gaia-core"] == "running"
        assert statuses["gaia-prime"] == "exited"

    @patch("gaia_web.routes.terminal._get_client")
    def test_docker_unavailable(self, mock_get_client):
        mock_get_client.return_value = None

        resp = client.get("/api/terminal/containers")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["error"].lower()
