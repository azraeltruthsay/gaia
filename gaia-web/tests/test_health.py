"""Tests for gaia-web health and root endpoints."""


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gaia-web"


def test_root_serves_dashboard(client):
    """Root / now serves the Mission Control dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
