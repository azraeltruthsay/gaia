"""Tests for blueprint API endpoints."""

import textwrap

import pytest


_MINIMAL_BLUEPRINT_YAML = textwrap.dedent("""\
    id: test-service
    version: "0.1"
    role: "Test Service"
    service_status: live
    runtime:
      port: 9999
      gpu: false
    interfaces:
      - id: health
        direction: inbound
        description: "Health check"
        status: active
        transport:
          type: http_rest
          path: /health
          method: GET
    dependencies:
      services: []
      volumes: []
      external_apis: []
    source_files: []
    failure_modes: []
    meta:
      status: candidate
      genesis: true
      generated_by: manual_seed
      blueprint_version: "0.1"
      schema_version: "1.0"
      confidence:
        runtime: high
        contract: high
        dependencies: medium
        failure_modes: medium
        intent: low
""")


@pytest.fixture
def tmp_blueprints(tmp_path, monkeypatch):
    """Create a temp blueprints directory with a minimal valid YAML and set env var."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "test-service.yaml").write_text(_MINIMAL_BLUEPRINT_YAML)

    monkeypatch.setenv("GAIA_BLUEPRINTS_ROOT", str(tmp_path))
    yield tmp_path


def test_list_blueprints(client, tmp_blueprints):
    resp = client.get("/api/blueprints")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    entry = next(e for e in data if e["id"] == "test-service")
    assert entry["role"] == "Test Service"
    assert entry["genesis"] is True
    assert "interface_count" in entry


def test_graph_returns_topology(client, tmp_blueprints):
    resp = client.get("/api/blueprints/graph?include_candidates=true")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "blueprint_count" in data
    assert data["blueprint_count"] >= 1
    node_ids = [n["id"] for n in data["nodes"]]
    assert "test-service" in node_ids


def test_get_blueprint_detail(client, tmp_blueprints):
    resp = client.get("/api/blueprints/test-service?candidate=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "test-service"
    assert data["role"] == "Test Service"
    assert "interfaces" in data
    assert len(data["interfaces"]) == 1


def test_get_blueprint_detail_auto_fallback(client, tmp_blueprints):
    """When candidate param is omitted, should fall back to candidate if no live."""
    resp = client.get("/api/blueprints/test-service")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "test-service"


def test_get_blueprint_markdown(client, tmp_blueprints):
    resp = client.get("/api/blueprints/test-service/markdown")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    text = resp.text
    assert "test-service" in text
    assert "Test Service" in text


def test_get_missing_blueprint_404(client, tmp_blueprints):
    resp = client.get("/api/blueprints/nonexistent")
    assert resp.status_code == 404


def test_dashboard_redirect(client):
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 307
    assert "/static/index.html" in resp.headers.get("location", "")


def test_root_includes_new_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "/dashboard" in data["endpoints"]
    assert "/api/blueprints" in data["endpoints"]
