"""
Tests for gaia_common.utils.blueprint_precheck

Uses a minimal test blueprint and synthetic source files via tmp_path.
"""

import pytest
from datetime import datetime
from pathlib import Path

from gaia_common.models.blueprint import (
    BlueprintMeta,
    BlueprintModel,
    BlueprintStatus,
    Dependencies,
    FailureMode,
    GeneratedBy,
    HttpRestInterface,
    Interface,
    InterfaceDirection,
    InterfaceStatus,
    Runtime,
    ServiceDependency,
    ServiceStatus,
    Severity,
    WebSocketInterface,
)
from gaia_common.utils.blueprint_precheck import (
    PreCheckItem,
    PreCheckResult,
    run_blueprint_precheck,
)


# ── Test fixtures ────────────────────────────────────────────────────────────

def _make_blueprint() -> BlueprintModel:
    """Create a minimal test blueprint with known endpoints, deps, and failure modes."""
    return BlueprintModel(
        id="test-service",
        version="0.1.0",
        role="Test Service",
        service_status=ServiceStatus.CANDIDATE,
        runtime=Runtime(port=8000),
        interfaces=[
            Interface(
                id="health",
                direction=InterfaceDirection.INBOUND,
                transport=HttpRestInterface(path="/health", method="GET"),
                description="Health check endpoint",
                status=InterfaceStatus.ACTIVE,
            ),
            Interface(
                id="process",
                direction=InterfaceDirection.INBOUND,
                transport=HttpRestInterface(path="/process", method="POST"),
                description="Process a request",
                status=InterfaceStatus.ACTIVE,
            ),
            Interface(
                id="delete-item",
                direction=InterfaceDirection.INBOUND,
                transport=HttpRestInterface(path="/items/{item_id}", method="DELETE"),
                description="Delete an item",
                status=InterfaceStatus.ACTIVE,
            ),
            Interface(
                id="ws-stream",
                direction=InterfaceDirection.INBOUND,
                transport=WebSocketInterface(path="/ws/stream"),
                description="WebSocket streaming",
                status=InterfaceStatus.ACTIVE,
            ),
            # Outbound — should NOT be checked as endpoint
            Interface(
                id="call-prime",
                direction=InterfaceDirection.OUTBOUND,
                transport=HttpRestInterface(path="/v1/completions", method="POST"),
                description="Call gaia-prime for inference",
                status=InterfaceStatus.ACTIVE,
            ),
        ],
        dependencies=Dependencies(
            services=[
                ServiceDependency(id="gaia-core", role="brain", required=True),
                ServiceDependency(id="gaia-prime", role="inference", required=True, fallback="groq-api"),
                ServiceDependency(id="gaia-phantom", role="nonexistent", required=False),
            ],
        ),
        failure_modes=[
            FailureMode(
                condition="gaia-prime unavailable",
                response="fallback to Groq API",
                severity=Severity.DEGRADED,
            ),
            FailureMode(
                condition="request timeout",
                response="return 504 Gateway Timeout",
                severity=Severity.PARTIAL,
            ),
            FailureMode(
                condition="cosmic_ray_bit_flip",
                response="reboot universe",
                severity=Severity.FATAL,
            ),
        ],
        meta=BlueprintMeta(
            status=BlueprintStatus.CANDIDATE,
            generated_by=GeneratedBy.MANUAL_SEED,
        ),
    )


def _write_source_files(tmp_path: Path) -> None:
    """Write synthetic source files that partially match the blueprint."""
    main_py = tmp_path / "main.py"
    main_py.write_text('''
from fastapi import APIRouter, WebSocket

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.post("/process")
async def process_request(data: dict):
    try:
        result = httpx.post("http://gaia-core:8000/turn", json=data)
        return result.json()
    except httpx.TimeoutException:
        return {"error": "timeout"}, 504
    except httpx.ConnectError:
        return {"error": "unavailable"}, 502

@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
''')

    client_py = tmp_path / "client.py"
    client_py.write_text('''
import httpx
from gaia_core.models import InferenceRequest

PRIME_ENDPOINT = "http://gaia-prime:8080/v1/completions"

async def call_prime(prompt: str):
    try:
        resp = await httpx.AsyncClient().post(PRIME_ENDPOINT, json={"prompt": prompt})
        return resp.json()
    except httpx.ConnectError:
        # fallback to groq
        return await call_groq_fallback(prompt)
''')


# ── Tests ────────────────────────────────────────────────────────────────────

class TestEndpointChecking:
    """Test endpoint detection in source files."""

    def test_found_endpoints(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["endpoint"])

        found = [i for i in result.items if i.status == "found"]
        found_claims = {i.blueprint_claim for i in found}
        assert "GET /health" in found_claims
        assert "POST /process" in found_claims
        assert "WS /ws/stream" in found_claims

    def test_missing_endpoint(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["endpoint"])

        missing = [i for i in result.items if i.status == "missing"]
        missing_claims = {i.blueprint_claim for i in missing}
        assert "DELETE /items/{item_id}" in missing_claims

    def test_outbound_not_checked(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["endpoint"])

        # Should only check inbound endpoints (4), not outbound (1)
        assert len(result.items) == 4

    def test_endpoint_source_file_reported(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["endpoint"])

        health = [i for i in result.items if "GET /health" in i.blueprint_claim][0]
        assert health.source_file is not None
        assert "main.py" in health.source_file


class TestDependencyChecking:
    """Test dependency reference detection."""

    def test_found_dependencies(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["dependency"])

        found = [i for i in result.items if i.status == "found"]
        found_claims = " ".join(i.blueprint_claim for i in found)
        assert "gaia-core" in found_claims
        assert "gaia-prime" in found_claims

    def test_missing_dependency(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["dependency"])

        missing = [i for i in result.items if i.status == "missing"]
        missing_claims = " ".join(i.blueprint_claim for i in missing)
        assert "gaia-phantom" in missing_claims


class TestFailureModeChecking:
    """Test failure mode handler detection."""

    def test_found_failure_modes(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["failure_mode"])

        found = [i for i in result.items if i.status == "found"]
        found_claims = {i.blueprint_claim for i in found}
        # "gaia-prime unavailable" should match ConnectError or fallback pattern
        assert "gaia-prime unavailable" in found_claims
        # "request timeout" should match TimeoutException
        assert "request timeout" in found_claims

    def test_missing_failure_mode(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["failure_mode"])

        missing = [i for i in result.items if i.status == "missing"]
        missing_claims = {i.blueprint_claim for i in missing}
        assert "cosmic_ray_bit_flip" in missing_claims


class TestPreCheckResult:
    """Test result structure and rendering."""

    def test_summary_counts(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path))

        assert result.summary.total == result.summary.found + result.summary.missing + result.summary.diverged
        assert result.summary.total > 0

    def test_service_id_preserved(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path))
        assert result.service_id == "test-service"

    def test_timestamp_set(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path))
        assert isinstance(result.timestamp, datetime)

    def test_to_prompt_text_format(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path))
        text = result.to_prompt_text()

        assert "## Mechanical Pre-Check Results: test-service" in text
        assert "[FOUND]" in text
        assert "[MISSING]" in text
        assert "### Summary" in text
        assert "Total checks:" in text
        assert "Structural completeness:" in text

    def test_to_prompt_text_sections(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path))
        text = result.to_prompt_text()

        assert "### Endpoints" in text
        assert "### Failure Modes" in text
        assert "### Dependencies" in text


class TestCategoryFiltering:
    """Test that category filtering works."""

    def test_single_category(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path), categories=["endpoint"])

        categories = {i.category for i in result.items}
        assert categories == {"endpoint"}

    def test_multiple_categories(self, tmp_path):
        _write_source_files(tmp_path)
        bp = _make_blueprint()
        result = run_blueprint_precheck(
            bp, str(tmp_path), categories=["endpoint", "dependency"]
        )

        categories = {i.category for i in result.items}
        assert "endpoint" in categories
        assert "dependency" in categories
        assert "failure_mode" not in categories


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_source_dir(self, tmp_path):
        bp = _make_blueprint()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = run_blueprint_precheck(bp, str(empty_dir))
        # All items should be missing (nothing found)
        assert all(i.status == "missing" for i in result.items)

    def test_nonexistent_source_dir(self, tmp_path):
        bp = _make_blueprint()
        result = run_blueprint_precheck(bp, str(tmp_path / "nonexistent"))
        assert all(i.status == "missing" for i in result.items)

    def test_empty_blueprint(self, tmp_path):
        _write_source_files(tmp_path)
        bp = BlueprintModel(
            id="empty-service",
            version="0.1.0",
            role="Empty",
            service_status=ServiceStatus.CANDIDATE,
            meta=BlueprintMeta(
                status=BlueprintStatus.CANDIDATE,
                generated_by=GeneratedBy.MANUAL_SEED,
            ),
        )
        result = run_blueprint_precheck(bp, str(tmp_path))
        assert result.summary.total == 0
        assert len(result.items) == 0
