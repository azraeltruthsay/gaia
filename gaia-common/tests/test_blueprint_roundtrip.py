"""
Integration test: Generate a brand-new blueprint from scratch, save it,
reload it, validate it, compute divergence, derive topology, render markdown.

Exercises the full blueprint lifecycle end-to-end.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from gaia_common.models.blueprint import (
    BlueprintMeta,
    BlueprintModel,
    Dependencies,
    ExternalApiDependency,
    FailureMode,
    GeneratedBy,
    HttpRestInterface,
    Intent,
    Interface,
    Runtime,
    SectionConfidence,
    ServiceDependency,
    Severity,
    SourceFile,
    VolumeAccess,
    VolumeDependency,
)
from gaia_common.utils.blueprint_io import (
    compute_divergence_score,
    derive_graph_topology,
    load_all_candidate_blueprints,
    load_blueprint,
    render_markdown,
    save_blueprint,
    validate_candidate_blueprint,
)


@pytest.fixture
def tmp_blueprints(bp_root):
    """Alias bp_root (from conftest) for compatibility with existing test names."""
    return bp_root


def _make_gaia_monitor_blueprint() -> BlueprintModel:
    """Build a realistic gaia-monitor blueprint from scratch — pure Python, no YAML."""
    return BlueprintModel(
        id="gaia-monitor",
        version="0.1",
        role="The Watcher (Health Monitoring)",
        service_status="live",
        runtime=Runtime(
            port=6420,
            base_image="python:3.11-slim",
            gpu=False,
            startup_cmd="uvicorn gaia_monitor.main:app --host 0.0.0.0 --port 6420",
            health_check="curl -f http://localhost:6420/health",
            dockerfile="gaia-monitor/Dockerfile",
            compose_service="gaia-monitor",
        ),
        interfaces=[
            Interface(
                id="health",
                direction="inbound",
                transport=HttpRestInterface(path="/health", method="GET"),
                description="Container health check endpoint.",
                status="active",
            ),
            Interface(
                id="dashboard",
                direction="inbound",
                transport=HttpRestInterface(path="/dashboard", method="GET"),
                description="Monitoring dashboard with service health grid.",
                status="active",
            ),
            Interface(
                id="metrics",
                direction="inbound",
                transport=HttpRestInterface(
                    path="/metrics",
                    method="GET",
                    output_schema="PrometheusMetrics",
                ),
                description="Prometheus-compatible metrics endpoint.",
                status="active",
            ),
            Interface(
                id="alerts_webhook",
                direction="inbound",
                transport=HttpRestInterface(
                    path="/alerts/webhook",
                    method="POST",
                    input_schema="AlertPayload",
                ),
                description="Receive alerts from external monitoring systems.",
                status="planned",
            ),
            # outbound: polls health endpoints of all other services
            Interface(
                id="probe_core_health",
                direction="outbound",
                transport=HttpRestInterface(path="/health", method="GET"),
                description="Poll gaia-core health endpoint every 30s.",
                status="active",
            ),
            Interface(
                id="probe_web_health",
                direction="outbound",
                transport=HttpRestInterface(path="/health", method="GET"),
                description="Poll gaia-web health endpoint every 30s.",
                status="active",
            ),
            Interface(
                id="probe_mcp_health",
                direction="outbound",
                transport=HttpRestInterface(path="/health", method="GET"),
                description="Poll gaia-mcp health endpoint every 30s.",
                status="active",
            ),
        ],
        dependencies=Dependencies(
            services=[
                ServiceDependency(
                    id="gaia-core",
                    role="monitored",
                    required=False,
                ),
                ServiceDependency(
                    id="gaia-web",
                    role="monitored",
                    required=False,
                ),
                ServiceDependency(
                    id="gaia-mcp",
                    role="monitored",
                    required=False,
                ),
            ],
            volumes=[
                VolumeDependency(
                    name="gaia-monitor-data",
                    access=VolumeAccess.RW,
                    mount_path="/data",
                    purpose="Persistent storage for metrics history.",
                ),
            ],
            external_apis=[
                ExternalApiDependency(
                    name="Discord Webhook",
                    purpose="Alert notifications sent to Discord channel.",
                    required=False,
                ),
            ],
        ),
        source_files=[
            SourceFile(path="gaia_monitor/main.py", role="entrypoint", file_type="python"),
            SourceFile(path="gaia_monitor/probes.py", role="core_logic", file_type="python"),
            SourceFile(path="gaia_monitor/alerts.py", role="core_logic", file_type="python"),
            SourceFile(path="Dockerfile", role="config", file_type="dockerfile"),
        ],
        failure_modes=[
            FailureMode(
                condition="All monitored services down",
                severity=Severity.FATAL,
                response="Continue logging metrics locally; retry probes with exponential backoff.",
            ),
            FailureMode(
                condition="Discord webhook unreachable",
                severity=Severity.DEGRADED,
                response="Queue alerts locally; batch-send when webhook recovers.",
            ),
            FailureMode(
                condition="Metrics storage full",
                severity=Severity.DEGRADED,
                response="Rotate oldest metrics; emit self-alert.",
            ),
        ],
        intent=Intent(
            purpose="Centralized health monitoring for all GAIA services. Provides a single dashboard showing service status, uptime trends, and alert history. Designed to be the one service that runs even when everything else is down.",
            cognitive_role="The Watcher",
            design_decisions=[
                "All service dependencies are optional — monitor must survive total stack failure",
                "Metrics stored locally (no external database) to minimize failure surface",
                "Probe interval configurable via env var PROBE_INTERVAL_SECONDS (default 30)",
                "Alert deduplication window prevents notification storms during cascading failures",
            ],
            open_questions=[
                "Should gaia-monitor also watch itself via a secondary health-check loop?",
                "Should metrics be exposed in OpenTelemetry format in addition to Prometheus?",
                "Integration with gaia-orchestrator for auto-restart of unhealthy services?",
            ],
        ),
        meta=BlueprintMeta(
            genesis=False,
            generated_by=GeneratedBy.DISCOVERY,
            confidence=SectionConfidence(
                runtime="high",
                contract="medium",
                dependencies="high",
                failure_modes="medium",
                intent="high",
            ),
        ),
    )


class TestBlueprintGeneration:
    """Verify we can create a blueprint entirely from Python code."""

    def test_model_validates(self):
        bp = _make_gaia_monitor_blueprint()
        assert bp.id == "gaia-monitor"
        assert bp.role == "The Watcher (Health Monitoring)"
        assert bp.is_validated()

    def test_interface_counts(self):
        bp = _make_gaia_monitor_blueprint()
        assert len(bp.inbound_interfaces()) == 4
        assert len(bp.outbound_interfaces()) == 3

    def test_graph_node_output(self):
        bp = _make_gaia_monitor_blueprint()
        node = bp.to_graph_node()
        assert node["id"] == "gaia-monitor"
        assert node["port"] == 6420
        assert node["gpu"] is False
        assert node["interface_count"] == 7
        assert node["open_question_count"] == 3

    def test_open_questions(self):
        bp = _make_gaia_monitor_blueprint()
        qs = bp.open_questions()
        assert len(qs) == 3
        assert any("OpenTelemetry" in q for q in qs)


class TestRoundTrip:
    """Save → load → validate → divergence → topology → markdown."""

    def test_save_and_reload(self, tmp_blueprints):
        bp = _make_gaia_monitor_blueprint()
        save_blueprint(bp, candidate=True)

        loaded = load_blueprint("gaia-monitor", candidate=True)
        assert loaded is not None
        assert loaded.id == bp.id
        assert loaded.role == bp.role
        assert len(loaded.interfaces) == len(bp.interfaces)

    def test_yaml_file_exists(self, tmp_blueprints):
        bp = _make_gaia_monitor_blueprint()
        save_blueprint(bp, candidate=True)

        yaml_path = tmp_blueprints / "candidates" / "gaia-monitor.yaml"
        assert yaml_path.exists()

        # Verify it's parseable YAML
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        assert data["id"] == "gaia-monitor"

    def test_markdown_sidecar_generated(self, tmp_blueprints):
        bp = _make_gaia_monitor_blueprint()
        save_blueprint(bp, candidate=True)

        md_path = tmp_blueprints / "candidates" / "gaia-monitor.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "gaia-monitor" in content
        assert "The Watcher" in content

    def test_validate_candidate(self, tmp_blueprints):
        bp = _make_gaia_monitor_blueprint()
        save_blueprint(bp, candidate=True)

        result = validate_candidate_blueprint("gaia-monitor")
        # Source files don't physically exist for this hypothetical service,
        # so validation correctly reports them as missing. What matters is
        # that the schema itself is valid — the only errors should be about
        # missing source files (a filesystem concern, not a schema concern).
        for err in result.errors:
            assert "Source file not found" in err, f"Unexpected error: {err}"

    def test_divergence_against_self_is_zero(self, tmp_blueprints):
        bp = _make_gaia_monitor_blueprint()
        score = compute_divergence_score(bp, bp)
        assert score == 0.0

    def test_divergence_detects_changes(self, tmp_blueprints):
        bp1 = _make_gaia_monitor_blueprint()
        bp2 = _make_gaia_monitor_blueprint()
        # Add an interface to bp2
        bp2.interfaces.append(
            Interface(
                id="extra_endpoint",
                direction="inbound",
                transport=HttpRestInterface(path="/extra", method="GET"),
                description="An extra endpoint not in bp1.",
                status="active",
            )
        )
        score = compute_divergence_score(bp1, bp2)
        assert score > 0.0

    def test_load_all_includes_new_blueprint(self, tmp_blueprints):
        bp = _make_gaia_monitor_blueprint()
        save_blueprint(bp, candidate=True)

        all_bps = load_all_candidate_blueprints()
        assert "gaia-monitor" in all_bps

    def test_topology_includes_new_service(self, tmp_blueprints):
        """Save gaia-monitor + a stub gaia-core, verify edges are derived."""
        monitor = _make_gaia_monitor_blueprint()
        save_blueprint(monitor, candidate=True)

        # Create a minimal gaia-core stub with matching inbound /health
        core = BlueprintModel(
            id="gaia-core",
            version="0.5",
            role="The Brain (Cognition)",
            service_status="live",
            runtime=Runtime(port=6415, base_image="python:3.11-slim", gpu=False),
            interfaces=[
                Interface(
                    id="health",
                    direction="inbound",
                    transport=HttpRestInterface(path="/health", method="GET"),
                    description="Health check.",
                    status="active",
                ),
            ],
            meta=BlueprintMeta(genesis=True, generated_by=GeneratedBy.MANUAL_SEED),
        )
        save_blueprint(core, candidate=True)

        # Pass both blueprints directly to derive_graph_topology
        all_bps = load_all_candidate_blueprints()
        topology = derive_graph_topology(blueprints=all_bps)
        node_ids = {n["id"] for n in topology.nodes}
        assert "gaia-monitor" in node_ids
        assert "gaia-core" in node_ids

        # gaia-monitor's outbound /health should connect to gaia-core's inbound /health
        monitor_to_core = [
            e for e in topology.edges
            if e.from_service == "gaia-monitor" and e.to_service == "gaia-core"
        ]
        assert len(monitor_to_core) > 0, "Expected edge from gaia-monitor to gaia-core"

    def test_render_markdown_from_generated(self):
        bp = _make_gaia_monitor_blueprint()
        md = render_markdown(bp)
        assert "# GAIA Service Blueprint: `gaia-monitor`" in md
        assert "The Watcher" in md
        assert "Prometheus" in md
        assert "Discord" in md
        assert "CANDIDATE" in md or "candidate" in md.lower()
        # Verify sections exist
        assert "## Runtime" in md
        assert "## Interfaces" in md
        assert "## Failure Modes" in md
        assert "Open Questions" in md
