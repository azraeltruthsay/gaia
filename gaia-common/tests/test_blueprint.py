"""Tests for the Blueprint schema (gaia_common.models.blueprint)."""

import pytest
from datetime import datetime, timezone

from gaia_common.models.blueprint import (
    BlueprintModel,
    BlueprintMeta,
    BlueprintStatus,
    ConfidenceLevel,
    Dependencies,
    ExternalApiDependency,
    FailureMode,
    GeneratedBy,
    GraphEdge,
    GraphTopology,
    HttpRestInterface,
    Intent,
    Interface,
    InterfaceDirection,
    InterfaceStatus,
    McpInterface,
    NegotiatedTransport,
    Runtime,
    SecurityConfig,
    SectionConfidence,
    ServiceDependency,
    ServiceStatus,
    Severity,
    SourceFile,
    SseInterface,
    TransportType,
    VolumeDependency,
    VolumeAccess,
    WebSocketInterface,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _minimal_meta(**overrides):
    defaults = dict(
        status=BlueprintStatus.CANDIDATE,
        genesis=True,
        generated_by=GeneratedBy.MANUAL_SEED,
    )
    defaults.update(overrides)
    return BlueprintMeta(**defaults)


def _minimal_blueprint(**overrides):
    defaults = dict(
        id="gaia-test",
        version="0.1",
        role="Test Service",
        meta=_minimal_meta(),
    )
    defaults.update(overrides)
    return BlueprintModel(**defaults)


def _http_interface(id, direction, path, method="POST", **kwargs):
    return Interface(
        id=id,
        direction=direction,
        transport=HttpRestInterface(path=path, method=method),
        description=f"Test interface {id}",
        **kwargs,
    )


# ── Schema: Enumerations ────────────────────────────────────────────────────


class TestEnumerations:
    def test_blueprint_status_values(self):
        assert BlueprintStatus.CANDIDATE.value == "candidate"
        assert BlueprintStatus.LIVE.value == "live"
        assert BlueprintStatus.ARCHIVED.value == "archived"

    def test_transport_type_values(self):
        assert TransportType.HTTP_REST.value == "http_rest"
        assert TransportType.MCP.value == "mcp"
        assert TransportType.SSE.value == "sse"
        assert TransportType.WEBSOCKET.value == "websocket"
        assert TransportType.EVENT.value == "event"
        assert TransportType.DIRECT_CALL.value == "direct_call"
        assert TransportType.GRPC.value == "grpc"

    def test_confidence_levels(self):
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"

    def test_severity_values(self):
        assert Severity.DEGRADED.value == "degraded"
        assert Severity.PARTIAL.value == "partial"
        assert Severity.FATAL.value == "fatal"


# ── Schema: Transport models ────────────────────────────────────────────────


class TestTransportModels:
    def test_http_rest(self):
        t = HttpRestInterface(path="/health", method="GET")
        assert t.type == TransportType.HTTP_REST
        assert t.path == "/health"

    def test_websocket(self):
        t = WebSocketInterface(path="/ws")
        assert t.type == TransportType.WEBSOCKET

    def test_sse(self):
        t = SseInterface(path="/stream", event_types=["token", "done"])
        assert t.type == TransportType.SSE
        assert len(t.event_types) == 2

    def test_mcp(self):
        t = McpInterface(target_service="gaia-mcp", methods=["run_shell", "read_file"])
        assert t.type == TransportType.MCP
        assert "run_shell" in t.methods

    def test_negotiated_transport(self):
        rest = HttpRestInterface(path="/api")
        ws = WebSocketInterface(path="/api")
        nt = NegotiatedTransport(
            transports=[rest, ws],
            preferred=TransportType.WEBSOCKET,
            upgrade_note="WebSocket when available",
        )
        assert len(nt.transports) == 2
        assert nt.preferred == TransportType.WEBSOCKET

    def test_negotiated_transport_requires_at_least_two(self):
        rest = HttpRestInterface(path="/api")
        with pytest.raises(Exception):
            NegotiatedTransport(transports=[rest], preferred=TransportType.HTTP_REST)


# ── Schema: Interface model ─────────────────────────────────────────────────


class TestInterface:
    def test_basic_interface(self):
        iface = _http_interface("health", InterfaceDirection.INBOUND, "/health", "GET")
        assert iface.id == "health"
        assert iface.direction == InterfaceDirection.INBOUND
        assert iface.status == InterfaceStatus.ACTIVE

    def test_interface_with_negotiated_transport(self):
        rest = HttpRestInterface(path="/data")
        ws = WebSocketInterface(path="/data")
        iface = Interface(
            id="data",
            direction=InterfaceDirection.INBOUND,
            transport=NegotiatedTransport(
                transports=[rest, ws],
                preferred=TransportType.WEBSOCKET,
            ),
            description="Data endpoint with upgrade path",
        )
        assert isinstance(iface.transport, NegotiatedTransport)


# ── Schema: Dependency models ───────────────────────────────────────────────


class TestDependencyModels:
    def test_service_dependency(self):
        dep = ServiceDependency(id="gaia-prime", role="inference", required=False, fallback="groq")
        assert dep.id == "gaia-prime"
        assert not dep.required
        assert dep.fallback == "groq"

    def test_volume_dependency_with_mount(self):
        vol = VolumeDependency(
            name="gaia-shared", access=VolumeAccess.RW,
            purpose="Session state", mount_path="/shared",
        )
        assert vol.mount_path == "/shared"
        assert vol.access == VolumeAccess.RW

    def test_volume_dependency_optional_mount(self):
        vol = VolumeDependency(name="logs", access=VolumeAccess.RW)
        assert vol.mount_path is None

    def test_external_api(self):
        api = ExternalApiDependency(name="groq", purpose="fallback_inference", required=False)
        assert not api.required


# ── Schema: Runtime + SecurityConfig ────────────────────────────────────────


class TestRuntime:
    def test_minimal_runtime(self):
        rt = Runtime()
        assert rt.port is None
        assert rt.gpu is False
        assert rt.security is None

    def test_full_runtime(self):
        rt = Runtime(
            port=6415,
            base_image="python:3.11-slim",
            gpu=False,
            startup_cmd="uvicorn main:app",
            health_check="curl -f http://localhost:6415/health",
            gpu_count=None,
            user="${UID}:${GID}",
            dockerfile="gaia-core/Dockerfile",
            compose_service="gaia-core",
            security=SecurityConfig(
                no_new_privileges=True,
                cap_drop=["ALL"],
                cap_add=["NET_BIND_SERVICE"],
            ),
        )
        assert rt.port == 6415
        assert rt.user == "${UID}:${GID}"
        assert rt.security.no_new_privileges is True
        assert "ALL" in rt.security.cap_drop

    def test_gpu_count_all_literal(self):
        rt = Runtime(gpu=True, gpu_count="all")
        assert rt.gpu_count == "all"

    def test_gpu_count_int(self):
        rt = Runtime(gpu=True, gpu_count=2)
        assert rt.gpu_count == 2


# ── Schema: SourceFile ──────────────────────────────────────────────────────


class TestSourceFile:
    def test_with_file_type(self):
        sf = SourceFile(path="gaia-core/main.py", role="entrypoint", file_type="python")
        assert sf.file_type == "python"

    def test_without_file_type(self):
        sf = SourceFile(path="gaia-core/main.py", role="entrypoint")
        assert sf.file_type is None


# ── Schema: BlueprintMeta ───────────────────────────────────────────────────


class TestBlueprintMeta:
    def test_defaults(self):
        meta = _minimal_meta()
        assert meta.genesis is True
        assert meta.blueprint_version == "0.1"
        assert meta.schema_version == "1.0"
        assert meta.last_reflected is None
        assert meta.promoted_at is None
        assert meta.divergence_score is None

    def test_created_at_uses_utc(self):
        meta = _minimal_meta()
        assert meta.created_at.tzinfo is not None

    def test_confidence_defaults(self):
        meta = _minimal_meta()
        assert meta.confidence.runtime == ConfidenceLevel.HIGH
        assert meta.confidence.intent == ConfidenceLevel.LOW

    def test_divergence_score_bounds(self):
        meta = _minimal_meta(divergence_score=0.5)
        assert meta.divergence_score == 0.5

        with pytest.raises(Exception):
            _minimal_meta(divergence_score=1.5)

        with pytest.raises(Exception):
            _minimal_meta(divergence_score=-0.1)


# ── Schema: BlueprintModel ──────────────────────────────────────────────────


class TestBlueprintModel:
    def test_minimal_valid(self):
        bp = _minimal_blueprint()
        assert bp.id == "gaia-test"
        assert bp.service_status == ServiceStatus.LIVE

    def test_interface_id_uniqueness(self):
        with pytest.raises(Exception):
            _minimal_blueprint(interfaces=[
                _http_interface("dup", InterfaceDirection.INBOUND, "/a"),
                _http_interface("dup", InterfaceDirection.INBOUND, "/b"),
            ])

    def test_inbound_outbound_helpers(self):
        bp = _minimal_blueprint(interfaces=[
            _http_interface("in1", InterfaceDirection.INBOUND, "/a"),
            _http_interface("in2", InterfaceDirection.INBOUND, "/b"),
            _http_interface("out1", InterfaceDirection.OUTBOUND, "/c"),
        ])
        assert len(bp.inbound_interfaces()) == 2
        assert len(bp.outbound_interfaces()) == 1

    def test_active_interfaces(self):
        bp = _minimal_blueprint(interfaces=[
            _http_interface("a", InterfaceDirection.INBOUND, "/a"),
            _http_interface("b", InterfaceDirection.INBOUND, "/b", status=InterfaceStatus.PLANNED),
        ])
        assert len(bp.active_interfaces()) == 1

    def test_open_questions_accessor(self):
        bp = _minimal_blueprint(intent=Intent(
            purpose="Test",
            open_questions=["Q1?", "Q2?"],
        ))
        assert bp.open_questions() == ["Q1?", "Q2?"]

    def test_open_questions_no_intent(self):
        bp = _minimal_blueprint()
        assert bp.open_questions() == []

    def test_is_validated(self):
        bp = _minimal_blueprint()
        assert not bp.is_validated()

        bp2 = _minimal_blueprint(meta=_minimal_meta(genesis=False))
        assert bp2.is_validated()

    def test_to_graph_node(self):
        bp = _minimal_blueprint(
            runtime=Runtime(port=6415, gpu=False),
            interfaces=[
                _http_interface("h", InterfaceDirection.INBOUND, "/health"),
            ],
        )
        node = bp.to_graph_node()
        assert node["id"] == "gaia-test"
        assert node["port"] == 6415
        assert node["gpu"] is False
        assert node["interface_count"] == 1
        assert node["genesis"] is True
        assert "confidence" in node


# ── Schema: GraphEdge + GraphTopology ───────────────────────────────────────


class TestGraphModels:
    def test_graph_edge(self):
        edge = GraphEdge(
            from_service="gaia-core",
            to_service="gaia-prime",
            interface_id_from="prime_inference",
            interface_id_to="chat_completions",
            transport_type=TransportType.HTTP_REST,
            status=InterfaceStatus.ACTIVE,
            description="LLM inference",
        )
        assert edge.from_service == "gaia-core"
        assert edge.has_fallback is False

    def test_graph_topology(self):
        topo = GraphTopology(
            nodes=[{"id": "gaia-core", "role": "Brain"}],
            edges=[],
            blueprint_count=1,
            pending_review_count=0,
        )
        assert topo.blueprint_count == 1
        assert topo.generated_at.tzinfo is not None
