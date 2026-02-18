"""Tests for the Blueprint I/O layer (gaia_common.utils.blueprint_io)."""

import os
import pytest
import yaml
from pathlib import Path

from gaia_common.models.blueprint import (
    BlueprintMeta,
    BlueprintModel,
    BlueprintStatus,
    ConfidenceLevel,
    Dependencies,
    FailureMode,
    GeneratedBy,
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
    TransportType,
    VolumeDependency,
    VolumeAccess,
    WebSocketInterface,
)
from gaia_common.utils.blueprint_io import (
    load_blueprint,
    load_all_live_blueprints,
    load_all_candidate_blueprints,
    save_blueprint,
    promote_blueprint,
    validate_candidate_blueprint,
    compute_divergence_score,
    derive_graph_topology,
    render_markdown,
    _interfaces_match,
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


@pytest.fixture
def bp_root(tmp_path, monkeypatch):
    """Set up a temporary blueprints directory and point the env var at it."""
    monkeypatch.setenv("GAIA_BLUEPRINTS_ROOT", str(tmp_path))
    (tmp_path / "candidates").mkdir()
    return tmp_path


# ── Load / Save round-trip ──────────────────────────────────────────────────


class TestLoadSave:
    def test_save_and_load_candidate(self, bp_root):
        bp = _minimal_blueprint()
        path = save_blueprint(bp, candidate=True)
        assert path.exists()
        assert "candidates" in str(path)

        loaded = load_blueprint("gaia-test", candidate=True)
        assert loaded is not None
        assert loaded.id == "gaia-test"
        assert loaded.version == "0.1"

    def test_save_generates_markdown(self, bp_root):
        bp = _minimal_blueprint()
        save_blueprint(bp, candidate=True)
        md_path = bp_root / "candidates" / "gaia-test.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "gaia-test" in content
        assert "Test Service" in content

    def test_load_nonexistent_returns_none(self, bp_root):
        result = load_blueprint("nonexistent", candidate=True)
        assert result is None

    def test_load_corrupt_raises(self, bp_root):
        bad_path = bp_root / "candidates" / "bad.yaml"
        bad_path.write_text("id: 123\nnot_a_field: true\n")
        with pytest.raises(ValueError, match="failed validation"):
            load_blueprint("bad", candidate=True)

    def test_round_trip_preserves_interfaces(self, bp_root):
        bp = _minimal_blueprint(interfaces=[
            _http_interface("h", InterfaceDirection.INBOUND, "/health", "GET"),
            _http_interface("pp", InterfaceDirection.INBOUND, "/process_packet", "POST"),
        ])
        save_blueprint(bp, candidate=True)
        loaded = load_blueprint("gaia-test", candidate=True)
        assert len(loaded.interfaces) == 2
        assert loaded.interfaces[0].id == "h"

    def test_save_live_requires_live_status(self, bp_root):
        bp = _minimal_blueprint()  # status=CANDIDATE
        with pytest.raises(ValueError, match="non-LIVE"):
            save_blueprint(bp, candidate=False)

    def test_save_live_downgrades_if_candidate_dir(self, bp_root):
        bp = _minimal_blueprint(meta=_minimal_meta(status=BlueprintStatus.LIVE))
        path = save_blueprint(bp, candidate=True)
        loaded = load_blueprint("gaia-test", candidate=True)
        # Should have been downgraded to CANDIDATE
        assert loaded.meta.status == BlueprintStatus.CANDIDATE


class TestLoadAll:
    def test_load_all_live(self, bp_root):
        bp1 = _minimal_blueprint(id="svc-a", meta=_minimal_meta(status=BlueprintStatus.LIVE))
        bp2 = _minimal_blueprint(id="svc-b", meta=_minimal_meta(status=BlueprintStatus.LIVE))
        save_blueprint(bp1, candidate=False)
        save_blueprint(bp2, candidate=False)
        all_live = load_all_live_blueprints()
        assert "svc-a" in all_live
        assert "svc-b" in all_live

    def test_load_all_candidates(self, bp_root):
        bp = _minimal_blueprint(id="svc-c")
        save_blueprint(bp, candidate=True)
        all_cand = load_all_candidate_blueprints()
        assert "svc-c" in all_cand

    def test_load_all_candidates_empty_dir(self, bp_root):
        result = load_all_candidate_blueprints()
        assert result == {}


# ── Validate ─────────────────────────────────────────────────────────────────


class TestValidate:
    def test_valid_candidate(self, bp_root, monkeypatch):
        monkeypatch.setenv("GAIA_ROOT", str(bp_root.parent))
        bp = _minimal_blueprint(
            interfaces=[_http_interface("h", InterfaceDirection.INBOUND, "/health")],
            intent=Intent(purpose="Test service"),
        )
        save_blueprint(bp, candidate=True)
        result = validate_candidate_blueprint("gaia-test")
        assert result.passed

    def test_missing_candidate(self, bp_root):
        result = validate_candidate_blueprint("missing")
        assert not result.passed
        assert any("No candidate" in e for e in result.errors)

    def test_no_interfaces_warns(self, bp_root, monkeypatch):
        monkeypatch.setenv("GAIA_ROOT", str(bp_root.parent))
        bp = _minimal_blueprint(interfaces=[])
        save_blueprint(bp, candidate=True)
        result = validate_candidate_blueprint("gaia-test")
        assert result.passed  # warning, not error
        assert any("island" in w for w in result.warnings)

    def test_missing_intent_warns(self, bp_root, monkeypatch):
        monkeypatch.setenv("GAIA_ROOT", str(bp_root.parent))
        bp = _minimal_blueprint()
        save_blueprint(bp, candidate=True)
        result = validate_candidate_blueprint("gaia-test")
        assert any("intent" in w.lower() for w in result.warnings)

    def test_id_mismatch_errors(self, bp_root, monkeypatch):
        monkeypatch.setenv("GAIA_ROOT", str(bp_root.parent))
        bp = _minimal_blueprint(id="wrong-id")
        # Save under a different filename
        path = bp_root / "candidates" / "gaia-test.yaml"
        with path.open("w") as f:
            yaml.dump(bp.model_dump(mode="json"), f, default_flow_style=False)
        result = validate_candidate_blueprint("gaia-test")
        assert not result.passed
        assert any("does not match" in e for e in result.errors)


# ── Promote ──────────────────────────────────────────────────────────────────


class TestPromote:
    def test_bootstrap_promote(self, bp_root, monkeypatch):
        monkeypatch.setenv("GAIA_ROOT", str(bp_root.parent))
        bp = _minimal_blueprint(
            interfaces=[_http_interface("h", InterfaceDirection.INBOUND, "/health")],
            intent=Intent(purpose="Test"),
        )
        save_blueprint(bp, candidate=True)

        promoted = promote_blueprint("gaia-test", bootstrap=True)
        assert promoted.meta.status == BlueprintStatus.LIVE
        assert promoted.meta.promoted_at is not None

        # Verify live file exists
        live = load_blueprint("gaia-test", candidate=False)
        assert live is not None
        assert live.meta.status == BlueprintStatus.LIVE

    def test_bootstrap_no_candidate_raises(self, bp_root):
        with pytest.raises(FileNotFoundError, match="No candidate"):
            promote_blueprint("missing", bootstrap=True)

    def test_bootstrap_invalid_candidate_raises(self, bp_root, monkeypatch):
        monkeypatch.setenv("GAIA_ROOT", str(bp_root.parent))
        bp = _minimal_blueprint(id="wrong-id")
        path = bp_root / "candidates" / "gaia-test.yaml"
        with path.open("w") as f:
            yaml.dump(bp.model_dump(mode="json"), f, default_flow_style=False)
        with pytest.raises(ValueError, match="validation failed"):
            promote_blueprint("gaia-test", bootstrap=True)

    def test_promote_without_bootstrap_needs_live(self, bp_root):
        with pytest.raises(FileNotFoundError, match="No live blueprint"):
            promote_blueprint("gaia-test", bootstrap=False)


# ── Divergence score ─────────────────────────────────────────────────────────


class TestDivergenceScore:
    def test_identical_blueprints_score_zero(self):
        bp = _minimal_blueprint(
            interfaces=[_http_interface("h", InterfaceDirection.INBOUND, "/health")],
            runtime=Runtime(port=6415, gpu=False),
        )
        assert compute_divergence_score(bp, bp) == 0.0

    def test_different_interfaces_increase_score(self):
        bp1 = _minimal_blueprint(interfaces=[
            _http_interface("h", InterfaceDirection.INBOUND, "/health"),
        ])
        bp2 = _minimal_blueprint(interfaces=[
            _http_interface("h", InterfaceDirection.INBOUND, "/health"),
            _http_interface("s", InterfaceDirection.INBOUND, "/status"),
        ])
        score = compute_divergence_score(bp1, bp2)
        assert score > 0.0

    def test_score_capped_at_one(self):
        bp1 = _minimal_blueprint(
            interfaces=[_http_interface("a", InterfaceDirection.INBOUND, "/a")],
            runtime=Runtime(port=6415, gpu=True),
            dependencies=Dependencies(services=[
                ServiceDependency(id="dep1", role="test"),
            ]),
        )
        bp2 = _minimal_blueprint(
            interfaces=[
                _http_interface("b", InterfaceDirection.INBOUND, "/b"),
                _http_interface("c", InterfaceDirection.INBOUND, "/c"),
            ],
            runtime=Runtime(port=9999, gpu=False),
            dependencies=Dependencies(services=[
                ServiceDependency(id="dep2", role="other"),
            ]),
        )
        score = compute_divergence_score(bp1, bp2)
        assert 0.0 <= score <= 1.0


# ── Interface matching ───────────────────────────────────────────────────────


class TestInterfaceMatching:
    def test_http_path_match(self):
        out = _http_interface("out", InterfaceDirection.OUTBOUND, "/health", "GET")
        inp = _http_interface("in", InterfaceDirection.INBOUND, "/health", "GET")
        assert _interfaces_match(out, inp) is True

    def test_http_path_mismatch(self):
        out = _http_interface("out", InterfaceDirection.OUTBOUND, "/health", "GET")
        inp = _http_interface("in", InterfaceDirection.INBOUND, "/status", "GET")
        assert _interfaces_match(out, inp) is False

    def test_direction_must_be_outbound_inbound(self):
        a = _http_interface("a", InterfaceDirection.INBOUND, "/health")
        b = _http_interface("b", InterfaceDirection.INBOUND, "/health")
        assert _interfaces_match(a, b) is False

    def test_transport_type_mismatch(self):
        out = Interface(
            id="out", direction=InterfaceDirection.OUTBOUND,
            transport=McpInterface(methods=["run_shell"]),
            description="MCP out",
        )
        inp = _http_interface("in", InterfaceDirection.INBOUND, "/jsonrpc")
        assert _interfaces_match(out, inp) is False

    def test_mcp_method_overlap(self):
        out = Interface(
            id="out", direction=InterfaceDirection.OUTBOUND,
            transport=McpInterface(target_service="gaia-mcp", methods=["run_shell", "read_file"]),
            description="MCP dispatch",
        )
        inp = Interface(
            id="in", direction=InterfaceDirection.INBOUND,
            transport=McpInterface(methods=["run_shell", "write_file"]),
            description="MCP receiver",
        )
        assert _interfaces_match(out, inp) is True

    def test_mcp_no_method_overlap(self):
        out = Interface(
            id="out", direction=InterfaceDirection.OUTBOUND,
            transport=McpInterface(methods=["run_shell"]),
            description="MCP dispatch",
        )
        inp = Interface(
            id="in", direction=InterfaceDirection.INBOUND,
            transport=McpInterface(methods=["write_file"]),
            description="MCP receiver",
        )
        assert _interfaces_match(out, inp) is False

    def test_negotiated_transport_matching(self):
        rest = HttpRestInterface(path="/data")
        ws = WebSocketInterface(path="/data")
        out = Interface(
            id="out", direction=InterfaceDirection.OUTBOUND,
            transport=NegotiatedTransport(
                transports=[rest, ws],
                preferred=TransportType.HTTP_REST,
            ),
            description="Negotiated out",
        )
        inp = _http_interface("in", InterfaceDirection.INBOUND, "/data")
        assert _interfaces_match(out, inp) is True


# ── Graph topology derivation ────────────────────────────────────────────────


class TestGraphTopology:
    def test_empty_blueprints(self):
        topo = derive_graph_topology({})
        assert len(topo.nodes) == 0
        assert len(topo.edges) == 0

    def test_two_services_with_matching_interfaces(self):
        core = _minimal_blueprint(
            id="gaia-core",
            interfaces=[
                _http_interface("prime_req", InterfaceDirection.OUTBOUND, "/v1/chat/completions"),
            ],
            meta=_minimal_meta(status=BlueprintStatus.LIVE),
        )
        prime = _minimal_blueprint(
            id="gaia-prime",
            role="The Mouth",
            interfaces=[
                _http_interface("chat_api", InterfaceDirection.INBOUND, "/v1/chat/completions"),
            ],
            meta=_minimal_meta(status=BlueprintStatus.LIVE),
        )
        topo = derive_graph_topology({"gaia-core": core, "gaia-prime": prime})
        assert len(topo.nodes) == 2
        assert len(topo.edges) == 1
        assert topo.edges[0].from_service == "gaia-core"
        assert topo.edges[0].to_service == "gaia-prime"

    def test_no_matching_interfaces_no_edges(self):
        core = _minimal_blueprint(
            id="gaia-core",
            interfaces=[
                _http_interface("out", InterfaceDirection.OUTBOUND, "/api/a"),
            ],
            meta=_minimal_meta(status=BlueprintStatus.LIVE),
        )
        web = _minimal_blueprint(
            id="gaia-web",
            role="The Voice",
            interfaces=[
                _http_interface("in", InterfaceDirection.INBOUND, "/api/b"),
            ],
            meta=_minimal_meta(status=BlueprintStatus.LIVE),
        )
        topo = derive_graph_topology({"gaia-core": core, "gaia-web": web})
        assert len(topo.edges) == 0

    def test_pending_review_count(self):
        bp1 = _minimal_blueprint(
            id="svc1",
            meta=_minimal_meta(status=BlueprintStatus.LIVE, genesis=True),
        )
        bp2 = _minimal_blueprint(
            id="svc2",
            meta=_minimal_meta(status=BlueprintStatus.LIVE, genesis=False),
        )
        topo = derive_graph_topology({"svc1": bp1, "svc2": bp2})
        assert topo.pending_review_count == 1

    def test_fallback_edge_detection(self):
        core = _minimal_blueprint(
            id="gaia-core",
            interfaces=[
                _http_interface("prime_req", InterfaceDirection.OUTBOUND, "/v1/chat/completions"),
            ],
            dependencies=Dependencies(services=[
                ServiceDependency(id="gaia-prime", role="inference", required=False, fallback="groq"),
            ]),
            meta=_minimal_meta(status=BlueprintStatus.LIVE),
        )
        prime = _minimal_blueprint(
            id="gaia-prime",
            interfaces=[
                _http_interface("chat_api", InterfaceDirection.INBOUND, "/v1/chat/completions"),
            ],
            meta=_minimal_meta(status=BlueprintStatus.LIVE),
        )
        topo = derive_graph_topology({"gaia-core": core, "gaia-prime": prime})
        assert topo.edges[0].has_fallback is True


# ── Markdown rendering ───────────────────────────────────────────────────────


class TestMarkdownRendering:
    def test_minimal_rendering(self):
        bp = _minimal_blueprint()
        md = render_markdown(bp)
        assert "# GAIA Service Blueprint: `gaia-test`" in md
        assert "CANDIDATE" in md

    def test_live_status_badge(self):
        bp = _minimal_blueprint(meta=_minimal_meta(status=BlueprintStatus.LIVE))
        md = render_markdown(bp)
        assert "LIVE" in md

    def test_intent_rendering(self):
        bp = _minimal_blueprint(intent=Intent(
            purpose="Test purpose",
            cognitive_role="The Thinker",
            design_decisions=["Decision A", "Decision B"],
            open_questions=["Why?"],
        ))
        md = render_markdown(bp)
        assert "Test purpose" in md
        assert "The Thinker" in md
        assert "Decision A" in md
        assert "Why?" in md
        assert "Open Questions" in md

    def test_interfaces_rendering(self):
        bp = _minimal_blueprint(interfaces=[
            _http_interface("h", InterfaceDirection.INBOUND, "/health", "GET"),
            _http_interface("out", InterfaceDirection.OUTBOUND, "/api"),
        ])
        md = render_markdown(bp)
        assert "Inbound" in md
        assert "Outbound" in md
        assert "/health" in md

    def test_runtime_security_rendering(self):
        bp = _minimal_blueprint(runtime=Runtime(
            port=6415,
            security=SecurityConfig(
                no_new_privileges=True,
                cap_drop=["ALL"],
            ),
        ))
        md = render_markdown(bp)
        assert "6415" in md
        assert "no_new_privileges" in md

    def test_volume_rendering(self):
        bp = _minimal_blueprint(dependencies=Dependencies(
            volumes=[
                VolumeDependency(name="shared", access=VolumeAccess.RW, mount_path="/shared", purpose="State"),
            ],
        ))
        md = render_markdown(bp)
        assert "Volume Dependencies" in md
        assert "/shared" in md

    def test_source_file_type_rendering(self):
        bp = _minimal_blueprint(source_files=[
            SourceFile(path="main.py", role="entrypoint", file_type="python"),
        ])
        md = render_markdown(bp)
        assert "[python]" in md

    def test_failure_modes_rendering(self):
        bp = _minimal_blueprint(failure_modes=[
            FailureMode(
                condition="Service down",
                response="Fallback to backup",
                severity=Severity.DEGRADED,
            ),
        ])
        md = render_markdown(bp)
        assert "Service down" in md
        assert "Fallback to backup" in md

    def test_confidence_table_rendering(self):
        bp = _minimal_blueprint()
        md = render_markdown(bp)
        assert "Blueprint Confidence" in md
        assert "Runtime" in md


# ── YAML parse: gaia-core.yaml seed ─────────────────────────────────────────


class TestGaiaCoreYamlSeed:
    """Verify the hand-authored gaia-core.yaml parses correctly against the schema."""

    @pytest.fixture
    def gaia_core_yaml(self):
        gaia_root = Path(os.environ.get("GAIA_ROOT", str(Path(__file__).parent.parent.parent.parent)))
        yaml_path = gaia_root / "knowledge" / "blueprints" / "candidates" / "gaia-core.yaml"
        if not yaml_path.exists():
            pytest.skip(f"gaia-core.yaml not found at {yaml_path}")
        with yaml_path.open() as f:
            return yaml.safe_load(f)

    def test_parses_into_blueprint(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        assert bp.id == "gaia-core"

    def test_correct_interface_count(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        assert len(bp.inbound_interfaces()) == 12
        assert len(bp.outbound_interfaces()) == 7

    def test_runtime_fields(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        assert bp.runtime.port == 6415
        assert bp.runtime.gpu is False
        assert bp.runtime.dockerfile == "gaia-core/Dockerfile"
        assert bp.runtime.compose_service == "gaia-core"

    def test_dependencies(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        svc_ids = {d.id for d in bp.dependencies.services}
        assert "gaia-prime" in svc_ids
        assert "gaia-mcp" in svc_ids
        assert "gaia-web" in svc_ids
        assert "gaia-orchestrator" in svc_ids
        assert len(bp.dependencies.volumes) == 5

    def test_source_files_have_types(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        for sf in bp.source_files:
            assert sf.file_type is not None, f"Missing file_type on {sf.path}"

    def test_failure_modes(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        assert len(bp.failure_modes) == 6

    def test_intent_present(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        assert bp.intent is not None
        assert bp.intent.cognitive_role == "The Brain"
        assert len(bp.intent.design_decisions) >= 5

    def test_meta_fields(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        assert bp.meta.status == BlueprintStatus.CANDIDATE
        assert bp.meta.genesis is True
        assert bp.meta.generated_by == GeneratedBy.MANUAL_SEED

    def test_graph_node_output(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        node = bp.to_graph_node()
        assert node["id"] == "gaia-core"
        assert node["role"] == "The Brain (Cognition)"
        assert node["port"] == 6415

    def test_markdown_renders_without_error(self, gaia_core_yaml):
        bp = BlueprintModel.model_validate(gaia_core_yaml)
        md = render_markdown(bp)
        assert len(md) > 500
        assert "gaia-core" in md
