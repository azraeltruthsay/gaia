"""
gaia_common/utils/blueprint_io.py

Load, save, validate, and derive topology from BlueprintModel instances.

Writers:    gaia-study (discovery & reflection cycles)
Readers:    gaia-web (graph API, markdown rendering), promote_candidate.sh
Gate:       promotion pipeline calls validate_candidate_blueprint() before
            allowing any candidate → live transition

Directory layout (enforced by this module — never write outside these paths):
  {BLUEPRINTS_ROOT}/                    ← live blueprints (graph renders from here)
    {service_id}.yaml
  {BLUEPRINTS_ROOT}/candidates/         ← candidate blueprints (graph ignores)
    {service_id}.yaml
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import ValidationError

from gaia_common.models.blueprint import (
    BlueprintMeta,
    BlueprintModel,
    BlueprintStatus,
    ConfidenceLevel,
    GeneratedBy,
    GraphEdge,
    GraphTopology,
    Interface,
    InterfaceDirection,
    InterfaceStatus,
    NegotiatedTransport,
    TransportType,
)

logger = logging.getLogger("GAIA.Blueprint.IO")

# Default root — override via GAIA_BLUEPRINTS_ROOT env var in each service
_DEFAULT_ROOT = Path("/shared/knowledge/blueprints")


def _blueprints_root() -> Path:
    import os
    root = os.environ.get("GAIA_BLUEPRINTS_ROOT", str(_DEFAULT_ROOT))
    return Path(root)


def _live_path(service_id: str) -> Path:
    return _blueprints_root() / f"{service_id}.yaml"


def _candidate_path(service_id: str) -> Path:
    return _blueprints_root() / "candidates" / f"{service_id}.yaml"


def _markdown_path(service_id: str, candidate: bool = False) -> Path:
    if candidate:
        return _blueprints_root() / "candidates" / f"{service_id}.md"
    return _blueprints_root() / f"{service_id}.md"


# ── Load ──────────────────────────────────────────────────────────────────────


def load_blueprint(service_id: str, candidate: bool = False) -> Optional[BlueprintModel]:
    """
    Load a blueprint from YAML. Returns None if not found.
    Validates against BlueprintModel schema on load — corrupt files raise ValueError.
    """
    path = _candidate_path(service_id) if candidate else _live_path(service_id)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
        return BlueprintModel.model_validate(data)
    except (yaml.YAMLError, ValidationError) as e:
        raise ValueError(f"Blueprint {path} failed validation: {e}") from e


def load_all_live_blueprints() -> Dict[str, BlueprintModel]:
    """Load all live blueprints. Skips malformed files with a warning."""
    root = _blueprints_root()
    blueprints: Dict[str, BlueprintModel] = {}
    for yaml_path in root.glob("*.yaml"):
        service_id = yaml_path.stem
        try:
            bp = load_blueprint(service_id, candidate=False)
            if bp:
                blueprints[service_id] = bp
        except ValueError as e:
            logger.warning("Skipping malformed blueprint %s: %s", service_id, e)
    return blueprints


def load_all_candidate_blueprints() -> Dict[str, BlueprintModel]:
    """Load all candidate blueprints. Skips malformed files with a warning."""
    candidate_dir = _blueprints_root() / "candidates"
    blueprints: Dict[str, BlueprintModel] = {}
    if not candidate_dir.exists():
        return blueprints
    for yaml_path in candidate_dir.glob("*.yaml"):
        service_id = yaml_path.stem
        try:
            bp = load_blueprint(service_id, candidate=True)
            if bp:
                blueprints[service_id] = bp
        except ValueError as e:
            logger.warning("Skipping malformed candidate blueprint %s: %s", service_id, e)
    return blueprints


# ── Save ──────────────────────────────────────────────────────────────────────


def save_blueprint(blueprint: BlueprintModel, candidate: bool = True) -> Path:
    """
    Persist a blueprint to YAML and regenerate its markdown.

    ENFORCED: study may only write candidate blueprints.
    Promotion to live status is performed exclusively by promote_blueprint().
    This preserves the invariant that the live graph always reflects
    human-approved, pipeline-validated blueprints.
    """
    if not candidate and blueprint.meta.status != BlueprintStatus.LIVE:
        raise ValueError(
            "Cannot write a non-LIVE blueprint to the live directory. "
            "Set candidate=True or use promote_blueprint()."
        )
    if candidate and blueprint.meta.status == BlueprintStatus.LIVE:
        # Automatically downgrade status to candidate when saving to candidate dir
        blueprint = blueprint.model_copy(
            update={"meta": blueprint.meta.model_copy(update={"status": BlueprintStatus.CANDIDATE})}
        )

    path = _candidate_path(blueprint.id) if candidate else _live_path(blueprint.id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        yaml.dump(
            blueprint.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # Always regenerate markdown alongside YAML
    md_path = _markdown_path(blueprint.id, candidate=candidate)
    md_path.write_text(render_markdown(blueprint))

    logger.info(
        "Blueprint saved: %s (candidate=%s, genesis=%s, generated_by=%s)",
        blueprint.id, candidate, blueprint.meta.genesis, blueprint.meta.generated_by
    )
    return path


def promote_blueprint(service_id: str, *, bootstrap: bool = False) -> BlueprintModel:
    """
    Promote a candidate blueprint to live status.

    Called by the promotion pipeline after code tests pass and discovery
    has generated a live blueprint from the promoted code.

    This is the only function that writes to the live blueprint directory.

    When bootstrap=False (default): expects a live blueprint to already exist
    (freshly generated by study's discovery process) and updates its status.

    When bootstrap=True: copies the candidate blueprint directly to live.
    This is the Phase 1 path for hand-authored seeds with no study discovery.
    """
    if bootstrap:
        candidate = load_blueprint(service_id, candidate=True)
        if candidate is None:
            raise FileNotFoundError(
                f"No candidate blueprint found for {service_id}. "
                "Create a candidate seed before bootstrapping."
            )
        validation = validate_candidate_blueprint(service_id)
        if not validation.passed:
            raise ValueError(
                f"Candidate blueprint validation failed: {validation.errors}"
            )
        promoted = candidate.model_copy(
            update={
                "meta": candidate.meta.model_copy(
                    update={
                        "status": BlueprintStatus.LIVE,
                        "promoted_at": datetime.now(timezone.utc),
                    }
                )
            }
        )
    else:
        live = load_blueprint(service_id, candidate=False)
        if live is None:
            raise FileNotFoundError(
                f"No live blueprint found for {service_id}. "
                "Run discovery before promoting, or use bootstrap=True for seeds."
            )
        if live.meta.status == BlueprintStatus.LIVE:
            logger.info("Blueprint %s already LIVE — refreshing promoted_at timestamp.", service_id)

        promoted = live.model_copy(
            update={
                "meta": live.meta.model_copy(
                    update={
                        "status": BlueprintStatus.LIVE,
                        "promoted_at": datetime.now(timezone.utc),
                    }
                )
            }
        )

    path = _live_path(service_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(promoted.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    md_path = _markdown_path(service_id, candidate=False)
    md_path.write_text(render_markdown(promoted))

    logger.info("Blueprint promoted to LIVE: %s (bootstrap=%s)", service_id, bootstrap)
    return promoted


# ── Validate ─────────────────────────────────────────────────────────────────


class BlueprintValidationResult:
    def __init__(self) -> None:
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"BlueprintValidation({status}, errors={self.errors}, warnings={self.warnings})"


def validate_candidate_blueprint(service_id: str) -> BlueprintValidationResult:
    """
    Promotion gate — called by promote_candidate.sh before allowing
    candidate → live transition. Fails promotion if result.passed is False.
    """
    result = BlueprintValidationResult()

    # 1. Candidate blueprint must exist
    bp = load_blueprint(service_id, candidate=True)
    if bp is None:
        result.error(f"No candidate blueprint found at candidates/{service_id}.yaml")
        return result

    # 2. Service ID must match filename
    if bp.id != service_id:
        result.error(f"Blueprint id '{bp.id}' does not match filename '{service_id}'")

    # 3. Must have at least one interface
    if not bp.interfaces:
        result.warn("No interfaces defined — service will be an island in the graph")

    # 4. Source files must exist on disk
    from pathlib import Path
    import os
    gaia_root = Path(os.environ.get("GAIA_ROOT", "/gaia/GAIA_Project"))
    for sf in bp.source_files:
        full_path = gaia_root / sf.path
        if not full_path.exists():
            result.error(f"Source file not found: {sf.path}")

    # 5. Warn if intent is missing (can't auto-derive design rationale)
    if bp.intent is None:
        result.warn("No intent section — design rationale will be missing from graph")

    # 6. Warn on low-confidence sections in a candidate that's about to be promoted
    low_confidence_sections = [
        section
        for section, level in bp.meta.confidence.model_dump().items()
        if level == ConfidenceLevel.LOW.value
    ]
    if low_confidence_sections:
        result.warn(
            f"Low confidence in sections: {low_confidence_sections}. "
            "Discovery cycle will attempt to resolve after promotion."
        )

    return result


def compute_divergence_score(candidate: BlueprintModel, live: BlueprintModel) -> float:
    """
    Compare a prescriptive candidate blueprint with the descriptive live blueprint
    produced after promotion and discovery.

    Returns a score in [0.0, 1.0]:
      0.0 = perfectly faithful implementation
      1.0 = completely diverged — flag for human review

    Used in the promotion report to surface implementation drift.
    """
    score = 0.0
    checks = 0

    def _check(weight: float, match: bool) -> None:
        nonlocal score, checks
        checks += 1
        if not match:
            score += weight

    # Interface count drift
    _check(0.25, len(candidate.interfaces) == len(live.interfaces))

    # Interface IDs preserved
    candidate_ids = {i.id for i in candidate.interfaces}
    live_ids = {i.id for i in live.interfaces}
    _check(0.25, candidate_ids == live_ids)

    # Runtime port preserved
    _check(0.15, candidate.runtime.port == live.runtime.port)

    # GPU requirement preserved
    _check(0.10, candidate.runtime.gpu == live.runtime.gpu)

    # Dependency service IDs preserved
    candidate_deps = {d.id for d in candidate.dependencies.services}
    live_deps = {d.id for d in live.dependencies.services}
    _check(0.25, candidate_deps == live_deps)

    return round(min(score, 1.0), 3)


# ── Graph topology derivation ─────────────────────────────────────────────────


def _interfaces_match(outbound: Interface, inbound: Interface) -> bool:
    """
    Two interfaces form a graph edge if their transports are compatible
    and their topics/paths/symbols match.

    Matching rules by transport type:
      http_rest / sse / websocket  → path must match
      event                        → topic must match
      mcp                          → target_service + method overlap
      grpc                         → rpc name must match
      direct_call                  → symbol must match
    """
    if outbound.direction != InterfaceDirection.OUTBOUND:
        return False
    if inbound.direction != InterfaceDirection.INBOUND:
        return False

    def _resolve_transport(iface: Interface):
        """Extract the concrete transport, unwrapping NegotiatedTransport if needed."""
        t = iface.transport
        if isinstance(t, NegotiatedTransport):
            # Use the preferred transport for matching
            for sub in t.transports:
                if getattr(sub, "type", None) == t.preferred:
                    return sub
            return t.transports[0]  # fallback to first
        return t

    def _transport_type(iface: Interface) -> Optional[str]:
        t = _resolve_transport(iface)
        return getattr(t, "type", None)

    out_type = _transport_type(outbound)
    in_type = _transport_type(inbound)
    if out_type != in_type:
        return False

    t = _resolve_transport(outbound)
    ti = _resolve_transport(inbound)

    if out_type in (TransportType.HTTP_REST, TransportType.SSE, TransportType.WEBSOCKET):
        return getattr(t, "path", None) == getattr(ti, "path", None)
    if out_type == TransportType.EVENT:
        return getattr(t, "topic", None) == getattr(ti, "topic", None)
    if out_type == TransportType.GRPC:
        return getattr(t, "rpc", None) == getattr(ti, "rpc", None)
    if out_type == TransportType.DIRECT_CALL:
        return getattr(t, "symbol", None) == getattr(ti, "symbol", None)
    if out_type == TransportType.MCP:
        out_methods = set(getattr(t, "methods", []))
        in_methods = set(getattr(ti, "methods", []))
        return bool(out_methods & in_methods)

    return False


def derive_graph_topology(blueprints: Optional[Dict[str, BlueprintModel]] = None) -> GraphTopology:
    """
    Derive the full graph topology from live blueprints.

    Edges self-assemble from interface matching — never manually defined.
    Adding a new service with matching interfaces automatically wires it in.

    Called by GET /api/blueprints/graph. Recomputed on every request.
    """
    if blueprints is None:
        blueprints = load_all_live_blueprints()

    nodes = [bp.to_graph_node() for bp in blueprints.values()]
    edges: List[GraphEdge] = []

    service_ids = list(blueprints.keys())

    for from_id in service_ids:
        from_bp = blueprints[from_id]
        for outbound in from_bp.outbound_interfaces():
            for to_id in service_ids:
                if to_id == from_id:
                    continue
                to_bp = blueprints[to_id]
                for inbound in to_bp.inbound_interfaces():
                    if _interfaces_match(outbound, inbound):
                        # Check if source marks this dependency as non-required
                        has_fallback = any(
                            dep.id == to_id and not dep.required
                            for dep in from_bp.dependencies.services
                        )
                        edge_status = (
                            outbound.status
                            if outbound.status == InterfaceStatus.PLANNED
                            else inbound.status
                        )
                        # Resolve transport type, unwrapping NegotiatedTransport
                        out_transport = outbound.transport
                        if isinstance(out_transport, NegotiatedTransport):
                            edge_transport = out_transport.preferred
                        else:
                            edge_transport = getattr(out_transport, "type", TransportType.HTTP_REST)
                        edges.append(GraphEdge(
                            from_service=from_id,
                            to_service=to_id,
                            interface_id_from=outbound.id,
                            interface_id_to=inbound.id,
                            transport_type=edge_transport,
                            status=edge_status,
                            description=outbound.description,
                            has_fallback=has_fallback,
                        ))

    pending_review = sum(
        1 for bp in blueprints.values() if bp.meta.genesis
    )

    return GraphTopology(
        nodes=nodes,
        edges=edges,
        blueprint_count=len(blueprints),
        pending_review_count=pending_review,
    )


# ── Markdown rendering ────────────────────────────────────────────────────────


def render_markdown(bp: BlueprintModel) -> str:
    """
    Generate human-readable markdown from a BlueprintModel.

    This is the only source of truth for the .md files in knowledge/blueprints/.
    Never edit the .md files by hand — they will be overwritten on the next
    save_blueprint() or promote_blueprint() call.
    """
    lines: List[str] = []
    status_badge = "🟢 LIVE" if bp.meta.status == BlueprintStatus.LIVE else "🟡 CANDIDATE"
    genesis_note = " · ⚠️ genesis — awaiting validation" if bp.meta.genesis else ""

    lines += [
        f"# GAIA Service Blueprint: `{bp.id}` ({bp.role})",
        f"",
        f"> **Status:** {status_badge}{genesis_note}  ",
        f"> **Service version:** {bp.version}  ",
        f"> **Blueprint version:** {bp.meta.blueprint_version}  ",
        f"> **Generated by:** `{bp.meta.generated_by.value if hasattr(bp.meta.generated_by, 'value') else bp.meta.generated_by}`  ",
        f"> **Last reflected:** {bp.meta.last_reflected.strftime('%Y-%m-%d %H:%M UTC') if bp.meta.last_reflected else 'never'}  ",
        f"",
    ]

    if bp.intent:
        lines += [
            "## Purpose",
            "",
            bp.intent.purpose,
            "",
        ]
        if bp.intent.cognitive_role:
            lines += [f"**Cognitive role:** {bp.intent.cognitive_role}", ""]

        if bp.intent.design_decisions:
            lines += ["### Design Decisions", ""]
            for d in bp.intent.design_decisions:
                lines.append(f"- {d}")
            lines.append("")

        if bp.intent.open_questions:
            lines += ["### ⚠️ Open Questions", ""]
            for q in bp.intent.open_questions:
                lines.append(f"- {q}")
            lines.append("")

    lines += [
        "## Runtime",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Port | `{bp.runtime.port or '—'}` |",
        f"| Base image | `{bp.runtime.base_image or '—'}` |",
        f"| GPU | {'Yes' if bp.runtime.gpu else 'No'} |",
        f"| Health check | `{bp.runtime.health_check or '—'}` |",
        f"| Startup | `{bp.runtime.startup_cmd or '—'}` |",
    ]
    if bp.runtime.gpu_count is not None:
        lines.append(f"| GPU count | `{bp.runtime.gpu_count}` |")
    if bp.runtime.user:
        lines.append(f"| User | `{bp.runtime.user}` |")
    if bp.runtime.dockerfile:
        lines.append(f"| Dockerfile | `{bp.runtime.dockerfile}` |")
    if bp.runtime.compose_service:
        lines.append(f"| Compose service | `{bp.runtime.compose_service}` |")
    if bp.runtime.security:
        sec = bp.runtime.security
        sec_parts = []
        if sec.no_new_privileges:
            sec_parts.append("no_new_privileges")
        if sec.cap_drop:
            sec_parts.append(f"cap_drop: {', '.join(sec.cap_drop)}")
        if sec.cap_add:
            sec_parts.append(f"cap_add: {', '.join(sec.cap_add)}")
        if sec_parts:
            lines.append(f"| Security | {'; '.join(sec_parts)} |")
    lines.append("")

    if bp.interfaces:
        lines += ["## Interfaces", ""]
        inbound = bp.inbound_interfaces()
        outbound = bp.outbound_interfaces()

        if inbound:
            lines += ["### Inbound", ""]
            for iface in inbound:
                t = iface.transport
                transport_str = getattr(t, "type", "unknown")
                if hasattr(transport_str, "value"):
                    transport_str = transport_str.value
                path_or_topic = (
                    getattr(t, "path", None)
                    or getattr(t, "topic", None)
                    or getattr(t, "symbol", None)
                    or getattr(t, "rpc", None)
                    or "—"
                )
                status_icon = "✅" if iface.status == InterfaceStatus.ACTIVE else "🔜"
                lines.append(
                    f"- **`{iface.id}`** {status_icon} `{transport_str}` `{path_or_topic}`  "
                )
                lines.append(f"  {iface.description}")
            lines.append("")

        if outbound:
            lines += ["### Outbound", ""]
            for iface in outbound:
                t = iface.transport
                transport_str = getattr(t, "type", "unknown")
                if hasattr(transport_str, "value"):
                    transport_str = transport_str.value
                path_or_topic = (
                    getattr(t, "path", None)
                    or getattr(t, "topic", None)
                    or getattr(t, "symbol", None)
                    or getattr(t, "rpc", None)
                    or "—"
                )
                status_icon = "✅" if iface.status == InterfaceStatus.ACTIVE else "🔜"
                lines.append(
                    f"- **`{iface.id}`** {status_icon} `{transport_str}` `{path_or_topic}`  "
                )
                lines.append(f"  {iface.description}")
            lines.append("")

    if bp.dependencies.services:
        lines += ["## Service Dependencies", ""]
        for dep in bp.dependencies.services:
            req = "required" if dep.required else f"optional (fallback: `{dep.fallback or '—'}`)"
            lines.append(f"- **`{dep.id}`** — {dep.role} · {req}")
        lines.append("")

    if bp.dependencies.external_apis:
        lines += ["## External API Dependencies", ""]
        for api in bp.dependencies.external_apis:
            req = "required" if api.required else "optional"
            lines.append(f"- **{api.name}** — {api.purpose} · {req}")
        lines.append("")

    if bp.dependencies.volumes:
        lines += ["## Volume Dependencies", ""]
        for vol in bp.dependencies.volumes:
            access_str = vol.access.value if hasattr(vol.access, "value") else vol.access
            mount = f" → `{vol.mount_path}`" if vol.mount_path else ""
            purpose = f" — {vol.purpose}" if vol.purpose else ""
            lines.append(f"- **{vol.name}** `{access_str}`{mount}{purpose}")
        lines.append("")

    if bp.failure_modes:
        lines += ["## Failure Modes", ""]
        for fm in bp.failure_modes:
            severity_icon = {"degraded": "🟡", "partial": "🟠", "fatal": "🔴"}.get(
                fm.severity.value if hasattr(fm.severity, "value") else fm.severity, "⚪"
            )
            lines.append(f"- {severity_icon} **{fm.condition}**")
            lines.append(f"  → {fm.response}")
        lines.append("")

    if bp.source_files:
        lines += ["## Source Files", ""]
        for sf in bp.source_files:
            ft = f" [{sf.file_type}]" if sf.file_type else ""
            lines.append(f"- `{sf.path}` _{sf.role}_{ft}")
        lines.append("")

    conf = bp.meta.confidence
    lines += [
        "## Blueprint Confidence",
        "",
        "| Section | Confidence |",
        "|---------|------------|",
    ]
    for section, level in conf.model_dump().items():
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "⚪")
        lines.append(f"| {section.replace('_', ' ').title()} | {icon} {level} |")
    lines.append("")

    lines += [
        "---",
        f"_Generated automatically from `{bp.id}.yaml`. Do not edit this file directly._",
    ]

    return "\n".join(lines)
