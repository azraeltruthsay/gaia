"""
gaia_common/utils/blueprint_generator.py

Deterministic blueprint generation from source code analysis.

Parses a service's Python source files, Dockerfile, and compose config to
produce a candidate BlueprintModel. No LLM inference — purely mechanical
extraction using AST analysis and regex.

Usage:
    from gaia_common.utils.blueprint_generator import generate_candidate_blueprint
    bp = generate_candidate_blueprint("gaia-audio", "/path/to/gaia_audio")
    from gaia_common.utils.blueprint_io import save_blueprint
    save_blueprint(bp, candidate=True)

Consumers:
  - sleep_task_scheduler: blueprint_discovery task (auto-generate for new services)
  - MCP tools: on-demand blueprint generation during conversation
  - promotion pipeline: pre-populate blueprint before validation
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    Runtime,
    SectionConfidence,
    ServiceDependency,
    ServiceStatus,
    Severity,
    SourceFile,
    VolumeDependency,
    VolumeAccess,
    WebSocketInterface,
)
from gaia_common.utils.ast_summarizer import ASTSummary, summarize_file

logger = logging.getLogger("GAIA.BlueprintGenerator")


# ── HTTP method mapping ──────────────────────────────────────────────────────

_ENDPOINT_METHOD_MAP = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "delete": "DELETE",
    "patch": "PATCH",
    "websocket": "WEBSOCKET",
}

# Known gaia service patterns in URLs / imports
_GAIA_SERVICE_PATTERNS: Dict[str, str] = {
    "gaia-core": "gaia.core|gaia_core|gaia-core",
    "gaia-web": "gaia.web|gaia_web|gaia-web",
    "gaia-prime": "gaia.prime|gaia_prime|gaia-prime",
    "gaia-mcp": "gaia.mcp|gaia_mcp|gaia-mcp",
    "gaia-study": "gaia.study|gaia_study|gaia-study",
    "gaia-orchestrator": "gaia.orchestrator|gaia_orchestrator|gaia-orchestrator",
    "gaia-audio": "gaia.audio|gaia_audio|gaia-audio",
}

# Roles inferred from URL patterns
_DEPENDENCY_ROLE_HINTS: Dict[str, str] = {
    "gaia-core": "cognition",
    "gaia-prime": "inference",
    "gaia-web": "web interface",
    "gaia-mcp": "tool dispatch",
    "gaia-study": "learning",
    "gaia-orchestrator": "orchestration",
    "gaia-audio": "audio processing",
}


# ── Public API ───────────────────────────────────────────────────────────────


def generate_candidate_blueprint(
    service_id: str,
    source_dir: str,
    *,
    dockerfile_path: str | None = None,
    compose_data: dict | None = None,
    role_hint: str = "",
) -> BlueprintModel:
    """Generate a candidate blueprint from source code analysis.

    Args:
        service_id: Service identifier (e.g. "gaia-audio")
        source_dir: Path to the service's Python source directory
        dockerfile_path: Optional path to Dockerfile (auto-detected if None)
        compose_data: Optional parsed compose service config dict
        role_hint: Optional human-readable role hint (e.g. "The Ears & Mouth")

    Returns:
        BlueprintModel with genesis=True, generated_by=DISCOVERY
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # 1. Summarize all Python files
    summaries = _summarize_all_python_files(source_path)
    logger.info("Summarized %d Python files for %s", len(summaries), service_id)

    # 2. Extract interfaces from endpoints
    interfaces = _extract_interfaces(summaries, service_id)

    # 3. Extract outbound interfaces and service dependencies
    outbound_interfaces, service_deps = _extract_outbound(summaries, service_id)
    interfaces.extend(outbound_interfaces)

    # 4. Parse Dockerfile for runtime config
    runtime = _extract_runtime(service_id, source_dir, dockerfile_path)

    # 5. Extract volume dependencies from compose data
    volume_deps = _extract_volume_deps(compose_data) if compose_data else []

    # 6. Extract failure modes from error handlers
    failure_modes = _extract_failure_modes(summaries)

    # 7. Build source file inventory
    source_files = _build_source_inventory(source_path, service_id)

    # 8. Extract intent from module docstrings
    intent = _extract_intent(summaries, service_id, role_hint)

    # 9. Assemble the blueprint
    bp = BlueprintModel(
        id=service_id,
        version="0.1",
        role=role_hint or intent.cognitive_role or service_id,
        service_status=ServiceStatus.CANDIDATE,
        runtime=runtime,
        interfaces=interfaces,
        dependencies=Dependencies(
            services=service_deps,
            volumes=volume_deps,
        ),
        source_files=source_files,
        failure_modes=failure_modes,
        intent=intent,
        meta=BlueprintMeta(
            status=BlueprintStatus.CANDIDATE,
            genesis=True,
            generated_by=GeneratedBy.DISCOVERY,
            confidence=SectionConfidence(
                runtime=ConfidenceLevel.HIGH,
                contract=ConfidenceLevel.HIGH,
                dependencies=ConfidenceLevel.MEDIUM,
                failure_modes=ConfidenceLevel.MEDIUM,
                intent=ConfidenceLevel.LOW,
            ),
        ),
    )

    logger.info(
        "Generated blueprint for %s: %d interfaces (%d inbound, %d outbound), "
        "%d dependencies, %d failure modes, %d source files",
        service_id,
        len(interfaces),
        len(bp.inbound_interfaces()),
        len(bp.outbound_interfaces()),
        len(service_deps),
        len(failure_modes),
        len(source_files),
    )

    return bp


# ── Extraction helpers ───────────────────────────────────────────────────────


def _summarize_all_python_files(source_path: Path) -> List[Tuple[str, ASTSummary]]:
    """Parse all .py files in source_dir and return (filename, summary) pairs."""
    results = []
    for py_file in sorted(source_path.rglob("*.py")):
        # Skip test files and __pycache__
        rel = py_file.relative_to(source_path)
        if "__pycache__" in str(rel):
            continue
        try:
            source_text = py_file.read_text(encoding="utf-8")
            summary = summarize_file(source_text, filename=str(rel))
            results.append((str(rel), summary))
        except (SyntaxError, UnicodeDecodeError) as exc:
            logger.warning("Could not parse %s: %s", rel, exc)
    return results


def _extract_interfaces(
    summaries: List[Tuple[str, ASTSummary]],
    service_id: str,
) -> List[Interface]:
    """Extract inbound interfaces from FastAPI endpoint decorators."""
    interfaces: List[Interface] = []
    seen_ids: set = set()

    for filename, summary in summaries:
        for ep in summary.endpoints:
            # Build a stable interface ID from the function name
            iface_id = ep.function_name.replace("_", "-")
            if iface_id in seen_ids:
                # Disambiguate with method prefix
                iface_id = f"{ep.method.lower()}-{iface_id}"
            seen_ids.add(iface_id)

            method_upper = _ENDPOINT_METHOD_MAP.get(ep.method.lower(), ep.method.upper())

            if method_upper == "WEBSOCKET":
                transport = WebSocketInterface(path=ep.path)
            else:
                transport = HttpRestInterface(
                    path=ep.path,
                    method=method_upper,
                )

            interfaces.append(Interface(
                id=iface_id,
                direction=InterfaceDirection.INBOUND,
                transport=transport,
                description=f"{method_upper} {ep.path} (from {filename}:{ep.line})",
                status=InterfaceStatus.ACTIVE,
            ))

    return interfaces


def _extract_outbound(
    summaries: List[Tuple[str, ASTSummary]],
    self_service_id: str,
) -> Tuple[List[Interface], List[ServiceDependency]]:
    """Extract outbound interfaces and service dependencies from HTTP client calls."""
    outbound: List[Interface] = []
    deps: Dict[str, ServiceDependency] = {}
    seen_outbound_ids: set = set()

    for filename, summary in summaries:
        for call in summary.http_calls:
            url = call.url_or_path or ""

            # Skip constructor calls (AsyncClient(), Client(), etc.)
            if not url or call.call_method in ("AsyncClient", "Client", "Session"):
                continue

            # Try to identify which gaia service this calls
            target_service = None
            for svc_id, pattern in _GAIA_SERVICE_PATTERNS.items():
                if svc_id == self_service_id:
                    continue
                if re.search(pattern, url, re.IGNORECASE):
                    target_service = svc_id
                    break

            # Also check variable names in f-string placeholders
            # e.g. "{config.core_endpoint}/sleep/wake" → gaia-core
            if not target_service and "{" in url:
                for svc_id, pattern in _GAIA_SERVICE_PATTERNS.items():
                    if svc_id == self_service_id:
                        continue
                    # Match against the variable name inside braces
                    svc_short = svc_id.replace("gaia-", "")
                    if svc_short + "_endpoint" in url or svc_short + "_url" in url:
                        target_service = svc_id
                        break

            # Extract the path portion (after the f-string variable)
            # "{config.core_endpoint}/sleep/wake" → "/sleep/wake"
            path = url
            fstring_path = re.search(r'\}(/[a-z_/\-]+)', url)
            if fstring_path:
                path = fstring_path.group(1)
            elif not url.startswith("/") and not url.startswith("{"):
                plain_path = re.search(r'(/[a-z_/\-]+)', url)
                path = plain_path.group(1) if plain_path else url

            # Build outbound interface
            func_name = call.enclosing_function or "unknown"
            iface_id = f"out-{func_name}".replace("_", "-")
            if iface_id in seen_outbound_ids:
                iface_id = f"{iface_id}-{call.line}"
            seen_outbound_ids.add(iface_id)

            method_upper = call.call_method.upper()
            if method_upper not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                method_upper = "POST"

            outbound.append(Interface(
                id=iface_id,
                direction=InterfaceDirection.OUTBOUND,
                transport=HttpRestInterface(
                    path=path if path.startswith("/") else f"/{path}",
                    method=method_upper,
                ),
                description=f"Outbound {method_upper} call from {func_name}() in {filename}",
                status=InterfaceStatus.ACTIVE,
            ))

            # Register service dependency
            if target_service and target_service not in deps:
                deps[target_service] = ServiceDependency(
                    id=target_service,
                    role=_DEPENDENCY_ROLE_HINTS.get(target_service, "unknown"),
                    required=False,  # conservative: assume optional
                )

        # Also check gaia imports for implicit dependencies
        for imp in summary.gaia_imports:
            for svc_id, pattern in _GAIA_SERVICE_PATTERNS.items():
                if svc_id == self_service_id:
                    continue
                if re.search(pattern, imp, re.IGNORECASE):
                    if svc_id not in deps:
                        deps[svc_id] = ServiceDependency(
                            id=svc_id,
                            role=_DEPENDENCY_ROLE_HINTS.get(svc_id, "library"),
                            required=True,
                        )

    return outbound, list(deps.values())


def _extract_runtime(
    service_id: str,
    source_dir: str,
    dockerfile_path: str | None = None,
) -> Runtime:
    """Extract runtime configuration from Dockerfile."""
    # Auto-detect Dockerfile location
    if dockerfile_path:
        df_path = Path(dockerfile_path)
    else:
        # Try common locations
        source_parent = Path(source_dir).parent
        candidates = [
            source_parent / "Dockerfile",
            Path(f"/gaia/GAIA_Project/candidates/{service_id}/Dockerfile"),
            Path(f"/gaia/GAIA_Project/{service_id}/Dockerfile"),
        ]
        df_path = next((p for p in candidates if p.exists()), None)

    runtime = Runtime(compose_service=service_id)

    if not df_path or not df_path.exists():
        logger.warning("No Dockerfile found for %s", service_id)
        return runtime

    try:
        content = df_path.read_text(encoding="utf-8")
    except OSError:
        return runtime

    runtime.dockerfile = str(df_path.relative_to("/gaia/GAIA_Project"))

    # Parse FROM
    from_match = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE)
    if from_match:
        base = from_match.group(1)
        # Strip build stage alias
        runtime.base_image = base.split(" AS ")[0].split(" as ")[0]

    # Parse EXPOSE
    expose_match = re.search(r"^EXPOSE\s+(\d+)", content, re.MULTILINE)
    if expose_match:
        runtime.port = int(expose_match.group(1))

    # Parse CMD for startup command
    cmd_match = re.search(r'^CMD\s+\[(.+)\]', content, re.MULTILINE)
    if cmd_match:
        parts = re.findall(r'"([^"]*)"', cmd_match.group(1))
        runtime.startup_cmd = " ".join(parts)

    # Parse HEALTHCHECK
    hc_match = re.search(r"HEALTHCHECK.*CMD\s+(.+?)(?:\s*\|\||$)", content, re.MULTILINE)
    if hc_match:
        runtime.health_check = hc_match.group(1).strip()

    # Detect GPU requirement
    runtime.gpu = bool(re.search(
        r"nvidia|cuda|torch\.cuda|gpu|NVIDIA_VISIBLE_DEVICES",
        content, re.IGNORECASE,
    ))

    # Parse USER
    user_match = re.search(r"^USER\s+(\S+)", content, re.MULTILINE)
    if user_match:
        runtime.user = user_match.group(1)

    return runtime


def _extract_volume_deps(compose_data: dict) -> List[VolumeDependency]:
    """Extract volume dependencies from compose service config."""
    volumes = []
    for vol_spec in compose_data.get("volumes", []):
        if isinstance(vol_spec, str):
            parts = vol_spec.split(":")
            if len(parts) >= 2:
                host_path = parts[0]
                mount_path = parts[1]
                access = VolumeAccess.RO if len(parts) > 2 and "ro" in parts[2] else VolumeAccess.RW
                name = Path(host_path).name
                volumes.append(VolumeDependency(
                    name=name,
                    access=access,
                    mount_path=mount_path,
                    purpose=f"Mounted from {host_path}",
                ))
    return volumes


def _extract_failure_modes(
    summaries: List[Tuple[str, ASTSummary]],
) -> List[FailureMode]:
    """Infer failure modes from error handlers and HTTP exceptions."""
    modes: List[FailureMode] = []
    seen: set = set()

    for filename, summary in summaries:
        for handler in summary.error_handlers:
            # Derive a condition description
            exc_types = ", ".join(handler.exception_types) if handler.exception_types else "Exception"
            func = handler.enclosing_function or "module"

            if handler.status_code:
                key = f"{handler.status_code}-{func}"
                if key in seen:
                    continue
                seen.add(key)

                severity = Severity.FATAL if handler.status_code >= 500 else Severity.DEGRADED
                modes.append(FailureMode(
                    condition=f"{exc_types} in {func}() (HTTP {handler.status_code})",
                    response=f"Returns HTTP {handler.status_code}",
                    severity=severity,
                    auto_recovers=True,
                ))
            else:
                key = f"{exc_types}-{func}"
                if key in seen:
                    continue
                seen.add(key)

                modes.append(FailureMode(
                    condition=f"{exc_types} in {func}()",
                    response="Error logged and handled",
                    severity=Severity.DEGRADED,
                    auto_recovers=True,
                ))

    return modes


def _build_source_inventory(
    source_path: Path,
    service_id: str,
) -> List[SourceFile]:
    """Build a list of source files with inferred roles."""
    files: List[SourceFile] = []
    candidate_prefix = f"candidates/{service_id}"

    for py_file in sorted(source_path.rglob("*.py")):
        rel = py_file.relative_to(source_path)
        if "__pycache__" in str(rel):
            continue

        name = rel.name
        role = _infer_file_role(name, str(rel))
        file_type = "python"

        files.append(SourceFile(
            path=f"{candidate_prefix}/{source_path.name}/{rel}",
            role=role,
            file_type=file_type,
        ))

    # Also check for Dockerfile
    dockerfile = source_path.parent / "Dockerfile"
    if dockerfile.exists():
        files.append(SourceFile(
            path=f"{candidate_prefix}/Dockerfile",
            role="dockerfile",
            file_type="dockerfile",
        ))

    return files


def _infer_file_role(filename: str, rel_path: str) -> str:
    """Infer a source file's role from its name."""
    if filename == "main.py":
        return "entrypoint"
    if filename == "config.py" or filename == "settings.py":
        return "config"
    if filename.startswith("test_") or filename.endswith("_test.py"):
        return "test"
    if filename == "__init__.py":
        return "package_init"
    if filename == "models.py" or filename == "schemas.py":
        return "models"
    if "route" in filename or "endpoint" in filename or "api" in filename:
        return "api"
    if "test" in rel_path.split("/"):
        return "test"
    return "core_logic"


def _extract_intent(
    summaries: List[Tuple[str, ASTSummary]],
    service_id: str,
    role_hint: str = "",
) -> Intent:
    """Extract intent from module docstrings and code structure."""
    # Collect all module docstrings
    docstrings = []
    for filename, summary in summaries:
        if summary.module_docstring and filename in ("main.py", "__init__.py"):
            docstrings.append(summary.module_docstring)

    purpose = docstrings[0] if docstrings else f"Service: {service_id}"

    # Infer cognitive role from common patterns
    cognitive_role = role_hint or None
    if not cognitive_role:
        purpose_lower = purpose.lower()
        if "audio" in purpose_lower or "stt" in purpose_lower or "tts" in purpose_lower:
            cognitive_role = "The Ears & Mouth (Audio)"
        elif "cognition" in purpose_lower or "brain" in purpose_lower:
            cognitive_role = "The Brain (Cognition)"
        elif "web" in purpose_lower or "discord" in purpose_lower:
            cognitive_role = "The Voice (Web Interface)"

    return Intent(
        purpose=purpose,
        cognitive_role=cognitive_role,
        design_decisions=[],
        open_questions=[
            f"Blueprint auto-generated by discovery — human review recommended.",
        ],
    )
