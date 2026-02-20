"""
gaia_common/utils/blueprint_precheck.py

Mechanical blueprint-vs-code validation. Checks structural presence of
blueprint-declared items in source code using regex and AST extraction.
Fast and deterministic — no LLM inference.

This is the refactored, standalone version of the validation logic
previously embedded in sleep_task_scheduler._run_blueprint_validation.

Two consumers:
  1. sleep_task_scheduler — calls during sleep cycles (same behavior as before)
  2. review_prompt_builder — produces structured pre-check annotations for LLM review

Direction: blueprint claims → check in source code
(The old sleep_task direction was code facts → check in blueprint text)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from gaia_common.models.blueprint import (
    BlueprintModel,
    HttpRestInterface,
    Interface,
    InterfaceDirection,
    NegotiatedTransport,
    SseInterface,
    WebSocketInterface,
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PreCheckItem:
    category: Literal["endpoint", "enum_member", "constant", "failure_mode", "dependency"]
    blueprint_claim: str
    status: Literal["found", "missing", "diverged"]
    source_file: Optional[str]
    detail: str


@dataclass
class PreCheckSummary:
    total: int
    found: int
    missing: int
    diverged: int


@dataclass
class PreCheckResult:
    service_id: str
    timestamp: datetime
    items: List[PreCheckItem]
    summary: PreCheckSummary

    def to_prompt_text(self) -> str:
        """Render as concise block for inclusion in an LLM review prompt."""
        lines: list[str] = []
        lines.append(f"## Mechanical Pre-Check Results: {self.service_id}")
        lines.append("")

        # Group items by category
        by_cat: Dict[str, list[PreCheckItem]] = {}
        for item in self.items:
            by_cat.setdefault(item.category, []).append(item)

        cat_labels = {
            "endpoint": "Endpoints",
            "enum_member": "Enum Members",
            "constant": "Constants",
            "failure_mode": "Failure Modes",
            "dependency": "Dependencies",
        }

        for cat in ("endpoint", "failure_mode", "dependency", "enum_member", "constant"):
            items = by_cat.get(cat, [])
            if not items:
                continue
            found_count = sum(1 for i in items if i.status == "found")
            label = cat_labels.get(cat, cat)
            lines.append(f"### {label} ({found_count}/{len(items)} found)")
            for item in items:
                tag = f"[{item.status.upper()}]"
                lines.append(f"  {tag:12s} {item.blueprint_claim}")
                if item.detail:
                    lines.append(f"              {item.detail}")
            lines.append("")

        pct = (self.summary.found / self.summary.total * 100) if self.summary.total else 0
        lines.append("### Summary")
        lines.append(
            f"  Total checks: {self.summary.total} | "
            f"Found: {self.summary.found} | "
            f"Missing: {self.summary.missing} | "
            f"Diverged: {self.summary.diverged}"
        )
        lines.append(f"  Structural completeness: {pct:.1f}%")
        lines.append("")

        return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────────

def run_blueprint_precheck(
    blueprint: BlueprintModel,
    source_dir: str,
    *,
    categories: Optional[List[str]] = None,
) -> PreCheckResult:
    """
    Run mechanical blueprint-vs-code validation.

    Checks structural presence of blueprint-declared items in source code
    using regex and AST extraction (NOT LLM inference). Fast and deterministic.

    Args:
        blueprint: The BlueprintModel to validate against
        source_dir: Path to the service source directory
        categories: Optional filter — only check these categories

    Returns:
        PreCheckResult with per-item status and summary counts
    """
    items: list[PreCheckItem] = []

    # Load all .py files from source_dir
    source_files = _load_source_files(source_dir)

    all_cats = {"endpoint", "enum_member", "constant", "failure_mode", "dependency"}
    active_cats = set(categories) if categories else all_cats

    if "endpoint" in active_cats:
        items.extend(_check_endpoints(blueprint, source_files))

    if "failure_mode" in active_cats:
        items.extend(_check_failure_modes(blueprint, source_files))

    if "dependency" in active_cats:
        items.extend(_check_dependencies(blueprint, source_files))

    if "enum_member" in active_cats:
        items.extend(_check_enums(blueprint, source_files))

    if "constant" in active_cats:
        items.extend(_check_constants(blueprint, source_files))

    found = sum(1 for i in items if i.status == "found")
    missing = sum(1 for i in items if i.status == "missing")
    diverged = sum(1 for i in items if i.status == "diverged")

    return PreCheckResult(
        service_id=blueprint.id,
        timestamp=datetime.now(timezone.utc),
        items=items,
        summary=PreCheckSummary(
            total=len(items),
            found=found,
            missing=missing,
            diverged=diverged,
        ),
    )


# ── Source file loading ──────────────────────────────────────────────────────

def _load_source_files(source_dir: str) -> Dict[str, str]:
    """Load all .py files in source_dir recursively. Returns {relative_path: content}."""
    files: Dict[str, str] = {}
    root = Path(source_dir)
    if not root.exists():
        return files
    for py_file in root.rglob("*.py"):
        rel = str(py_file.relative_to(root))
        try:
            files[rel] = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return files


# ── Endpoint checking ────────────────────────────────────────────────────────

# Regex patterns for FastAPI route decorators
_HTTP_ROUTE_RE = re.compile(
    r"@(?:router|app)\.(get|post|put|delete|patch)\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_WS_ROUTE_RE = re.compile(
    r"@(?:router|app)\.websocket\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_SSE_ROUTE_RE = _HTTP_ROUTE_RE  # SSE typically uses GET endpoint


def _check_endpoints(
    blueprint: BlueprintModel, source_files: Dict[str, str]
) -> list[PreCheckItem]:
    """Check that each blueprint interface endpoint exists in source."""
    items: list[PreCheckItem] = []

    # Collect all declared endpoints from blueprint
    for iface in blueprint.interfaces:
        if iface.direction != InterfaceDirection.INBOUND:
            continue

        transport = iface.transport
        if isinstance(transport, NegotiatedTransport):
            transport = transport.transports[0]

        if isinstance(transport, HttpRestInterface):
            claim = f"{transport.method.upper()} {transport.path}"
            found_file, found_line = _find_http_endpoint(
                transport.method, transport.path, source_files
            )
            if found_file:
                items.append(PreCheckItem(
                    category="endpoint",
                    blueprint_claim=claim,
                    status="found",
                    source_file=found_file,
                    detail=f"→ {found_file}:{found_line}",
                ))
            else:
                items.append(PreCheckItem(
                    category="endpoint",
                    blueprint_claim=claim,
                    status="missing",
                    source_file=None,
                    detail="no matching route decorator found",
                ))

        elif isinstance(transport, WebSocketInterface):
            claim = f"WS {transport.path}"
            found_file, found_line = _find_ws_endpoint(transport.path, source_files)
            if found_file:
                items.append(PreCheckItem(
                    category="endpoint",
                    blueprint_claim=claim,
                    status="found",
                    source_file=found_file,
                    detail=f"→ {found_file}:{found_line}",
                ))
            else:
                items.append(PreCheckItem(
                    category="endpoint",
                    blueprint_claim=claim,
                    status="missing",
                    source_file=None,
                    detail="no matching websocket decorator found",
                ))

        elif isinstance(transport, SseInterface):
            claim = f"SSE {transport.path}"
            found_file, found_line = _find_http_endpoint("GET", transport.path, source_files)
            if found_file:
                items.append(PreCheckItem(
                    category="endpoint",
                    blueprint_claim=claim,
                    status="found",
                    source_file=found_file,
                    detail=f"→ {found_file}:{found_line}",
                ))
            else:
                items.append(PreCheckItem(
                    category="endpoint",
                    blueprint_claim=claim,
                    status="missing",
                    source_file=None,
                    detail="no matching GET route for SSE found",
                ))

    return items


def _find_http_endpoint(
    method: str, path: str, source_files: Dict[str, str]
) -> tuple[Optional[str], int]:
    """Find a matching @router.{method}("{path}") in source files."""
    method_lower = method.lower()
    for filename, content in source_files.items():
        for i, line in enumerate(content.splitlines(), 1):
            match = _HTTP_ROUTE_RE.search(line)
            if match:
                route_method = match.group(1).lower()
                route_path = match.group(2)
                if route_method == method_lower and _paths_match(route_path, path):
                    return filename, i
    return None, 0


def _find_ws_endpoint(
    path: str, source_files: Dict[str, str]
) -> tuple[Optional[str], int]:
    """Find a matching @router.websocket("{path}") in source files."""
    for filename, content in source_files.items():
        for i, line in enumerate(content.splitlines(), 1):
            match = _WS_ROUTE_RE.search(line)
            if match:
                route_path = match.group(1)
                if _paths_match(route_path, path):
                    return filename, i
    return None, 0


def _paths_match(source_path: str, blueprint_path: str) -> bool:
    """Compare route paths, treating {param} and path parameters as equivalent."""
    # Normalize: strip leading/trailing slashes for comparison
    sp = source_path.strip("/")
    bp = blueprint_path.strip("/")
    if sp == bp:
        return True
    # Try segment-by-segment comparison treating {x} as wildcard
    sp_parts = sp.split("/")
    bp_parts = bp.split("/")
    if len(sp_parts) != len(bp_parts):
        return False
    for s, b in zip(sp_parts, bp_parts):
        if s == b:
            continue
        if s.startswith("{") or b.startswith("{"):
            continue
        return False
    return True


# ── Failure mode checking ────────────────────────────────────────────────────

def _check_failure_modes(
    blueprint: BlueprintModel, source_files: Dict[str, str]
) -> list[PreCheckItem]:
    """Check that each blueprint failure mode has a matching handler in source."""
    items: list[PreCheckItem] = []
    all_source = "\n".join(source_files.values())

    for fm in blueprint.failure_modes:
        claim = fm.condition
        # Build search patterns from the failure mode condition
        patterns = _failure_mode_patterns(fm.condition, fm.response)
        found = False
        found_file = None
        detail = ""

        for filename, content in source_files.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    line_num = content[:match.start()].count("\n") + 1
                    found = True
                    found_file = filename
                    detail = f"→ pattern match in {filename}:{line_num}"
                    break
            if found:
                break

        items.append(PreCheckItem(
            category="failure_mode",
            blueprint_claim=claim,
            status="found" if found else "missing",
            source_file=found_file,
            detail=detail if found else "no matching handler found",
        ))

    return items


def _failure_mode_patterns(condition: str, response: str) -> list[str]:
    """Generate regex patterns from a failure mode condition/response pair."""
    patterns: list[str] = []

    # Common exception types derived from conditions
    _CONDITION_PATTERNS = {
        "unavailable": [r"ConnectError", r"ConnectionError", r"ConnectionRefusedError"],
        "timeout": [r"TimeoutException", r"ReadTimeout", r"Timeout"],
        "error": [r"except\s+\w*Error", r"HTTPStatusError"],
        "invalid": [r"ValueError", r"ValidationError"],
        "parse": [r"JSONDecodeError", r"ParseError", r"yaml\.YAMLError"],
        "auth": [r"401", r"403", r"Unauthorized", r"Forbidden"],
    }

    # Try condition-based pattern matching
    condition_lower = condition.lower()
    for keyword, pats in _CONDITION_PATTERNS.items():
        if keyword in condition_lower:
            patterns.extend(pats)

    # Extract service name from condition for import/URL checks
    # e.g., "gaia-prime unavailable" → look for "gaia.prime" or "gaia-prime" refs
    words = condition.lower().replace("-", "_").split()
    for word in words:
        if word.startswith("gaia_"):
            patterns.append(re.escape(word))

    # HTTP status codes from response text
    status_codes = re.findall(r"\b([45]\d{2})\b", response)
    for code in status_codes:
        patterns.append(rf"status_code\s*=\s*{code}")

    # Fallback keywords from response
    if "fallback" in response.lower() or "retry" in response.lower():
        patterns.append(r"(?:fallback|retry|backoff)")

    if "degrade" in response.lower() or "graceful" in response.lower():
        patterns.append(r"(?:degrad|graceful)")

    # If no patterns generated, try exact condition words
    if not patterns:
        sanitized = re.escape(condition)
        patterns.append(sanitized)

    return patterns


# ── Dependency checking ──────────────────────────────────────────────────────

def _check_dependencies(
    blueprint: BlueprintModel, source_files: Dict[str, str]
) -> list[PreCheckItem]:
    """Check that each declared service dependency appears in source."""
    items: list[PreCheckItem] = []

    for dep in blueprint.dependencies.services:
        claim = f"{dep.id} ({dep.role})"
        # Build patterns: import references, URL patterns, env vars
        patterns = _dependency_patterns(dep.id)
        found = False
        found_file = None
        detail = ""

        for filename, content in source_files.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    line_num = content[:match.start()].count("\n") + 1
                    found = True
                    found_file = filename
                    detail = f"→ {filename}:{line_num}"
                    break
            if found:
                break

        items.append(PreCheckItem(
            category="dependency",
            blueprint_claim=claim,
            status="found" if found else "missing",
            source_file=found_file,
            detail=detail if found else "no import or URL reference found",
        ))

    return items


def _dependency_patterns(service_id: str) -> list[str]:
    """Generate regex patterns to detect a dependency reference."""
    patterns: list[str] = []
    # gaia-core → gaia_core (Python import style)
    py_name = service_id.replace("-", "_")
    patterns.append(rf"(?:from|import)\s+{re.escape(py_name)}")
    # URL pattern: http://gaia-core:PORT or similar
    patterns.append(rf"(?:http|https)://\s*{re.escape(service_id)}")
    # Env var pattern: CORE_ENDPOINT, PRIME_ENDPOINT, etc.
    short_name = service_id.replace("gaia-", "").upper()
    patterns.append(rf"{short_name}_(?:ENDPOINT|URL|HOST)")
    # Direct string reference
    patterns.append(re.escape(service_id))
    return patterns


# ── Enum checking ────────────────────────────────────────────────────────────

_ENUM_CLASS_RE = re.compile(r"class\s+(\w+)\(.*Enum.*\):")
_ENUM_MEMBER_RE = re.compile(r"^\s+(\w+)\s*=\s*", re.MULTILINE)


def _check_enums(
    blueprint: BlueprintModel, source_files: Dict[str, str]
) -> list[PreCheckItem]:
    """
    Check enum references from blueprint interfaces (input/output schemas).

    Scans for schema names that correspond to enum classes in source code.
    """
    items: list[PreCheckItem] = []

    # Collect schema references from interfaces that might be enums
    schema_refs: set[str] = set()
    for iface in blueprint.interfaces:
        transport = iface.transport
        if isinstance(transport, NegotiatedTransport):
            transport = transport.transports[0]
        if isinstance(transport, HttpRestInterface):
            if transport.input_schema:
                schema_refs.add(transport.input_schema)
            if transport.output_schema:
                schema_refs.add(transport.output_schema)

    # Find all enums in source
    source_enums: Dict[str, list[str]] = {}
    for filename, content in source_files.items():
        for match in _ENUM_CLASS_RE.finditer(content):
            enum_name = match.group(1)
            # Extract members from the class body
            start = match.end()
            # Find next class/function definition or end of file
            next_def = re.search(r"\n(?:class |def |async def )", content[start:])
            end = start + next_def.start() if next_def else len(content)
            body = content[start:end]
            members = _ENUM_MEMBER_RE.findall(body)
            source_enums[enum_name] = members

    # Check each schema ref that looks like it could be an enum
    for schema in schema_refs:
        if schema in source_enums:
            items.append(PreCheckItem(
                category="enum_member",
                blueprint_claim=f"schema: {schema}",
                status="found",
                source_file=None,
                detail=f"enum {schema} with members: {', '.join(source_enums[schema][:5])}",
            ))

    return items


# ── Constant checking ────────────────────────────────────────────────────────

_CONSTANT_RE = re.compile(r"^([A-Z][A-Z_]{2,})\s*=\s*(.+)", re.MULTILINE)


def _check_constants(
    blueprint: BlueprintModel, source_files: Dict[str, str]
) -> list[PreCheckItem]:
    """
    Check that key constants exist in source files.

    This is a lightweight check — it scans for UPPER_CASE constants and
    reports their presence. Since blueprints don't usually declare specific
    constants, this mainly serves as a supplementary annotation for the
    LLM reviewer.
    """
    # For now, constants checking is informational — blueprints don't
    # declare specific constant values. This will be extended when the
    # blueprint schema gains a constants section.
    return []
