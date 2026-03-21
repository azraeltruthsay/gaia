#!/usr/bin/env python3
"""
compile_registry.py — Compile GAIA blueprint YAMLs into a single service registry JSON.

Loads all live blueprints via the existing blueprint_io module, derives the
graph topology (wiring edges), validates wiring, and writes a stdlib-readable
JSON file to /shared/registry/service_registry.json.

The compiled JSON is consumed by:
  - gaia-doctor (stdlib json.load — no pydantic)
  - validate_wiring.py (standalone wiring checker)
  - gaia-web /api/registry/validation endpoint

Usage:
    python scripts/compile_registry.py [--output PATH] [--blueprints-root PATH]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path so we can import gaia_common
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gaia-common"))

from gaia_common.utils.blueprint_io import (
    load_all_live_blueprints,
    derive_graph_topology,
)
from gaia_common.models.blueprint import InterfaceDirection

logger = logging.getLogger("GAIA.Registry.Compiler")

DEFAULT_REGISTRY_PATH = "/shared/registry/service_registry.json"


def _enum_val(obj: Any) -> str:
    """Convert a Pydantic enum to its string value, or str() as fallback."""
    return obj.value if hasattr(obj, "value") else str(obj)


def _transport_endpoint(transport: Any) -> str | None:
    """Extract the path/topic/symbol/rpc from any transport type."""
    return (
        getattr(transport, "path", None)
        or getattr(transport, "topic", None)
        or getattr(transport, "symbol", None)
        or getattr(transport, "rpc", None)
    )


def compile_registry(blueprints_root: str | None = None,
                     output_path: str | None = None) -> dict[str, Any]:
    """Compile all live blueprints into a single registry dict.

    Args:
        blueprints_root: Override GAIA_BLUEPRINTS_ROOT env var.
        output_path: Output file (default: /shared/registry/service_registry.json).

    Returns:
        Dict with keys: compiled_at, blueprint_count, edge_count, services, edges, validation.
    """
    if blueprints_root:
        os.environ["GAIA_BLUEPRINTS_ROOT"] = blueprints_root

    blueprints = load_all_live_blueprints()
    if not blueprints:
        logger.warning("No live blueprints found")

    topology = derive_graph_topology(blueprints)

    # Build services dict (stdlib-friendly — no pydantic objects)
    services: dict[str, Any] = {}
    for service_id, bp in blueprints.items():
        inbound: list[dict[str, Any]] = []
        outbound: list[dict[str, Any]] = []
        for iface in bp.interfaces:
            iface_dict = {
                "id": iface.id,
                "transport": _enum_val(getattr(iface.transport, "type", "unknown")),
                "endpoint": _transport_endpoint(iface.transport),
                "method": getattr(iface.transport, "method", None),
                "description": iface.description,
                "status": _enum_val(iface.status),
            }
            if iface.direction == InterfaceDirection.INBOUND:
                inbound.append(iface_dict)
            else:
                outbound.append(iface_dict)

        deps = [
            {"id": d.id, "role": d.role, "required": d.required, "fallback": d.fallback}
            for d in bp.dependencies.services
        ]

        services[service_id] = {
            "id": service_id,
            "role": bp.role,
            "port": bp.runtime.port,
            "gpu": bp.runtime.gpu,
            "health_check": bp.runtime.health_check,
            "service_status": _enum_val(bp.service_status),
            "inbound": inbound,
            "outbound": outbound,
            "dependencies": deps,
        }

    # Build edges list (stdlib-friendly)
    edges = [
        {
            "from_service": e.from_service,
            "to_service": e.to_service,
            "interface_from": e.interface_id_from,
            "interface_to": e.interface_id_to,
            "transport": _enum_val(e.transport_type),
            "status": _enum_val(e.status),
            "description": e.description,
            "has_fallback": e.has_fallback,
        }
        for e in topology.edges
    ]

    validation = _validate_wiring(services, edges)

    registry: dict[str, Any] = {
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "blueprint_count": len(services),
        "edge_count": len(edges),
        "services": services,
        "edges": edges,
        "validation": validation,
    }

    # Write output
    out = Path(output_path or DEFAULT_REGISTRY_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w") as f:
            json.dump(registry, f, indent=2, default=str)
    except OSError as e:
        logger.error("Failed to write registry to %s: %s", out, e)
        raise

    logger.info("Registry compiled: %d services, %d edges → %s", len(services), len(edges), out)
    # Also print for CLI visibility
    print(f"Registry compiled: {len(services)} services, {len(edges)} edges → {out}")
    if validation["orphaned_outbound"]:
        print(f"  WARNING: {len(validation['orphaned_outbound'])} orphaned outbound calls")
    if validation["uncalled_inbound"]:
        print(f"  INFO: {len(validation['uncalled_inbound'])} uncalled inbound endpoints")

    return registry


def _validate_wiring(services: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate wiring — check for orphaned outbound and uncalled inbound."""
    all_outbound = {
        (sid, iface["id"], iface["endpoint"], iface["transport"])
        for sid, svc in services.items()
        for iface in svc["outbound"]
    }

    matched_outbound = {(e["from_service"], e["interface_from"]) for e in edges}
    matched_inbound = {(e["to_service"], e["interface_to"]) for e in edges}

    orphaned_outbound = [
        {"service": sid, "interface": iid, "endpoint": ep, "transport": tp}
        for sid, iid, ep, tp in all_outbound
        if (sid, iid) not in matched_outbound
    ]

    uncalled_inbound = [
        {"service": sid, "interface": iface["id"], "endpoint": iface["endpoint"]}
        for sid, svc in services.items()
        for iface in svc["inbound"]
        if (sid, iface["id"]) not in matched_inbound
    ]

    return {
        "status": "clean" if not orphaned_outbound else "warnings",
        "orphaned_outbound": orphaned_outbound,
        "uncalled_inbound": uncalled_inbound,
        "matched_edges": len(edges),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compile GAIA service registry from blueprints")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: /shared/registry/service_registry.json)")
    parser.add_argument("--blueprints-root", "-b", default=None, help="Blueprints root directory")
    args = parser.parse_args()
    compile_registry(blueprints_root=args.blueprints_root, output_path=args.output)


if __name__ == "__main__":
    main()
