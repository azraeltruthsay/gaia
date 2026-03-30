#!/usr/bin/env python3
"""
validate_wiring.py — Validate GAIA service wiring from compiled registry.

Reads the compiled service_registry.json (stdlib only — no pydantic required)
and reports:
  - Orphaned outbound calls (no matching inbound endpoint in any service)
  - Uncalled inbound endpoints (no outbound points to them)

Exit codes:
  0 = clean (no issues)
  1 = warnings (orphaned outbound or uncalled inbound)
  2 = errors (registry file missing or corrupt)

Usage:
    python scripts/validate_wiring.py [--registry PATH] [--strict] [--json]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = "/shared/registry/service_registry.json"
_STATUS_LABELS = {0: "CLEAN", 1: "WARNINGS", 2: "ERRORS"}


def load_registry(path: str) -> dict[str, Any] | None:
    """Load compiled registry JSON. Returns None on failure."""
    try:
        with Path(path).open() as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Registry not found at {path}", file=sys.stderr)
        print("Run: python scripts/compile_registry.py", file=sys.stderr)
        return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Failed to read registry at {path}: {e}", file=sys.stderr)
        return None


def validate(registry: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """Extract pre-computed wiring validation from compiled registry and set exit code."""
    validation = registry.get("validation", {})
    orphaned = validation.get("orphaned_outbound", [])
    uncalled = validation.get("uncalled_inbound", [])

    exit_code = 0
    if orphaned:
        exit_code = 1
    if strict and uncalled:
        exit_code = 1

    return {
        "services_count": len(registry.get("services", {})),
        "edges_count": len(registry.get("edges", [])),
        "orphaned_outbound": orphaned,
        "uncalled_inbound": uncalled,
        "exit_code": exit_code,
    }


def print_report(result: dict[str, Any], as_json: bool = False) -> None:
    """Print human-readable or JSON validation report."""
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print("GAIA Service Wiring Validation")
    print("=" * 40)
    print(f"Services:  {result.get('services_count', '?')}")
    print(f"Edges:     {result.get('edges_count', '?')}")
    print()

    orphaned = result.get("orphaned_outbound", [])
    uncalled = result.get("uncalled_inbound", [])

    if not orphaned and not uncalled:
        print("Status: CLEAN — all wiring validated")
        return

    if orphaned:
        print(f"ORPHANED OUTBOUND ({len(orphaned)}):")
        print("  These outbound calls have no matching inbound endpoint in any service.")
        for o in orphaned:
            print(f"  - {o.get('service', '?')}.{o.get('interface', '?')} → {o.get('transport', '?')} {o.get('endpoint', '?')}")
        print()

    if uncalled:
        print(f"UNCALLED INBOUND ({len(uncalled)}):")
        print("  These inbound endpoints are not called by any outbound interface.")
        for u in uncalled:
            print(f"  - {u.get('service', '?')}.{u.get('interface', '?')} ← {u.get('endpoint', '?')}")
        print()

    print(f"Status: {_STATUS_LABELS.get(result['exit_code'], 'UNKNOWN')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GAIA service wiring")
    parser.add_argument("--registry", "-r", default=DEFAULT_REGISTRY, help="Registry JSON path")
    parser.add_argument("--strict", action="store_true", help="Treat uncalled inbound as warnings too")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if registry is None:
        sys.exit(2)

    result = validate(registry, strict=args.strict)
    print_report(result, as_json=args.json)
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
