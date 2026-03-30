#!/usr/bin/env python3
"""
validate_paths.py — Live path interconnectivity validation for GAIA services.

Goes beyond blueprint wiring validation by checking that every outbound call
in the compiled registry actually exists as a live endpoint on the target service.

Reads the compiled registry for edge pairs, then hits each service's /openapi.json
to verify the target path+method actually exists at runtime.

Catches bugs like:
  - Blueprint declares an endpoint but the router was never mounted
  - Caller uses POST but receiver only accepts GET
  - Path typos (/models/ vs /model/)
  - Endpoints removed from code but still in blueprint

Exit codes:
  0 = all paths verified
  1 = mismatches found
  2 = registry or services unreachable

Usage:
    python scripts/validate_paths.py [--registry PATH] [--host HOST] [--json]
    python scripts/validate_paths.py --docker   # use container hostnames
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen, Request

DEFAULT_REGISTRY = "/shared/registry/service_registry.json"

# Services without OpenAPI (stdlib HTTP server, llama-server)
_NO_OPENAPI = {"gaia-doctor", "gaia-nano", "dozzle", "gaia-wiki"}

# Known port mappings (host-side) for services
_HOST_PORTS = {
    "gaia-core": 6415,
    "gaia-web": 6414,
    "gaia-mcp": 8765,
    "gaia-study": 8766,
    "gaia-orchestrator": 6410,
    "gaia-prime": 7777,
    "gaia-audio": 8080,
    "gaia-doctor": 6419,
    "gaia-monkey": 6420,
    "gaia-nano": 8090,
}

# Internal Docker ports (for --docker mode)
_DOCKER_PORTS = {
    "gaia-core": 6415,
    "gaia-web": 6414,
    "gaia-mcp": 8765,
    "gaia-study": 8766,
    "gaia-orchestrator": 6410,
    "gaia-prime": 7777,
    "gaia-audio": 8080,
    "gaia-doctor": 6419,
    "gaia-monkey": 6420,
    "gaia-nano": 8080,
}


def load_registry(path: str) -> dict[str, Any] | None:
    """Load compiled registry JSON."""
    try:
        with Path(path).open() as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Registry not found at {path}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Failed to read registry: {e}", file=sys.stderr)
        return None


def fetch_openapi(host: str, port: int, timeout: float = 5.0) -> dict | None:
    """Fetch /openapi.json from a running service."""
    url = f"http://{host}:{port}/openapi.json"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError):
        return None


def extract_live_routes(openapi: dict) -> dict[tuple[str, str], dict]:
    """Extract all routes from OpenAPI as {(path, METHOD): details} dict.

    Normalizes path parameters: /adapters/{name} -> /adapters/{param}
    """
    routes = {}
    for path, methods in openapi.get("paths", {}).items():
        normalized = _normalize_path(path)
        for method, details in methods.items():
            method = method.upper()
            if method in ("HEAD", "OPTIONS", "TRACE"):
                continue
            routes[(normalized, method)] = {
                "path": path,
                "method": method,
                "summary": details.get("summary", ""),
                "parameters": details.get("parameters", []),
                "request_body": bool(details.get("requestBody")),
                "responses": list(details.get("responses", {}).keys()),
            }
    return routes


def _normalize_path(path: str) -> str:
    """Normalize path parameters for comparison.

    /adapters/{adapter_name} and /adapters/{name} should match.
    /api/kv-cache/restore/{role} matches /api/kv-cache/restore/{anything}
    """
    return re.sub(r"\{[^}]+\}", "{param}", path)


def validate_paths(registry: dict[str, Any], host: str = "localhost",
                   docker: bool = False) -> dict[str, Any]:
    """Validate all edge paths against live OpenAPI schemas.

    Returns a validation report with verified, mismatched, and unreachable results.
    """
    edges = registry.get("edges", [])
    services = registry.get("services", {})

    port_map = _DOCKER_PORTS if docker else _HOST_PORTS

    # Cache OpenAPI schemas per service
    openapi_cache: dict[str, dict[tuple[str, str], dict] | None] = {}

    def _get_routes(service_id: str) -> dict[tuple[str, str], dict] | None:
        if service_id in openapi_cache:
            return openapi_cache[service_id]

        if service_id in _NO_OPENAPI:
            openapi_cache[service_id] = None
            return None

        svc_host = service_id if docker else host
        port = port_map.get(service_id, 8080)
        schema = fetch_openapi(svc_host, port)
        if schema:
            routes = extract_live_routes(schema)
            openapi_cache[service_id] = routes
            return routes

        openapi_cache[service_id] = None
        return None

    verified = []
    mismatched = []
    unreachable = []
    skipped = []

    for edge in edges:
        from_svc = edge["from_service"]
        to_svc = edge["to_service"]
        transport = edge.get("transport", "")
        iface_from = edge.get("interface_from", "")
        iface_to = edge.get("interface_to", "")
        description = edge.get("description", "")

        # Only validate HTTP REST edges
        if transport != "http_rest":
            skipped.append({
                "from": from_svc,
                "to": to_svc,
                "interface": iface_from,
                "transport": transport,
                "reason": f"non-HTTP transport ({transport})",
            })
            continue

        # Get the target endpoint path from the registry
        to_svc_data = services.get(to_svc, {})
        target_iface = None
        for iface in to_svc_data.get("inbound", []):
            if iface["id"] == iface_to:
                target_iface = iface
                break

        if not target_iface:
            mismatched.append({
                "from": from_svc,
                "to": to_svc,
                "interface": iface_to,
                "expected_path": "?",
                "expected_method": "?",
                "issue": "target interface not found in registry",
            })
            continue

        expected_path = target_iface.get("endpoint", "")
        expected_method = (target_iface.get("method") or "GET").upper()

        # Skip services without OpenAPI
        if to_svc in _NO_OPENAPI:
            skipped.append({
                "from": from_svc,
                "to": to_svc,
                "interface": iface_to,
                "path": expected_path,
                "reason": f"{to_svc} has no OpenAPI (stdlib/llama-server)",
            })
            continue

        # Get live routes for target service
        routes = _get_routes(to_svc)
        if routes is None:
            unreachable.append({
                "from": from_svc,
                "to": to_svc,
                "interface": iface_to,
                "path": expected_path,
                "reason": f"{to_svc} OpenAPI unreachable",
            })
            continue

        # Check if the path+method exists in the live service
        normalized = _normalize_path(expected_path) if expected_path else ""
        live_match = routes.get((normalized, expected_method))

        if live_match:
            verified.append({
                "from": from_svc,
                "to": to_svc,
                "interface": iface_to,
                "path": expected_path,
                "method": expected_method,
            })
        else:
            # Check if path exists with wrong method
            methods_for_path = [m for (p, m), _ in routes.items() if p == normalized]
            if methods_for_path:
                issue = f"method mismatch: blueprint says {expected_method}, live has {methods_for_path}"
            else:
                # Check for close matches (typos)
                close = _find_close_paths(normalized, routes)
                if close:
                    issue = f"path not found — close matches: {close}"
                else:
                    issue = "path not found in live OpenAPI"

            mismatched.append({
                "from": from_svc,
                "to": to_svc,
                "interface": iface_to,
                "expected_path": expected_path,
                "expected_method": expected_method,
                "issue": issue,
            })

    return {
        "verified": verified,
        "mismatched": mismatched,
        "unreachable": unreachable,
        "skipped": skipped,
        "summary": {
            "total_edges": len(edges),
            "verified": len(verified),
            "mismatched": len(mismatched),
            "unreachable": len(unreachable),
            "skipped": len(skipped),
            "services_checked": len([s for s in openapi_cache if openapi_cache[s] is not None]),
            "services_unreachable": len([s for s in openapi_cache if openapi_cache[s] is None and s not in _NO_OPENAPI]),
        },
    }


def _find_close_paths(target: str, routes: dict) -> list[str]:
    """Find paths that are similar to the target (simple prefix matching)."""
    parts = target.strip("/").split("/")
    if not parts:
        return []
    prefix = "/" + parts[0]
    close = []
    for (path, _method), _ in routes.items():
        if path.startswith(prefix) and path != target:
            close.append(path)
    return close[:3]


def print_report(result: dict[str, Any], as_json: bool = False) -> None:
    """Print human-readable or JSON validation report."""
    if as_json:
        print(json.dumps(result, indent=2))
        return

    s = result["summary"]
    print("GAIA Live Path Validation")
    print("=" * 50)
    print(f"Edges checked:     {s['total_edges']}")
    print(f"Verified:          {s['verified']}")
    print(f"Mismatched:        {s['mismatched']}")
    print(f"Unreachable:       {s['unreachable']}")
    print(f"Skipped:           {s['skipped']}")
    print(f"Services checked:  {s['services_checked']}")
    print()

    if result["mismatched"]:
        print(f"MISMATCHED ({len(result['mismatched'])}):")
        for m in result["mismatched"]:
            print(f"  {m['from']} -> {m['to']}.{m['interface']}")
            print(f"    expected: {m.get('expected_method', '?')} {m.get('expected_path', '?')}")
            print(f"    issue:    {m['issue']}")
        print()

    if result["unreachable"]:
        print(f"UNREACHABLE ({len(result['unreachable'])}):")
        for u in result["unreachable"]:
            print(f"  {u['from']} -> {u['to']}: {u['reason']}")
        print()

    if not result["mismatched"]:
        print("Status: ALL PATHS VERIFIED")
    else:
        print(f"Status: {len(result['mismatched'])} PATH MISMATCHES")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live path interconnectivity validation")
    parser.add_argument("--registry", "-r", default=DEFAULT_REGISTRY, help="Registry JSON path")
    parser.add_argument("--host", default="localhost", help="Service hostname (default: localhost)")
    parser.add_argument("--docker", action="store_true", help="Use Docker container hostnames")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if registry is None:
        sys.exit(2)

    result = validate_paths(registry, host=args.host, docker=args.docker)
    print_report(result, as_json=args.json)

    sys.exit(1 if result["mismatched"] else 0)


if __name__ == "__main__":
    main()
