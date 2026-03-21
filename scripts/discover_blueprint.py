#!/usr/bin/env python3
"""
discover_blueprint.py — Auto-generate or refresh a GAIA service blueprint from
a running service's OpenAPI schema.

Connects to a service's /openapi.json endpoint (FastAPI default), extracts all
routes with methods, and produces a BlueprintModel-compatible YAML file.

Modes:
  DISCOVERY  — Generate a new blueprint for a service with no existing YAML.
               Writes to knowledge/blueprints/candidates/{service_id}.yaml
  REFRESH    — Compare a running service against its existing blueprint.
               Reports added/removed endpoints. Optionally updates the YAML.

Usage:
    # Discover a new service
    python scripts/discover_blueprint.py gaia-core --port 6415

    # Refresh an existing blueprint (diff only)
    python scripts/discover_blueprint.py gaia-core --port 6415 --refresh

    # Refresh and auto-update the YAML
    python scripts/discover_blueprint.py gaia-core --port 6415 --refresh --update

    # Discover all known services
    python scripts/discover_blueprint.py --all

    # Use Docker network hostnames instead of localhost
    python scripts/discover_blueprint.py gaia-core --host gaia-core --port 6415
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLUEPRINTS_ROOT = PROJECT_ROOT / "knowledge" / "blueprints"
_SKIP_METHODS = {"HEAD", "OPTIONS", "TRACE"}

# Known services and their default ports (for --all mode)
KNOWN_SERVICES = {
    "gaia-core": {"port": 6415, "role": "The Brain (Cognition)", "gpu": False},
    "gaia-web": {"port": 6414, "role": "The Face (Dashboard + Gateway)", "gpu": False},
    "gaia-mcp": {"port": 8765, "role": "The Hands (Tool Execution)", "gpu": False},
    "gaia-study": {"port": 8766, "role": "The Subconscious (Training + Indexing)", "gpu": True},
    "gaia-orchestrator": {"port": 6410, "role": "The Coordinator (GPU Lifecycle)", "gpu": False},
    "gaia-prime": {"port": 7777, "role": "The Voice (GAIA Engine Inference)", "gpu": True},
    "gaia-audio": {"port": 8080, "role": "The Ears & Mouth (STT + TTS)", "gpu": True},
    "gaia-doctor": {"port": 6419, "role": "The Immune System (Health Watchdog)", "gpu": False},
    "gaia-monkey": {"port": 6420, "role": "The Adversary (Chaos Testing)", "gpu": False},
    "gaia-nano": {"port": 8090, "role": "The Reflex (Nano Triage)", "gpu": True},
}


def fetch_openapi(host: str, port: int, timeout: float = 5.0) -> dict | None:
    """Fetch /openapi.json from a running service. Returns None on failure."""
    url = f"http://{host}:{port}/openapi.json"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as e:
        print(f"  Could not reach {url}: {e}", file=sys.stderr)
        return None


def fetch_health(host: str, port: int, timeout: float = 3.0) -> bool:
    """Check if a service is healthy."""
    url = f"http://{host}:{port}/health"
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def extract_interfaces(openapi: dict) -> list[dict]:
    """Extract interfaces from OpenAPI paths."""
    interfaces = []
    seen_ids = set()

    for path, methods in sorted(openapi.get("paths", {}).items()):
        for method, details in methods.items():
            method = method.upper()
            if method in _SKIP_METHODS:
                continue

            # Generate a stable interface ID from path
            iface_id = _path_to_id(path, method)
            if iface_id in seen_ids:
                # Append method suffix to disambiguate
                iface_id = f"{iface_id}_{method.lower()}"
            seen_ids.add(iface_id)

            summary = details.get("summary", "")
            description = details.get("description", summary or f"{method} {path}")
            # Truncate to first sentence
            if description and ". " in description:
                description = description.split(". ")[0] + "."

            interfaces.append({
                "id": iface_id,
                "direction": "inbound",
                "transport": {
                    "type": "http_rest",
                    "path": path,
                    "method": method,
                    "input_schema": None,
                    "output_schema": None,
                },
                "description": description,
                "status": "active",
            })

    return interfaces


def _path_to_id(path: str, method: str) -> str:
    """Convert a URL path to a stable interface ID."""
    # /api/kv-cache/compact/{role} → kv_cache_compact
    # /health → health
    # /v1/chat/completions → v1_chat_completions
    parts = path.strip("/").split("/")
    # Remove path parameters
    parts = [p for p in parts if not p.startswith("{")]
    # Remove common prefixes
    if parts and parts[0] == "api":
        parts = parts[1:]
    name = "_".join(parts).replace("-", "_")
    if not name:
        name = "root"
    return name


def load_existing_blueprint(service_id: str) -> dict | None:
    """Load existing blueprint YAML as raw dict (no pydantic)."""
    import yaml
    live_path = BLUEPRINTS_ROOT / f"{service_id}.yaml"
    try:
        with live_path.open() as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None
    except (yaml.YAMLError, OSError) as e:
        print(f"  WARNING: Failed to load {live_path}: {e}", file=sys.stderr)
        return None


def diff_interfaces(existing: list[dict], discovered: list[dict]) -> dict:
    """Compare existing vs discovered interfaces. Returns added/removed."""
    def _iface_key(iface):
        t = iface.get("transport", {})
        path = t.get("path") or t.get("topic") or t.get("symbol") or ""
        method = t.get("method", "GET")
        return (path, method)

    existing_keys = {_iface_key(i) for i in existing if i.get("direction") == "inbound"}
    discovered_keys = {_iface_key(i) for i in discovered}

    added = discovered_keys - existing_keys
    removed = existing_keys - discovered_keys

    added_details = [i for i in discovered if _iface_key(i) in added]
    removed_details = [(path, method) for path, method in removed]

    return {
        "added": added_details,
        "removed": removed_details,
        "existing_count": len(existing_keys),
        "discovered_count": len(discovered_keys),
    }


def generate_blueprint_yaml(service_id: str, interfaces: list[dict],
                             role: str = "", gpu: bool = False,
                             port: int = 0) -> dict:
    """Generate a complete blueprint dict suitable for YAML serialization."""
    return {
        "id": service_id,
        "version": "0.1",
        "role": role or f"Discovered service ({service_id})",
        "service_status": "live",
        "runtime": {
            "port": port,
            "base_image": "python:3.11-slim",
            "gpu": gpu,
            "startup_cmd": None,
            "health_check": f"curl -f http://localhost:{port}/health",
            "gpu_count": None,
            "user": None,
            "dockerfile": f"{service_id}/Dockerfile",
            "compose_service": service_id,
            "security": None,
        },
        "interfaces": interfaces,
        "dependencies": {"services": [], "volumes": [], "external_apis": []},
        "source_files": [],
        "failure_modes": [],
        "intent": {
            "purpose": f"Auto-discovered blueprint for {service_id}. Review and enrich.",
            "design_decisions": [],
            "open_questions": [
                "Blueprint auto-discovered from OpenAPI — review interfaces and add outbound calls",
                "Add service dependencies, volumes, and failure modes",
                "Write intent/purpose description",
            ],
            "cognitive_role": None,
        },
        "meta": {
            "status": "candidate",
            "genesis": True,
            "generated_by": "discovery",
            "blueprint_version": "0.1",
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_reflected": None,
            "promoted_at": None,
            "confidence": {
                "runtime": "high",
                "contract": "medium",
                "dependencies": "low",
                "failure_modes": "low",
                "intent": "low",
            },
            "reflection_notes": f"Auto-discovered from /openapi.json on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "divergence_score": None,
        },
    }


def discover_service(service_id: str, host: str = "localhost", port: int = 0,
                     refresh: bool = False, update: bool = False) -> dict | None:
    """Discover or refresh a single service blueprint."""
    import yaml

    svc_info = KNOWN_SERVICES.get(service_id, {})
    if not port:
        port = svc_info.get("port", 8080)

    print(f"\n{'=' * 60}")
    print(f"  {service_id} ({host}:{port})")
    print(f"{'=' * 60}")

    # Check health first
    if not fetch_health(host, port):
        print(f"  SKIP: {service_id} not healthy at {host}:{port}")
        return None

    # Fetch OpenAPI
    openapi = fetch_openapi(host, port)
    if not openapi:
        print(f"  SKIP: No OpenAPI schema at {host}:{port}/openapi.json")
        return None

    # Extract interfaces
    discovered = extract_interfaces(openapi)
    print(f"  Discovered {len(discovered)} inbound endpoints")

    existing = load_existing_blueprint(service_id) if refresh else None
    if existing:
        existing_ifaces = existing.get("interfaces", [])
        diff = diff_interfaces(existing_ifaces, discovered)
        print(f"  Existing: {diff['existing_count']} inbound, Discovered: {diff['discovered_count']}")

        if diff["added"]:
            print(f"\n  ADDED ({len(diff['added'])}):")
            for iface in diff["added"]:
                t = iface["transport"]
                print(f"    + {t['method']} {t['path']}")

        if diff["removed"]:
            print(f"\n  REMOVED ({len(diff['removed'])}):")
            for path, method in diff["removed"]:
                print(f"    - {method} {path}")

        if not diff["added"] and not diff["removed"]:
            print("  No changes detected")
            return {"service": service_id, "status": "up_to_date"}

        if update and diff["added"]:
            existing["interfaces"] = existing_ifaces + diff["added"]
            out_path = BLUEPRINTS_ROOT / f"{service_id}.yaml"
            try:
                with out_path.open("w") as f:
                    yaml.dump(existing, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"\n  UPDATED: {out_path}")
            except OSError as e:
                print(f"  ERROR: Failed to write {out_path}: {e}", file=sys.stderr)
            return {"service": service_id, "status": "updated", "added": len(diff["added"])}

        return {
            "service": service_id,
            "status": "drift_detected",
            "added": len(diff["added"]),
            "removed": len(diff["removed"]),
        }

    # No existing blueprint or not in refresh mode — full discovery
    if refresh:
        print("  No existing blueprint — treating as discovery")
    role = svc_info.get("role", "")
    gpu = svc_info.get("gpu", False)
    blueprint = generate_blueprint_yaml(service_id, discovered, role=role, gpu=gpu, port=port)

    candidates_dir = BLUEPRINTS_ROOT / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    out_path = candidates_dir / f"{service_id}.yaml"
    try:
        with out_path.open("w") as f:
            yaml.dump(blueprint, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  WROTE: {out_path}")
    except OSError as e:
        print(f"  ERROR: Failed to write {out_path}: {e}", file=sys.stderr)
    return {"service": service_id, "status": "discovered", "endpoints": len(discovered)}


def discover_all(host: str = "localhost", refresh: bool = False, update: bool = False) -> list[dict]:
    """Discover/refresh all known services."""
    results = []
    for service_id, info in KNOWN_SERVICES.items():
        result = discover_service(
            service_id,
            host=host,
            port=info["port"],
            refresh=refresh,
            update=update,
        )
        if result:
            results.append(result)

    print(f"\n{'=' * 60}")
    print(f"  Summary: {len(results)} services processed")
    for r in results:
        print(f"    {r['service']}: {r['status']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-discover or refresh GAIA service blueprints")
    parser.add_argument("service", nargs="?", help="Service ID (e.g., gaia-core)")
    parser.add_argument("--host", default="localhost", help="Service hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=0, help="Service port (default: from KNOWN_SERVICES)")
    parser.add_argument("--all", action="store_true", help="Discover all known services")
    parser.add_argument("--refresh", action="store_true", help="Compare running service against existing blueprint")
    parser.add_argument("--update", action="store_true", help="Auto-update blueprint with new endpoints (requires --refresh)")
    args = parser.parse_args()

    if args.all:
        discover_all(host=args.host, refresh=args.refresh, update=args.update)
    elif args.service:
        discover_service(args.service, host=args.host, port=args.port,
                        refresh=args.refresh, update=args.update)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
