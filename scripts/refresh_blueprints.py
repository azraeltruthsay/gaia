#!/usr/bin/env python3
"""
refresh_blueprints.py — Full blueprint lifecycle automation.

Runs the complete refresh cycle:
  1. Discover endpoints from all running services (via /openapi.json)
  2. Diff discovered endpoints against existing blueprints
  3. Optionally auto-update blueprints with new endpoints
  4. Compile registry (blueprints -> JSON)
  5. Validate wiring

This is the single command that keeps blueprints in sync with reality.

Usage:
    # Dry run — show diff, don't change anything
    python scripts/refresh_blueprints.py

    # Auto-update blueprints + recompile registry
    python scripts/refresh_blueprints.py --update

    # Use Docker network hostnames (from inside a container)
    python scripts/refresh_blueprints.py --docker
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gaia-common"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from discover_blueprint import discover_all, discover_service, KNOWN_SERVICES
    from compile_registry import compile_registry
    from validate_wiring import validate, print_report
except ImportError as e:
    print(f"FATAL: Missing dependency script: {e}", file=sys.stderr)
    print("Ensure compile_registry.py, validate_wiring.py, discover_blueprint.py exist in scripts/", file=sys.stderr)
    sys.exit(127)


def main() -> int:
    parser = argparse.ArgumentParser(description="Full blueprint refresh cycle")
    parser.add_argument("--update", action="store_true", help="Auto-update blueprints with discovered endpoints")
    parser.add_argument("--docker", action="store_true", help="Use Docker network hostnames instead of localhost")
    parser.add_argument("--output", default=None, help="Registry output path")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    blueprints_root = str(PROJECT_ROOT / "knowledge" / "blueprints")
    output = args.output or "/tmp/service_registry.json"

    print("=" * 60)
    print("  GAIA Blueprint Refresh Cycle")
    print("=" * 60)

    # Phase 1: Discover
    print("\n--- Phase 1: Discover endpoints from running services ---")
    if not args.docker:
        results = discover_all(host="localhost", refresh=True, update=args.update)
    else:
        # Docker mode — use per-service hostnames
        results = []
        for service_id, info in KNOWN_SERVICES.items():
            r = discover_service(
                service_id,
                host=service_id,
                port=info["port"],
                refresh=True,
                update=args.update,
            )
            if r:
                results.append(r)

    drift_count = sum(1 for r in results if r.get("status") == "drift_detected")
    updated_count = sum(1 for r in results if r.get("status") == "updated")

    # Phase 2: Compile registry
    print("\n--- Phase 2: Compile registry from blueprints ---")
    try:
        registry = compile_registry(blueprints_root=blueprints_root, output_path=output)
    except Exception as e:
        print(f"ERROR: Registry compilation failed: {e}", file=sys.stderr)
        return 2

    # Phase 3: Validate wiring
    print("\n--- Phase 3: Validate wiring ---")
    result = validate(registry)
    print_report(result, as_json=args.json)

    # Summary
    print(f"\n{'=' * 60}")
    print("  Refresh Summary")
    print(f"{'=' * 60}")
    print(f"  Services scanned:  {len(results)}")
    print(f"  Drift detected:    {drift_count}")
    print(f"  Blueprints updated: {updated_count}")
    print(f"  Registry services: {registry.get('blueprint_count', '?')}")
    print(f"  Registry edges:    {registry.get('edge_count', '?')}")
    print(f"  Wiring status:     {registry.get('validation', {}).get('status', '?')}")

    # Copy to shared dir if accessible
    shared_path = Path("/shared/registry/service_registry.json")
    try:
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, shared_path)
        print(f"  Registry synced to {shared_path}")
    except OSError:
        pass  # /shared not accessible on host — normal

    if drift_count > 0 and not args.update:
        print("\n  TIP: Run with --update to auto-apply discovered changes")

    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
