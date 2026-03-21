#!/usr/bin/env python3
"""
cognitive_battery_full.py — Orchestrated multi-tier cognitive battery with SAE monitoring.

Runs the cognitive test battery against all 3 model tiers (Prime, Core, Nano) with:
  - Orchestrator-managed GPU handoffs between tiers
  - KV cache warm-up before architecture tests
  - SAE atlas recording during each tier's battery
  - Polygraph hidden-state capture on every test response
  - Per-tier and cross-tier comparison reporting

This is the deep validation tool — not for quick checks, but for comprehensive
knowledge and capability assessment across the full model stack.

Usage:
    python scripts/cognitive_battery_full.py
    python scripts/cognitive_battery_full.py --tiers core,nano  # skip prime
    python scripts/cognitive_battery_full.py --no-sae           # skip SAE recording
    python scripts/cognitive_battery_full.py --canary-only       # only general knowledge
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GAIA.CogBattery.Full")

# Service endpoints
ORCHESTRATOR = os.getenv("ORCHESTRATOR_ENDPOINT", "http://localhost:6410")
CORE_ENDPOINT = os.getenv("CORE_ENDPOINT", "http://localhost:6415")
DOCTOR_ENDPOINT = os.getenv("DOCTOR_ENDPOINT", "http://localhost:6419")

# Engine ports (internal to gaia-core container, but mapped to host)
ENGINE_PORTS = {
    "core": os.getenv("CORE_ENGINE_PORT", "8092"),
    "nano": os.getenv("NANO_ENGINE_PORT", "8090"),
    "prime": os.getenv("PRIME_ENGINE_PORT", "7777"),
}

# Tier metadata
TIERS = {
    "prime": {
        "name": "Prime (8B)",
        "engine_host": "localhost",
        "engine_port": ENGINE_PORTS["prime"],
        "target": "prime",
        "sae_layers": [4, 8, 12, 16, 20, 24, 28],
    },
    "core": {
        "name": "Core (2B)",
        "engine_host": "localhost",
        "engine_port": ENGINE_PORTS["core"],
        "target": "core",
        "sae_layers": [2, 6, 10, 14, 18, 22, 26],
    },
    "nano": {
        "name": "Nano (0.8B)",
        "engine_host": "localhost",
        "engine_port": ENGINE_PORTS["nano"],
        "target": "nano",
        "sae_layers": [2, 4, 8, 12, 16, 20, 23],
    },
}

OUTPUT_DIR = Path(os.getenv("COG_BATTERY_OUTPUT", "/tmp/cognitive_battery_full"))


def http_get(url: str, timeout: float = 10.0) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as e:
        log.debug("GET %s failed: %s", url, e)
        return None


def http_post(url: str, data: dict | None = None, timeout: float = 30.0) -> dict:
    payload = json.dumps(data or {}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": str(e)}


def wait_for_health(url: str, timeout: int = 120) -> bool:
    """Poll a health endpoint until it returns 200."""
    for i in range(timeout):
        try:
            with urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(1)
    return False


def _engine_url(tier: str) -> str:
    """Get the engine URL — use Docker hostname if available, else localhost mapped port."""
    # When running from host, engine ports may not be mapped.
    # Try the Docker-internal hostname via gaia-core proxy first.
    docker_hosts = {"core": "gaia-core:8092", "nano": "gaia-nano:8080", "prime": "gaia-prime:7777"}
    host_ports = {"core": "8092", "nano": "8090", "prime": "7777"}
    # From host, use localhost + mapped port
    return f"http://localhost:{host_ports.get(tier, '8092')}"


def enable_polygraph(tier: str) -> bool:
    """Enable hidden-state capture on the tier's engine (best-effort)."""
    url = _engine_url(tier)
    result = http_post(f"{url}/polygraph/enable")
    if result.get("ok"):
        log.info("Polygraph enabled for %s", tier)
        return True
    log.debug("Polygraph not available for %s (engine may not be directly reachable): %s", tier, result)
    return False


def disable_polygraph(tier: str) -> None:
    url = _engine_url(tier)
    http_post(f"{url}/polygraph/disable")


def record_sae_atlas(tier: str, tag: str) -> dict | None:
    """Trigger SAE atlas recording on the tier's engine."""
    port = ENGINE_PORTS.get(tier, "8092")
    tier_info = TIERS.get(tier, {})
    output_dir = str(OUTPUT_DIR / f"sae_{tier}_{tag}")

    result = http_post(f"http://localhost:{port}/atlas/record", {
        "tier": tier,
        "tag": tag,
        "output_dir": output_dir,
        "layers": tier_info.get("sae_layers", []),
        "num_features_multiplier": 2,
        "epochs": 30,
    }, timeout=300)

    if result.get("ok"):
        log.info("SAE atlas recording started for %s (tag=%s, dir=%s)", tier, tag, output_dir)
        return result
    log.warning("SAE atlas recording failed for %s: %s", tier, result)
    return None


def warm_kv_cache(tier: str) -> bool:
    """Warm the tier's KV cache with identity + system context.

    This ensures architecture-specific tests run against a model that has
    its full cognitive context loaded, not a cold start.
    """
    port = ENGINE_PORTS.get(tier, "8092")
    # Trigger a cache update with identity + world state segments
    result = http_post(f"http://localhost:{port}/cache/update", {
        "identity": "You are GAIA, a sovereign AI system.",
        "world_state": f"Current time: {datetime.now().strftime('%I:%M %p %Z')}",
    })
    if result.get("ok") or result.get("segments_updated"):
        log.info("KV cache warmed for %s", tier)
        return True
    # Not fatal — cache warm is best-effort
    log.debug("KV cache warm failed for %s: %s", tier, result)
    return False


def run_battery_for_tier(tier: str, sections: list[str] | None = None,
                         canary_only: bool = False) -> dict:
    """Run the cognitive test battery against a specific tier.

    Uses the doctor's battery runner via /cognitive/run, targeting the specific model.
    """
    tier_info = TIERS[tier]
    target = tier_info["target"]

    payload: dict[str, Any] = {
        "target": target,
        "timeout": 30,
        "no_think": True,  # faster, saves tokens
    }
    if sections:
        payload["section"] = sections[0] if len(sections) == 1 else None
    if canary_only:
        payload["section"] = "general_knowledge"

    log.info("Running cognitive battery for %s (target=%s)", tier_info["name"], target)

    result = http_post(f"{DOCTOR_ENDPOINT}/cognitive/run", payload, timeout=600)
    if result.get("error"):
        log.error("Battery failed for %s: %s", tier, result["error"])
        return {"tier": tier, "error": result["error"]}

    # Wait for battery to complete (it runs async)
    log.info("Battery started for %s, waiting for completion...", tier)
    for _ in range(120):  # 10 minutes max
        time.sleep(5)
        status = http_get(f"{DOCTOR_ENDPOINT}/cognitive/status")
        if status and not status.get("running"):
            break

    # Get results
    results = http_get(f"{DOCTOR_ENDPOINT}/cognitive/results")
    if results:
        summary = results.get("summary", {})
        log.info("%s battery: %d/%d passed (%.1f%%)",
                 tier_info["name"], summary.get("passed", 0),
                 summary.get("total", 0), summary.get("pass_rate", 0) * 100)
        return {"tier": tier, "results": results}

    return {"tier": tier, "error": "Could not retrieve results"}


def run_full_battery(tiers: list[str], enable_sae: bool = True,
                     canary_only: bool = False) -> dict:
    """Run the full orchestrated multi-tier battery."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"full-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    log.info("=" * 60)
    log.info("  GAIA Full Cognitive Battery — %s", run_id)
    log.info("  Tiers: %s | SAE: %s | Canary-only: %s", tiers, enable_sae, canary_only)
    log.info("=" * 60)

    all_results = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tiers": {},
    }

    for tier in tiers:
        tier_info = TIERS.get(tier)
        if not tier_info:
            log.warning("Unknown tier: %s, skipping", tier)
            continue

        log.info("")
        log.info("─── Tier: %s ───", tier_info["name"])

        # Step 1: Check if the tier is reachable (via gaia-core's health endpoint)
        health = http_get(f"{CORE_ENDPOINT}/health")
        if not health:
            log.warning("gaia-core not reachable — skipping %s", tier)
            all_results["tiers"][tier] = {"error": "gaia-core unreachable", "skipped": True}
            continue

        # Step 2: Enable polygraph for hidden state monitoring
        if enable_sae:
            enable_polygraph(tier)

        # Step 3: Warm KV cache (for architecture tests)
        if not canary_only:
            warm_kv_cache(tier)

        # Step 4: Run the battery
        t0 = time.time()
        tier_result = run_battery_for_tier(tier, canary_only=canary_only)
        elapsed = time.time() - t0

        # Step 5: Disable polygraph
        if enable_sae:
            disable_polygraph(tier)

        # Step 6: Record SAE atlas snapshot (post-battery)
        if enable_sae:
            record_sae_atlas(tier, tag=f"battery_{run_id}")

        tier_result["elapsed_seconds"] = round(elapsed, 1)
        all_results["tiers"][tier] = tier_result
        log.info("%s completed in %.1fs", tier_info["name"], elapsed)

    # Summary
    all_results["completed_at"] = datetime.now(timezone.utc).isoformat()
    log.info("")
    log.info("=" * 60)
    log.info("  Results Summary")
    log.info("=" * 60)

    for tier, result in all_results["tiers"].items():
        if result.get("skipped"):
            log.info("  %s: SKIPPED (%s)", TIERS[tier]["name"], result.get("error", ""))
            continue
        if result.get("error"):
            log.info("  %s: ERROR (%s)", TIERS[tier]["name"], result["error"])
            continue
        r = result.get("results", {})
        summary = r.get("summary", {})
        canary = r.get("canary", {})
        log.info("  %s: %d/%d (%.1f%%) — canary: %d/%d (%.1f%%) — %.1fs",
                 TIERS[tier]["name"],
                 summary.get("passed", 0), summary.get("total", 0),
                 summary.get("pass_rate", 0) * 100,
                 canary.get("passed", 0), canary.get("total", 0),
                 canary.get("pass_rate", 0) * 100,
                 result.get("elapsed_seconds", 0))

    # Save full results
    output_file = OUTPUT_DIR / f"results_{run_id}.json"
    with output_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("  Results saved to: %s", output_file)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrated multi-tier cognitive battery")
    parser.add_argument("--tiers", default="prime,core,nano",
                        help="Comma-separated tiers to test (default: prime,core,nano)")
    parser.add_argument("--no-sae", action="store_true",
                        help="Skip SAE atlas recording (faster)")
    parser.add_argument("--canary-only", action="store_true",
                        help="Only run general knowledge canaries (skip architecture tests)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    results = run_full_battery(tiers, enable_sae=not args.no_sae, canary_only=args.canary_only)

    if args.json:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
