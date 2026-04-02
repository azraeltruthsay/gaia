#!/usr/bin/env python3
"""
Training Lifecycle Manager — proper setup/teardown for GAIA training sessions.

Ensures the system is never left in a half-broken state after training.
Snapshots current state, unloads models for VRAM, runs training, then
restores everything to exactly how it was.

Usage as a library:
    from training_lifecycle import TrainingLifecycle

    with TrainingLifecycle() as lc:
        lc.start_study()
        # ... run training ...
        # teardown happens automatically, even on exception

Usage as a CLI wrapper:
    python training_lifecycle.py -- python /path/to/train_script.py [args]
    python training_lifecycle.py --skip-restore -- python train.py --skip-sae
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] lifecycle: %(message)s",
)
logger = logging.getLogger("training-lifecycle")

# ── Configuration ──────────────────────────────────────────────────────────

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:6410")
NANO_URL = os.getenv("NANO_URL", "http://localhost:8090")
CORE_URL = os.getenv("CORE_URL", "http://localhost:8092")
PRIME_URL = os.getenv("PRIME_URL", "http://localhost:7777")

# Timeouts
HTTP_TIMEOUT = 15
MODEL_LOAD_TIMEOUT = 180
HEALTH_CHECK_RETRIES = 5
HEALTH_CHECK_DELAY = 3


# ── Data Types ─────────────────────────────────────────────────────────────

@dataclass
class TierSnapshot:
    """Snapshot of a single tier's state."""
    name: str
    url: str
    model_path: str = ""
    device: str = "unloaded"  # "gpu", "cpu", "unloaded"
    was_loaded: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class SystemSnapshot:
    """Full system state snapshot for restoration."""
    timestamp: str = ""
    tiers: Dict[str, TierSnapshot] = field(default_factory=dict)
    study_was_running: bool = False

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "study_was_running": self.study_was_running,
        }


# ── HTTP Helpers ───────────────────────────────────────────────────────────

def _get(url: str, timeout: int = HTTP_TIMEOUT) -> Optional[dict]:
    """GET JSON from URL, return None on failure."""
    try:
        req = Request(url)
        resp = urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except Exception as e:
        logger.debug("GET %s failed: %s", url, e)
        return None


def _post(url: str, data: dict = None, timeout: int = HTTP_TIMEOUT) -> Optional[dict]:
    """POST JSON to URL, return response or None on failure."""
    try:
        body = json.dumps(data or {}).encode() if data else b"{}"
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except Exception as e:
        logger.debug("POST %s failed: %s", url, e)
        return None


# ── Lifecycle Manager ──────────────────────────────────────────────────────

class TrainingLifecycle:
    """
    Context manager for training sessions.

    Snapshots system state on enter, restores on exit.
    Handles exceptions gracefully — always attempts restoration.
    """

    def __init__(self, skip_restore: bool = False):
        self.snapshot: Optional[SystemSnapshot] = None
        self.skip_restore = skip_restore
        self._entered = False
        self._core_was_stopped = False
        self._nano_was_stopped = False

    def __enter__(self):
        self._entered = True
        self.snapshot = self._take_snapshot()
        self._log_snapshot("BEFORE")
        self._unload_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Training failed with %s: %s", exc_type.__name__, exc_val)

        if self.skip_restore:
            logger.info("Skipping restoration (--skip-restore)")
        else:
            self._restore(self.snapshot)
            self._verify_health()

        self._stop_study()
        return False  # Don't suppress exceptions

    # ── Snapshot ───────────────────────────────────────────────────────

    def _take_snapshot(self) -> SystemSnapshot:
        """Capture current state of all tiers."""
        logger.info("Taking system snapshot...")
        snap = SystemSnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Check each tier via its engine status endpoint
        for name, url in [("nano", NANO_URL), ("prime", PRIME_URL)]:
            tier = TierSnapshot(name=name, url=url)
            status = _get(f"{url}/status")
            if status and status.get("model"):
                tier.model_path = status["model"]
                tier.device = status.get("device", "cpu")
                tier.was_loaded = True
            elif status:
                # Engine running but no model
                tier.was_loaded = False
            snap.tiers[name] = tier

        # Core is embedded — try /status (proxied to worker), fall back to /model/info
        core_tier = TierSnapshot(name="core", url=CORE_URL)
        core_status = _get(f"{CORE_URL}/status")
        if core_status and core_status.get("model"):
            core_tier.model_path = core_status["model"]
            core_tier.device = core_status.get("device", "cpu")
            core_tier.was_loaded = True
        else:
            # Try model/info (manager endpoint, always available)
            core_info = _get(f"{CORE_URL}/model/info")
            if core_info and core_info.get("model_loaded"):
                core_tier.model_path = core_info.get("model_path", "")
                core_tier.device = core_info.get("device", "cpu")
                core_tier.was_loaded = True
            else:
                # Check orchestrator's view as last resort
                orch_state = _get(f"{ORCHESTRATOR_URL}/lifecycle/state")
                if orch_state:
                    core_orch = orch_state.get("tiers", {}).get("core", {})
                    if core_orch.get("model_loaded"):
                        core_tier.model_path = core_orch.get("model_path", "")
                        core_tier.device = core_orch.get("device", "cpu")
                        core_tier.was_loaded = True
        snap.tiers["core"] = core_tier

        # Check study container
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", "gaia-study"],
                capture_output=True, text=True, timeout=5
            )
            snap.study_was_running = result.stdout.strip() == "true"
        except Exception:
            snap.study_was_running = False

        return snap

    def _log_snapshot(self, label: str):
        """Log the snapshot in a readable format."""
        if not self.snapshot:
            return
        logger.info("── System State %s ──", label)
        for name, tier in self.snapshot.tiers.items():
            if tier.was_loaded:
                logger.info("  %s: %s on %s", name, tier.model_path or "(unknown)", tier.device)
            else:
                logger.info("  %s: unloaded", name)
        logger.info("  study: %s", "running" if self.snapshot.study_was_running else "stopped")

    # ── Teardown (pre-training) ────────────────────────────────────────

    def _unload_all(self):
        """Unload all models to free VRAM for training."""
        logger.info("Unloading all models for training...")

        # Unload all tiers including Core's managed engine
        for name, url in [("nano", NANO_URL), ("core", CORE_URL), ("prime", PRIME_URL)]:
            tier = self.snapshot.tiers.get(name)
            if tier and tier.was_loaded:
                result = _post(f"{url}/model/unload")
                if result and result.get("ok"):
                    logger.info("  %s unloaded", name)
                else:
                    logger.warning("  %s unload failed: %s", name, result)
            else:
                # Try anyway — Core may be loaded by orchestrator even if snapshot missed it
                result = _post(f"{url}/model/unload")
                if result and result.get("ok"):
                    logger.info("  %s unloaded (wasn't in snapshot)", name)

        # Release GPU lease so orchestrator doesn't auto-reload
        result = _post(f"{ORCHESTRATOR_URL}/gpu/release")
        if result and result.get("success"):
            logger.info("  GPU released via orchestrator")

        # Transition to MEDITATION to prevent orchestrator from reloading
        result = _post(f"{ORCHESTRATOR_URL}/consciousness/training", timeout=30)
        if result:
            logger.info("  Consciousness set to training mode")
        else:
            logger.warning("  Could not set training mode (orchestrator may reload models)")

        # Wait for VRAM to settle
        time.sleep(3)

        # Check if GPU is actually free — if not, stop containers as nuclear option
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            gpu_line = result.stdout.strip()
            logger.info("  GPU after soft unload: %s", gpu_line)

            # Parse VRAM: "13745 MiB, 2090 MiB" → used, free
            parts = gpu_line.replace(" MiB", "").split(",")
            if len(parts) == 2:
                used_mb = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                # If less than 10GB free, containers are still hogging GPU
                if free_mb < 10000:
                    logger.warning("  Only %dMB free — stopping gaia-core to force GPU release", free_mb)
                    subprocess.run(["docker", "stop", "gaia-core"], capture_output=True, timeout=30)
                    self._core_was_stopped = True
                    time.sleep(3)
                    # Also stop nano if still eating VRAM
                    subprocess.run(["docker", "stop", "gaia-nano"], capture_output=True, timeout=15)
                    self._nano_was_stopped = True
                    time.sleep(2)
                    # Re-check
                    result2 = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5
                    )
                    logger.info("  GPU after container stop: %s", result2.stdout.strip())
        except Exception as e:
            logger.warning("  GPU check failed: %s", e)

        logger.info("Models unloaded, ready for training")

    # ── Study container management ─────────────────────────────────────

    def start_study(self):
        """Start the gaia-study container for training."""
        logger.info("Starting gaia-study container...")
        try:
            subprocess.run(
                ["docker", "start", "gaia-study"],
                capture_output=True, text=True, timeout=30
            )
            # Wait for healthy
            for i in range(HEALTH_CHECK_RETRIES):
                time.sleep(HEALTH_CHECK_DELAY)
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Health.Status}}", "gaia-study"],
                    capture_output=True, text=True, timeout=5
                )
                status = result.stdout.strip()
                if status == "healthy":
                    logger.info("  gaia-study is healthy")
                    return True
                logger.debug("  gaia-study health: %s (attempt %d)", status, i + 1)
            logger.warning("  gaia-study didn't become healthy in time")
            return True  # Proceed anyway — some containers don't have health checks
        except Exception as e:
            logger.error("  Failed to start gaia-study: %s", e)
            return False

    def _stop_study(self):
        """Stop the gaia-study container if we started it."""
        if self.snapshot and not self.snapshot.study_was_running:
            logger.info("Stopping gaia-study (wasn't running before)...")
            try:
                subprocess.run(
                    ["docker", "stop", "gaia-study"],
                    capture_output=True, text=True, timeout=30
                )
                logger.info("  gaia-study stopped")
            except Exception as e:
                logger.warning("  Failed to stop gaia-study: %s", e)

    # ── Restoration (post-training) ────────────────────────────────────

    def _restore(self, snapshot: Optional[SystemSnapshot]):
        """Restore system to pre-training state."""
        if not snapshot:
            logger.warning("No snapshot to restore from")
            return

        logger.info("Restoring system state...")

        # Restart containers if we stopped them for VRAM
        if self._core_was_stopped:
            logger.info("  Restarting gaia-core (was stopped for training)...")
            subprocess.run(["docker", "start", "gaia-core"], capture_output=True, timeout=30)
            time.sleep(10)  # Wait for Core to initialize
        if self._nano_was_stopped:
            logger.info("  Restarting gaia-nano (was stopped for training)...")
            subprocess.run(["docker", "start", "gaia-nano"], capture_output=True, timeout=15)
            time.sleep(5)

        # Exit training mode — tell orchestrator to wake up
        result = _post(f"{ORCHESTRATOR_URL}/consciousness/awake", timeout=30)
        if result:
            logger.info("  Consciousness set to awake")

        # Restore each tier (including Core)
        for name in ["nano", "core", "prime"]:
            tier = snapshot.tiers.get(name)
            if not tier or not tier.was_loaded:
                logger.info("  %s: was unloaded, skipping", name)
                continue

            logger.info("  %s: restoring %s on %s...", name, tier.model_path, tier.device)

            # Map device names for the engine
            device = tier.device
            if device in ("gpu_safetensors", "gpu", "cuda"):
                device = "gpu"
            elif device in ("cpu_gguf", "cpu"):
                device = "cpu"

            result = _post(f"{tier.url}/model/load", {
                "model_path": tier.model_path,
                "device": device,
            }, timeout=MODEL_LOAD_TIMEOUT)

            if result and result.get("ok"):
                logger.info("  %s: restored", name)
            else:
                # Try swap in case something is already loaded
                result = _post(f"{tier.url}/model/swap", {
                    "model_path": tier.model_path,
                    "device": device,
                }, timeout=MODEL_LOAD_TIMEOUT)
                if result and result.get("ok"):
                    logger.info("  %s: restored via swap", name)
                else:
                    logger.error("  %s: FAILED to restore — %s", name, result)

        # Poke orchestrator to refresh its view
        _post(f"{ORCHESTRATOR_URL}/consciousness/probe", timeout=30)
        logger.info("Restoration complete")

    # ── Verification ───────────────────────────────────────────────────

    def _verify_health(self):
        """Verify all services are healthy after restoration."""
        logger.info("Verifying system health...")
        all_ok = True

        # Check orchestrator
        status = _get(f"{ORCHESTRATOR_URL}/status")
        if status:
            logger.info("  orchestrator: %s", status.get("gpu_state", "unknown"))
        else:
            logger.warning("  orchestrator: not responding")
            all_ok = False

        # Check each tier
        for name, url in [("nano", NANO_URL), ("prime", PRIME_URL)]:
            tier = self.snapshot.tiers.get(name) if self.snapshot else None
            if tier and tier.was_loaded:
                status = _get(f"{url}/status")
                if status and status.get("model"):
                    logger.info("  %s: loaded (%s)", name, status.get("device", "?"))
                else:
                    logger.warning("  %s: NOT loaded after restore", name)
                    all_ok = False

        # Check core health
        health = _get("http://localhost:6415/health")
        if health and health.get("status") == "healthy":
            logger.info("  core: healthy")
        else:
            logger.warning("  core: unhealthy or not responding")
            all_ok = False

        if all_ok:
            logger.info("All systems nominal")
        else:
            logger.warning("Some systems need attention — check dashboard")

        return all_ok


# ── CLI Wrapper ────────────────────────────────────────────────────────────

def main():
    """
    Wrap any training command with proper lifecycle management.

    Usage:
        python training_lifecycle.py -- docker exec gaia-study python /path/to/train.py
        python training_lifecycle.py --skip-restore -- python train.py
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Training lifecycle wrapper — snapshot, train, restore",
        usage="%(prog)s [options] -- COMMAND [ARGS...]"
    )
    parser.add_argument("--skip-restore", action="store_true",
                        help="Don't restore system state after training")
    parser.add_argument("--no-study", action="store_true",
                        help="Don't start/stop gaia-study container")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Training command to run (after --)")

    args = parser.parse_args()

    # Strip leading "--" from command
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        parser.print_help()
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training Lifecycle Manager")
    logger.info("  Command: %s", " ".join(cmd))
    logger.info("=" * 60)

    exit_code = 1
    with TrainingLifecycle(skip_restore=args.skip_restore) as lc:
        if not args.no_study:
            lc.start_study()

        logger.info("Running training command...")
        logger.info("─" * 40)

        try:
            result = subprocess.run(cmd, timeout=1800)  # 30 min max
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            logger.error("Training command timed out after 30 minutes")
            exit_code = 124
        except Exception as e:
            logger.error("Training command failed: %s", e)
            exit_code = 1

        logger.info("─" * 40)
        logger.info("Training command exited with code %d", exit_code)

    logger.info("=" * 60)
    logger.info("Lifecycle complete (exit %d)", exit_code)
    logger.info("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
