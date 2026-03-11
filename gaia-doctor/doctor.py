#!/usr/bin/env python3
"""
gaia-doctor — Persistent HA watchdog service.

Monitors GAIA service health and automatically restarts crashed or
misconfigured HA candidates via docker compose with the HA overlay.

Zero external dependencies — stdlib only.
"""

import ast
import difflib
import hashlib
import json
import logging
import os
import random
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen, Request

# Cognitive test battery (stdlib only — lives alongside doctor.py)
try:
    from cognitive_test_battery import run_battery as _run_cognitive_battery, get_test_metadata as _get_test_metadata
    _BATTERY_AVAILABLE = True
except ImportError:
    _BATTERY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "60"))
FAILURE_THRESHOLD = int(os.environ.get("FAILURE_THRESHOLD", "2"))
RESTART_COOLDOWN = int(os.environ.get("RESTART_COOLDOWN", "300"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "6419"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
MAINTENANCE_FLAG = Path(os.environ.get("SHARED_DIR", "/shared")) / "ha_maintenance"
STATUS_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "status.json"
ALARMS_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "alarms.json"
COMPOSE_DIR = os.environ.get("COMPOSE_DIR", "/compose")
COMPOSE_PROJECT = os.environ.get("COMPOSE_PROJECT_NAME", "gaia_project")

# Circuit breaker for production restarts: max N restarts within a rolling window
PROD_RESTART_MAX = int(os.environ.get("PROD_RESTART_MAX", "2"))
PROD_RESTART_WINDOW = int(os.environ.get("PROD_RESTART_WINDOW", "1800"))  # 30 minutes

# Atomic file hashing — vital organs and service coverage
VITAL_ORGANS = [
    "gaia-core/gaia_core/main.py",
    "gaia-core/gaia_core/cognition/agent_core.py",
    "gaia-core/gaia_core/utils/prompt_builder.py",
    "gaia-core/gaia_core/model_server.py",
    "gaia-web/gaia_web/main.py",
    "gaia-web/gaia_web/discord_interface.py",
    "gaia-mcp/gaia_mcp/tools.py",
    "gaia-common/gaia_common/utils/immune_system.py",
    "gaia-common/gaia_common/protocols/cognition_packet.py",
    "gaia-orchestrator/gaia_orchestrator/main.py",
    "gaia-study/gaia_study/qlora_trainer.py",
    "gaia-study/gaia_study/merge_and_requantize.py",
]

HASHED_SERVICES = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-common", "gaia-study", "gaia-orchestrator"]
HASH_REGISTRY_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "file_hashes.json"

MONKEY_ENDPOINT = os.environ.get("MONKEY_ENDPOINT", "http://gaia-monkey:6420")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [gaia-doctor] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("gaia-doctor")

# ---------------------------------------------------------------------------
# Service registry
# ---------------------------------------------------------------------------

SERVICES = {
    # name: (health_url, remediation)
    # remediation: None = observe only, "restart" = docker restart, "ha" = compose HA overlay
    "gaia-core": ("http://gaia-core:6415/health", "restart"),
    "gaia-web": ("http://gaia-web:6414/health", "restart"),
    "gaia-mcp": ("http://gaia-mcp:8765/health", "restart"),
    "gaia-prime": ("http://gaia-prime:7777/health", None),
    "gaia-nano": ("http://gaia-nano:8080/health", None),
    "gaia-audio": ("http://gaia-audio:8080/health", None),
    "gaia-study": ("http://gaia-study:8766/health", None),
    "gaia-orchestrator": ("http://gaia-orchestrator:6410/health", None),
    "gaia-monkey": ("http://gaia-monkey:6420/health", None),
    "gaia-wiki": ("http://gaia-wiki:8080", None),
    "gaia-core-candidate": ("http://gaia-core-candidate:6415/health", "ha"),
    "gaia-mcp-candidate": ("http://gaia-mcp-candidate:8765/health", "ha"),
}

# Orchestrator endpoint for GPU status enrichment
ORCHESTRATOR_ENDPOINT = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")

# Pipeline state file for training pipeline monitoring
PIPELINE_STATE_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "pipeline" / "self_awareness_state.json"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_start_time = time.monotonic()
_service_state: dict[str, dict] = {}
_consecutive_failures: dict[str, int] = {}
_last_restart: dict[str, float] = {}
_restart_history: dict[str, list] = {}   # timestamps of recent restarts per service
_alarmed_services: set = set()           # services currently in alarm state
_remediation_log: list[dict] = []
_active_alarms: list[dict] = []
_irritations: list[dict] = []            # detected log errors/irritations
_last_log_offsets: dict[str, int] = {}   # service -> last read byte offset
_code_mtimes: dict[str, float] = {}      # service -> last seen mtime of its code dir
_dissonance_report: dict = {}            # module-level divergence detection
_hash_registry: dict = {}               # file -> {hash, mtime} for atomic hashing
_cognitive_running: bool = False         # True while cognitive battery is executing
_cognitive_last_result: dict | None = None  # cached last battery run result
# ---------------------------------------------------------------------------
# gaia-monkey integration — delegate chaos/serenity/meditation to monkey service
# ---------------------------------------------------------------------------

def _notify_monkey_break_serenity(reason: str):
    """Called when vital organ fails — notify monkey to break serenity."""
    try:
        req = Request(
            f"{MONKEY_ENDPOINT}/serenity/break",
            data=json.dumps({"reason": reason}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=3)
    except Exception:
        pass  # non-blocking; monkey may be down


def _is_meditation_active() -> bool:
    """Check if Defensive Meditation is active by reading the shared flag file."""
    try:
        flag_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "defensive_meditation.json"
        if flag_path.exists():
            data = json.loads(flag_path.read_text())
            return bool(data.get("active", False))
    except Exception:
        pass
    return False


def _get_serenity_report() -> dict:
    """Read serenity state from shared file (written by gaia-monkey)."""
    try:
        serenity_file = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "serenity.json"
        if serenity_file.exists():
            return json.loads(serenity_file.read_text())
    except Exception:
        pass
    return {"serene": False, "score": 0.0, "threshold": 5.0}


# ---------------------------------------------------------------------------
# Atomic File Hash Registry
# ---------------------------------------------------------------------------

def _load_hash_registry() -> dict:
    """Load persisted hash registry from shared volume."""
    global _hash_registry
    try:
        if HASH_REGISTRY_PATH.exists():
            _hash_registry = json.loads(HASH_REGISTRY_PATH.read_text())
    except Exception:
        log.debug("Failed to load hash registry", exc_info=True)
        _hash_registry = {}
    return _hash_registry


def _save_hash_registry(registry: dict):
    """Atomic write of hash registry via .tmp rename."""
    try:
        HASH_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = HASH_REGISTRY_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(registry, indent=2))
        tmp_path.rename(HASH_REGISTRY_PATH)
    except Exception:
        log.debug("Failed to save hash registry", exc_info=True)


def _init_state():
    for name in SERVICES:
        _service_state[name] = {"healthy": None, "last_check": None}
        _consecutive_failures[name] = 0
        _restart_history[name] = []
        _last_log_offsets[name] = 0
        _code_mtimes[name] = _get_service_mtime(name)


# ---------------------------------------------------------------------------
# Code Audit (Test-before-Restart)
# ---------------------------------------------------------------------------

GAIA_PROJECT_ROOT = Path("/gaia/GAIA_Project")

SERVICE_CODE_DIRS = {
    "gaia-core": GAIA_PROJECT_ROOT / "gaia-core",
    "gaia-web": GAIA_PROJECT_ROOT / "gaia-web",
    "gaia-mcp": GAIA_PROJECT_ROOT / "gaia-mcp",
    "gaia-study": GAIA_PROJECT_ROOT / "gaia-study",
    "gaia-orchestrator": GAIA_PROJECT_ROOT / "gaia-orchestrator",

    # Candidates
    "gaia-core-candidate": GAIA_PROJECT_ROOT / "candidates" / "gaia-core",
    "gaia-mcp-candidate": GAIA_PROJECT_ROOT / "candidates" / "gaia-mcp",
}


def get_dissonance_report() -> dict:
    """Atomic file-level hashing across all .py files in HASHED_SERVICES.

    Compares live vs candidate directories with mtime-based caching.
    Classifies divergent files as vital (high-severity) or standard (informational).
    Persists hash registry to JSON for cross-cycle consistency.
    """
    global _hash_registry
    if not _hash_registry:
        _load_hash_registry()

    vital_set = set(VITAL_ORGANS)
    vital_divergent = []
    standard_divergent = []
    total_files = 0
    matches = 0
    skip_dirs = {"__pycache__", ".pytest_cache", "venv", ".venv", "node_modules"}

    def _hash_file(path: Path) -> str:
        if not path.exists():
            return "MISSING"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _cached_hash(path: Path) -> str:
        """Return hash, using mtime cache to avoid redundant reads."""
        key = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return "MISSING"
        cached = _hash_registry.get(key)
        if cached and cached.get("mtime") == mtime:
            return cached["hash"]
        h = _hash_file(path)
        _hash_registry[key] = {"hash": h, "mtime": mtime}
        return h

    for svc in HASHED_SERVICES:
        live_dir = GAIA_PROJECT_ROOT / svc
        cand_dir = GAIA_PROJECT_ROOT / "candidates" / svc
        if not live_dir.exists():
            continue

        for py_file in live_dir.rglob("*.py"):
            # Skip excluded directories
            if any(part in skip_dirs for part in py_file.parts):
                continue

            rel = str(py_file.relative_to(GAIA_PROJECT_ROOT))
            cand_path = cand_dir / py_file.relative_to(live_dir)
            total_files += 1

            h_live = _cached_hash(py_file)
            h_cand = _cached_hash(cand_path)

            if h_live == h_cand and h_live != "MISSING":
                matches += 1
            else:
                entry = {
                    "file": rel,
                    "live_hash": h_live[:12],
                    "cand_hash": h_cand[:12],
                    "status": "MISSING" if "MISSING" in (h_live, h_cand) else "DIVERGENT",
                }
                if rel in vital_set:
                    vital_divergent.append(entry)
                else:
                    standard_divergent.append(entry)

    _save_hash_registry(_hash_registry)

    parity = (matches / total_files * 100.0) if total_files > 0 else 100.0
    return {
        "vital_divergent": vital_divergent,
        "standard_divergent": standard_divergent,
        "parity_percent": round(parity, 1),
        "total_files": total_files,
        # Backwards compat — old callers used "divergent_files"
        "divergent_files": vital_divergent + standard_divergent,
    }


def _get_service_mtime(name: str) -> float:
    """Get the maximum mtime of all .py files in a service directory."""
    code_dir = SERVICE_CODE_DIRS.get(name)
    if not code_dir or not code_dir.exists():
        return 0.0
    
    max_mtime = 0.0
    try:
        for p in code_dir.rglob("*.py"):
            mtime = p.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
    except Exception:
        pass
    return max_mtime


def audit_code():
    """Check for code changes and restart if tests pass."""
    for name in SERVICE_CODE_DIRS:
        # Only audit services we can remediate
        if SERVICES.get(name) and SERVICES[name][1] is None:
            continue

        current_mtime = _get_service_mtime(name)
        last_mtime = _code_mtimes.get(name, 0.0)

        # Skip if we just initialized and saw the mtime for the first time
        if last_mtime == 0.0:
            _code_mtimes[name] = current_mtime
            continue

        if current_mtime > last_mtime:
            log.info("CODE CHANGE detected for %s. Auditing...", name)
            _code_mtimes[name] = current_mtime
            
            if run_service_tests(name):
                log.info("Tests PASSED for %s. Triggering auto-restart.", name)
                remediation = SERVICES[name][1]
                if remediation == "restart":
                    docker_restart(name)
                elif remediation == "ha":
                    restart_candidate(name)
            else:
                log.warning("Tests FAILED for %s code changes. Auto-restart ABORTED.", name)
                _record_irritation(name, "Code changes detected but tests failed", "CodeAudit: Tests Failed")


def run_service_tests(name: str) -> bool:
    """Run ruff and pytest inside the container to validate code changes."""
    # 1. Fast Lint Check (Fatal errors only: F821 Undefined Name, F811 Redefined)
    log.info("Running fast lint audit for %s...", name)
    try:
        lint_cmd = ["docker", "exec", name, "python", "-m", "ruff", "check", "/app", "--select", "F821,F811", "--no-cache"]
        lint_res = subprocess.run(lint_cmd, capture_output=True, text=True, timeout=30)
        if lint_res.returncode != 0:
            log.error("FATAL LINT ERROR in %s:\n%s", name, lint_res.stdout)
            _record_irritation(name, f"Fatal lint error: {lint_res.stdout[:100]}", "CodeAudit: Lint Fatal")
            return False
    except Exception as e:
        log.warning("Fast lint audit failed for %s: %s", name, e)

    # 2. Standard Unit Tests
    log.info("Running pytest for %s before restart...", name)
    try:
        # Standard GAIA test command: python -m pytest <path> -v --tb=short
        # We run it against the /app directory inside the container
        cmd = ["docker", "exec", name, "python", "-m", "pytest", "/app", "-v", "--tb=short", "-m", "not integration"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            return True
        else:
            log.warning("Tests failed for %s:\n%s", name, result.stdout + result.stderr)
            return False
    except subprocess.TimeoutExpired:
        log.error("Testing %s timed out (>180s)", name)
        return False
    except Exception as e:
        log.error("Failed to run tests for %s: %s", name, e)
        return False


# ---------------------------------------------------------------------------
# Irritation Monitoring (Log Scanning)
# ---------------------------------------------------------------------------

IRRITATION_PATTERNS = [
    "PermissionError",
    "TimeoutError",
    "httpx.ReadTimeout",
    "httpx.ConnectTimeout",
    "HTTPStatusError",
    "Sovereign Shield: Cannot save",
    "BLAST SHIELD blocked",
    "Circuit breaker triggered",
    "HEALING_REQUIRED",
    "llama-server exited",
    "GPU acquire failed",
    "MERGE_4B failed",
    "TRAIN_4B failed",
    "GGUF conversion failed",
    "Pipeline halted",
    "model/release failed",
    "model/reload failed",
    "Warm pool sync failed",
    "OOM",
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
]

SERVICE_LOGS = {
    "gaia-core": "/logs/gaia-core.log",
    "gaia-web": "/logs/gaia-web.log",
    "gaia-mcp": "/logs/gaia-mcp.log",
    "gaia-study": "/logs/gaia-study.log",
}


def scan_logs():
    """Scan service logs for irritation patterns."""
    for service, log_path in SERVICE_LOGS.items():
        p = Path(log_path)
        if not p.exists():
            continue

        try:
            file_size = p.stat().st_size
            last_offset = _last_log_offsets.get(service, 0)

            # If file was rotated or truncated, reset offset
            if file_size < last_offset:
                last_offset = 0

            if file_size > last_offset:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(last_offset)
                    # Don't read more than 1MB at once to avoid memory issues
                    chunk = f.read(1024 * 1024)
                    _last_log_offsets[service] = f.tell()

                    for line in chunk.splitlines():
                        for pattern in IRRITATION_PATTERNS:
                            if pattern in line:
                                _record_irritation(service, line, pattern)

        except Exception as e:
            log.debug(f"Failed to scan log {log_path}: {e}")


def _record_irritation(service: str, line: str, pattern: str):
    """Record a detected irritation in the state."""
    # Avoid duplicate recordings of the same line if multiple patterns match
    entry = {
        "service": service,
        "time": datetime.now(timezone.utc).isoformat(),
        "pattern": pattern,
        "message": line.strip()[:500],
    }
    _irritations.append(entry)
    
    # Keep only the last 100 irritations
    if len(_irritations) > 100:
        _irritations.pop(0)
    
    log.warning("IRRITATION detected in %s: %s", service, pattern)


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------

def check_health(name: str, url: str) -> bool:
    """HTTP GET the health endpoint. Returns True if 200."""
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def inspect_container(name: str) -> dict | None:
    """Use docker CLI to inspect a container. Returns parsed JSON or None."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format",
             '{"status":"{{.State.Status}}","restart":"{{.HostConfig.RestartPolicy.Name}}","exit_code":{{.State.ExitCode}}}',
             name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# Remediation
# ---------------------------------------------------------------------------

def restart_candidate(name: str) -> bool:
    """Restart an HA candidate via docker compose with the HA overlay."""
    now = time.monotonic()
    last = _last_restart.get(name, 0)
    if now - last < RESTART_COOLDOWN:
        remaining = int(RESTART_COOLDOWN - (now - last))
        log.info("Cooldown active for %s (%ds remaining), skipping restart", name, remaining)
        return False

    if MAINTENANCE_FLAG.exists():
        log.info("Maintenance mode active, skipping restart of %s", name)
        return False

    # 1. Structural Integrity Check (The "Quarantine" Gate)
    if not run_structural_audit(name):
        return False

    log.warning("REMEDIATION: Restarting %s via HA compose overlay", name)
    try:
        # Override network for candidate stack if needed
        env = os.environ.copy()
        env["GAIA_NETWORK"] = "gaia-network"
        
        result = subprocess.run(
            ["docker", "compose",
             "-p", COMPOSE_PROJECT,
             "-f", f"{COMPOSE_DIR}/docker-compose.candidate.yml",
             "-f", f"{COMPOSE_DIR}/docker-compose.ha.yml",
             "--profile", "ha",
             "up", "-d", name],
            capture_output=True, text=True, timeout=120,
            env=env
        )
        _last_restart[name] = time.monotonic()
        entry = {
            "service": name,
            "time": datetime.now(timezone.utc).isoformat(),
            "success": result.returncode == 0,
            "output": (result.stdout + result.stderr).strip()[:500],
        }
        _remediation_log.append(entry)
        if len(_remediation_log) > 50:
            _remediation_log.pop(0)

        if result.returncode == 0:
            log.info("Successfully restarted %s", name)
            url = SERVICES[name][0]
            _verify_recovery(name, url)
            return True
        else:
            log.error("Failed to restart %s: %s", name, result.stderr.strip()[:200])
            return False
    except subprocess.TimeoutExpired:
        log.error("Restart of %s timed out (>120s)", name)
        _last_restart[name] = time.monotonic()
        return False


def raise_alarm(name: str, reason: str):
    """Record an alarm for a service that has exceeded restart limits."""
    _alarmed_services.add(name)
    entry = {
        "service": name,
        "time": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    _active_alarms.append(entry)
    if len(_active_alarms) > 50:
        _active_alarms.pop(0)

    log.error("[ALARM] %s: %s — manual intervention required", name, reason)

    try:
        ALARMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ALARMS_FILE.write_text(json.dumps(_active_alarms, indent=2))
    except Exception:
        log.debug("Failed to write alarms file", exc_info=True)


def run_structural_audit(name: str) -> bool:
    """Perform pre-restart structural validation and autonomous repair."""
    code_dir = SERVICE_CODE_DIRS.get(name)
    if not code_dir or not code_dir.exists():
        return True

    log.info("Performing structural audit for %s...", name)
    
    try:
        # Use ast.parse for a guaranteed read-only check
        # Iterative check to find the specific failing file path
        audit_script = f"""
import ast
from pathlib import Path
import sys
broken = False
for p in Path('{code_dir}').rglob('*.py'):
    try:
        ast.parse(p.read_text())
    except Exception as e:
        print(f"FILE:{{p}}")
        print(e)
        broken = True
        break
if broken: sys.exit(1)
"""
        audit_res = subprocess.run(
            ["python3", "-c", audit_script],
            capture_output=True, text=True, timeout=60
        )
        
        if audit_res.returncode != 0:
            log.warning("Structural error detected in %s. Attempting repair...", name)
            
            # Extract broken file path from stdout
            broken_file = None
            for line in audit_res.stdout.splitlines():
                if line.startswith("FILE:"):
                    broken_file = Path(line.replace("FILE:", "").strip())
                    break
            
            if broken_file:
                log.info("Targeting broken file for repair: %s", broken_file)
                
                # Tier 1 Repair: Ruff --fix (if available)
                if subprocess.run(["which", "ruff"], capture_output=True).returncode == 0:
                    subprocess.run(["ruff", "check", "--fix", str(broken_file)], capture_output=True)
                
                # Re-audit
                audit_res = subprocess.run(
                    ["python3", "-c", f"import ast; ast.parse(open('{str(broken_file)}').read())"],
                    capture_output=True, text=True, timeout=10
                )
                
                if audit_res.returncode == 0:
                    log.info("✅ Tier 1 repair successful for %s", name)
                    return True
                else:
                    # Tier 2 Repair: High-Availability Surgery
                    # gaia-core has rw access to the project root; this container's
                    # project mount is ro.  Send file_path so gaia-core validates
                    # and writes the fix itself.
                    log.warning("Tier 1 repair failed for %s. Escalating to Tier 2 (HA Surgery)...", name)
                    try:
                        broken_content = broken_file.read_text()

                        repair_url = "http://gaia-core:6415/api/repair/structural"
                        repair_data = json.dumps({
                            "service": name,
                            "broken_code": broken_content,
                            "error_msg": audit_res.stdout + audit_res.stderr,
                            "file_path": str(broken_file),
                        }).encode("utf-8")

                        req = Request(
                            repair_url,
                            data=repair_data,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )

                        with urlopen(req, timeout=120) as response:
                            if response.status == 200:
                                res_body = json.loads(response.read().decode("utf-8"))
                                if res_body.get("status") == "repaired":
                                    log.info("✅ Tier 2 repair (HA Surgery) successful for %s", name)
                                    return True
                                log.error("❌ HA Surgery did not confirm write for %s: %s", name, res_body)
                                return False
                            else:
                                log.error("HA surgery API failed: %d", response.status)
                                return False
                    except Exception as e:
                        log.error("Exception during Tier 2 surgery: %s", e)
                        return False
            
            log.critical("⛔ QUARANTINE: %s has fatal syntax errors.", name)
            return False
            
        return True
    except Exception as e:
        log.error("Error during structural audit for %s: %s", name, e)
        return True

def _verify_recovery(name: str, url: str, delay: int = 5):
    """Post-remediation health check — confirms recovery immediately instead of waiting for next poll."""
    time.sleep(delay)
    healthy = check_health(name, url)
    if healthy:
        _consecutive_failures[name] = 0
        _service_state[name]["healthy"] = True
        _alarmed_services.discard(name)
        log.info("POST-REMEDIATION: %s confirmed healthy", name)
        # Recovery tracked by gaia-monkey via serenity shared file
    else:
        log.warning("POST-REMEDIATION: %s still unhealthy after %ds — next poll will retry", name, delay)


def docker_restart(name: str) -> bool:
    """Restart a production service via `docker restart`. Enforces structural audit, reload guard, and circuit breaker."""
    if MAINTENANCE_FLAG.exists():
        log.info("Maintenance mode active, skipping restart of %s", name)
        return False

    now = time.monotonic()

    # 1. Reload Guard: Detect high-frequency restart loops
    _restart_history[name] = [t for t in _restart_history[name] if now - t < PROD_RESTART_WINDOW]
    if len(_restart_history[name]) >= PROD_RESTART_MAX and not _is_meditation_active():
        log.critical("🚨 RELOAD LOOP DETECTED for %s. Quarantine active.", name)
    elif len(_restart_history[name]) >= PROD_RESTART_MAX and _is_meditation_active():
        log.info("🧘 Defensive Meditation active — restart limit bypassed for %s", name)
    if len(_restart_history[name]) >= PROD_RESTART_MAX and not _is_meditation_active():
        if name not in _alarmed_services:
            raise_alarm(
                name,
                f"Recursive restart loop detected: {len(_restart_history[name])} restarts in "
                f"{PROD_RESTART_WINDOW // 60}min. Manual intervention required."
            )
            
            # Autonomous Diagnostics
            try:
                log.info("Dispatching autonomous diagnostics for %s...", name)
                # Extract last 100 lines of logs
                log_res = subprocess.run(["docker", "logs", "--tail", "100", name], capture_output=True, text=True)
                logs = log_res.stdout + log_res.stderr
                
                diag_data = json.dumps({"service": name, "logs": logs}).encode("utf-8")
                diag_req = Request(
                    "http://gaia-core:6415/api/doctor/diagnose",
                    data=diag_data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urlopen(diag_req, timeout=10)
            except Exception as diag_err:
                log.error("Failed to dispatch diagnostics: %s", diag_err)
                
        return False

    # 2. Structural Integrity Check (The "Quarantine" Gate)
    if not run_structural_audit(name):
        return False

    # Cooldown between individual restarts (reuse RESTART_COOLDOWN)

    now = time.monotonic()

    # Trim restart history to the rolling window
    _restart_history[name] = [t for t in _restart_history[name] if now - t < PROD_RESTART_WINDOW]

    if len(_restart_history[name]) >= PROD_RESTART_MAX and not _is_meditation_active():
        if name not in _alarmed_services:
            raise_alarm(
                name,
                f"restarted {len(_restart_history[name])} times in the last "
                f"{PROD_RESTART_WINDOW // 60}min — circuit breaker tripped",
            )
        return False

    # Cooldown between individual restarts (reuse RESTART_COOLDOWN)
    last = _last_restart.get(name, 0)
    if now - last < RESTART_COOLDOWN:
        remaining = int(RESTART_COOLDOWN - (now - last))
        log.info("Cooldown active for %s (%ds remaining), skipping restart", name, remaining)
        return False

    log.warning("REMEDIATION: docker compose recreate %s (attempt %d/%d in window)",
                name, len(_restart_history[name]) + 1, PROD_RESTART_MAX)
    try:
        # Use compose up --force-recreate to ensure correct container naming.
        # Plain `docker restart` preserves mangled names (e.g. "2a85f751fcd3_gaia-web").
        # Compose service name = container_name (without project prefix) in our setup.
        project_root = str(GAIA_PROJECT_ROOT)
        result = subprocess.run(
            ["docker", "compose",
             "-p", COMPOSE_PROJECT,
             "-f", f"{project_root}/docker-compose.yml",
             "-f", f"{project_root}/docker-compose.override.yml",
             "up", "-d", "--force-recreate", name],
            capture_output=True, text=True, timeout=120,
        )

        ts = time.monotonic()
        _last_restart[name] = ts
        _restart_history[name].append(ts)

        entry = {
            "service": name,
            "time": datetime.now(timezone.utc).isoformat(),
            "mode": "docker_restart",
            "success": result.returncode == 0,
            "output": (result.stdout + result.stderr).strip()[:500],
        }
        _remediation_log.append(entry)
        if len(_remediation_log) > 50:
            _remediation_log.pop(0)

        if result.returncode == 0:
            log.info("Successfully restarted %s", name)
            _alarmed_services.discard(name)
            url = SERVICES[name][0]
            _verify_recovery(name, url)
            return True
        else:
            log.error("Failed to restart %s: %s", name, result.stderr.strip()[:200])
            return False
    except subprocess.TimeoutExpired:
        log.error("Restart of %s timed out (>60s)", name)
        _last_restart[name] = time.monotonic()
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def sovereign_promote(divergent_files: list) -> bool:
    """Doctor-mediated sovereign promotion with GAIA cognitive review.

    Flow: validate syntax → restart candidate → health check → generate diffs →
    POST to gaia-core for GAIA review → if approved, use gaia-core's structural
    repair endpoint to write files (doctor is read-only) → restart production →
    final health verification.
    """
    log.info("🔱 SOVEREIGN PROMOTE: Evaluating %d divergent files", len(divergent_files))

    # Step 1: Validate candidate syntax (ast.parse on disk)
    for f in divergent_files:
        cand_path = GAIA_PROJECT_ROOT / "candidates" / f["file"]
        if not cand_path.exists():
            log.warning("Sovereign promote: candidate file missing — %s", f["file"])
            continue
        try:
            ast.parse(cand_path.read_text())
        except SyntaxError as e:
            log.error("🔱 Sovereign promote ABORTED: syntax error in candidate %s — %s", f["file"], e)
            return False

    # Step 2: Generate diffs (±40 lines context)
    diffs = []
    for f in divergent_files:
        live_path = GAIA_PROJECT_ROOT / f["file"]
        cand_path = GAIA_PROJECT_ROOT / "candidates" / f["file"]
        try:
            live_lines = live_path.read_text().splitlines(keepends=True) if live_path.exists() else []
            cand_lines = cand_path.read_text().splitlines(keepends=True) if cand_path.exists() else []
            diff = list(difflib.unified_diff(
                live_lines, cand_lines,
                fromfile=f"live/{f['file']}",
                tofile=f"candidate/{f['file']}",
                n=40,  # 40 lines of context
            ))
            if diff:
                diffs.append({
                    "file": f["file"],
                    "diff": "".join(diff)[:8000],  # Cap at 8K per file
                    "vital": f["file"] in set(VITAL_ORGANS),
                })
        except Exception as e:
            log.warning("Failed to generate diff for %s: %s", f["file"], e)

    if not diffs:
        log.info("🔱 Sovereign promote: no meaningful diffs — skipping")
        return False

    # Step 3: Cognitive review — auto-approve if Serene, otherwise GAIA reviews
    if _get_serenity_report().get("serene", False):
        log.info("🔱🪷 SERENITY AUTO-APPROVE: Skipping cognitive review — GAIA has proven resilience")
        result = {"approved": True, "reason": "Auto-approved: system is Serene"}
    else:
        log.info("🔱 Submitting %d diffs for GAIA cognitive review...", len(diffs))
        try:
            review_payload = json.dumps({
                "diffs": diffs,
                "source": "doctor_sovereign_promote",
                "file_count": len(diffs),
            }).encode("utf-8")

            req = Request(
                "http://gaia-core:6415/api/doctor/review",
                data=review_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=120) as resp:
                if resp.status != 200:
                    log.error("🔱 Cognitive review returned status %d", resp.status)
                    return False
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            log.error("🔱 Cognitive review request failed: %s", e)
        return False

    if not result.get("approved"):
        log.warning("🔱 GAIA VETOED promotion: %s", result.get("reason", "no reason given"))
        return False

    log.info("🔱 GAIA APPROVED promotion — writing %d files via structural repair endpoint", len(diffs))

    # Step 4: Write files via gaia-core's structural repair endpoint (doctor is ro)
    for d in diffs:
        cand_path = GAIA_PROJECT_ROOT / "candidates" / d["file"]
        live_path = GAIA_PROJECT_ROOT / d["file"]
        try:
            cand_content = cand_path.read_text()
            write_payload = json.dumps({
                "service": "sovereign_promote",
                "broken_code": cand_content,  # Not broken — reusing endpoint for writes
                "error_msg": "sovereign_promote: candidate → production copy",
                "file_path": str(live_path),
            }).encode("utf-8")

            req = Request(
                "http://gaia-core:6415/api/repair/structural",
                data=write_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    log.error("🔱 File write failed for %s (status %d)", d["file"], resp.status)
                    return False
                res_body = json.loads(resp.read().decode("utf-8"))
                if res_body.get("status") != "repaired":
                    log.error("🔱 File write not confirmed for %s: %s", d["file"], res_body)
                    return False
        except Exception as e:
            log.error("🔱 Exception writing %s: %s", d["file"], e)
            return False

    # Step 5: Restart production services that had files changed
    affected_services = set()
    for d in diffs:
        svc = d["file"].split("/")[0]  # e.g. "gaia-core" from "gaia-core/gaia_core/main.py"
        if svc in SERVICES:
            affected_services.add(svc)

    for svc in affected_services:
        log.info("🔱 Restarting %s after sovereign promotion...", svc)
        docker_restart(svc)

    # Step 6: Final health verification
    time.sleep(8)
    all_healthy = True
    for svc in affected_services:
        url = SERVICES.get(svc, (None,))[0]
        if url and not check_health(svc, url):
            log.error("🔱 POST-PROMOTION: %s is unhealthy!", svc)
            all_healthy = False

    if all_healthy:
        log.info("🔱 SOVEREIGN PROMOTION COMPLETE — all affected services healthy")
    else:
        log.warning("🔱 SOVEREIGN PROMOTION COMPLETE with warnings — some services unhealthy")

    return all_healthy


def poll_cycle():
    """Run one health check cycle across all services."""
    global _gpu_status_cache, _model_server_cache, _pipeline_status_cache

    # Check for cognitive dissonance (drift between PROD and CAND)
    try:
        from gaia_common.utils.immune_system import ImmuneSystem
        imm = ImmuneSystem("/logs")
        global _dissonance_report
        _dissonance_report = imm.get_dissonance_report()
    except Exception:
        log.debug("Failed to generate dissonance report", exc_info=True)

    # Enrich status with GPU, model server, and pipeline state
    _gpu_status_cache = _fetch_gpu_status()
    _model_server_cache = _fetch_model_server_status()
    _pipeline_status_cache = _fetch_pipeline_status()

    # First scan logs for irritations
    scan_logs()

    # Audit code for disk/memory mismatches
    audit_code()

    # Then check HTTP health
    for name, (url, remediation) in SERVICES.items():
        healthy = check_health(name, url)
        _service_state[name]["last_check"] = datetime.now(timezone.utc).isoformat()

        if healthy:
            _consecutive_failures[name] = 0
            if _service_state[name]["healthy"] is False:
                log.info("%s recovered", name)
                _alarmed_services.discard(name)
                # Recovery tracked by gaia-monkey
            elif name in _alarmed_services:
                # Clear stale alarm if the restart window has expired naturally
                now_t = time.monotonic()
                in_window = [t for t in _restart_history.get(name, []) if now_t - t < PROD_RESTART_WINDOW]
                if not in_window:
                    log.info("%s alarm cleared — restart window expired", name)
                    _alarmed_services.discard(name)
            _service_state[name]["healthy"] = True
        else:
            _consecutive_failures[name] += 1
            failures = _consecutive_failures[name]

            if failures >= FAILURE_THRESHOLD:
                if _service_state[name]["healthy"] is not False:
                    log.warning("%s is DOWN (%d consecutive failures)", name, failures)
                    # Break serenity for vital services going down (not during meditation)
                    if not _is_meditation_active() and name in ("gaia-core", "gaia-web", "gaia-mcp", "gaia-nano", "gaia-orchestrator"):
                        _notify_monkey_break_serenity(f"Vital service {name} went DOWN")
                _service_state[name]["healthy"] = False

                # Enforce structural audit before ANY remediation
                if not run_structural_audit(name):
                    log.error("Structural audit failed for %s. Quarantine active.", name)
                    if not _is_meditation_active():
                        _notify_monkey_break_serenity(f"Structural audit failed for {name}")
                    continue

                if remediation == "ha":
                    info = inspect_container(name)
                    needs_restart = (
                        info is None
                        or info.get("status") != "running"
                        or info.get("restart") not in ("unless-stopped", "always")
                    )
                    if needs_restart:
                        restart_candidate(name)
                elif remediation == "restart":
                    docker_restart(name)
            else:
                log.debug("%s failed check %d/%d", name, failures, FAILURE_THRESHOLD)

    # Run dissonance probe (atomic file-level hashing)
    _dissonance_report = get_dissonance_report()
    if _dissonance_report.get("vital_divergent"):
        log.warning("VITAL ORGAN DISSONANCE: %d vital files diverged — %s",
                     len(_dissonance_report["vital_divergent"]),
                     [f["file"] for f in _dissonance_report["vital_divergent"]])
    if _dissonance_report.get("standard_divergent"):
        log.info("Standard dissonance: %d files diverged (parity %.1f%%)",
                 len(_dissonance_report["standard_divergent"]),
                 _dissonance_report["parity_percent"])

    # Sovereign promotion: if candidate is healthy and files have diverged, trigger review
    all_divergent = _dissonance_report.get("vital_divergent", []) + _dissonance_report.get("standard_divergent", [])
    if all_divergent:
        try:
            candidate_healthy = all(
                check_health(name, url)
                for name, (url, _) in SERVICES.items()
                if "candidate" in name
            )
            if candidate_healthy:
                sovereign_promote(all_divergent)
        except Exception:
            log.debug("Sovereign promotion check failed", exc_info=True)

    _write_status()


def _write_status():
    """Write current state to shared status file."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        status = _build_status()
        STATUS_FILE.write_text(json.dumps(status, indent=2))
    except Exception:
        log.debug("Failed to write status file", exc_info=True)


def _fetch_gpu_status() -> dict | None:
    """Pull GPU status from gaia-orchestrator (non-blocking, best-effort)."""
    try:
        url = f"{ORCHESTRATOR_ENDPOINT}/gpu/status"
        resp = urlopen(url, timeout=3)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_model_server_status() -> dict | None:
    """Pull embedded llama-server status from gaia-core (non-blocking)."""
    try:
        resp = urlopen("http://gaia-core:6415/model/status", timeout=3)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_pipeline_status() -> dict | None:
    """Read self-awareness pipeline state file."""
    try:
        if PIPELINE_STATE_FILE.exists():
            data = json.loads(PIPELINE_STATE_FILE.read_text())
            # Summarize: find current/last stage
            stages = data.get("stages", {})
            running = [s for s, v in stages.items() if v.get("status") == "running"]
            failed = [s for s, v in stages.items() if v.get("status") == "failed"]
            completed = [s for s, v in stages.items() if v.get("status") == "completed"]
            return {
                "pipeline_id": data.get("pipeline_id"),
                "started_at": data.get("started_at"),
                "stages_completed": len(completed),
                "stages_total": len(stages),
                "current_stage": running[0] if running else None,
                "failed_stages": failed,
                "pre_eval_f1": data.get("pre_eval", {}).get("core_avg_f1"),
                "post_eval_f1": data.get("post_eval", {}).get("core_avg_f1"),
            }
    except Exception:
        pass
    return None


_gpu_status_cache: dict | None = None
_model_server_cache: dict | None = None
_pipeline_status_cache: dict | None = None


def _build_status() -> dict:
    uptime = int(time.monotonic() - _start_time)
    now = time.monotonic()
    status = {
        "service": "gaia-doctor",
        "uptime_seconds": uptime,
        "poll_interval": POLL_INTERVAL,
        "maintenance_mode": MAINTENANCE_FLAG.exists(),
        "active_alarms": list(_alarmed_services),
        "irritation_count": len(_irritations),
        "dissonance": _dissonance_report,
        "services": {
            name: {
                "healthy": state["healthy"],
                "last_check": state["last_check"],
                "consecutive_failures": _consecutive_failures.get(name, 0),
                "remediation": SERVICES[name][1],
                "alarmed": name in _alarmed_services,
                "restarts_in_window": len([
                    t for t in _restart_history.get(name, [])
                    if now - t < PROD_RESTART_WINDOW
                ]),
            }
            for name, state in _service_state.items()
        },
        "recent_remediations": _remediation_log[-10:],
        "recent_alarms": _active_alarms[-10:],
        "recent_irritations": _irritations[-5:],
        "dissonance_report": _dissonance_report,
        "serenity": _get_serenity_report(),
        "defensive_meditation": _is_meditation_active(),
        # Enriched subsystem status (populated per poll cycle)
        "gpu": _gpu_status_cache,
        "model_server": _model_server_cache,
        "training_pipeline": _pipeline_status_cache,
    }
    return status


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class DoctorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "healthy", "service": "gaia-doctor"})
        elif self.path == "/status":
            self._json_response(200, _build_status())
        elif self.path == "/alarms":
            self._json_response(200, {
                "alarmed_services": list(_alarmed_services),
                "alarms": _active_alarms[-20:],
            })
        elif self.path == "/irritations":
            self._json_response(200, {
                "irritations": _irritations[-50:],
            })
        elif self.path == "/dissonance":
            self._json_response(200, _dissonance_report or {})
        elif self.path == "/serenity":
            # Read serenity from shared file (written by gaia-monkey)
            self._json_response(200, _get_serenity_report())
        elif self.path == "/gpu":
            self._json_response(200, _gpu_status_cache or {"error": "no data yet"})
        elif self.path == "/model":
            self._json_response(200, _model_server_cache or {"error": "no data yet"})
        elif self.path == "/pipeline":
            self._json_response(200, _pipeline_status_cache or {"status": "no pipeline running"})
        elif self.path == "/cognitive/status":
            self._cognitive_status()
        elif self.path == "/cognitive/results":
            self._cognitive_results()
        elif self.path == "/cognitive/tests":
            self._cognitive_tests()
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/cognitive/run":
            self._cognitive_run()
        else:
            # Chaos, meditation, and serenity management moved to gaia-monkey:6420
            self._json_response(404, {"error": "not found — chaos endpoints moved to gaia-monkey:6420"})

    def _cognitive_status(self):
        global _cognitive_last_result
        if not _BATTERY_AVAILABLE:
            self._json_response(501, {"error": "cognitive_test_battery not available"})
            return
        summary = {}
        alignment = "UNTRAINED"
        if _cognitive_last_result:
            summary = _cognitive_last_result.get("summary", {})
            summary["run_id"] = _cognitive_last_result.get("run_id")
            summary["completed_at"] = _cognitive_last_result.get("completed_at")
            alignment = _cognitive_last_result.get("alignment", "UNTRAINED")
        self._json_response(200, {"running": _cognitive_running, "alignment": alignment, "last_run": summary})

    def _cognitive_results(self):
        global _cognitive_last_result
        if not _BATTERY_AVAILABLE:
            self._json_response(501, {"error": "cognitive_test_battery not available"})
            return
        # Try loading from file if we don't have cached results
        if not _cognitive_last_result:
            results_path = os.environ.get("COGNITIVE_RESULTS_PATH", "/shared/doctor/cognitive_test_results.json")
            try:
                with open(results_path) as f:
                    _cognitive_last_result = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        if _cognitive_last_result:
            self._json_response(200, _cognitive_last_result)
        else:
            self._json_response(200, {"message": "no results available — run a battery first"})

    def _cognitive_tests(self):
        if not _BATTERY_AVAILABLE:
            self._json_response(501, {"error": "cognitive_test_battery not available"})
            return
        self._json_response(200, {"tests": _get_test_metadata()})

    def _cognitive_run(self):
        global _cognitive_running, _cognitive_last_result
        if not _BATTERY_AVAILABLE:
            self._json_response(501, {"error": "cognitive_test_battery not available"})
            return
        if _cognitive_running:
            self._json_response(409, {"error": "battery already running"})
            return
        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            params = {}
        section = params.get("section")
        ids = params.get("ids")
        timeout = params.get("timeout", 30)
        wake_prime = params.get("wake_prime", False)
        full_pipeline = params.get("full_pipeline", False)
        target = params.get("target", "core")
        no_think = params.get("no_think", False)

        def _run():
            global _cognitive_running, _cognitive_last_result
            try:
                _cognitive_running = True
                # Optionally wake gaia-prime before running tests
                if wake_prime:
                    log.info("Cognitive battery: sending wake signal to gaia-core (prime)")
                    try:
                        data = json.dumps({}).encode()
                        req = Request(
                            "http://gaia-core:6415/sleep/wake",
                            data=data,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        urlopen(req, timeout=10)
                        # Wait for core to be active
                        import time as _time
                        for _ in range(30):
                            _time.sleep(2)
                            try:
                                resp = urlopen("http://gaia-core:6415/health", timeout=5)
                                health = json.loads(resp.read().decode())
                                if health.get("sleep_state") == "active":
                                    break
                            except Exception:
                                pass
                    except Exception as e:
                        log.warning("Wake signal failed: %s", e)
                result = _run_cognitive_battery(section=section, ids=ids, timeout=timeout, full_pipeline=full_pipeline, target=target, no_think=no_think)
                _cognitive_last_result = result
            except Exception:
                log.error("Cognitive battery failed", exc_info=True)
            finally:
                _cognitive_running = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        self._json_response(202, {"status": "started", "section": section, "ids": ids, "full_pipeline": full_pipeline, "target": target})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logging


def start_http_server():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), DoctorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("HTTP server listening on port %d", HTTP_PORT)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("gaia-doctor starting (poll=%ds, threshold=%d, cooldown=%ds)",
             POLL_INTERVAL, FAILURE_THRESHOLD, RESTART_COOLDOWN)
    _init_state()
    start_http_server()

    while True:
        try:
            poll_cycle()
        except Exception:
            log.error("Error in poll cycle", exc_info=True)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
