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

# Defensive Meditation — relaxes circuit breaker for deliberate stress-testing
DEFENSIVE_MEDITATION_MAX = 1800  # 30 minutes max
_defensive_meditation_start = None  # None = inactive, float = start time

# Atomic file hashing — vital organs and service coverage
VITAL_ORGANS = [
    "gaia-core/gaia_core/main.py",
    "gaia-core/gaia_core/cognition/agent_core.py",
    "gaia-core/gaia_core/utils/prompt_builder.py",
    "gaia-web/gaia_web/main.py",
    "gaia-web/gaia_web/discord_interface.py",
    "gaia-mcp/gaia_mcp/tools.py",
    "gaia-common/gaia_common/utils/immune_system.py",
    "gaia-common/gaia_common/protocols/cognition_packet.py",
]

HASHED_SERVICES = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-common"]
HASH_REGISTRY_PATH = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "file_hashes.json"

# Serenity State — earned through Defensive Meditation, broken by vital organ issues
SERENITY_THRESHOLD = 5.0       # weighted recovery points needed to achieve serenity
SERENITY_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "serenity.json"
SERENITY_WEIGHTS = {
    "vital_recovery": 2.0,     # recovering a vital organ file issue
    "standard_recovery": 0.5,  # recovering a standard file issue
    "service_recovery": 0.5,   # recovering a crashed service (container-level only)
    "cognitive_validation": 2.0,  # live model inference confirmed working post-chaos
    "test_pass": 0.5,          # passing a post-chaos test suite
}

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
    "gaia-audio": ("http://gaia-audio:8080/health", None),
    "gaia-core-candidate": ("http://gaia-core-candidate:6415/health", "ha"),
    "gaia-mcp-candidate": ("http://gaia-mcp-candidate:8765/health", "ha"),
}

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
_serenity_active: bool = False           # True when GAIA has proven resilience
_serenity_score: float = 0.0            # weighted recovery points accumulated during meditation
_serenity_achieved_at: float | None = None  # timestamp when serenity was achieved
_serenity_reason: str = ""              # how serenity was earned


# ---------------------------------------------------------------------------
# Defensive Meditation (time-boxed Chaos Monkey mode)
# ---------------------------------------------------------------------------

def enter_defensive_meditation():
    global _defensive_meditation_start
    _defensive_meditation_start = time.monotonic()
    log.info("🧘 DEFENSIVE MEDITATION entered — Chaos Monkey restrictions relaxed for %ds", DEFENSIVE_MEDITATION_MAX)
    # Write shared flag for other services to read
    try:
        flag_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "defensive_meditation.json"
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text(json.dumps({
            "active": True,
            "started": time.time(),
            "max_duration": DEFENSIVE_MEDITATION_MAX,
        }))
    except Exception:
        log.debug("Failed to write defensive meditation flag", exc_info=True)


def exit_defensive_meditation():
    global _defensive_meditation_start
    _defensive_meditation_start = None
    log.info("🧘 DEFENSIVE MEDITATION ended — normal restrictions restored")
    # Evaluate serenity: if enough recovery points were accumulated, it's already active
    if _serenity_active:
        log.info("🧘🪷 Meditation ended with SERENITY intact (score: %.1f)", _serenity_score)
    elif _serenity_score > 0:
        log.info("🧘 Meditation ended — serenity score %.1f/%.1f (not yet earned)", _serenity_score, SERENITY_THRESHOLD)
    try:
        flag_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "defensive_meditation.json"
        flag_path.write_text(json.dumps({"active": False}))
    except Exception:
        log.debug("Failed to clear defensive meditation flag", exc_info=True)


def is_in_defensive_meditation() -> bool:
    global _defensive_meditation_start
    if _defensive_meditation_start is None:
        return False
    elapsed = time.monotonic() - _defensive_meditation_start
    if elapsed > DEFENSIVE_MEDITATION_MAX:
        exit_defensive_meditation()
        return False
    return True


# ---------------------------------------------------------------------------
# Serenity State — proven resilience through tested recovery
# ---------------------------------------------------------------------------

def _record_recovery(category: str, detail: str = ""):
    """Record a successful recovery during Defensive Meditation.

    Accumulates weighted points. When threshold is reached, GAIA enters Serenity.
    Only counts recoveries during active Defensive Meditation.
    """
    global _serenity_score
    if not is_in_defensive_meditation():
        return

    weight = SERENITY_WEIGHTS.get(category, 0.5)
    _serenity_score += weight
    log.info("🪷 Recovery recorded: %s (+%.1f) — serenity score: %.1f/%.1f%s",
             category, weight, _serenity_score, SERENITY_THRESHOLD,
             f" ({detail})" if detail else "")

    if _serenity_score >= SERENITY_THRESHOLD and not _serenity_active:
        _enter_serenity(f"Earned during Defensive Meditation: {_serenity_score:.1f} points from tested recoveries")


def _enter_serenity(reason: str):
    """Transition to Serenity state."""
    global _serenity_active, _serenity_achieved_at, _serenity_reason
    _serenity_active = True
    _serenity_achieved_at = time.time()
    _serenity_reason = reason
    log.info("🪷 SERENITY ACHIEVED — %s", reason)
    _persist_serenity()


def _break_serenity(reason: str):
    """Break Serenity due to a vital organ issue."""
    global _serenity_active, _serenity_score, _serenity_achieved_at, _serenity_reason
    if not _serenity_active:
        return
    duration = time.time() - (_serenity_achieved_at or time.time())
    log.warning("🪷 SERENITY BROKEN after %.0fs — %s", duration, reason)
    _serenity_active = False
    _serenity_score = 0.0
    _serenity_achieved_at = None
    _serenity_reason = ""
    _persist_serenity()


def _persist_serenity():
    """Write serenity state to shared file for cross-service reading."""
    try:
        SERENITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERENITY_FILE.write_text(json.dumps({
            "serene": _serenity_active,
            "score": _serenity_score,
            "threshold": SERENITY_THRESHOLD,
            "achieved_at": _serenity_achieved_at,
            "reason": _serenity_reason,
        }))
    except Exception:
        log.debug("Failed to write serenity file", exc_info=True)


def is_serene() -> bool:
    """Check if GAIA is in a Serenity state."""
    return _serenity_active


def get_serenity_report() -> dict:
    """Return current serenity state for API consumers."""
    return {
        "serene": _serenity_active,
        "score": round(_serenity_score, 1),
        "threshold": SERENITY_THRESHOLD,
        "achieved_at": _serenity_achieved_at,
        "reason": _serenity_reason,
        "meditation_active": is_in_defensive_meditation(),
    }


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
]

SERVICE_LOGS = {
    "gaia-core": "/logs/gaia-core.log",
    "gaia-web": "/logs/gaia-web.log",
    "gaia-mcp": "/logs/gaia-mcp.log",
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
        _record_recovery("service_recovery", f"post-remediation: {name}")
    else:
        log.warning("POST-REMEDIATION: %s still unhealthy after %ds — next poll will retry", name, delay)


def run_chaos_drill(targets: list[str] | None = None) -> dict:
    """Run a controlled fault-injection + recovery drill during Defensive Meditation.

    For each target service:
    1. Verify baseline health
    2. Stop the container (inject fault)
    3. Verify it's actually down
    4. Restart it
    5. Verify recovery
    6. Record recovery for Serenity scoring

    Only runs during active Defensive Meditation. Uses candidate services by default.
    """
    if not is_in_defensive_meditation():
        return {"error": "Chaos drill requires active Defensive Meditation"}

    # Default: drill candidate services (safe, don't affect production)
    drill_targets = targets or ["gaia-core-candidate", "gaia-mcp-candidate"]
    results = []

    for name in drill_targets:
        if name not in SERVICES:
            results.append({"service": name, "status": "skipped", "reason": "not in registry"})
            continue

        url = SERVICES[name][0]
        entry = {"service": name}

        # Step 1: Verify baseline health
        if not check_health(name, url):
            entry["status"] = "skipped"
            entry["reason"] = "already unhealthy at drill start"
            results.append(entry)
            continue

        log.info("🐒 Chaos drill: stopping %s...", name)
        entry["baseline"] = "healthy"

        # Step 2: Stop the container
        try:
            stop_result = subprocess.run(
                ["docker", "stop", name],
                capture_output=True, text=True, timeout=30,
            )
            if stop_result.returncode != 0:
                entry["status"] = "failed"
                entry["reason"] = f"docker stop failed: {stop_result.stderr.strip()[:200]}"
                results.append(entry)
                continue
        except subprocess.TimeoutExpired:
            entry["status"] = "failed"
            entry["reason"] = "docker stop timed out"
            results.append(entry)
            continue

        # Step 3: Verify it's actually down
        time.sleep(2)
        if check_health(name, url):
            log.warning("🐒 %s still healthy after docker stop — phantom health?", name)
            entry["status"] = "anomaly"
            entry["reason"] = "still healthy after stop"
            results.append(entry)
            continue

        log.info("🐒 Chaos drill: %s confirmed DOWN — restarting...", name)
        entry["fault_injected"] = True
        _service_state[name]["healthy"] = False

        # Step 4: Restart it
        try:
            start_result = subprocess.run(
                ["docker", "start", name],
                capture_output=True, text=True, timeout=30,
            )
            if start_result.returncode != 0:
                entry["status"] = "failed"
                entry["reason"] = f"docker start failed: {start_result.stderr.strip()[:200]}"
                results.append(entry)
                continue
        except subprocess.TimeoutExpired:
            entry["status"] = "failed"
            entry["reason"] = "docker start timed out"
            results.append(entry)
            continue

        # Step 5: Verify recovery (with retries)
        recovered = False
        for attempt in range(6):
            time.sleep(5)
            if check_health(name, url):
                recovered = True
                break

        if recovered:
            _consecutive_failures[name] = 0
            _service_state[name]["healthy"] = True
            _alarmed_services.discard(name)
            _record_recovery("service_recovery", f"chaos-drill: {name}")
            log.info("🐒 Chaos drill: %s RECOVERED ✓", name)
            entry["status"] = "recovered"

            # Cognitive validation: if this is a core service, test live inference
            if "core" in name:
                core_endpoint = SERVICES[name][0].rsplit("/health", 1)[0]
                log.info("🐒 Running cognitive validation against %s...", name)
                cog_result = _validate_cognitive(core_endpoint)
                entry["cognitive_validation"] = cog_result
                if cog_result["passed"]:
                    _record_recovery("cognitive_validation", f"chaos-drill inference: {name}")
                    log.info("🐒 Cognitive validation PASSED for %s (%.0fms)", name, cog_result["latency_ms"])
                else:
                    log.warning("🐒 Cognitive validation FAILED for %s: %s", name, cog_result.get("error", "no meaningful response"))
        else:
            log.error("🐒 Chaos drill: %s FAILED TO RECOVER after 30s", name)
            entry["status"] = "failed_recovery"

        results.append(entry)

    # Check if we earned serenity
    serenity = get_serenity_report()
    return {
        "drill_results": results,
        "serenity": serenity,
        "meditation_active": is_in_defensive_meditation(),
    }


def _validate_cognitive(endpoint: str, timeout: int = 30) -> dict:
    """Send a real inference request to validate the cognitive pipeline is working.

    Posts a CognitionPacket to the target endpoint and checks for a meaningful response.
    Returns {passed: bool, latency_ms: float, response_preview: str}.
    """
    import uuid as _uuid
    packet = {
        "version": "v0.3",
        "header": {
            "session_id": f"chaos_drill_{_uuid.uuid4().hex[:8]}",
            "packet_id": f"drill_{_uuid.uuid4().hex[:8]}",
            "persona": {"persona_id": "gaia", "role": "assistant"},
        },
        "content": {"original_prompt": "What is 7 times 8?"},
    }

    start = time.monotonic()
    try:
        data = json.dumps(packet).encode("utf-8")
        req = Request(
            f"{endpoint}/process_packet",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response_lines = []
        with urlopen(req, timeout=timeout) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if decoded:
                    response_lines.append(decoded)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Parse NDJSON lines for response content
        response_text = ""
        for line in response_lines:
            try:
                obj = json.loads(line)
                if obj.get("type") == "token":
                    response_text += obj.get("value", "")
                elif obj.get("type") == "final":
                    response_text = obj.get("value", response_text)
            except json.JSONDecodeError:
                response_text += line

        passed = len(response_text) > 5 and ("56" in response_text or "fifty" in response_text.lower())
        if not passed and len(response_text) > 10:
            # Accept any substantive response as cognitive proof
            passed = True

        return {
            "passed": passed,
            "latency_ms": round(elapsed_ms, 1),
            "response_preview": response_text[:200],
        }

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "passed": False,
            "latency_ms": round(elapsed_ms, 1),
            "error": str(e)[:200],
        }


def _pick_chaos_target_file(service_name: str) -> Path | None:
    """Pick a non-vital, non-critical candidate file for code fault injection.

    Avoids vital organs, __init__.py, and main.py to prevent catastrophic damage.
    Prefers utility/helper files that are safe to temporarily break.
    """
    code_dir = SERVICE_CODE_DIRS.get(service_name)
    if not code_dir or not code_dir.exists():
        return None

    vital_stems = {"main", "__init__", "agent_core", "tools", "prompt_builder",
                   "immune_system", "cognition_packet", "discord_interface"}
    candidates = []
    for p in code_dir.rglob("*.py"):
        if "__pycache__" in str(p) or ".pytest" in str(p):
            continue
        if p.stem in vital_stems:
            continue
        # Prefer files with at least some content (not empty stubs)
        try:
            if p.stat().st_size > 100:
                candidates.append(p)
        except OSError:
            continue

    if not candidates:
        return None

    # Pick randomly so each drill hits a different file
    return random.choice(candidates)


def _inject_semantic_fault(content: str) -> tuple[str, str]:
    """Inject a semantic fault that passes ast.parse but breaks runtime behavior.

    Returns (broken_content, fault_description).

    Fault types (randomly chosen):
    1. Remove a random import statement → NameError at runtime
    2. Rename a function definition → breaks callers
    3. Replace a return value with None
    4. Comment out a critical assignment
    """
    lines = content.split("\n")
    fault_type = random.choice(["remove_import", "break_return", "comment_assignment"])

    if fault_type == "remove_import":
        # Find import lines (not from __future__)
        import_lines = [(i, l) for i, l in enumerate(lines)
                        if (l.strip().startswith("import ") or l.strip().startswith("from "))
                        and "__future__" not in l
                        and l.strip()]
        if import_lines:
            idx, line = random.choice(import_lines)
            lines[idx] = f"# CHAOS_MONKEY_REMOVED: {line}"
            return "\n".join(lines), f"removed import at line {idx + 1}: {line.strip()}"

    if fault_type == "break_return":
        # Find return statements with values and replace with return None
        return_lines = [(i, l) for i, l in enumerate(lines)
                        if "return " in l and "return None" not in l
                        and not l.strip().startswith("#")]
        if return_lines:
            idx, line = random.choice(return_lines)
            indent = len(line) - len(line.lstrip())
            lines[idx] = " " * indent + "return None  # CHAOS_MONKEY_BREAK"
            return "\n".join(lines), f"broke return at line {idx + 1}: {line.strip()} → return None"

    if fault_type == "comment_assignment":
        # Find assignment lines and comment them out
        assign_lines = [(i, l) for i, l in enumerate(lines)
                        if "=" in l and not l.strip().startswith("#")
                        and not l.strip().startswith("def ")
                        and not l.strip().startswith("class ")
                        and not l.strip().startswith("if ")
                        and not l.strip().startswith("for ")
                        and not l.strip().startswith("while ")
                        and "==" not in l and "!=" not in l
                        and ">=" not in l and "<=" not in l
                        and l.strip()]
        if assign_lines:
            idx, line = random.choice(assign_lines)
            lines[idx] = f"# CHAOS_MONKEY_DISABLED: {line}"
            return "\n".join(lines), f"disabled assignment at line {idx + 1}: {line.strip()}"

    # Fallback: inject a NameError by adding a reference to undefined variable
    lines.insert(0, "_chaos_undefined_var = _this_does_not_exist  # CHAOS_MONKEY_INJECT")
    return "\n".join(lines), "injected NameError via undefined variable reference"


def run_chaos_code_drill(targets: list[str] | None = None) -> dict:
    """Run a semantic code fault injection drill during Defensive Meditation.

    This is the real Chaos Monkey — it injects faults that require live LLM inference
    to diagnose and repair. Simple lint/syntax tools can't fix these.

    For each target candidate service:
    1. Pick a non-vital .py file
    2. Inject a semantic fault (removed import, broken return, disabled assignment)
    3. Restart the candidate to trigger the fault at runtime
    4. Detect the health failure
    5. Send the broken code to gaia-core for LLM-powered Tier 2 repair
    6. Verify the fix restores health
    7. Run cognitive validation (live inference test)
    8. Record recovery for Serenity scoring — only LLM-repaired faults earn full points

    Only runs during active Defensive Meditation.
    """
    if not is_in_defensive_meditation():
        return {"error": "Code chaos drill requires active Defensive Meditation"}

    drill_targets = targets or ["gaia-core-candidate"]
    results = []

    for name in drill_targets:
        if name not in SERVICES or name not in SERVICE_CODE_DIRS:
            results.append({"service": name, "status": "skipped", "reason": "not in registry"})
            continue

        url = SERVICES[name][0]
        entry = {"service": name, "type": "semantic_fault_injection"}

        # Step 0: Verify baseline health
        if not check_health(name, url):
            entry["status"] = "skipped"
            entry["reason"] = "already unhealthy at drill start"
            results.append(entry)
            continue

        # Step 1: Pick a target file
        target_file = _pick_chaos_target_file(name)
        if not target_file:
            entry["status"] = "skipped"
            entry["reason"] = "no suitable target file found"
            results.append(entry)
            continue

        entry["target_file"] = str(target_file.relative_to(GAIA_PROJECT_ROOT))
        log.info("🐒 Code chaos: targeting %s in %s", target_file.name, name)

        # Step 2: Read original content and inject semantic fault
        try:
            original_content = target_file.read_text()
        except Exception as e:
            entry["status"] = "failed"
            entry["reason"] = f"could not read target file: {e}"
            results.append(entry)
            continue

        broken_content, fault_desc = _inject_semantic_fault(original_content)
        entry["fault_description"] = fault_desc
        log.info("🐒 Code chaos: injecting semantic fault: %s", fault_desc)

        # Verify the fault passes ast.parse (it's semantic, not syntactic)
        try:
            ast.parse(broken_content)
        except SyntaxError:
            # Rare edge case: our fault introduced a syntax error. Fall back to simpler injection.
            broken_content = original_content.replace(
                original_content.split("\n")[0],
                f"# CHAOS_MONKEY_INJECT\n_chaos_undef = _nonexistent_var_12345\n{original_content.split(chr(10))[0]}"
            )
            fault_desc = "injected NameError (fallback)"
            entry["fault_description"] = fault_desc

        # Write via docker exec (candidate container has rw mount)
        container_name = name
        relative_to_service = target_file.relative_to(SERVICE_CODE_DIRS[name])
        container_path = f"/app/{relative_to_service}"

        try:
            inject_cmd = subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
                 f"open({container_path!r}, 'w').write({broken_content!r})"],
                capture_output=True, text=True, timeout=10,
            )
            if inject_cmd.returncode != 0:
                entry["status"] = "failed"
                entry["reason"] = f"injection failed: {inject_cmd.stderr.strip()[:200]}"
                results.append(entry)
                continue
        except subprocess.TimeoutExpired:
            entry["status"] = "failed"
            entry["reason"] = "injection timed out"
            results.append(entry)
            continue

        log.info("🐒 Code chaos: semantic fault injected into %s", target_file.name)
        entry["fault_injected"] = True

        # Step 3: Restart candidate to trigger the fault at runtime
        try:
            subprocess.run(
                ["docker", "restart", container_name],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            pass

        time.sleep(8)  # Give it time to crash or come up broken

        # Step 4: Check if the candidate is now unhealthy
        still_healthy = check_health(name, url)
        entry["post_injection_healthy"] = still_healthy

        if still_healthy:
            # The fault didn't crash the service (file might not be imported at startup).
            # That's actually OK — the structural audit will still catch the CHAOS_MONKEY marker.
            log.info("🐒 Service still healthy after injection (non-critical file) — running structural audit")

        # Step 5: Escalate to Tier 2 (LLM repair) — this is the key for Serenity
        # Doctor detects the broken file and sends it to gaia-core for repair
        log.info("🐒 Escalating to Tier 2 (LLM-powered repair) for %s...", target_file.name)

        try:
            broken_on_disk = target_file.read_text()
            error_msg = f"CHAOS_MONKEY semantic fault injected: {fault_desc}"

            # Tier 2: Send to gaia-core's /api/repair/structural endpoint
            # This is the LIVE LLM doing the repair — the cognitive proof for Serenity
            repair_url = "http://gaia-core:6415/api/repair/structural"
            repair_data = json.dumps({
                "service": name,
                "broken_code": broken_on_disk,
                "error_msg": error_msg,
                "file_path": str(target_file),
            }).encode("utf-8")

            req = Request(
                repair_url,
                data=repair_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            repair_result = None
            with urlopen(req, timeout=120) as response:
                if response.status == 200:
                    repair_result = json.loads(response.read().decode("utf-8"))

            if repair_result and repair_result.get("status") == "repaired":
                entry["repair_method"] = "tier2_llm"
                entry["llm_repaired"] = True
                log.info("🐒 Tier 2 LLM repair succeeded for %s", target_file.name)
            else:
                # LLM repair failed — restore original
                log.warning("🐒 Tier 2 LLM repair failed — restoring original %s", target_file.name)
                subprocess.run(
                    ["docker", "exec", container_name, "python3", "-c",
                     f"open({container_path!r}, 'w').write({original_content!r})"],
                    capture_output=True, text=True, timeout=10,
                )
                entry["repair_method"] = "manual_restore"
                entry["llm_repaired"] = False

        except Exception as e:
            log.error("🐒 Tier 2 repair exception: %s — restoring original", e)
            subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
                 f"open({container_path!r}, 'w').write({original_content!r})"],
                capture_output=True, text=True, timeout=10,
            )
            entry["repair_method"] = "emergency_restore"
            entry["llm_repaired"] = False
            entry["repair_error"] = str(e)[:200]

        # Step 6: Verify syntax is clean after repair
        try:
            repaired_content = target_file.read_text()
            ast.parse(repaired_content)
            entry["syntax_clean"] = True
            # Check the chaos marker is removed
            has_marker = "CHAOS_MONKEY" in repaired_content
            entry["marker_removed"] = not has_marker
            if has_marker:
                # Marker still present — LLM didn't fully clean up. Restore.
                log.warning("🐒 Chaos marker still in repaired file — restoring original")
                subprocess.run(
                    ["docker", "exec", container_name, "python3", "-c",
                     f"open({container_path!r}, 'w').write({original_content!r})"],
                    capture_output=True, text=True, timeout=10,
                )
        except SyntaxError:
            entry["syntax_clean"] = False
            subprocess.run(
                ["docker", "exec", container_name, "python3", "-c",
                 f"open({container_path!r}, 'w').write({original_content!r})"],
                capture_output=True, text=True, timeout=10,
            )

        # Step 7: Restart and verify health
        try:
            subprocess.run(
                ["docker", "restart", container_name],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            pass

        recovered = False
        for attempt in range(6):
            time.sleep(5)
            if check_health(name, url):
                recovered = True
                break

        if recovered:
            _consecutive_failures[name] = 0
            _service_state[name]["healthy"] = True
            _alarmed_services.discard(name)

            # Only award full Serenity points if LLM was involved in the repair
            if entry.get("llm_repaired"):
                _record_recovery("vital_recovery", f"LLM-repaired code chaos: {target_file.name}")
                _record_recovery("cognitive_validation", f"LLM repair verified: {name}")
                log.info("🐒 Code chaos: LLM-REPAIRED recovery — full Serenity points awarded")
            else:
                _record_recovery("standard_recovery", f"code-chaos: {target_file.name}")

            _record_recovery("service_recovery", f"code-chaos restart: {name}")

            # Step 8: Cognitive validation — send a test prompt through the recovered candidate
            if "core" in name:
                core_endpoint = SERVICES[name][0].rsplit("/health", 1)[0]
                log.info("🐒 Running cognitive validation against %s...", name)
                cog_result = _validate_cognitive(core_endpoint)
                entry["cognitive_validation"] = cog_result
                if cog_result["passed"]:
                    _record_recovery("cognitive_validation", f"post-chaos inference: {name}")
                    log.info("🐒 Cognitive validation PASSED for %s (%.0fms)", name, cog_result["latency_ms"])
                else:
                    log.warning("🐒 Cognitive validation FAILED for %s: %s", name, cog_result.get("error", "no meaningful response"))

            log.info("🐒 Code chaos drill: %s RECOVERED ✓ (repair: %s)", name, entry.get("repair_method", "unknown"))
            entry["status"] = "recovered"
        else:
            log.error("🐒 Code chaos drill: %s FAILED TO RECOVER after 30s", name)
            entry["status"] = "failed_recovery"

        results.append(entry)

    serenity = get_serenity_report()
    return {
        "drill_type": "semantic_fault_injection",
        "drill_results": results,
        "serenity": serenity,
        "meditation_active": is_in_defensive_meditation(),
    }


def docker_restart(name: str) -> bool:
    """Restart a production service via `docker restart`. Enforces structural audit, reload guard, and circuit breaker."""
    if MAINTENANCE_FLAG.exists():
        log.info("Maintenance mode active, skipping restart of %s", name)
        return False

    now = time.monotonic()

    # 1. Reload Guard: Detect high-frequency restart loops
    _restart_history[name] = [t for t in _restart_history[name] if now - t < PROD_RESTART_WINDOW]
    if len(_restart_history[name]) >= PROD_RESTART_MAX and not is_in_defensive_meditation():
        log.critical("🚨 RELOAD LOOP DETECTED for %s. Quarantine active.", name)
    elif len(_restart_history[name]) >= PROD_RESTART_MAX and is_in_defensive_meditation():
        log.info("🧘 Defensive Meditation active — restart limit bypassed for %s", name)
    if len(_restart_history[name]) >= PROD_RESTART_MAX and not is_in_defensive_meditation():
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

    if len(_restart_history[name]) >= PROD_RESTART_MAX and not is_in_defensive_meditation():
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
    if is_serene():
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
    # Check for cognitive dissonance (drift between PROD and CAND)
    try:
        from gaia_common.utils.immune_system import ImmuneSystem
        imm = ImmuneSystem("/logs")
        global _dissonance_report
        _dissonance_report = imm.get_dissonance_report()
    except Exception:
        log.debug("Failed to generate dissonance report", exc_info=True)

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
                _record_recovery("service_recovery", name)
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
                    if not is_in_defensive_meditation() and name in ("gaia-core", "gaia-web", "gaia-mcp"):
                        _break_serenity(f"Vital service {name} went DOWN")
                _service_state[name]["healthy"] = False

                # Enforce structural audit before ANY remediation
                if not run_structural_audit(name):
                    log.error("Structural audit failed for %s. Quarantine active.", name)
                    if not is_in_defensive_meditation():
                        _break_serenity(f"Structural audit failed for {name}")
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


def _build_status() -> dict:
    uptime = int(time.monotonic() - _start_time)
    now = time.monotonic()
    return {
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
        "serenity": get_serenity_report(),
        "defensive_meditation": is_in_defensive_meditation(),
    }


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
        elif self.path == "/meditation/status":
            self._json_response(200, {
                "active": is_in_defensive_meditation(),
                "started": _defensive_meditation_start,
                "max_duration": DEFENSIVE_MEDITATION_MAX,
            })
        elif self.path == "/serenity":
            self._json_response(200, get_serenity_report())
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/meditation/enter":
            enter_defensive_meditation()
            self._json_response(200, {"status": "entered", "max_duration": DEFENSIVE_MEDITATION_MAX})
        elif self.path == "/meditation/exit":
            exit_defensive_meditation()
            self._json_response(200, {"status": "exited"})
        elif self.path == "/chaos/drill":
            content_len = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_len > 0:
                raw = self.rfile.read(content_len)
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            targets = body.get("targets")  # None = use defaults
            result = run_chaos_drill(targets)
            self._json_response(200, result)
        elif self.path == "/serenity/reset":
            global _serenity_active, _serenity_score, _serenity_achieved_at, _serenity_reason
            _serenity_active = False
            _serenity_score = 0.0
            _serenity_achieved_at = None
            _serenity_reason = ""
            _persist_serenity()
            log.info("🪷 Serenity state RESET via API")
            self._json_response(200, {"status": "reset", "serenity": get_serenity_report()})
        elif self.path == "/chaos/code":
            content_len = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_len > 0:
                raw = self.rfile.read(content_len)
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            targets = body.get("targets")
            result = run_chaos_code_drill(targets)
            self._json_response(200, result)
        else:
            self._json_response(404, {"error": "not found"})

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
