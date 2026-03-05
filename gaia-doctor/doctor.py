#!/usr/bin/env python3
"""
gaia-doctor — Persistent HA watchdog service.

Monitors GAIA service health and automatically restarts crashed or
misconfigured HA candidates via docker compose with the HA overlay.

Zero external dependencies — stdlib only.
"""

import json
import logging
import os
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
    "gaia-web": ("http://localhost:6414/health", "restart"),
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
                _record_irritation(name, f"Code changes detected but tests failed", "CodeAudit: Tests Failed")


def run_service_tests(name: str) -> bool:
    """Run ruff and pytest inside the container to validate code changes."""
    # 1. Fast Lint Check (Fatal errors only: F821 Undefined Name, E999 Syntax)
    log.info("Running fast lint audit for %s...", name)
    try:
        lint_cmd = ["docker", "exec", name, "python", "-m", "ruff", "check", "/app", "--select", "F821,E999"]
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

    log.warning("REMEDIATION: Restarting %s via HA compose overlay", name)
    try:
        result = subprocess.run(
            ["docker", "compose",
             "-p", COMPOSE_PROJECT,
             "-f", f"{COMPOSE_DIR}/docker-compose.candidate.yml",
             "-f", f"{COMPOSE_DIR}/docker-compose.ha.yml",
             "--profile", "ha",
             "up", "-d", name],
            capture_output=True, text=True, timeout=120,
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


def docker_restart(name: str) -> bool:
    """Restart a production service via `docker restart`. Enforces circuit breaker."""
    if MAINTENANCE_FLAG.exists():
        log.info("Maintenance mode active, skipping restart of %s", name)
        return False

    now = time.monotonic()

    # Trim restart history to the rolling window
    _restart_history[name] = [t for t in _restart_history[name] if now - t < PROD_RESTART_WINDOW]

    if len(_restart_history[name]) >= PROD_RESTART_MAX:
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

    log.warning("REMEDIATION: docker restart %s (attempt %d/%d in window)",
                name, len(_restart_history[name]) + 1, PROD_RESTART_MAX)
    try:
        result = subprocess.run(
            ["docker", "restart", name],
            capture_output=True, text=True, timeout=60,
        )
        
        # Fallback: if direct name fails, search for container ID (handles renames)
        if result.returncode != 0:
            log.info("Direct restart failed for %s, searching for container ID...", name)
            find_cmd = ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.ID}}"]
            find_res = subprocess.run(find_cmd, capture_output=True, text=True, timeout=30)
            container_ids = find_res.stdout.strip().split("\n")
            if container_ids and container_ids[0]:
                target_id = container_ids[0]
                log.warning("Found container ID %s for %s. Restarting ID...", target_id, name)
                result = subprocess.run(["docker", "restart", target_id], capture_output=True, text=True, timeout=60)

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
            # Clear alarm state on successful restart if previously alarmed
            _alarmed_services.discard(name)
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
                _service_state[name]["healthy"] = False

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
