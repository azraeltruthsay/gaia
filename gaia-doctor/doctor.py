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
COMPOSE_DIR = os.environ.get("COMPOSE_DIR", "/compose")
COMPOSE_PROJECT = os.environ.get("COMPOSE_PROJECT_NAME", "gaia_project")

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
    # name: (health_url, can_remediate)
    "gaia-core": ("http://gaia-core:6415/health", False),
    "gaia-prime": ("http://gaia-prime:7777/health", False),
    "gaia-core-candidate": ("http://gaia-core-candidate:6415/health", True),
    "gaia-mcp-candidate": ("http://gaia-mcp-candidate:8765/health", True),
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_start_time = time.monotonic()
_service_state: dict[str, dict] = {}
_consecutive_failures: dict[str, int] = {}
_last_restart: dict[str, float] = {}
_remediation_log: list[dict] = []


def _init_state():
    for name in SERVICES:
        _service_state[name] = {"healthy": None, "last_check": None}
        _consecutive_failures[name] = 0


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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def poll_cycle():
    """Run one health check cycle across all services."""
    for name, (url, can_remediate) in SERVICES.items():
        healthy = check_health(name, url)
        _service_state[name]["last_check"] = datetime.now(timezone.utc).isoformat()

        if healthy:
            _consecutive_failures[name] = 0
            if _service_state[name]["healthy"] is False:
                log.info("%s recovered", name)
            _service_state[name]["healthy"] = True
        else:
            _consecutive_failures[name] += 1
            failures = _consecutive_failures[name]

            if failures >= FAILURE_THRESHOLD:
                if _service_state[name]["healthy"] is not False:
                    log.warning("%s is DOWN (%d consecutive failures)", name, failures)
                _service_state[name]["healthy"] = False

                if can_remediate:
                    info = inspect_container(name)
                    needs_restart = (
                        info is None
                        or info.get("status") != "running"
                        or info.get("restart") not in ("unless-stopped", "always")
                    )
                    if needs_restart:
                        restart_candidate(name)
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
    return {
        "service": "gaia-doctor",
        "uptime_seconds": uptime,
        "poll_interval": POLL_INTERVAL,
        "maintenance_mode": MAINTENANCE_FLAG.exists(),
        "services": {
            name: {
                "healthy": state["healthy"],
                "last_check": state["last_check"],
                "consecutive_failures": _consecutive_failures.get(name, 0),
                "can_remediate": SERVICES[name][1],
            }
            for name, state in _service_state.items()
        },
        "recent_remediations": _remediation_log[-10:],
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
