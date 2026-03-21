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
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen, Request

# Cognitive test battery (stdlib only — lives alongside doctor.py)
try:
    from cognitive_test_battery import run_battery as _run_cognitive_battery, get_test_metadata as _get_test_metadata, generate_rubric as _generate_rubric
    _BATTERY_AVAILABLE = True
except ImportError:
    _BATTERY_AVAILABLE = False

# Error registry (lazy import — stdlib-safe, lives in gaia-common)
_error_registry = None

def _get_error_registry():
    """Lazy-import the GAIA error registry to enrich log entries with hints."""
    global _error_registry
    if _error_registry is not None:
        return _error_registry
    try:
        from gaia_common.errors import lookup, all_errors
        _error_registry = {"lookup": lookup, "all_errors": all_errors}
    except ImportError:
        _error_registry = {"lookup": lambda _: None, "all_errors": lambda: {}}
    return _error_registry

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "60"))
FAILURE_THRESHOLD = int(os.environ.get("FAILURE_THRESHOLD", "2"))
RESTART_COOLDOWN = int(os.environ.get("RESTART_COOLDOWN", "300"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "6419"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# Dry-run mode: observe and report but never restart containers or promote code.
# Used by gaia-doctor-candidate to safely shadow production doctor.
DRY_RUN = os.environ.get("DOCTOR_DRY_RUN", "0") in ("1", "true", "yes")
MAINTENANCE_FLAG = Path(os.environ.get("SHARED_DIR", "/shared")) / "ha_maintenance"

# Structured maintenance mode (new — backward compat with MAINTENANCE_FLAG)
try:
    from gaia_common.utils.maintenance import (
        is_maintenance_active,
        get_maintenance_info,
        enter_maintenance,
        exit_maintenance,
    )
except ImportError:
    # Inline stdlib-only implementation (doctor can't import gaia-common)
    _MAINT_FLAG_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "maintenance_mode.json"

    def is_maintenance_active():
        try:
            if _MAINT_FLAG_FILE.exists():
                data = json.loads(_MAINT_FLAG_FILE.read_text())
                return data.get("active", False)
        except (json.JSONDecodeError, OSError):
            pass
        return MAINTENANCE_FLAG.exists()

    def get_maintenance_info():
        if not is_maintenance_active():
            return None
        try:
            if _MAINT_FLAG_FILE.exists():
                return json.loads(_MAINT_FLAG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
        if MAINTENANCE_FLAG.exists():
            return {"active": True, "entered_at": "unknown", "entered_by": "legacy", "reason": "ha_maintenance flag"}
        return None

    def enter_maintenance(reason="manual", entered_by="unknown"):
        _MAINT_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active": True,
            "entered_at": datetime.now(timezone.utc).isoformat(),
            "entered_by": entered_by,
            "reason": reason,
        }
        _MAINT_FLAG_FILE.write_text(json.dumps(data, indent=2))
        MAINTENANCE_FLAG.touch()
        return data

    def exit_maintenance():
        info = get_maintenance_info()
        duration = None
        if info and info.get("entered_at", "unknown") != "unknown":
            try:
                entered = datetime.fromisoformat(info["entered_at"])
                duration = (datetime.now(timezone.utc) - entered).total_seconds()
            except (ValueError, TypeError):
                pass
        try:
            _MAINT_FLAG_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            MAINTENANCE_FLAG.unlink(missing_ok=True)
        except OSError:
            pass
        return {"active": False, "exited_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": duration, "previous": info}
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

# KV cache pressure monitoring — independent of gaia-core
KV_CACHE_ENDPOINTS = {
    "reflex": "http://gaia-nano:8080/slots",
    "core": "http://localhost:8092/slots",
}
KV_CACHE_DOCTOR_THRESHOLD = float(os.environ.get("KV_CACHE_DOCTOR_THRESHOLD", "0.90"))

MONKEY_ENDPOINT = os.environ.get("MONKEY_ENDPOINT", "http://gaia-monkey:6420")
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415")
ES_ENDPOINT = os.environ.get("ES_ENDPOINT", "http://elasticsearch:9200")

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
    "gaia-nano": ("http://gaia-nano:8080/health", "restart"),
    "gaia-audio": ("http://gaia-audio:8080/health", "restart"),
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

# Sovereign promotion rate limiter
_last_sovereign_attempt: float = 0.0

# Cognitive monitor (heartbeat probe) state
_cognitive_monitor_interval = int(os.environ.get("COGNITIVE_MONITOR_INTERVAL", "300"))
_cognitive_monitor_last_run: float = 0.0
_cognitive_monitor_failures: int = 0
_cognitive_monitor_last_result: dict | None = None
COGNITIVE_MONITOR_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "cognitive_monitor.json"

# ---------------------------------------------------------------------------
# Surgeon Approval Queue
# ---------------------------------------------------------------------------

SURGEON_CONFIG_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "surgeon_config.json"
_surgeon_approval_required: bool = False
_surgeon_queue: list[dict] = []       # pending repair proposals
_surgeon_history: list[dict] = []     # completed/rejected (last 50)


def _load_surgeon_config():
    """Load surgeon config from shared file on startup."""
    global _surgeon_approval_required
    try:
        if SURGEON_CONFIG_FILE.exists():
            data = json.loads(SURGEON_CONFIG_FILE.read_text())
            _surgeon_approval_required = data.get("approval_required", False)
    except (json.JSONDecodeError, OSError):
        pass


def _save_surgeon_config():
    """Persist surgeon config to shared file."""
    try:
        SURGEON_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        SURGEON_CONFIG_FILE.write_text(json.dumps({
            "approval_required": _surgeon_approval_required,
        }, indent=2))
    except OSError as e:
        log.warning("Failed to save surgeon config: %s", e)

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
# KV Cache Pressure Monitoring
# ---------------------------------------------------------------------------

_kv_cache_pressure: dict = {}  # role -> {pressure, n_past, n_ctx, checked_at}


def poll_kv_cache_pressure():
    """Poll llama-server /slots endpoints for KV cache fill ratio.

    If pressure >= doctor threshold (90%), request compaction via gaia-core.
    If compaction fails AND the service supports restart, restart as last resort.
    """
    global _kv_cache_pressure

    for role, slots_url in KV_CACHE_ENDPOINTS.items():
        try:
            resp = urlopen(slots_url, timeout=5)
            data = json.loads(resp.read().decode())
            # /slots returns a list of slot dicts
            slot = data[0] if isinstance(data, list) and data else data
            n_ctx = slot.get("n_ctx", 0)
            n_past = slot.get("n_past", 0)
            pressure = n_past / n_ctx if n_ctx > 0 else 0.0

            _kv_cache_pressure[role] = {
                "pressure": round(pressure, 4),
                "n_past": n_past,
                "n_ctx": n_ctx,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

            if pressure >= KV_CACHE_DOCTOR_THRESHOLD:
                log.warning(
                    "KV cache pressure HIGH for '%s': %.1f%% (%d/%d) — requesting compaction",
                    role, pressure * 100, n_past, n_ctx,
                )
                compacted = _request_compact(role)
                if not compacted:
                    # Last resort: restart the service if it supports restart
                    svc_name = "gaia-nano" if role == "reflex" else "gaia-core"
                    svc_entry = SERVICES.get(svc_name)
                    if svc_entry and svc_entry[1] == "restart":
                        log.warning(
                            "Compaction failed for '%s' — restarting %s as last resort",
                            role, svc_name,
                        )
                        docker_restart(svc_name)
            elif pressure >= 0.7:
                log.info("KV cache pressure ELEVATED for '%s': %.1f%%", role, pressure * 100)

        except Exception:
            _kv_cache_pressure[role] = {
                "pressure": -1,
                "error": "unreachable",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            log.debug("KV cache poll failed for '%s'", role, exc_info=True)


def _request_compact(role: str) -> bool:
    """Request KV cache compaction from gaia-core's API. Returns True on success."""
    try:
        url = f"{CORE_ENDPOINT}/api/kv-cache/compact/{role}"
        req = Request(url, data=b"", method="POST",
                      headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=15)
        result = json.loads(resp.read().decode())
        if result.get("status") == "compacted":
            log.info("KV cache compaction succeeded for '%s' via gaia-core", role)
            return True
        log.warning("KV cache compaction response for '%s': %s", role, result)
        return False
    except Exception as e:
        log.warning("KV cache compaction request failed for '%s': %s", role, e)
        return False


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

    # Load surgeon approval config from persistent file
    _load_surgeon_config()
    log.info("Surgeon approval mode: %s", "ON" if _surgeon_approval_required else "OFF")

    # Generate cognitive rubric for observer scoring
    if _BATTERY_AVAILABLE:
        try:
            _generate_rubric()
            log.info("Cognitive rubric generated on startup")
        except Exception:
            log.warning("Failed to generate cognitive rubric on startup", exc_info=True)


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
                # If candidate + meditation active -> enter doctor self-repair loop
                if "candidate" in name and _is_meditation_active():
                    log.info("Meditation active — entering self-repair for divergent candidate %s", name)
                    _repair_divergent_candidate(name)


def run_service_tests(name: str) -> bool:
    """Run ruff and pytest inside the container to validate code changes."""
    # 1. Lint Check (all F-rules: F401 unused import, F811 redef, F821 undef, etc.)
    log.info("Running lint audit for %s...", name)
    try:
        lint_cmd = ["docker", "exec", name, "python", "-m", "ruff", "check", "/app", "--select", "F", "--no-cache"]
        lint_res = subprocess.run(lint_cmd, capture_output=True, text=True, timeout=30)
        if lint_res.returncode != 0:
            log.error("LINT ERROR in %s:\n%s", name, lint_res.stdout)
            # Attempt auto-fix for safe rules before recording irritation
            if _attempt_lint_autofix(name):
                lint_res = subprocess.run(lint_cmd, capture_output=True, text=True, timeout=30)
                if lint_res.returncode == 0:
                    log.info("Lint autofix resolved all errors in %s", name)
                else:
                    _record_irritation(name, f"Lint error (post-autofix): {lint_res.stdout[:100]}", "CodeAudit: Lint Fatal")
                    return False
            else:
                _record_irritation(name, f"Lint error: {lint_res.stdout[:100]}", "CodeAudit: Lint Fatal")
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
# Lint Auto-Fix (safe subset of F-rules)
# ---------------------------------------------------------------------------

# Rules safe for unattended auto-fix (ruff --fix handles these deterministically):
_LINT_AUTOFIX_RULES = "F401,F541,F811,F841"


def _attempt_lint_autofix(name: str) -> bool:
    """Try to auto-fix trivial lint errors (unused imports, etc.) in a container.

    Returns True if fixes were applied successfully.
    If surgeon approval is required, queues a proposal and returns False.
    """
    # 1. Preview fixable issues
    try:
        preview_cmd = [
            "docker", "exec", name, "python", "-m", "ruff", "check", "/app",
            "--select", _LINT_AUTOFIX_RULES, "--no-cache",
        ]
        preview = subprocess.run(preview_cmd, capture_output=True, text=True, timeout=30)
        if preview.returncode == 0:
            return False  # Nothing to fix in the safe subset
    except Exception as e:
        log.warning("Lint autofix preview failed for %s: %s", name, e)
        return False

    fixable_summary = preview.stdout.strip()[:500]
    log.info("Fixable lint issues in %s:\n%s", name, fixable_summary)

    # 2. Surgeon approval gate
    if _surgeon_approval_required:
        repair_id = f"lint_{hashlib.md5(f'{name}:{time.time()}'.encode()).hexdigest()[:12]}"
        proposal = {
            "repair_id": repair_id,
            "service": name,
            "file": "(multiple — ruff --fix)",
            "container_path": "/app",
            "broken_code": fixable_summary,
            "fixed_code": "(ruff auto-fix)",
            "error_msg": fixable_summary,
            "method": "lint_autofix",
            "rules": _LINT_AUTOFIX_RULES,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        _surgeon_queue.append(proposal)
        log.info("Surgeon approval required — queued lint autofix %s for %s", repair_id, name)
        return False

    # 3. Apply fixes
    try:
        fix_cmd = [
            "docker", "exec", name, "python", "-m", "ruff", "check", "/app",
            "--select", _LINT_AUTOFIX_RULES, "--fix", "--no-cache",
        ]
        fix_res = subprocess.run(fix_cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        log.warning("Lint autofix apply failed for %s: %s", name, e)
        return False

    # 4. Validate — full F-rule check
    try:
        verify_cmd = [
            "docker", "exec", name, "python", "-m", "ruff", "check", "/app",
            "--select", "F", "--no-cache",
        ]
        verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        log.warning("Lint autofix verify failed for %s: %s", name, e)
        return False

    if verify.returncode == 0:
        log.info("LINT AUTOFIX succeeded for %s — all F-rules clean", name)
        _record_lint_autofix(name, fixable_summary)
        return True
    else:
        log.warning("LINT AUTOFIX partial for %s — safe rules fixed but unfixable errors remain:\n%s",
                     name, verify.stdout.strip()[:300])
        _record_lint_autofix(name, f"partial: {fixable_summary}")
        return True  # Safe fixes were applied; caller re-checks full lint


def _record_lint_autofix(name: str, summary: str):
    """Record a successful lint autofix in the remediation log."""
    entry = {
        "service": name,
        "time": datetime.now(timezone.utc).isoformat(),
        "action": "lint_autofix",
        "detail": summary[:300],
    }
    _remediation_log.append(entry)
    if len(_remediation_log) > 50:
        _remediation_log.pop(0)
    log.info("Recorded lint autofix remediation for %s", name)


# ---------------------------------------------------------------------------
# Self-Repair Loop (Chaos Monkey integration)
# ---------------------------------------------------------------------------

def _get_test_errors(name: str) -> str:
    """Run pytest and return the error output (last 3K chars) for LLM context."""
    try:
        cmd = ["docker", "exec", name, "python", "-m", "pytest", "/app",
               "-v", "--tb=short", "-m", "not integration"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = (result.stdout + "\n" + result.stderr).strip()
        return output[-3000:]  # Last 3K chars
    except subprocess.TimeoutExpired:
        return "pytest timed out (>180s)"
    except Exception as e:
        return f"failed to run pytest: {e}"


def _read_container_file(container: str, container_path: str) -> str | None:
    """Read a file from inside a running container via docker exec."""
    try:
        result = subprocess.run(
            ["docker", "exec", container, "cat", container_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        log.warning("Failed to read %s from %s: %s", container_path, container, e)
    return None


def _write_container_file(container: str, container_path: str, content: str) -> bool:
    """Write content to a file inside a running container via docker exec."""
    try:
        result = subprocess.run(
            ["docker", "exec", container, "python3", "-c",
             f"open({container_path!r}, 'w').write({content!r})"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception as e:
        log.warning("Failed to write %s in %s: %s", container_path, container, e)
        return False


def _notify_monkey_repair_success(service: str, file_path: str):
    """Notify monkey of successful LLM repair — full serenity points."""
    try:
        data = json.dumps({
            "category": "vital_recovery",
            "detail": f"Doctor LLM-repaired: {file_path} in {service}",
        }).encode()
        req = Request(
            f"{MONKEY_ENDPOINT}/serenity/record_recovery",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=5)
    except Exception:
        pass


def _notify_monkey_repair_restored(service: str, file_path: str):
    """Notify monkey of restore-only recovery — partial serenity credit."""
    try:
        data = json.dumps({
            "category": "standard_recovery",
            "detail": f"Doctor restored from production: {file_path} in {service}",
        }).encode()
        req = Request(
            f"{MONKEY_ENDPOINT}/serenity/record_recovery",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=5)
    except Exception:
        pass


def restore_from_production(candidate_path: str) -> dict:
    """Restore a candidate file from its production counterpart.

    Checks for divergence between production and git HEAD before restoring.
    Falls back to git HEAD if production file is corrupt.
    """
    # Derive production path: candidates/gaia-core/x.py -> gaia-core/x.py
    cand = Path(candidate_path)
    try:
        # Find the "candidates/" prefix and strip it
        parts = cand.parts
        candidates_idx = parts.index("candidates")
        prod_rel = Path(*parts[candidates_idx + 1:])
        prod_path = GAIA_PROJECT_ROOT / prod_rel
    except (ValueError, IndexError):
        return {"status": "error", "reason": f"cannot derive production path from {candidate_path}"}

    git_rel = str(prod_rel)

    # Divergence check: compare production file vs git HEAD
    try:
        git_content = subprocess.run(
            ["git", "-C", str(GAIA_PROJECT_ROOT), "show", f"HEAD:{git_rel}"],
            capture_output=True, text=True, timeout=10,
        )
        if git_content.returncode == 0:
            git_text = git_content.stdout
            if prod_path.exists():
                prod_text = prod_path.read_text()
                similarity = difflib.SequenceMatcher(None, prod_text, git_text).ratio()
                if similarity < 0.50:
                    log.error("DIVERGENCE ABORT: production vs git HEAD similarity=%.2f for %s", similarity, git_rel)
                    return {
                        "status": "divergence_abort",
                        "reason": f"production vs git HEAD similarity {similarity:.2f} < 0.50 — needs human review",
                        "file": git_rel,
                    }
        else:
            git_text = None
            log.warning("git show failed for %s — proceeding without divergence check", git_rel)
    except Exception as e:
        git_text = None
        log.warning("Git divergence check failed: %s — proceeding", e)

    # Try production file first
    source = "production"
    try:
        if prod_path.exists():
            content = prod_path.read_text()
            ast.parse(content)
        else:
            raise FileNotFoundError(f"production file not found: {prod_path}")
    except (SyntaxError, FileNotFoundError) as e:
        log.warning("Production file unusable (%s) — falling back to git HEAD", e)
        if git_text:
            try:
                ast.parse(git_text)
                content = git_text
                source = "git_head"
            except SyntaxError:
                return {"status": "error", "reason": "both production and git HEAD have syntax errors"}
        else:
            return {"status": "error", "reason": f"production file unusable and git HEAD unavailable: {e}"}

    # Derive the container name and container path for writing
    # candidate_path like: candidates/gaia-core/gaia_core/x.py
    # service name: gaia-core-candidate
    # container path: /app/gaia_core/x.py
    service_name = str(prod_rel).split("/")[0]  # e.g., "gaia-core"
    container_name = f"{service_name}-candidate"
    code_dir = SERVICE_CODE_DIRS.get(container_name)
    if code_dir:
        try:
            relative = cand.relative_to(code_dir)
            container_path = f"/app/{relative}"
        except ValueError:
            container_path = f"/app/{cand.name}"
    else:
        container_path = f"/app/{cand.name}"

    # Write via docker exec into the candidate container
    if _write_container_file(container_name, container_path, content):
        log.info("Restored %s from %s via docker exec", candidate_path, source)
        return {"status": "restored", "source": source, "file": git_rel}

    # Fallback: try structural repair endpoint
    try:
        repair_data = json.dumps({
            "service": "restore",
            "broken_code": content,
            "error_msg": "RESTORE_FROM_PRODUCTION — write this content as-is",
            "file_path": str(cand),
            "restore_mode": True,
        }).encode("utf-8")
        req = Request(
            f"{CORE_ENDPOINT}/api/repair/structural",
            data=repair_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                log.info("Restored %s from %s via structural endpoint", candidate_path, source)
                return {"status": "restored", "source": source, "file": git_rel}
    except Exception as e:
        log.error("All restore methods failed: %s", e)

    return {"status": "error", "reason": "all restore methods failed"}


def _demarker(code: str) -> str:
    """Deterministic removal of all CHAOS_MONKEY markers.

    Handles:
      - '# CHAOS_MONKEY_DISABLED: <original line>'  -> restore original line
      - '# CHAOS_MONKEY_REMOVED: <import line>'     -> restore import line
      - 'return None  # CHAOS_MONKEY_BREAK'         -> cannot restore (original unknown)
      - '<line>  # CHAOS_MONKEY_SWAP'                -> cannot restore (original unknown)
      - '_chaos_undefined_var = ... # CHAOS_MONKEY_INJECT' -> remove entire line

    Returns demarked code. Lines it can't deterministically fix are left
    for the LLM tier.
    """
    import re as _re
    out = []
    for line in code.split("\n"):
        # DISABLED / REMOVED: uncomment and strip marker
        m = _re.match(r'^(\s*)#\s*CHAOS_MONKEY_(?:DISABLED|REMOVED):\s?(.*)', line)
        if m:
            out.append(m.group(1) + m.group(2))
            continue
        # INJECT: remove marker line and the injected variable line
        if "CHAOS_MONKEY_INJECT" in line:
            continue
        if _re.match(r'^\s*_chaos_\w+\s*=', line):
            continue
        # BREAK / SWAP: strip the trailing marker comment but leave the (broken) code
        # — these need LLM help to restore the original logic
        stripped = _re.sub(r'\s*#\s*CHAOS_MONKEY_(?:BREAK|SWAP)\b.*$', '', line)
        out.append(stripped)
    return "\n".join(out)


def _validate_repair(service: str, container_path: str) -> dict:
    """Check if a repaired file is clean: no marker, lint passes, AST valid."""
    content = _read_container_file(service, container_path)
    if not content:
        return {"valid": False, "reason": "cannot read file"}
    has_marker = "CHAOS_MONKEY" in content
    if has_marker:
        return {"valid": False, "reason": "marker still present"}
    # AST check
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {"valid": False, "reason": f"syntax error: {e}"}
    # Lint check — just this file
    try:
        lint_cmd = ["docker", "exec", service, "python", "-m", "ruff", "check",
                    container_path, "--select", "F821,F811", "--no-cache"]
        lint_res = subprocess.run(lint_cmd, capture_output=True, text=True, timeout=30)
        if lint_res.returncode != 0:
            return {"valid": False, "reason": f"lint errors: {lint_res.stdout[:200]}"}
    except Exception as e:
        return {"valid": False, "reason": f"lint check failed: {e}"}
    return {"valid": True}


def repair_candidate_file(service: str, file_path: str, max_retries: int = 3) -> dict:
    """Two-tier repair of a broken candidate file.

    Tier 1 (deterministic): Regex demarker strips CHAOS_MONKEY comments.
      Handles DISABLED, REMOVED, INJECT markers instantly.
    Tier 2 (LLM): For faults the demarker can't fix (BREAK, SWAP, multi-fault
      residue), escalates to the structural surgeon with error context.

    After all retries -> restore from production.
    """
    attempts = []

    # Derive container path
    code_dir = SERVICE_CODE_DIRS.get(service)
    full_path = GAIA_PROJECT_ROOT / file_path
    if code_dir:
        try:
            relative = full_path.relative_to(code_dir)
            container_path = f"/app/{relative}"
        except ValueError:
            container_path = f"/app/{full_path.name}"
    else:
        container_path = f"/app/{full_path.name}"

    # ── Tier 0: Check if already clean (race guard) ─────────────────────
    broken_code = _read_container_file(service, container_path)
    if broken_code is None:
        try:
            broken_code = full_path.read_text()
        except Exception as e:
            return {"status": "error", "reason": f"cannot read broken file: {e}"}

    if "CHAOS_MONKEY" not in broken_code:
        check = _validate_repair(service, container_path)
        if check["valid"]:
            log.info("File already clean (no markers, lint OK) — likely fixed by parallel path: %s", file_path)
            return {"status": "already_clean", "file": file_path, "service": service}

    # ── Tier 1: Deterministic demarker ──────────────────────────────────
    if "CHAOS_MONKEY" in broken_code:
        demarked = _demarker(broken_code)
        if _write_container_file(service, container_path, demarked):
            log.info("Tier 1 demarker applied to %s:%s", service, container_path)
            check = _validate_repair(service, container_path)
            if check["valid"]:
                log.info("TIER 1 SUCCESS (demarker) for %s — clean on first pass", file_path)
                _notify_monkey_repair_success(service, file_path)
                return {
                    "status": "repaired",
                    "method": "demarker",
                    "attempts": [{"attempt": 0, "status": "demarker_success"}],
                    "file": file_path,
                    "service": service,
                }
            else:
                log.info("Demarker applied but validation failed (%s) — escalating to Tier 2",
                         check["reason"])
                attempts.append({"attempt": 0, "status": "demarker_partial", "reason": check["reason"]})
        else:
            log.warning("Demarker write failed — escalating to Tier 2")

    # ── Tier 2: LLM structural surgeon ──────────────────────────────────
    escalation_prompts = [
        "Fix the broken code. Lines containing 'CHAOS_MONKEY' markers are injected faults — "
        "restore the original logic. For 'return None  # CHAOS_MONKEY_BREAK', reconstruct "
        "the correct return value. For '# CHAOS_MONKEY_SWAP', swap the arguments back.",
        "Previous fix attempt failed. The code still has issues. Focus on the specific "
        "error lines. Be precise — only change what's broken.",
        "FINAL ATTEMPT: Be extremely conservative. Fix only the lines causing errors. "
        "Do not add new code, imports, or refactor anything.",
    ]

    for attempt in range(1, max_retries + 1):
        log.info("Tier 2 LLM repair attempt %d/%d for %s in %s", attempt, max_retries, file_path, service)

        test_errors = _get_test_errors(service)
        prompt = escalation_prompts[min(attempt - 1, len(escalation_prompts) - 1)]

        current_code = _read_container_file(service, container_path)
        if current_code is None:
            attempts.append({"attempt": attempt, "status": "error", "reason": "cannot read file"})
            continue

        try:
            repair_data = json.dumps({
                "service": service,
                "broken_code": current_code,
                "error_msg": f"{prompt}\n\nTest/lint errors:\n{test_errors}",
            }).encode("utf-8")
            req = Request(
                f"{CORE_ENDPOINT}/api/repair/structural",
                data=repair_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
        except Exception as e:
            attempts.append({"attempt": attempt, "status": "error", "reason": f"repair endpoint failed: {e}"})
            continue

        fixed_code = result.get("fixed_code")
        if not fixed_code:
            attempts.append({"attempt": attempt, "status": "repair_rejected", "detail": str(result)[:200]})
            continue

        # ── Surgeon approval gate ────────────────────────────────
        if _surgeon_approval_required:
            repair_id = f"repair_{hashlib.md5(f'{service}:{file_path}:{time.time()}'.encode()).hexdigest()[:12]}"
            test_errors = _get_test_errors(service)
            proposal = {
                "repair_id": repair_id,
                "service": service,
                "file": file_path,
                "container_path": container_path,
                "broken_code": current_code,
                "fixed_code": fixed_code,
                "error_msg": test_errors[:2000] if test_errors else "",
                "method": "llm_surgeon",
                "attempt": attempt,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
            }
            _surgeon_queue.append(proposal)
            log.info("Surgeon approval required — queued repair %s for %s:%s", repair_id, service, file_path)
            return {
                "status": "pending_approval",
                "repair_id": repair_id,
                "attempts": attempts + [{"attempt": attempt, "status": "queued_for_approval"}],
                "file": file_path,
                "service": service,
            }

        if not _write_container_file(service, container_path, fixed_code):
            attempts.append({"attempt": attempt, "status": "write_failed"})
            continue
        log.info("Wrote LLM fix to %s:%s (attempt %d)", service, container_path, attempt)

        check = _validate_repair(service, container_path)
        if check["valid"]:
            log.info("TIER 2 SUCCESS (LLM) on attempt %d for %s", attempt, file_path)
            _notify_monkey_repair_success(service, file_path)
            return {
                "status": "repaired",
                "method": "llm_surgeon",
                "attempts": attempts + [{"attempt": attempt, "status": "success"}],
                "file": file_path,
                "service": service,
            }
        else:
            log.info("LLM fix validation failed (attempt %d): %s", attempt, check["reason"])
            attempts.append({"attempt": attempt, "status": "fix_incomplete", "reason": check["reason"]})

    # All retries exhausted — restore from production
    log.warning("All %d repair attempts failed for %s — restoring from production", max_retries, file_path)
    restore_result = restore_from_production(str(GAIA_PROJECT_ROOT / file_path))

    if restore_result.get("status") == "restored":
        _notify_monkey_repair_restored(service, file_path)
        # Restart container after restore
        try:
            subprocess.run(["docker", "restart", service], capture_output=True, text=True, timeout=30)
        except Exception:
            pass

    return {
        "status": "restored" if restore_result.get("status") == "restored" else "failed",
        "attempts": attempts,
        "restore": restore_result,
        "file": file_path,
        "service": service,
    }


def _handle_chaos_notification(data: dict):
    """Handle notification from monkey about injected chaos fault.

    Sleeps briefly for file write to settle, runs tests, then enters repair loop if needed.
    """
    service = data.get("service", "")
    file_path = data.get("file", "")
    fault = data.get("fault", "unknown")
    difficulty = data.get("difficulty", 1)

    log.info("Chaos notification received: %s in %s (difficulty=%d, fault=%s)",
             file_path, service, difficulty, fault)

    # Brief sleep for file write to settle on disk
    time.sleep(3)

    # Skip full test suite — we know a fault was injected, go straight to repair
    log.info("Entering self-repair loop for %s (fault=%s, difficulty=%d)", service, fault, difficulty)
    result = repair_candidate_file(service, file_path)
    log.info("Self-repair result for %s: %s", file_path, result.get("status", "unknown"))


def _repair_divergent_candidate(name: str):
    """Called from audit_code() when a candidate file fails tests during meditation.

    Only runs repair if we find a file with a CHAOS_MONKEY marker, confirming
    this is an injected fault rather than a pre-existing test failure.
    """
    code_dir = SERVICE_CODE_DIRS.get(name)
    if not code_dir or not code_dir.exists():
        return

    # Find recently modified .py files (within last 60s) that have chaos markers
    now = time.time()
    chaos_files = []
    for p in code_dir.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        try:
            if now - p.stat().st_mtime < 60:
                # Check inside the container for chaos markers
                rel = p.relative_to(code_dir)
                container_path = f"/app/{rel}"
                content = _read_container_file(name, container_path)
                if content and "CHAOS_MONKEY" in content:
                    chaos_files.append(p)
        except (OSError, ValueError):
            continue

    if not chaos_files:
        log.debug("No chaos-marked files found for %s — skipping repair (pre-existing test failures)", name)
        return

    for changed in chaos_files:
        try:
            rel_path = str(changed.relative_to(GAIA_PROJECT_ROOT))
        except ValueError:
            continue
        log.info("Attempting organic repair for %s in %s", rel_path, name)
        result = repair_candidate_file(name, rel_path)
        log.info("Organic repair result: %s", result.get("status", "unknown"))


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
    "vLLM returned 503",
    "Cannot reach vLLM server",
    "model not loaded",
    "GPU zombie detected",
    "GAIA-CORE-075",
    "GAIA-CORE-076",
    "GAIA-CORE-077",
    "GAIA-CORE-078",
    "GAIA-CORE-080",
    "Inference stream interrupted",
    "Model swap failed",
    "Loop detector",
    "Plan generation failed",
    "GAIA-WEB-040",
    "GAIA-WEB-045",
    "GAIA-WEB-050",
    "GAIA-WEB-055",
    "GAIA-WEB-060",
    "Discord message send failed",
    "Voice processing loop",
    "Speech playback failed",
    "Transcribe request failed",
    "inference degraded",
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

    # ── OOM Auto-Resolution: negotiate VRAM via Orchestrator ──
    # When a GPU service hits OOM, Doctor presses the orchestrator buttons
    # to free VRAM. Glass stays on — no manual intervention needed.
    if pattern in ("OOM", "CUDA out of memory", "torch.cuda.OutOfMemoryError"):
        _attempt_oom_resolution(service, line)

    # ── Audio Auto-Resolution: restart gaia-audio on voice/STT failures ──
    if pattern in ("GAIA-WEB-050", "GAIA-WEB-055", "GAIA-WEB-060",
                    "Voice processing loop", "Speech playback failed",
                    "Transcribe request failed"):
        _attempt_audio_restart(service, line, pattern)

    # ── Inference Auto-Resolution: request Prime wake on stream failures ──
    if pattern in ("GAIA-CORE-075", "GAIA-CORE-080",
                    "Inference stream interrupted"):
        _attempt_inference_recovery(service, line, pattern)


# ── OOM Resolution ────────────────────────────────────────────────────────

_oom_cooldown: dict = {}  # service → last_attempt_time
OOM_COOLDOWN_SECONDS = 60  # Don't spam orchestrator

def _attempt_oom_resolution(service: str, error_line: str):
    """Ask Orchestrator to free VRAM when a service hits OOM.

    Resolution strategy (escalating):
    1. Backoff Nano to CPU (frees ~1.5 GB)
    2. If still insufficient, request full GPU release (frees Core too)
    3. Emit CodeMind detection for investigation

    The Pinball Machine pattern: Doctor detects the problem, presses
    Orchestrator's buttons, and watches the resolution through the glass.
    """
    now = time.monotonic()
    last = _oom_cooldown.get(service, 0)
    if now - last < OOM_COOLDOWN_SECONDS:
        log.debug("OOM resolution for %s on cooldown (%ds remaining)",
                  service, int(OOM_COOLDOWN_SECONDS - (now - last)))
        return
    _oom_cooldown[service] = now

    log.warning("OOM resolution triggered for %s — negotiating VRAM via Orchestrator", service)

    # Step 1: Check current GPU state
    try:
        req = Request(f"{ORCHESTRATOR_ENDPOINT}/watch/state", method="GET")
        with urlopen(req, timeout=5) as resp:
            watch_state = json.loads(resp.read().decode())
        tiers = watch_state.get("tiers", {})
        log.info("OOM: Current GPU state — %s",
                 {k: f"{v.get('device')}({v.get('vram_mb')}MB)" for k, v in tiers.items()})
    except Exception as e:
        log.warning("OOM: Could not read Orchestrator watch state: %s", e)
        return

    # Step 2: Identify what can be offloaded
    nano_on_gpu = tiers.get("nano", {}).get("device", "") != "unloaded"
    core_on_gpu = tiers.get("core", {}).get("device", "") != "unloaded"

    resolution_log = []

    # Step 2a: Backoff Nano first (smallest, safest)
    if nano_on_gpu:
        try:
            log.info("OOM: Requesting Nano backoff to CPU...")
            data = json.dumps({"reason": f"oom_resolution:{service}"}).encode()
            req = Request(
                f"{ORCHESTRATOR_ENDPOINT}/nano/backoff",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            if result.get("ok"):
                resolution_log.append("nano_backoff:success")
                log.info("OOM: Nano backed off to CPU — freed ~1.5GB VRAM")
            else:
                resolution_log.append(f"nano_backoff:failed:{result.get('error', 'unknown')}")
        except Exception as e:
            resolution_log.append(f"nano_backoff:error:{e}")
            log.warning("OOM: Nano backoff failed: %s", e)

    # Step 2b: If the OOM service is Prime and Core is also on GPU, release Core
    if service in ("gaia-prime", "gaia-study") and core_on_gpu:
        try:
            log.info("OOM: Requesting GPU sleep (release Core)...")
            data = json.dumps({"reason": f"oom_resolution:{service}"}).encode()
            req = Request(
                f"{ORCHESTRATOR_ENDPOINT}/gpu/sleep",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            if result.get("ok"):
                resolution_log.append("gpu_sleep:success")
                log.info("OOM: GPU sleep — Core released VRAM")
            else:
                resolution_log.append(f"gpu_sleep:failed:{result.get('error', 'unknown')}")
        except Exception as e:
            resolution_log.append(f"gpu_sleep:error:{e}")
            log.warning("OOM: GPU sleep failed: %s", e)

    # Step 3: Emit to CodeMind detect queue for deeper investigation
    try:
        _emit_codemind_oom(service, error_line, resolution_log)
    except Exception:
        pass

    # Step 4: Write resolution to shared state for dashboard
    _write_oom_resolution(service, error_line, resolution_log)

    log.info("OOM resolution complete for %s: %s", service, resolution_log)


def _emit_codemind_oom(service: str, error_line: str, resolution_log: list):
    """Emit OOM event to CodeMind detect queue for investigation."""
    try:
        queue_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "codemind" / "detect_queue.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "immune_irritation",
            "issue_type": "oom_error",
            "file_path": "",
            "description": f"OOM in {service}: {error_line[:200]}. Resolution: {resolution_log}",
            "severity": "warn",
            "priority": 2,
            "metadata": {"service": service, "resolution": resolution_log},
        }
        with open(queue_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.debug("Failed to emit CodeMind OOM detection: %s", e)


def _write_oom_resolution(service: str, error_line: str, resolution_log: list):
    """Write OOM resolution record to shared state for dashboard visibility."""
    try:
        state_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "oom_resolutions.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {"history": []}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        state["history"].insert(0, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "error": error_line[:300],
            "resolution": resolution_log,
        })
        state["history"] = state["history"][:20]  # Keep last 20
        state["last"] = state["history"][0] if state["history"] else None

        state_path.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.debug("Failed to write OOM resolution state: %s", e)


# ── Audio Auto-Resolution ─────────────────────────────────────────────────

_audio_restart_cooldown: dict = {}
AUDIO_RESTART_COOLDOWN = 120  # seconds — longer than OOM since restarts are heavier

def _attempt_audio_restart(service: str, error_line: str, pattern: str):
    """Restart gaia-audio when voice/STT failures are detected.

    Pinball Machine: Doctor sees voice failures in gaia-web logs,
    presses the gaia-audio restart button, watches through the glass.
    """
    now = time.monotonic()
    last = _audio_restart_cooldown.get("gaia-audio", 0)
    if now - last < AUDIO_RESTART_COOLDOWN:
        log.debug("Audio restart on cooldown (%ds remaining)",
                  int(AUDIO_RESTART_COOLDOWN - (now - last)))
        return
    _audio_restart_cooldown["gaia-audio"] = now

    log.warning("Audio restart triggered by %s in %s — restarting gaia-audio", pattern, service)

    try:
        docker_restart("gaia-audio")
    except Exception as e:
        log.warning("Audio restart failed: %s", e)

    # Emit to CodeMind for pattern analysis
    try:
        queue_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "codemind" / "detect_queue.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "immune_irritation",
            "issue_type": "audio_failure",
            "file_path": "",
            "description": f"Audio restart in {service} due to {pattern}: {error_line[:200]}",
            "severity": "warn",
            "priority": 2,
            "metadata": {"service": service, "pattern": pattern, "action": "restart_gaia-audio"},
        }
        with open(queue_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.debug("Failed to emit CodeMind audio detection: %s", e)


# ── Inference Auto-Resolution ─────────────────────────────────────────────

_inference_recovery_cooldown: dict = {}
INFERENCE_RECOVERY_COOLDOWN = 90  # seconds

def _attempt_inference_recovery(service: str, error_line: str, pattern: str):
    """Request Prime wake when inference stream failures are detected.

    When gaia-core logs a stream interruption, the most common cause is
    Prime being in standby or crashed. Doctor asks Orchestrator to wake it.
    """
    now = time.monotonic()
    last = _inference_recovery_cooldown.get("inference", 0)
    if now - last < INFERENCE_RECOVERY_COOLDOWN:
        log.debug("Inference recovery on cooldown (%ds remaining)",
                  int(INFERENCE_RECOVERY_COOLDOWN - (now - last)))
        return
    _inference_recovery_cooldown["inference"] = now

    log.warning("Inference recovery triggered by %s — requesting Prime wake via Orchestrator", pattern)

    # Step 1: Check if Prime is actually down
    try:
        req = Request("http://gaia-prime:7777/health", method="GET")
        with urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                health = json.loads(resp.read().decode())
                if health.get("model_loaded"):
                    log.info("Inference recovery: Prime is healthy and loaded — stream error was transient")
                    return
                log.info("Inference recovery: Prime healthy but model not loaded — requesting wake")
    except Exception:
        log.info("Inference recovery: Prime unreachable — requesting wake")

    # Step 2: Try to load Prime's model directly
    # The orchestrator's /watch/focus is broken (can't docker compose from
    # inside a container). gaia-prime accepts POST /model/load directly.
    # If VRAM is insufficient, this will fail — that's OK, Core handles it.
    try:
        prime_endpoint = os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777")
        data = json.dumps({
            "model_path": os.environ.get(
                "PRIME_MODEL_PATH",
                "/warm_pool/Huihui-Qwen3-8B-GAIA-Prime-adaptive",
            ),
        }).encode()
        req = Request(
            f"{prime_endpoint}/model/load",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        if result.get("ok") or result.get("model_loaded"):
            log.info("Inference recovery: Prime model load accepted")
        else:
            log.warning("Inference recovery: Prime model load response: %s", result)
    except Exception as e:
        log.warning("Inference recovery: direct Prime model load failed: %s", e)
        # Fallback: ask orchestrator to wake (marks GPU available at least)
        try:
            wake_data = json.dumps({"reason": f"inference_recovery:{pattern}"}).encode()
            req = Request(
                f"{ORCHESTRATOR_ENDPOINT}/gpu/wake",
                data=wake_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            log.info("Inference recovery: fallback /gpu/wake result: %s", result)
        except Exception:
            pass
    except Exception as e:
        log.warning("Inference recovery: Could not reach Orchestrator: %s", e)

    # Step 3: Emit to CodeMind
    try:
        queue_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "codemind" / "detect_queue.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "immune_irritation",
            "issue_type": "inference_failure",
            "file_path": "",
            "description": f"Inference recovery in {service} due to {pattern}: {error_line[:200]}",
            "severity": "warn",
            "priority": 2,
            "metadata": {"service": service, "pattern": pattern, "action": "wake_prime"},
        }
        with open(queue_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.debug("Failed to emit CodeMind inference detection: %s", e)


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------

def check_health(name: str, url: str) -> bool:
    """HTTP GET the health endpoint. Returns True if healthy.

    For gaia-core: also checks the inference_ok field — a "degraded"
    service (API alive but inference dead) is treated as unhealthy.
    """
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False
            # For services that report inference health, check it
            try:
                body = json.loads(resp.read().decode())
                if body.get("status") == "degraded":
                    log.warning(
                        "%s is degraded: inference_ok=%s detail=%s",
                        name, body.get("inference_ok"), body.get("inference_detail", ""),
                    )
                    _record_irritation(name, f"inference degraded: {body.get('inference_detail', '')}", "inference degraded")
                    return False
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Non-JSON health response — just check HTTP status
            return True
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


def check_container_naming(name: str) -> bool:
    """Check if a container's actual name matches the expected service name.

    Docker compose can produce mangled names (e.g. 'e682e64949a5_gaia-core')
    when a previous recreate failed or the container was orphaned.  If detected,
    force-recreate via compose to fix the naming.

    Returns True if naming is correct (or service not found), False if a
    fix was attempted.
    """
    try:
        # Find containers matching the service name (partial match)
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return True  # container not found — nothing to fix

        for container_name in result.stdout.strip().split("\n"):
            container_name = container_name.strip()
            if not container_name:
                continue
            # Skip candidate containers when checking production service
            if "candidate" in container_name and "candidate" not in name:
                continue
            # The expected name should match exactly
            if container_name != name:
                log.warning(
                    "NAMING ANOMALY: container '%s' should be '%s' — force-recreating",
                    container_name, name,
                )
                # Stop and remove the misnamed container first
                subprocess.run(
                    ["docker", "stop", container_name],
                    capture_output=True, text=True, timeout=30,
                )
                subprocess.run(
                    ["docker", "rm", container_name],
                    capture_output=True, text=True, timeout=10,
                )
                # Recreate with correct naming via compose
                docker_restart(name)
                return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        log.debug("Container naming check failed for %s", name, exc_info=True)
    return True


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

    if is_maintenance_active():
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
            log.error("[GAIA-DOCTOR-001] Failed to restart %s: %s", name, result.stderr.strip()[:200])
            return False
    except subprocess.TimeoutExpired:
        log.error("[GAIA-DOCTOR-010] Restart of %s timed out (>120s)", name)
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
    if DRY_RUN:
        log.info("[DRY_RUN] Would restart %s — skipping (candidate mode)", name)
        return False
    if is_maintenance_active():
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
    if DRY_RUN:
        log.info("[DRY_RUN] Would evaluate %d divergent files for promotion — skipping (candidate mode)", len(divergent_files))
        return False
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


# ---------------------------------------------------------------------------
# Cognitive Monitor — lightweight heartbeat via GAIA Engine
# ---------------------------------------------------------------------------
# Instead of routing through the full 20-stage cognitive pipeline,
# this pings the GAIA Engine directly: one identity question,
# one response check, one polygraph reading. ~400ms, ~12 tokens.

# Engine endpoints for direct tier probes
_CORE_ENGINE = os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092")
_NANO_ENGINE = os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080")

# Expected identity neurons per tier (from SAE atlas)
_IDENTITY_NEURONS = {
    "core": {"layer": "layer_23", "neuron": 1201, "min_strength": 1.0},
    "nano": {"layer": "layer_23", "neuron": 0, "min_strength": 0.5},
}


def _run_cognitive_monitor():
    """Lightweight cognitive heartbeat — identity check + polygraph validation.

    For each active tier:
    1. Ask "Who are you?" via the GAIA Engine (~12 tokens)
    2. Check response contains "GAIA" (string match)
    3. Check polygraph: identity neuron active? (SAE validation)
    Total: ~400ms per tier, no full pipeline needed.
    """
    global _cognitive_monitor_failures, _cognitive_monitor_last_result

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
        "tiers": {},
        "consecutive_failures": _cognitive_monitor_failures,
        "interval_seconds": _cognitive_monitor_interval,
    }

    try:
        # Skip if maintenance
        if is_maintenance_active():
            result["status"] = "skipped"
            result["reason"] = "maintenance_mode"
            _cognitive_monitor_last_result = result
            _persist_monitor_result(result)
            return

        tiers_checked = 0
        tiers_passed = 0

        for tier_name, endpoint in [("core", _CORE_ENGINE), ("nano", _NANO_ENGINE)]:
            tier_result = {"status": "unknown"}
            try:
                # Step 1: Identity probe — direct to engine, ~12 tokens
                payload = json.dumps({
                    "messages": [
                        {"role": "system", "content": "You are GAIA."},
                        {"role": "user", "content": "Who are you?"},
                    ],
                    "max_tokens": 30,
                    "temperature": 0.0,
                }).encode()
                req = Request(
                    f"{endpoint}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                resp = urlopen(req, timeout=10)
                data = json.loads(resp.read().decode("utf-8"))
                response = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Step 2: Identity check — does response contain "GAIA"?
                identity_present = "gaia" in response.lower()

                # Step 3: Polygraph check — is identity neuron firing?
                polygraph_ok = False
                try:
                    poly_resp = urlopen(f"{endpoint}/polygraph/activations", timeout=5)
                    poly_data = json.loads(poly_resp.read().decode("utf-8"))
                    activations = poly_data.get("activations", {})

                    expected = _IDENTITY_NEURONS.get(tier_name, {})
                    layer = expected.get("layer", "layer_23")
                    neuron = expected.get("neuron", -1)
                    min_str = expected.get("min_strength", 0.5)

                    if layer in activations:
                        top_indices = activations[layer].get("top_5_indices", [])
                        top_values = activations[layer].get("top_5_values", [])
                        if neuron in top_indices:
                            idx = top_indices.index(neuron)
                            strength = top_values[idx] if idx < len(top_values) else 0
                            polygraph_ok = strength >= min_str
                            tier_result["identity_neuron"] = neuron
                            tier_result["identity_strength"] = round(strength, 3)
                except Exception as e:
                    tier_result["polygraph_error"] = str(e)[:60]

                tier_result["identity_present"] = identity_present
                tier_result["polygraph_ok"] = polygraph_ok
                tier_result["response"] = response[:80]

                # Step 4: Time awareness probe — ask time, compare to actual
                time_ok = False
                try:
                    time_payload = json.dumps({
                        "messages": [
                            {"role": "system", "content": "You are GAIA. Read the Clock line for the current time."},
                            {"role": "user", "content": "What time is it?"},
                        ],
                        "max_tokens": 40,
                        "temperature": 0.0,
                    }).encode()
                    time_req = Request(
                        f"{endpoint}/v1/chat/completions",
                        data=time_payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    time_resp = urlopen(time_req, timeout=10)
                    time_data = json.loads(time_resp.read().decode("utf-8"))
                    time_response = time_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # Extract time and compare to actual (±10 min tolerance)
                    import re as _re
                    now_utc = datetime.now(timezone.utc)
                    pacific = timezone(timedelta(hours=int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))))
                    now_local = now_utc.astimezone(pacific)

                    # Try 12h format first, then 24h
                    m12 = _re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)', time_response.lower())
                    m24 = _re.search(r'(\d{1,2}):(\d{2})', time_response)
                    ext_h, ext_m = None, None
                    if m12:
                        ext_h, ext_m = int(m12.group(1)), int(m12.group(2))
                        if m12.group(3) == "pm" and ext_h != 12: ext_h += 12
                        elif m12.group(3) == "am" and ext_h == 12: ext_h = 0
                    elif m24:
                        ext_h, ext_m = int(m24.group(1)), int(m24.group(2))

                    if ext_h is not None:
                        ext_total = ext_h * 60 + ext_m
                        # Check against both UTC and local
                        best_diff = 9999
                        for actual in [now_utc, now_local]:
                            act_total = actual.hour * 60 + actual.minute
                            diff = abs(ext_total - act_total)
                            if diff > 720: diff = 1440 - diff
                            best_diff = min(best_diff, diff)
                        time_ok = best_diff <= 10
                        tier_result["time_accuracy_min"] = best_diff
                    tier_result["time_response"] = time_response[:80]
                except Exception as e:
                    tier_result["time_error"] = str(e)[:60]
                tier_result["time_ok"] = time_ok

                if identity_present:
                    tier_result["status"] = "pass"
                    tiers_passed += 1
                else:
                    tier_result["status"] = "fail"
                    tier_result["error"] = "identity_missing"

                tiers_checked += 1

            except Exception as e:
                tier_result["status"] = "unreachable"
                tier_result["error"] = str(e)[:100]

            result["tiers"][tier_name] = tier_result

        # Overall status
        if tiers_checked == 0:
            result["status"] = "fail"
            result["error"] = "no_tiers_reachable"
            _handle_monitor_failure(result)
        elif tiers_passed == tiers_checked:
            result["status"] = "pass"
            _handle_monitor_success(result)
        elif tiers_passed > 0:
            result["status"] = "degraded"
            _handle_monitor_success(result)  # Partial pass still clears alarm
        else:
            result["status"] = "fail"
            result["error"] = "all_tiers_failed_identity"
            _handle_monitor_failure(result)

    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)[:200]
        _handle_monitor_failure(result)


def _handle_monitor_failure(result: dict):
    """Increment failures, alarm after threshold."""
    global _cognitive_monitor_failures, _cognitive_monitor_last_result
    _cognitive_monitor_failures += 1
    result["consecutive_failures"] = _cognitive_monitor_failures
    _cognitive_monitor_last_result = result
    _persist_monitor_result(result)

    if _cognitive_monitor_failures >= FAILURE_THRESHOLD:
        if "cognitive_monitor" not in _alarmed_services:
            log.warning("COGNITIVE MONITOR ALARM: %d consecutive failures — %s",
                        _cognitive_monitor_failures, result.get("error", "unknown"))
            _alarmed_services.add("cognitive_monitor")
            _active_alarms.append({
                "service": "cognitive_monitor",
                "time": datetime.now(timezone.utc).isoformat(),
                "reason": f"cognitive probe failed {_cognitive_monitor_failures}x: {result.get('error', 'unknown')}",
            })
    else:
        log.info("Cognitive monitor probe failed (%d/%d): %s",
                 _cognitive_monitor_failures, FAILURE_THRESHOLD, result.get("error", "unknown"))


def _handle_monitor_success(result: dict):
    """Clear failures and alarm on success."""
    global _cognitive_monitor_failures, _cognitive_monitor_last_result
    if _cognitive_monitor_failures > 0:
        log.info("Cognitive monitor recovered after %d failures", _cognitive_monitor_failures)
    _cognitive_monitor_failures = 0
    result["consecutive_failures"] = 0
    _cognitive_monitor_last_result = result
    _alarmed_services.discard("cognitive_monitor")
    _persist_monitor_result(result)


def _persist_monitor_result(result: dict):
    """Write monitor result to shared JSON file."""
    try:
        COGNITIVE_MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        COGNITIVE_MONITOR_FILE.write_text(json.dumps(result, indent=2))
    except Exception:
        log.debug("Failed to persist cognitive monitor result", exc_info=True)


# ── GPU Zombie Cleanup ─────────────────────────────────────────────────────

_zombie_cleanup_cooldown = 0.0

def _cleanup_gpu_zombies():
    """Detect and kill GPU processes not owned by any running container.

    Runs nvidia-smi to find GPU consumers, cross-references with running
    containers, and kills any orphans. This prevents VRAM leaks from
    crashed or stopped containers that left GPU processes behind.
    """
    global _zombie_cleanup_cooldown
    now = time.monotonic()
    if now - _zombie_cleanup_cooldown < 120:  # Check every 2 min max
        return
    _zombie_cleanup_cooldown = now

    try:
        # Get GPU processes
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if gpu_result.returncode != 0:
            return
        gpu_pids = [int(p.strip()) for p in gpu_result.stdout.strip().split("\n") if p.strip()]
        if not gpu_pids:
            return

        # Get running container PIDs
        container_pids = set()
        ps_result = subprocess.run(
            ["docker", "ps", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        for cid in ps_result.stdout.strip().split("\n"):
            cid = cid.strip()
            if not cid:
                continue
            try:
                inspect = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Pid}}", cid],
                    capture_output=True, text=True, timeout=5,
                )
                pid = int(inspect.stdout.strip())
                if pid > 0:
                    container_pids.add(pid)
            except (ValueError, subprocess.TimeoutExpired):
                pass

        # Find orphans: GPU PIDs not matching any container's main PID
        # Check if the GPU process's cgroup matches a running container
        for gpu_pid in gpu_pids:
            try:
                cgroup_path = f"/proc/{gpu_pid}/cgroup"
                with open(cgroup_path, "r") as f:
                    cgroup = f.read()
                # Extract docker container ID from cgroup
                if "docker-" in cgroup:
                    container_hash = cgroup.split("docker-")[-1].split(".scope")[0][:12]
                    # Check if this container is still running
                    check = subprocess.run(
                        ["docker", "inspect", "--format", "{{.State.Running}}", container_hash],
                        capture_output=True, text=True, timeout=5,
                    )
                    if check.stdout.strip() == "true":
                        continue  # Container is running, GPU process is legitimate

                # Orphan detected — container stopped but GPU process remains
                log.warning("GPU zombie detected: PID %d (container stopped). Killing.", gpu_pid)
                subprocess.run(["docker", "rm", "-f", container_hash], capture_output=True, timeout=5)
                os.kill(gpu_pid, 9)
                _record_irritation("gpu", f"Killed GPU zombie PID {gpu_pid}", "GPUZombie")

                # Notify orchestrator
                try:
                    data = json.dumps({"reason": f"zombie_cleanup:pid_{gpu_pid}"}).encode()
                    req = Request(
                        f"{ORCHESTRATOR_ENDPOINT}/gpu/release",
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urlopen(req, timeout=5)
                except Exception:
                    pass

            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass  # Process already gone
            except Exception as e:
                log.debug("Zombie check failed for PID %d: %s", gpu_pid, e)

    except FileNotFoundError:
        pass  # nvidia-smi not available
    except Exception:
        log.debug("GPU zombie cleanup error", exc_info=True)


# ── VRAM Reconciliation ────────────────────────────────────────────────────

_vram_reconcile_cooldown = 0.0

def _reconcile_vram():
    """Compare orchestrator's expected GPU state vs actual nvidia-smi usage.

    If the orchestrator thinks the GPU is idle but nvidia-smi shows heavy usage
    (or vice versa), log the discrepancy. This catches state drift between
    the orchestrator's model and reality.
    """
    global _vram_reconcile_cooldown
    now = time.monotonic()
    if now - _vram_reconcile_cooldown < 300:  # Check every 5 min
        return
    _vram_reconcile_cooldown = now

    try:
        # Get orchestrator's view
        req = Request(f"{ORCHESTRATOR_ENDPOINT}/watch/state", method="GET")
        with urlopen(req, timeout=5) as resp:
            watch = json.loads(resp.read().decode())

        expected_vram = sum(
            t.get("vram_mb", 0)
            for t in watch.get("tiers", {}).values()
        )

        # Get actual GPU usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return
        actual_vram = int(result.stdout.strip())

        # Compare — allow 2GB tolerance for overhead, KV cache, etc.
        drift = abs(actual_vram - expected_vram)
        if drift > 2000:  # More than 2GB discrepancy
            log.warning(
                "VRAM drift: orchestrator expects %dMB, nvidia-smi shows %dMB (drift %dMB)",
                expected_vram, actual_vram, drift,
            )
            _record_irritation(
                "gpu", f"VRAM drift: expected {expected_vram}MB, actual {actual_vram}MB",
                "VRAMDrift",
            )

    except (URLError, OSError, TimeoutError):
        pass
    except FileNotFoundError:
        pass  # nvidia-smi not available
    except Exception:
        log.debug("VRAM reconciliation error", exc_info=True)


# ── Prime Model Health ─────────────────────────────────────────────────────

_prime_model_warned = False

def _check_prime_model_health():
    """Monitor Prime's model state for dashboard visibility.

    Prime in standby is a VALID state — the GPU can't hold all tiers.
    Doctor observes and reports but does NOT auto-wake. Waking Prime is
    Core's job when a real request needs it (via _request_prime_load
    in vllm_remote_model.py on 503 retry).
    """
    global _prime_model_warned
    try:
        req = Request("http://gaia-prime:7777/health", method="GET")
        with urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read().decode())

        if health.get("model_loaded"):
            if _prime_model_warned:
                log.info("Prime model loaded — active")
                _prime_model_warned = False
        else:
            if not _prime_model_warned:
                log.info("Prime in standby (model_loaded=false) — valid state, Core will wake on demand")
                _prime_model_warned = True
            # NOT an irritation — standby is normal. Just track for dashboard.

    except (URLError, OSError, TimeoutError):
        pass


_audio_stt_warned = False  # Track if we've already warned about STT being down

def _check_audio_sensory_health():
    """Deep health check for gaia-audio: verify STT model is loaded.

    The container can be healthy (HTTP 200) but with STT unloaded (no model).
    This checks /status and attempts a wake if STT is missing.
    """
    global _audio_stt_warned
    try:
        req = Request("http://gaia-audio:8080/status", method="GET")
        with urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read().decode())

        stt_model = status.get("stt_model")
        tts_engine = status.get("tts_engine")

        if stt_model and "ASR" in stt_model:
            # STT is loaded — all good
            if _audio_stt_warned:
                log.info("Audio STT recovered: %s", stt_model)
                _audio_stt_warned = False
            return

        # STT not loaded — attempt wake
        if not _audio_stt_warned:
            log.warning("Audio STT not loaded (stt_model=%s). Attempting wake...", stt_model)
            _audio_stt_warned = True
            _record_irritation("gaia-audio", f"STT model not loaded: {stt_model}", "AudioSTTDown")

        try:
            wake_req = Request(
                "http://gaia-audio:8080/wake",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(wake_req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            log.info("Audio wake requested: %s", result)
        except Exception as e:
            log.debug("Audio wake request failed: %s", e)

    except (URLError, OSError, TimeoutError):
        pass  # Container unreachable — normal health check will catch this


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

    # Poll KV cache pressure (independent of gaia-core)
    try:
        poll_kv_cache_pressure()
    except Exception:
        log.debug("KV cache pressure poll failed", exc_info=True)

    # Prime deep health — verify model is loaded, not just container healthy
    try:
        _check_prime_model_health()
    except Exception:
        log.debug("Prime model health check failed", exc_info=True)

    # Audio sensory health — verify STT model is loaded (not just container up)
    try:
        _check_audio_sensory_health()
    except Exception:
        log.debug("Audio sensory health check failed", exc_info=True)

    # GPU zombie cleanup — detect orphaned processes not owned by any container
    try:
        _cleanup_gpu_zombies()
    except Exception:
        log.debug("GPU zombie cleanup failed", exc_info=True)

    # VRAM reconciliation — compare orchestrator's expected state vs actual nvidia-smi
    try:
        _reconcile_vram()
    except Exception:
        log.debug("VRAM reconciliation failed", exc_info=True)

    # First scan logs for irritations (skip scoring in maintenance mode)
    if not is_maintenance_active():
        scan_logs()

    # Audit code for disk/memory mismatches (skip in maintenance mode)
    if not is_maintenance_active():
        audit_code()

    # Check for container naming anomalies (mangled names from failed recreates)
    for name, (_url, remediation) in SERVICES.items():
        if remediation == "restart" and "candidate" not in name:
            check_container_naming(name)

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
    # Rate-limited to prevent flooding the cognitive pipeline with 16K-token review packets.
    # Skip entirely in maintenance mode — promotion during dev sessions causes contention.
    global _last_sovereign_attempt
    all_divergent = _dissonance_report.get("vital_divergent", []) + _dissonance_report.get("standard_divergent", [])
    _sovereign_cooldown = int(os.environ.get("SOVEREIGN_COOLDOWN", "3600"))  # 60 min — sovereign review now replaced by lightweight cognitive monitor
    _sovereign_enabled = os.environ.get("SOVEREIGN_REVIEW_ENABLED", "0") == "1"
    if _sovereign_enabled and all_divergent and not is_maintenance_active() and (time.time() - _last_sovereign_attempt > _sovereign_cooldown):
        _last_sovereign_attempt = time.time()
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

    # Cognitive monitor heartbeat probe (runs in background thread)
    global _cognitive_monitor_last_run
    now_cm = time.time()
    if now_cm - _cognitive_monitor_last_run >= _cognitive_monitor_interval:
        _cognitive_monitor_last_run = now_cm
        threading.Thread(target=_run_cognitive_monitor, daemon=True).start()

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
                "alignment_status": data.get("alignment_status", "UNTRAINED"),
                "stages": stages,
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
        "maintenance_mode": is_maintenance_active(),
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
        "cognitive_monitor": _cognitive_monitor_last_result,
        "kv_cache_pressure": _kv_cache_pressure or None,
    }
    return status


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Elasticsearch log query (stdlib only)
# ---------------------------------------------------------------------------

def _es_query(query_body: dict, index: str = "gaia-logs-*", size: int = 50) -> dict:
    """Query Elasticsearch using stdlib urllib. Returns parsed JSON or error dict."""
    url = f"{ES_ENDPOINT}/{index}/_search"
    payload = json.dumps(query_body).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=10)
        return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": f"[GAIA-DOCTOR-020] ES unreachable: {e}"}
    except Exception as e:
        return {"error": f"[GAIA-DOCTOR-020] ES query failed: {e}"}


def _es_recent_errors(minutes: int = 60, service: str | None = None, size: int = 50) -> list[dict]:
    """Fetch recent ERROR-level logs from Elasticsearch."""
    must_clauses = [
        {"range": {"@timestamp": {"gte": f"now-{minutes}m"}}},
    ]
    # Match ERROR in parsed JSON field or raw message
    should_clauses = [
        {"match": {"log_level": "ERROR"}},
        {"match": {"gaia.level": "ERROR"}},
        {"match_phrase": {"message": "ERROR"}},
    ]
    must_clauses.append({"bool": {"should": should_clauses, "minimum_should_match": 1}})

    if service:
        must_clauses.append({
            "bool": {
                "should": [
                    {"match": {"gaia_service": service}},
                    {"match": {"service_name": service}},
                    {"wildcard": {"container.name": f"*{service}*"}},
                ],
                "minimum_should_match": 1,
            }
        })

    query = {
        "size": size,
        "sort": [{"@timestamp": {"order": "desc"}}],
        "query": {"bool": {"must": must_clauses}},
        "_source": ["@timestamp", "message", "log_level", "gaia", "gaia_service", "service_name", "container"],
    }
    result = _es_query(query, size=size)
    if "error" in result:
        return [result]
    hits = result.get("hits", {}).get("hits", [])
    return [h.get("_source", {}) for h in hits]


def _es_service_error_counts(minutes: int = 60) -> dict:
    """Get error counts per service from the last N minutes."""
    query = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": f"now-{minutes}m"}}},
                    {"bool": {
                        "should": [
                            {"match": {"log_level": "ERROR"}},
                            {"match": {"gaia.level": "ERROR"}},
                        ],
                        "minimum_should_match": 1,
                    }},
                ],
            }
        },
        "aggs": {
            "by_service": {
                "terms": {"field": "service_name.keyword", "size": 20}
            }
        },
    }
    result = _es_query(query)
    if "error" in result:
        return result
    buckets = result.get("aggregations", {}).get("by_service", {}).get("buckets", [])
    return {b["key"]: b["doc_count"] for b in buckets}


def _es_health() -> dict:
    """Check if Elasticsearch is reachable."""
    try:
        resp = urlopen(f"{ES_ENDPOINT}/_cluster/health", timeout=5)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


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
        elif self.path == "/kv-cache":
            self._json_response(200, _kv_cache_pressure or {"status": "no data yet"})
        elif self.path == "/cognitive/status":
            self._cognitive_status()
        elif self.path == "/cognitive/results":
            self._cognitive_results()
        elif self.path == "/cognitive/tests":
            self._cognitive_tests()
        elif self.path == "/cognitive/monitor":
            self._json_response(200, {
                "last_result": _cognitive_monitor_last_result,
                "consecutive_failures": _cognitive_monitor_failures,
                "interval_seconds": _cognitive_monitor_interval,
                "alarmed": "cognitive_monitor" in _alarmed_services,
            })
        elif self.path == "/maintenance/status":
            info = get_maintenance_info()
            self._json_response(200, info or {"active": False})
        elif self.path == "/surgeon/config":
            self._json_response(200, {"approval_required": _surgeon_approval_required})
        elif self.path == "/surgeon/queue":
            self._json_response(200, {"queue": _surgeon_queue})
        elif self.path == "/surgeon/history":
            self._json_response(200, {"history": _surgeon_history[-50:]})
        elif self.path == "/logs/health":
            self._json_response(200, _es_health())
        elif self.path.startswith("/logs"):
            self._handle_logs()
        elif self.path == "/errors":
            self._handle_errors()
        elif self.path == "/oom/history":
            try:
                state_path = Path(os.environ.get("SHARED_DIR", "/shared")) / "doctor" / "oom_resolutions.json"
                if state_path.exists():
                    self._json_response(200, json.loads(state_path.read_text()))
                else:
                    self._json_response(200, {"history": [], "last": None})
            except Exception:
                self._json_response(200, {"history": [], "last": None})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/cognitive/run":
            self._cognitive_run()
        elif self.path == "/pipeline/run":
            self._pipeline_run()
        elif self.path == "/maintenance/enter":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                params = json.loads(body)
            except json.JSONDecodeError:
                params = {}
            reason = params.get("reason", "manual")
            entered_by = params.get("entered_by", "api")
            result = enter_maintenance(reason=reason, entered_by=entered_by)
            log.warning("🔧 MAINTENANCE MODE ENTERED: %s (by %s)", reason, entered_by)
            self._json_response(200, result)
        elif self.path == "/maintenance/exit":
            result = exit_maintenance()
            log.warning("🔧 MAINTENANCE MODE EXITED (duration: %s sec)",
                        result.get("duration_seconds", "unknown"))
            self._json_response(200, result)
        elif self.path == "/surgeon/config":
            self._handle_surgeon_config()
        elif self.path == "/surgeon/approve":
            self._handle_surgeon_approve()
        elif self.path == "/surgeon/reject":
            self._handle_surgeon_reject()
        elif self.path == "/notify/chaos_injection":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {}
            # Spawn background thread to handle repair — return immediately
            thread = threading.Thread(target=_handle_chaos_notification, args=(data,), daemon=True)
            thread.start()
            self._json_response(200, {"status": "acknowledged", "service": data.get("service", ""), "file": data.get("file", "")})
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

    def _pipeline_run(self):
        """Proxy POST /pipeline/run to gaia-study."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            study_url = "http://gaia-study:8766/pipeline/run"
            req = Request(
                study_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=15)
            result = json.loads(resp.read().decode())
            self._json_response(resp.status, result)
        except Exception as e:
            log.error("Pipeline run proxy failed: %s", e)
            self._json_response(502, {"error": f"gaia-study unreachable: {e}"})

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
        target = params.get("target", "prime")
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
                # Regenerate rubric after battery (tests may have been updated)
                try:
                    _generate_rubric()
                except Exception:
                    log.debug("Rubric regeneration after battery failed", exc_info=True)
            except Exception:
                log.error("[GAIA-DOCTOR-025] Cognitive battery failed", exc_info=True)
            finally:
                _cognitive_running = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        self._json_response(202, {"status": "started", "section": section, "ids": ids, "full_pipeline": full_pipeline, "target": target})

    def _handle_logs(self):
        """Query Elasticsearch for recent logs.

        GET /logs?minutes=60&service=gaia-core&size=50  → recent errors
        GET /logs/counts?minutes=60                     → error counts by service
        """
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        minutes = int(params.get("minutes", ["60"])[0])
        service = params.get("service", [None])[0]
        size = int(params.get("size", ["50"])[0])

        if parsed.path == "/logs/counts":
            result = _es_service_error_counts(minutes=minutes)
            self._json_response(200, result)
        else:
            errors = _es_recent_errors(minutes=minutes, service=service, size=size)
            # Enrich log entries with hints from the error registry
            import re as _re
            _gaia_code_re = _re.compile(r'GAIA-[A-Z]+-\d{3}')
            registry = _get_error_registry()
            for entry in errors:
                msg = entry.get("message", "")
                # Check for error_code already in structured JSON fields
                code = entry.get("error_code") or entry.get("gaia", {}).get("error_code", "")
                if not code:
                    # Try to extract from message text
                    m = _gaia_code_re.search(msg)
                    if m:
                        code = m.group(0)
                if code:
                    defn = registry["lookup"](code)
                    if defn:
                        entry["error_code"] = code
                        entry["error_hint"] = defn.hint
                        entry["error_category"] = defn.category.value
            self._json_response(200, {"count": len(errors), "errors": errors})

    def _handle_errors(self):
        """List all registered GAIA error codes with hints."""
        registry = _get_error_registry()
        errors = {}
        for code, defn in registry["all_errors"]().items():
            errors[code] = {
                "message": defn.message,
                "hint": defn.hint,
                "level": logging.getLevelName(defn.level),
                "category": defn.category.value,
            }
        self._json_response(200, {"count": len(errors), "errors": errors})

    def _handle_surgeon_config(self):
        global _surgeon_approval_required
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            params = {}
        if "approval_required" in params:
            _surgeon_approval_required = bool(params["approval_required"])
            _save_surgeon_config()
            log.info("Surgeon approval mode: %s", "ON" if _surgeon_approval_required else "OFF")
        self._json_response(200, {"approval_required": _surgeon_approval_required})

    def _handle_surgeon_approve(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON"})
            return
        repair_id = params.get("repair_id")
        if not repair_id:
            self._json_response(400, {"error": "repair_id required"})
            return
        # Find in queue
        proposal = None
        for i, p in enumerate(_surgeon_queue):
            if p["repair_id"] == repair_id:
                proposal = _surgeon_queue.pop(i)
                break
        if not proposal:
            self._json_response(404, {"error": f"repair {repair_id} not found in queue"})
            return
        # Apply the fix
        service = proposal["service"]
        container_path = proposal["container_path"]
        fixed_code = proposal["fixed_code"]
        file_path = proposal["file"]

        # ── Lint autofix branch — run ruff --fix instead of writing fixed_code ──
        if proposal.get("method") == "lint_autofix":
            rules = proposal.get("rules", _LINT_AUTOFIX_RULES)
            try:
                fix_cmd = [
                    "docker", "exec", service, "python", "-m", "ruff", "check", "/app",
                    "--select", rules, "--fix", "--no-cache",
                ]
                subprocess.run(fix_cmd, capture_output=True, text=True, timeout=30)
                verify_cmd = [
                    "docker", "exec", service, "python", "-m", "ruff", "check", "/app",
                    "--select", "F", "--no-cache",
                ]
                verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
            except Exception as e:
                proposal["status"] = "apply_failed"
                proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
                _surgeon_history.append(proposal)
                if len(_surgeon_history) > 50:
                    _surgeon_history[:] = _surgeon_history[-50:]
                self._json_response(500, {"error": f"lint autofix failed: {e}", "repair_id": repair_id})
                return

            proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
            if verify.returncode == 0:
                proposal["status"] = "approved"
                _surgeon_history.append(proposal)
                if len(_surgeon_history) > 50:
                    _surgeon_history[:] = _surgeon_history[-50:]
                _record_lint_autofix(service, f"surgeon-approved: {proposal.get('error_msg', '')[:200]}")
                log.info("Surgeon lint autofix APPROVED for %s: %s", repair_id, service)
                self._json_response(200, {"status": "approved", "repair_id": repair_id, "validation": "passed"})
            else:
                proposal["status"] = "partial"
                proposal["remaining_errors"] = verify.stdout.strip()[:300]
                _surgeon_history.append(proposal)
                if len(_surgeon_history) > 50:
                    _surgeon_history[:] = _surgeon_history[-50:]
                _record_lint_autofix(service, f"surgeon-approved (partial): {proposal.get('error_msg', '')[:200]}")
                log.info("Surgeon lint autofix PARTIAL for %s — safe rules fixed, unfixable remain", repair_id)
                self._json_response(200, {"status": "partial", "repair_id": repair_id, "remaining": verify.stdout.strip()[:300]})
            return

        if not _write_container_file(service, container_path, fixed_code):
            proposal["status"] = "apply_failed"
            proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _surgeon_history.append(proposal)
            if len(_surgeon_history) > 50:
                _surgeon_history[:] = _surgeon_history[-50:]
            self._json_response(500, {"error": "failed to write fixed code", "repair_id": repair_id})
            return

        check = _validate_repair(service, container_path)
        if check["valid"]:
            # Restart container
            try:
                subprocess.run(["docker", "restart", service], capture_output=True, text=True, timeout=30)
            except Exception:
                pass
            proposal["status"] = "approved"
            proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _surgeon_history.append(proposal)
            if len(_surgeon_history) > 50:
                _surgeon_history[:] = _surgeon_history[-50:]
            _notify_monkey_repair_success(service, file_path)
            log.info("Surgeon repair APPROVED and applied: %s for %s", repair_id, file_path)
            self._json_response(200, {"status": "approved", "repair_id": repair_id, "validation": "passed"})
        else:
            # Validation failed — revert and report
            proposal["status"] = "apply_failed"
            proposal["validation_error"] = check["reason"]
            proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _surgeon_history.append(proposal)
            if len(_surgeon_history) > 50:
                _surgeon_history[:] = _surgeon_history[-50:]
            log.warning("Surgeon repair approved but validation FAILED for %s: %s", repair_id, check["reason"])
            self._json_response(200, {"status": "validation_failed", "repair_id": repair_id, "reason": check["reason"]})

    def _handle_surgeon_reject(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON"})
            return
        repair_id = params.get("repair_id")
        if not repair_id:
            self._json_response(400, {"error": "repair_id required"})
            return
        proposal = None
        for i, p in enumerate(_surgeon_queue):
            if p["repair_id"] == repair_id:
                proposal = _surgeon_queue.pop(i)
                break
        if not proposal:
            self._json_response(404, {"error": f"repair {repair_id} not found in queue"})
            return
        proposal["status"] = "rejected"
        proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
        _surgeon_history.append(proposal)
        if len(_surgeon_history) > 50:
            _surgeon_history[:] = _surgeon_history[-50:]
        log.info("Surgeon repair REJECTED: %s for %s", repair_id, proposal["file"])
        self._json_response(200, {"status": "rejected", "repair_id": repair_id})

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
