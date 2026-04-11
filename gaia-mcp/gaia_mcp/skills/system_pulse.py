"""
Memento Skill: System Pulse (Phase 5i — Sovereign Awareness)

GAIA's first self-reflective skill. Reads her own vitals across four
pulse domains and returns a unified Sovereign Status assessment.

Pulse Domains:
  1. Biological  — Heartbeat time-check canary (shared/heartbeat/time_check.json)
  2. Structural  — Immune system status (logs/immune_status.json)
  3. Cognitive   — EthicalSentinel loop counter + CPR tier (gaia-core /status)
  4. Security    — Recent adversarial_summary hits (gaia-core /status)

Usage (via MCP SkillManager):
    result = await skill_manager.execute_skill("system_pulse", {})
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.Skill.SystemPulse")

# ── Endpoint Configuration ─────────────────────────────────────────────

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
HEARTBEAT_PATH = SHARED_DIR / "heartbeat" / "time_check.json"
IMMUNE_STATUS_PATH = Path("/logs/immune_status.json")
IMMUNE_STATUS_FALLBACK = Path("./logs/immune_status.json")

CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415")
DOCTOR_ENDPOINT = os.environ.get("DOCTOR_ENDPOINT", "http://gaia-doctor:6419")
ORCHESTRATOR_ENDPOINT = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")

HEALING_LOCK_PATH = SHARED_DIR / "HEALING_REQUIRED.lock"


# ── Pulse Readers ──────────────────────────────────────────────────────

def _read_biological_pulse() -> Dict[str, Any]:
    """Read the heartbeat time-check canary (Biological Pulse).

    Returns last heartbeat status: pass/fail, drift, timestamp.
    """
    try:
        path = HEARTBEAT_PATH
        if not path.exists():
            return {"status": "UNKNOWN", "reason": "heartbeat file not found"}

        with open(path, "r") as f:
            data = json.load(f)

        last_check = data.get("last_check", {})
        passed = last_check.get("passed", False)
        drift = last_check.get("drift_minutes", None)
        timestamp = last_check.get("timestamp", "")
        consecutive_fails = data.get("consecutive_failures", 0)

        # Stale check: if last heartbeat is older than 15 minutes, flag it
        stale = False
        if timestamp:
            try:
                from datetime import datetime, timezone
                last_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age_seconds = (datetime.now(timezone.utc) - last_dt).total_seconds()
                stale = age_seconds > 900  # 15 minutes
            except Exception:
                pass

        if stale:
            status = "STALE"
        elif passed:
            status = "HEALTHY"
        elif consecutive_fails >= 3:
            status = "CRITICAL"
        else:
            status = "DEGRADED"

        return {
            "status": status,
            "passed": passed,
            "drift_minutes": drift,
            "consecutive_failures": consecutive_fails,
            "last_check": timestamp,
            "stale": stale,
        }
    except Exception as e:
        logger.debug("Biological pulse read failed: %s", e)
        return {"status": "ERROR", "reason": str(e)}


def _read_structural_pulse() -> Dict[str, Any]:
    """Read the immune system status (Structural Pulse).

    Returns irritation count, active alarms, remediation history.
    """
    try:
        # Try container path first, then local fallback
        path = IMMUNE_STATUS_PATH if IMMUNE_STATUS_PATH.parent.exists() else IMMUNE_STATUS_FALLBACK
        if not path.exists():
            # Fall back to doctor endpoint
            return _read_structural_from_doctor()

        with open(path, "r") as f:
            data = json.load(f)

        irritation_count = data.get("irritation_count", 0)
        active_alarms = data.get("active_alarms", [])
        last_scan = data.get("last_scan", "")

        if len(active_alarms) >= 3:
            status = "CRITICAL"
        elif irritation_count > 5:
            status = "IRRITATED"
        elif active_alarms:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "irritation_count": irritation_count,
            "active_alarms": active_alarms[:5],  # cap for report size
            "last_scan": last_scan,
        }
    except Exception as e:
        logger.debug("Structural pulse file read failed, trying doctor: %s", e)
        return _read_structural_from_doctor()


def _read_structural_from_doctor() -> Dict[str, Any]:
    """Fallback: read structural pulse from gaia-doctor /status endpoint."""
    try:
        req = Request(f"{DOCTOR_ENDPOINT}/status", headers={"Accept": "application/json"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        irritation_count = data.get("irritation_count", 0)
        active_alarms = data.get("active_alarms", [])
        services = data.get("services", {})

        unhealthy = [name for name, info in services.items()
                     if isinstance(info, dict) and not info.get("healthy", True)]

        if len(unhealthy) >= 3:
            status = "CRITICAL"
        elif irritation_count > 5:
            status = "IRRITATED"
        elif unhealthy:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "irritation_count": irritation_count,
            "active_alarms": active_alarms[:5],
            "unhealthy_services": unhealthy[:5],
        }
    except Exception as e:
        logger.debug("Doctor endpoint unreachable: %s", e)
        return {"status": "UNKNOWN", "reason": f"doctor unreachable: {e}"}


def _read_cognitive_pulse() -> Dict[str, Any]:
    """Read the EthicalSentinel state (Cognitive Pulse).

    Checks: loop counter, CPR tier status, HEALING_REQUIRED.lock.
    Uses gaia-core internal API or falls back to file-based detection.
    """
    result: Dict[str, Any] = {
        "status": "HEALTHY",
        "loop_counter": 0,
        "cpr_tier": 0,
        "healing_lock": False,
    }

    # Check for HEALING_REQUIRED.lock (Tier 3 — most severe)
    try:
        if HEALING_LOCK_PATH.exists():
            result["healing_lock"] = True
            result["status"] = "LOCKED"
            result["cpr_tier"] = 3
            try:
                result["lock_content"] = HEALING_LOCK_PATH.read_text()[:200]
            except Exception:
                pass
            return result
    except Exception:
        pass

    # Query gaia-core for sentinel state
    try:
        req = Request(
            f"{CORE_ENDPOINT}/api/diagnostics/sentinel",
            headers={"Accept": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        loop_counter = data.get("loop_counter", 0)
        tier_1_fired = data.get("tier_1_fired", False)
        tier_2_fired = data.get("tier_2_fired", False)

        result["loop_counter"] = loop_counter

        if tier_2_fired:
            result["cpr_tier"] = 2
            result["status"] = "RECOVERING"
        elif tier_1_fired:
            result["cpr_tier"] = 1
            result["status"] = "RECOVERING"
        elif loop_counter > 10:
            result["status"] = "ELEVATED"
        else:
            result["status"] = "HEALTHY"

        return result
    except Exception as e:
        logger.debug("Core sentinel endpoint unavailable: %s", e)
        # If we can't query but no lock exists, assume healthy
        result["reason"] = f"sentinel endpoint unavailable: {e}"
        return result


def _read_security_pulse() -> Dict[str, Any]:
    """Scan for recent adversarial_summary hits (Security Pulse).

    Checks gaia-core for recent injection detections.
    """
    try:
        req = Request(
            f"{CORE_ENDPOINT}/api/diagnostics/security",
            headers={"Accept": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        recent_attacks = data.get("recent_adversarial_events", [])
        total_blocked = data.get("total_blocked", 0)

        if recent_attacks:
            # Check if any are within last 5 minutes
            recent_count = len(recent_attacks)
            status = "UNDER_ATTACK" if recent_count >= 3 else "ALERTED"
        else:
            status = "CLEAR"

        return {
            "status": status,
            "recent_events": recent_attacks[:5],
            "total_blocked": total_blocked,
        }
    except Exception as e:
        logger.debug("Core security endpoint unavailable: %s", e)
        # Fall back to checking shared adversarial log
        return _read_security_from_log()


def _read_security_from_log() -> Dict[str, Any]:
    """Fallback: check adversarial event log on shared volume."""
    try:
        log_path = SHARED_DIR / "security" / "adversarial_events.jsonl"
        if not log_path.exists():
            return {"status": "CLEAR", "recent_events": [], "total_blocked": 0}

        recent = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recent.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Keep only last 10
        recent = recent[-10:]

        # Check recency
        now = time.time()
        recent_5min = [e for e in recent
                       if now - e.get("timestamp", 0) < 300]

        if len(recent_5min) >= 3:
            status = "UNDER_ATTACK"
        elif recent_5min:
            status = "ALERTED"
        else:
            status = "CLEAR"

        return {
            "status": status,
            "recent_events": [e.get("summary", "unknown") for e in recent[-5:]],
            "total_blocked": len(recent),
        }
    except Exception as e:
        return {"status": "UNKNOWN", "reason": str(e)}


# ── Sovereign Status Assessment ────────────────────────────────────────

def _assess_sovereign_status(
    biological: Dict, structural: Dict, cognitive: Dict, security: Dict
) -> str:
    """Synthesize individual pulse statuses into a single Sovereign Status.

    Priority (highest to lowest):
      LOCKED      — HEALING_REQUIRED.lock active (Tier 3)
      UNDER_ATTACK — active adversarial assault
      CRITICAL    — multiple system failures
      RECOVERING  — CPR tiers active (self-healing in progress)
      IRRITATED   — elevated irritation count
      DEGRADED    — partial failures but functional
      ELEVATED    — loop counter rising but no CPR yet
      STABLE      — all systems nominal
    """
    statuses = {
        "biological": biological.get("status", "UNKNOWN"),
        "structural": structural.get("status", "UNKNOWN"),
        "cognitive": cognitive.get("status", "UNKNOWN"),
        "security": security.get("status", "UNKNOWN"),
    }

    # Tier 3: full lockdown
    if cognitive.get("healing_lock"):
        return "LOCKED"

    # Active attack
    if statuses["security"] == "UNDER_ATTACK":
        return "UNDER_ATTACK"

    # Critical failures
    critical_count = sum(1 for s in statuses.values() if s == "CRITICAL")
    if critical_count >= 2:
        return "CRITICAL"

    # Self-healing
    if statuses["cognitive"] == "RECOVERING":
        return "RECOVERING"

    # Irritation
    if statuses["structural"] == "IRRITATED":
        return "IRRITATED"

    # Degradation
    degraded_count = sum(1 for s in statuses.values() if s in ("DEGRADED", "CRITICAL"))
    if degraded_count > 0:
        return "DEGRADED"

    # Elevated loop counter
    if statuses["cognitive"] == "ELEVATED":
        return "ELEVATED"

    # Security alert (but not under active attack)
    if statuses["security"] == "ALERTED":
        return "ALERTED"

    return "STABLE"


# ── Skill Entry Point ──────────────────────────────────────────────────

def execute(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute the System Pulse skill.

    Delegates to GaiaVitals (Phase 5-C unified vitals manager) for all
    health monitoring. Falls back to inline readers if GaiaVitals is
    unavailable (e.g., gaia-common not on sys.path).

    Args:
        params: Optional dict. Supports:
            - verbose (bool): Include full pulse details (default True)

    Returns:
        Dict with sovereign_status, irritation_score, and pulse data.
    """
    params = params or {}
    verbose = params.get("verbose", True)

    # Prefer GaiaVitals (unified, authoritative)
    try:
        from gaia_common.utils.vitals import GaiaVitals
        vitals = GaiaVitals()
        report = vitals.get_sovereign_health(verbose=verbose)
        report["ok"] = True
        return report
    except ImportError:
        logger.debug("GaiaVitals not available, using inline readers")
    except Exception:
        logger.debug("GaiaVitals failed, falling back to inline readers", exc_info=True)

    # Fallback: inline readers (original system_pulse logic)
    start = time.time()

    biological = _read_biological_pulse()
    structural = _read_structural_pulse()
    cognitive = _read_cognitive_pulse()
    security = _read_security_pulse()

    sovereign_status = _assess_sovereign_status(
        biological, structural, cognitive, security
    )

    elapsed_ms = round((time.time() - start) * 1000)

    report = {
        "ok": True,
        "sovereign_status": sovereign_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_ms": elapsed_ms,
    }

    if verbose:
        report["pulses"] = {
            "biological": biological,
            "structural": structural,
            "cognitive": cognitive,
            "security": security,
        }

    logger.info("System Pulse: %s (took %dms)", sovereign_status, elapsed_ms)
    return report
