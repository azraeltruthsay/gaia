"""
GaiaVitals — Unified Sovereign Health Monitor (Phase 5-C, Proposal 01)

Consolidates four previously independent health-monitoring systems into a
single authoritative class:

  Biological  — inference-chain heartbeat (time_check.json)
  Structural  — log MRI + syntax/lint scanning (immune_status.json)
  Cognitive   — loop counter, CPR tier, resource pressure (sentinel state)
  Security    — adversarial event history (force field log)

Usage:
    vitals = GaiaVitals()
    health = vitals.get_sovereign_health()
    # health["sovereign_status"]   -> "STABLE" | "IRRITATED" | "RECOVERING" | ...
    # health["irritation_score"]   -> 0.0 - 100.0
    # health["pulses"]             -> per-domain details

Design: Proposal 01 (Vitals Manager Consolidation)
"""

from __future__ import annotations

import json
import logging
import math
import os
import psutil
import re
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.Vitals")


# ── Configuration ──────────────────────────────────────────────────────

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
HEARTBEAT_PATH = SHARED_DIR / "heartbeat" / "time_check.json"
HEALING_LOCK_PATH = SHARED_DIR / "HEALING_REQUIRED.lock"
ADVERSARIAL_LOG_PATH = SHARED_DIR / "security" / "adversarial_events.jsonl"

NANO_ENDPOINT = os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080")
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://gaia-core:6415")
DOCTOR_ENDPOINT = os.environ.get("DOCTOR_ENDPOINT", "http://gaia-doctor:6419")

# Immune system priority map (from immune_system.py)
PRIORITY_MAP = {
    r"ModuleNotFoundError": 3.0,
    r"NameResolutionError": 0.5,
    r"ConnectionError": 0.5,
    r"Permission denied": 2.0,
    r"not found in configuration": 1.0,
    r"Model path does not exist": 2.5,
    r"SyntaxError": 4.0,
    r"LintError": 2.0,
    r"Root not allowed": 0.2,
    r"timeout": 0.8,
    r"uid not found": 2.0,
    r"cpuinfo": 0.1,
}

# Irritation score weights per domain
_WEIGHT_BIOLOGICAL = 0.20
_WEIGHT_STRUCTURAL = 0.35
_WEIGHT_COGNITIVE = 0.30
_WEIGHT_SECURITY = 0.15

# Sovereign status thresholds (irritation score)
_THRESHOLD_CRITICAL = 70.0
_THRESHOLD_IRRITATED = 30.0
_THRESHOLD_ELEVATED = 10.0

# Staleness thresholds
_HEARTBEAT_STALE_SECONDS = 900   # 15 minutes
_IMMUNE_STALE_SECONDS = 600      # 10 minutes

LOG_WINDOW_SECONDS = 1800  # 30 minutes for log scanning


# ── Sovereign Status Enum ──────────────────────────────────────────────

SOVEREIGN_STATUSES = [
    "LOCKED",        # HEALING_REQUIRED.lock active
    "UNDER_ATTACK",  # active adversarial assault
    "CRITICAL",      # multiple system failures
    "RECOVERING",    # CPR tiers active
    "IRRITATED",     # elevated irritation
    "DEGRADED",      # partial failures
    "ELEVATED",      # loop counter rising
    "ALERTED",       # security events detected
    "STABLE",        # all nominal
]


# ── GaiaVitals ─────────────────────────────────────────────────────────

class GaiaVitals:
    """Unified sovereign health monitor.

    Single class that reads all four pulse domains and computes a
    weighted irritation score and sovereign status.

    Designed to be instantiated fresh per query (stateless) or cached
    as a singleton. All reads are from shared-volume files or HTTP
    endpoints — no in-process state coupling.
    """

    def __init__(self, log_dir: str = "/logs"):
        self.log_dir = Path(log_dir)
        self._last_status: Optional[str] = None
        self._doc_sentinel = None

    # ── Public API ─────────────────────────────────────────────────────

    def get_sovereign_health(self, verbose: bool = True) -> Dict[str, Any]:
        """Read all vitals and return a unified health report.

        Args:
            verbose: Include per-domain pulse details (default True).

        Returns:
            Dict with sovereign_status, irritation_score, timestamp,
            and optionally per-domain pulse data.
        """
        start = time.time()

        biological = self._check_inference_chain()
        structural = self._check_system_integrity()
        cognitive = self._check_reasoning_loops()
        security = self._check_adversarial_state()

        irritation = self.calculate_irritation_score(
            biological, structural, cognitive, security
        )
        sovereign_status = self._assess_status(
            biological, structural, cognitive, security, irritation
        )

        elapsed_ms = round((time.time() - start) * 1000)

        report: Dict[str, Any] = {
            "sovereign_status": sovereign_status,
            "irritation_score": round(irritation, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": elapsed_ms,
        }

        if verbose:
            report["pulses"] = {
                "biological": biological,
                "structural": structural,
                "cognitive": cognitive,
                "security": security,
            }

        logger.info(
            "Sovereign Health: %s (irritation=%.1f, took %dms)",
            sovereign_status, irritation, elapsed_ms,
        )

        # DocSentinel: auto-record incident on state transition
        self._record_transition_if_needed(report)

        return report

    def _record_transition_if_needed(self, report: Dict[str, Any]) -> None:
        """Fire DocSentinel incident report on sovereign status change."""
        current = report.get("sovereign_status", "STABLE")
        if self._last_status is not None and current != self._last_status:
            # Only record escalations (not recovery-to-stable)
            if current != "STABLE":
                try:
                    if self._doc_sentinel is None:
                        from gaia_common.utils.doc_sentinel import DocSentinel
                        self._doc_sentinel = DocSentinel()
                    self._doc_sentinel.record_event(
                        status=current,
                        reason=f"Sovereign status transition: {self._last_status} -> {current}",
                        previous_status=self._last_status,
                        vitals_snapshot=report,
                    )
                    logger.info("DocSentinel: recorded transition %s -> %s", self._last_status, current)
                except Exception:
                    logger.debug("DocSentinel: failed to record transition", exc_info=True)
        self._last_status = current

    def calculate_irritation_score(
        self,
        biological: Optional[Dict] = None,
        structural: Optional[Dict] = None,
        cognitive: Optional[Dict] = None,
        security: Optional[Dict] = None,
    ) -> float:
        """Compute a weighted irritation score [0.0 - 100.0].

        Each domain contributes a normalized score (0-100) which is
        weighted and summed.
        """
        bio = biological or self._check_inference_chain()
        struct = structural or self._check_system_integrity()
        cog = cognitive or self._check_reasoning_loops()
        sec = security or self._check_adversarial_state()

        bio_score = self._normalize_biological(bio)
        struct_score = self._normalize_structural(struct)
        cog_score = self._normalize_cognitive(cog)
        sec_score = self._normalize_security(sec)

        total = (
            _WEIGHT_BIOLOGICAL * bio_score
            + _WEIGHT_STRUCTURAL * struct_score
            + _WEIGHT_COGNITIVE * cog_score
            + _WEIGHT_SECURITY * sec_score
        )
        return min(100.0, max(0.0, total))

    def get_compact_summary(self) -> str:
        """One-line health summary for prompt injection or CLI display."""
        health = self.get_sovereign_health(verbose=False)
        return (
            f"Sovereign: {health['sovereign_status']} "
            f"(irritation={health['irritation_score']})"
        )

    # ── Biological Pulse (Inference Chain Heartbeat) ───────────────────

    def _check_inference_chain(self) -> Dict[str, Any]:
        """Read the heartbeat time-check canary.

        Best-of: jittered interval awareness, focus multiplier logic.
        """
        try:
            if not HEARTBEAT_PATH.exists():
                return {"status": "UNKNOWN", "reason": "heartbeat file not found"}

            data = json.loads(HEARTBEAT_PATH.read_text())
            current = data.get("current", {})
            stats = data.get("stats", {})

            passed = current.get("status") == "pass"
            drift = current.get("drift_minutes")
            timestamp = current.get("timestamp", "")
            error = current.get("error")

            # Staleness check
            stale = False
            age_seconds = None
            if timestamp:
                try:
                    last_dt = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                    age_seconds = (
                        datetime.now(timezone.utc) - last_dt
                    ).total_seconds()
                    stale = age_seconds > _HEARTBEAT_STALE_SECONDS
                except Exception:
                    pass

            # Consecutive failure tracking from stats
            total = stats.get("total", 0)
            fails = stats.get("fails", 0)
            errors = stats.get("errors", 0)
            consecutive_failures = 0
            for entry in data.get("history", [])[:5]:
                if entry.get("status") != "pass":
                    consecutive_failures += 1
                else:
                    break

            if stale:
                status = "STALE"
            elif passed:
                status = "HEALTHY"
            elif consecutive_failures >= 3:
                status = "CRITICAL"
            elif error:
                status = "ERROR"
            else:
                status = "DEGRADED"

            return {
                "status": status,
                "passed": passed,
                "drift_minutes": drift,
                "consecutive_failures": consecutive_failures,
                "last_check": timestamp,
                "stale": stale,
                "age_seconds": round(age_seconds) if age_seconds else None,
                "lifetime_stats": {
                    "total": total, "passes": stats.get("passes", 0),
                    "fails": fails, "errors": errors,
                },
            }
        except Exception as e:
            logger.debug("Biological pulse failed: %s", e)
            return {"status": "ERROR", "reason": str(e)}

    # ── Structural Pulse (Log MRI + Syntax) ────────────────────────────

    def _check_system_integrity(self) -> Dict[str, Any]:
        """Read immune system status.

        Best-of: weighted priority map for triage, MRI diagnostics.
        """
        # Try cached status file first (written by BackgroundImmuneSystem)
        for path in [Path("/logs/immune_status.json"), Path("./logs/immune_status.json")]:
            try:
                if path.exists():
                    data = json.loads(path.read_text())
                    age = time.time() - data.get("timestamp", 0)
                    if age < _IMMUNE_STALE_SECONDS:
                        score = data.get("score", 0.0)
                        diagnostics = data.get("diagnostics", [])
                        summary = data.get("summary", "")

                        if score > 25:
                            status = "CRITICAL"
                        elif score > 8:
                            status = "IRRITATED"
                        elif score > 2:
                            status = "MINOR_NOISE"
                        else:
                            status = "HEALTHY"

                        return {
                            "status": status,
                            "immune_score": round(score, 1),
                            "diagnostic_count": len(diagnostics),
                            "summary": summary[:200],
                            "stale": False,
                        }
            except Exception:
                continue

        # Fallback: query gaia-doctor
        try:
            req = Request(
                f"{DOCTOR_ENDPOINT}/status",
                headers={"Accept": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            irritation_count = data.get("irritation_count", 0)
            active_alarms = data.get("active_alarms", [])
            services = data.get("services", {})
            unhealthy = [
                name for name, info in services.items()
                if isinstance(info, dict) and not info.get("healthy", True)
            ]

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
                "immune_score": float(irritation_count),
                "active_alarms": active_alarms[:5],
                "unhealthy_services": unhealthy[:5],
                "stale": False,
            }
        except Exception as e:
            logger.debug("Structural pulse failed: %s", e)
            return {"status": "UNKNOWN", "reason": str(e)}

    # ── Cognitive Pulse (Loops + Resources) ────────────────────────────

    def _check_reasoning_loops(self) -> Dict[str, Any]:
        """Check sentinel state and system resources.

        Best-of: CPR tier tracking, resource pressure sensing.
        """
        result: Dict[str, Any] = {
            "status": "HEALTHY",
            "loop_counter": 0,
            "cpr_tier": 0,
            "healing_lock": False,
            "cpu_percent": None,
            "memory_percent": None,
        }

        # Check HEALING_REQUIRED.lock (Tier 3 — highest severity)
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

        # System resources (from psutil — available in all containers)
        try:
            result["cpu_percent"] = round(psutil.cpu_percent(interval=0.1), 1)
            result["memory_percent"] = round(
                psutil.virtual_memory().percent, 1
            )
        except Exception:
            pass

        # Query sentinel state via gaia-core diagnostics
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
        except Exception as e:
            logger.debug("Sentinel endpoint unavailable: %s", e)
            result["reason"] = str(e)

        # Resource pressure can override to STRESSED
        cpu = result.get("cpu_percent") or 0
        mem = result.get("memory_percent") or 0
        if cpu > 90 or mem > 90:
            if result["status"] == "HEALTHY":
                result["status"] = "STRESSED"

        return result

    # ── Security Pulse (Adversarial State) ─────────────────────────────

    def _check_adversarial_state(self) -> Dict[str, Any]:
        """Check for recent adversarial events.

        Best-of: adversarial awareness summaries from force field.
        """
        # Try gaia-core security diagnostics endpoint
        try:
            req = Request(
                f"{CORE_ENDPOINT}/api/diagnostics/security",
                headers={"Accept": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            recent = data.get("recent_adversarial_events", [])
            total = data.get("total_blocked", 0)

            if len(recent) >= 3:
                status = "UNDER_ATTACK"
            elif recent:
                status = "ALERTED"
            else:
                status = "CLEAR"

            return {
                "status": status,
                "recent_events": recent[:5],
                "total_blocked": total,
            }
        except Exception:
            pass

        # Fallback: read adversarial event log from shared volume
        try:
            if not ADVERSARIAL_LOG_PATH.exists():
                return {"status": "CLEAR", "recent_events": [], "total_blocked": 0}

            recent = []
            with open(ADVERSARIAL_LOG_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recent.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            recent = recent[-10:]
            now = time.time()
            recent_5min = [
                e for e in recent if now - e.get("timestamp", 0) < 300
            ]

            if len(recent_5min) >= 3:
                status = "UNDER_ATTACK"
            elif recent_5min:
                status = "ALERTED"
            else:
                status = "CLEAR"

            return {
                "status": status,
                "recent_events": [
                    e.get("summary", "unknown") for e in recent[-5:]
                ],
                "total_blocked": len(recent),
            }
        except Exception as e:
            return {"status": "UNKNOWN", "reason": str(e)}

    # ── Normalization (per-domain → 0-100) ─────────────────────────────

    @staticmethod
    def _normalize_biological(pulse: Dict) -> float:
        """Map biological status to irritation contribution [0-100]."""
        status = pulse.get("status", "UNKNOWN")
        if status == "HEALTHY":
            return 0.0
        elif status == "STALE":
            return 30.0
        elif status == "DEGRADED":
            return 50.0
        elif status in ("CRITICAL", "ERROR"):
            return 90.0
        return 15.0  # UNKNOWN

    @staticmethod
    def _normalize_structural(pulse: Dict) -> float:
        """Map structural status to irritation contribution [0-100]."""
        immune_score = pulse.get("immune_score", 0.0)
        # Map the immune system's log-weighted score to 0-100
        # Immune score: 0=clean, 2=noise, 8=irritated, 25+=critical
        if immune_score <= 0:
            return 0.0
        elif immune_score <= 2:
            return 5.0
        elif immune_score <= 8:
            return 25.0
        elif immune_score <= 25:
            return 60.0
        return 90.0

    @staticmethod
    def _normalize_cognitive(pulse: Dict) -> float:
        """Map cognitive status to irritation contribution [0-100]."""
        status = pulse.get("status", "HEALTHY")
        loop_counter = pulse.get("loop_counter", 0)
        cpu = pulse.get("cpu_percent") or 0
        mem = pulse.get("memory_percent") or 0

        base = {
            "HEALTHY": 0.0,
            "STRESSED": 30.0,
            "ELEVATED": 40.0,
            "RECOVERING": 60.0,
            "LOCKED": 100.0,
        }.get(status, 15.0)

        # Loop counter adds linear pressure
        loop_pressure = min(30.0, loop_counter * 0.6)

        # Resource pressure
        resource_pressure = 0.0
        if cpu > 80:
            resource_pressure += (cpu - 80) * 0.5
        if mem > 80:
            resource_pressure += (mem - 80) * 0.5

        return min(100.0, base + loop_pressure + resource_pressure)

    @staticmethod
    def _normalize_security(pulse: Dict) -> float:
        """Map security status to irritation contribution [0-100]."""
        status = pulse.get("status", "CLEAR")
        return {
            "CLEAR": 0.0,
            "ALERTED": 30.0,
            "UNDER_ATTACK": 80.0,
        }.get(status, 10.0)

    # ── Status Assessment ──────────────────────────────────────────────

    @staticmethod
    def _assess_status(
        biological: Dict,
        structural: Dict,
        cognitive: Dict,
        security: Dict,
        irritation: float,
    ) -> str:
        """Synthesize a single sovereign status from all pulses.

        Priority (highest → lowest):
          LOCKED       — HEALING_REQUIRED.lock (CPR Tier 3)
          UNDER_ATTACK — active adversarial assault
          CRITICAL     — irritation > 70 or multiple failures
          RECOVERING   — CPR tiers active
          IRRITATED    — irritation > 30
          DEGRADED     — partial failures
          ELEVATED     — irritation > 10
          ALERTED      — security events detected
          STABLE       — all nominal
        """
        # Hard overrides (check state, not just score)
        if cognitive.get("healing_lock"):
            return "LOCKED"
        if security.get("status") == "UNDER_ATTACK":
            return "UNDER_ATTACK"

        # Score-driven assessment
        if irritation >= _THRESHOLD_CRITICAL:
            return "CRITICAL"
        if cognitive.get("status") == "RECOVERING":
            return "RECOVERING"
        if irritation >= _THRESHOLD_IRRITATED:
            return "IRRITATED"

        # Partial failures
        degraded_count = sum(
            1 for s in [
                biological.get("status"),
                structural.get("status"),
            ]
            if s in ("DEGRADED", "CRITICAL", "ERROR")
        )
        if degraded_count > 0:
            return "DEGRADED"

        if irritation >= _THRESHOLD_ELEVATED:
            return "ELEVATED"
        if security.get("status") == "ALERTED":
            return "ALERTED"

        return "STABLE"


# ── Module-level helpers ───────────────────────────────────────────────

_singleton: Optional[GaiaVitals] = None


def get_vitals(log_dir: str = "/logs") -> GaiaVitals:
    """Get or create the GaiaVitals singleton."""
    global _singleton
    if _singleton is None:
        _singleton = GaiaVitals(log_dir)
    return _singleton


def get_sovereign_health(verbose: bool = True) -> Dict[str, Any]:
    """Module-level shortcut for GaiaVitals.get_sovereign_health()."""
    return get_vitals().get_sovereign_health(verbose=verbose)


def get_irritation_score() -> float:
    """Module-level shortcut for the irritation score."""
    return get_vitals().calculate_irritation_score()


def is_system_irritated(threshold: float = 30.0) -> bool:
    """Check if the system is above the irritation threshold."""
    return get_irritation_score() >= threshold
