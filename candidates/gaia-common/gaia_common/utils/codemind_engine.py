"""CodeMind Engine — GAIA's autonomous self-improvement state machine.

Implements the DETECT → ANALYZE → PROPOSE → VALIDATE → APPLY → VERIFY → PROMOTE
loop with circuit breaker, scope tier enforcement, and safety gates at every stage.

All operations target candidates/ only (Production Lock). Promotion requires
separate approval for vital organs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.CodeMind")


# ── State Machine ────────────────────────────────────────────────────────

class CodeMindState(str, Enum):
    IDLE = "IDLE"
    DETECT = "DETECT"
    ANALYZE = "ANALYZE"
    PROPOSE = "PROPOSE"
    VALIDATE = "VALIDATE"
    APPLY = "APPLY"
    VERIFY = "VERIFY"
    PROMOTE = "PROMOTE"


class ScopeTier(int, Enum):
    """What CodeMind can touch — higher tier = more dangerous."""
    TIER1_AUTO = 1        # Knowledge, config, curriculum
    TIER2_SUPERVISED = 2  # Candidate Python code
    TIER3_GATED = 3       # Promotion to production


class TriggerSource(str, Enum):
    USER_REQUEST = "user_request"
    IMMUNE_IRRITATION = "immune_irritation"
    DRIFT_DETECTION = "drift_detection"
    SLEEP_CYCLE = "sleep_cycle"


# Files that require Azrael approval before promotion
VITAL_ORGANS = frozenset({
    "main.py", "agent_core.py", "mcp_client.py", "tools.py", "immune_system.py",
})

# Tier 1: auto-approvable paths (knowledge, config, curriculum)
TIER1_PATTERNS = (
    "/knowledge/", "/curricula/", "gaia_constants.json",
    "/awareness/", "/personas/",
)


@dataclass
class CodeMindChange:
    """A single proposed change within a cycle."""
    file_path: str
    issue: str
    scope_tier: int
    diff_summary: str = ""
    validation_result: Optional[Dict[str, Any]] = None
    applied: bool = False
    verified: bool = False
    promoted: bool = False
    error: Optional[str] = None


@dataclass
class CycleContext:
    """Tracks state for a single CodeMind cycle."""
    cycle_id: str = ""
    trigger: str = ""
    state: str = CodeMindState.IDLE.value
    started_at: str = ""
    changes: List[Dict[str, Any]] = field(default_factory=list)
    dry_run: bool = True
    errors: List[str] = field(default_factory=list)


# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "enabled": False,
    "max_changes_per_cycle": 3,
    "auto_promote": False,
    "dry_run": True,
    "scope_tiers": {
        "tier1_auto": True,
        "tier2_supervised": True,
        "tier3_gated": False,
    },
    "validation": {
        "py_compile": True,
        "ruff": True,
        "ast_parse": True,
        "pytest": False,
    },
    "triggers": {
        "sleep_cycle": True,
        "immune_irritation": True,
        "drift_detection": False,
        "user_request": True,
    },
    "cognitive_battery_gate": 0.85,
    "regression_threshold": 0.05,
}


def _load_config(constants: dict | None = None) -> dict:
    """Merge CODEMIND block from constants with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if constants and "CODEMIND" in constants:
        user_cfg = constants["CODEMIND"]
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# ── Scope Classification ────────────────────────────────────────────────

def classify_scope(file_path: str) -> ScopeTier:
    """Determine the scope tier for a given file path."""
    norm = file_path.replace("\\", "/")
    for pattern in TIER1_PATTERNS:
        if pattern in norm:
            return ScopeTier.TIER1_AUTO
    if "/candidates/" in norm or norm.startswith("candidates/"):
        return ScopeTier.TIER2_SUPERVISED
    return ScopeTier.TIER3_GATED


def is_vital_organ(file_path: str) -> bool:
    """Check if a file is a vital organ requiring Azrael approval."""
    return Path(file_path).name in VITAL_ORGANS


# ── Circuit Breaker ──────────────────────────────────────────────────────

class CircuitBreaker:
    """Limits changes per cycle to prevent runaway self-modification."""

    def __init__(self, max_changes: int = 3):
        self.max_changes = max_changes
        self._count = 0
        self._tripped = False

    def increment(self) -> bool:
        """Record a change. Returns True if still within limits."""
        self._count += 1
        if self._count >= self.max_changes:
            self._tripped = True
            logger.warning(
                "Circuit breaker tripped: %d/%d changes",
                self._count, self.max_changes,
            )
        return not self._tripped

    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._count = 0
        self._tripped = False

    @property
    def remaining(self) -> int:
        return max(0, self.max_changes - self._count)

    def status(self) -> dict:
        return {
            "count": self._count,
            "max": self.max_changes,
            "remaining": self.remaining,
            "tripped": self._tripped,
        }


# ── Engine ───────────────────────────────────────────────────────────────

class CodeMindEngine:
    """State machine for autonomous code improvement cycles."""

    VALID_TRANSITIONS = {
        CodeMindState.IDLE: {CodeMindState.DETECT},
        CodeMindState.DETECT: {CodeMindState.ANALYZE, CodeMindState.IDLE},
        CodeMindState.ANALYZE: {CodeMindState.PROPOSE, CodeMindState.IDLE},
        CodeMindState.PROPOSE: {CodeMindState.VALIDATE, CodeMindState.IDLE},
        CodeMindState.VALIDATE: {CodeMindState.APPLY, CodeMindState.IDLE},
        CodeMindState.APPLY: {CodeMindState.VERIFY, CodeMindState.IDLE},
        CodeMindState.VERIFY: {CodeMindState.PROMOTE, CodeMindState.IDLE},
        CodeMindState.PROMOTE: {CodeMindState.IDLE},
    }

    # Shared adapter with code-architect persona — same coding skill, different context
    ADAPTER_NAME = "code-architect"

    def __init__(self, constants: dict | None = None):
        self.config = _load_config(constants)
        self.state = CodeMindState.IDLE
        self.circuit_breaker = CircuitBreaker(self.config["max_changes_per_cycle"])
        self._current_cycle: Optional[CycleContext] = None
        self._changelog_path = os.environ.get(
            "CODEMIND_CHANGELOG_PATH",
            "/shared/codemind/changelog.jsonl",
        )

    # ── State transitions ────────────────────────────────────────────

    def transition(self, new_state: CodeMindState) -> bool:
        """Attempt a state transition. Returns False if invalid."""
        if new_state not in self.VALID_TRANSITIONS.get(self.state, set()):
            logger.error(
                "Invalid transition: %s → %s", self.state.value, new_state.value,
            )
            return False
        old = self.state
        self.state = new_state
        if self._current_cycle:
            self._current_cycle.state = new_state.value
        logger.debug("CodeMind: %s → %s", old.value, new_state.value)
        return True

    def reset(self) -> None:
        """Return to IDLE and reset circuit breaker."""
        self.state = CodeMindState.IDLE
        self.circuit_breaker.reset()
        self._current_cycle = None

    # ── Scope enforcement ────────────────────────────────────────────

    def is_scope_allowed(self, file_path: str) -> bool:
        """Check if the engine's config allows changes at this scope tier."""
        tier = classify_scope(file_path)
        tier_map = {
            ScopeTier.TIER1_AUTO: "tier1_auto",
            ScopeTier.TIER2_SUPERVISED: "tier2_supervised",
            ScopeTier.TIER3_GATED: "tier3_gated",
        }
        key = tier_map.get(tier, "tier3_gated")
        return self.config["scope_tiers"].get(key, False)

    # ── Trigger gate ─────────────────────────────────────────────────

    def is_trigger_allowed(self, source: TriggerSource) -> bool:
        """Check if this trigger source is enabled in config."""
        return self.config["triggers"].get(source.value, False)

    # ── Cycle management ─────────────────────────────────────────────

    def start_cycle(self, trigger: TriggerSource, context: str = "") -> CycleContext:
        """Begin a new CodeMind cycle."""
        now = datetime.now(timezone.utc)
        cycle = CycleContext(
            cycle_id=f"cm-{now.strftime('%Y%m%d-%H%M%S')}",
            trigger=trigger.value,
            state=CodeMindState.DETECT.value,
            started_at=now.isoformat(),
            dry_run=self.config["dry_run"],
        )
        self._current_cycle = cycle
        self.circuit_breaker.reset()
        self.transition(CodeMindState.DETECT)
        logger.info(
            "CodeMind cycle started: %s (trigger=%s, dry_run=%s)",
            cycle.cycle_id, trigger.value, cycle.dry_run,
        )
        return cycle

    def record_change(self, change: CodeMindChange) -> None:
        """Record a change in the current cycle and increment circuit breaker."""
        if self._current_cycle:
            self._current_cycle.changes.append(asdict(change))
        self.circuit_breaker.increment()

    def end_cycle(self, outcome: str = "complete") -> dict:
        """Finalize and log the cycle."""
        self.transition(CodeMindState.IDLE)
        result = {
            "cycle_id": self._current_cycle.cycle_id if self._current_cycle else "unknown",
            "outcome": outcome,
            "changes_count": len(self._current_cycle.changes) if self._current_cycle else 0,
            "dry_run": self._current_cycle.dry_run if self._current_cycle else True,
            "errors": self._current_cycle.errors if self._current_cycle else [],
        }
        # Write to changelog
        if self._current_cycle:
            self._write_changelog(result)
        self.reset()
        return result

    # ── Fix Prompt Construction ───────────────────────────────────────

    @staticmethod
    def build_fix_prompt(
        file_path: str,
        issue_description: str,
        file_content: str,
        suggestion: str = "",
    ) -> str:
        """Build the LLM prompt for CodeMind's PROPOSE stage.

        Uses the code-architect adapter but with a surgical fix context,
        not a generative blueprint context.
        """
        suggestion_line = f"\nSUGGESTED APPROACH: {suggestion}" if suggestion else ""
        return f"""You are CodeMind, GAIA's code self-improvement layer. Fix the specific error below.

RULES:
- Make the MINIMAL change needed to fix the error
- Do NOT add comments, docstrings, or refactoring beyond the fix
- Do NOT reorganize code that isn't broken
- Prefer deletion over addition
- If the fix requires changes to multiple files, respond: CANNOT_FIX: requires multi-file change
- If the error is ambiguous, respond: CANNOT_FIX: <reason>

FILE: {file_path}
ERROR: {issue_description}{suggestion_line}

CURRENT CODE:
```python
{file_content}
```

Respond with ONLY the complete fixed file content. No markdown fences. No explanation.
Start with the first line of the file."""

    # ── Awareness Prompt Construction ─────────────────────────────────

    @staticmethod
    def build_awareness_prompt(
        gap_description: str,
        current_awareness: str,
    ) -> str:
        """Build LLM prompt for awareness file additions (not code fixes).

        Used when CodeMind detects a capability_gap — generates a markdown
        addition to an awareness file rather than a code diff.
        """
        return f"""You are CodeMind, GAIA's self-awareness layer. A capability gap was detected.

RULES:
- Add a SINGLE concise markdown bullet or short paragraph to the awareness file
- State what GAIA CANNOT do or does NOT have access to
- Do NOT remove or modify existing content
- Do NOT add speculative capabilities — only document confirmed limitations
- Keep additions under 3 lines
- If this gap is already documented, respond: ALREADY_DOCUMENTED

GAP DETECTED:
{gap_description}

CURRENT AWARENESS FILE:
{current_awareness}

Respond with ONLY the new markdown content to APPEND (no fences, no explanation).
Start with a bullet point (- )."""

    # ── Changelog ────────────────────────────────────────────────────

    def _write_changelog(self, result: dict) -> None:
        """Append cycle result to changelog JSONL."""
        try:
            from gaia_common.utils.codemind_changelog import append_entry
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle_id": result.get("cycle_id", "unknown"),
                "trigger": self._current_cycle.trigger if self._current_cycle else "",
                "outcome": result.get("outcome", "unknown"),
                "changes": self._current_cycle.changes if self._current_cycle else [],
                "dry_run": result.get("dry_run", True),
                "errors": result.get("errors", []),
            }
            append_entry(entry, self._changelog_path)
        except Exception as e:
            logger.warning("Failed to write changelog: %s", e)

    # ── Status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        """Current engine status for dashboard/API."""
        return {
            "state": self.state.value,
            "enabled": self.config["enabled"],
            "dry_run": self.config["dry_run"],
            "auto_promote": self.config["auto_promote"],
            "circuit_breaker": self.circuit_breaker.status(),
            "current_cycle": asdict(self._current_cycle) if self._current_cycle else None,
            "scope_tiers": self.config["scope_tiers"],
            "triggers": self.config["triggers"],
        }
