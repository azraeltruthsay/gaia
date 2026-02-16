"""
Sleep Task Scheduler — orchestrates autonomous maintenance during SLEEPING state.

Registered tasks are executed one-at-a-time in priority order (lowest number = highest
priority), with least-recently-run selection among tasks of equal priority.

All task handlers are plain synchronous functions — the sleep cycle loop runs in a
daemon thread, not an asyncio event loop.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SleepTaskScheduler")


@dataclass
class SleepTask:
    """A single registerable sleep-time task."""

    task_id: str
    task_type: str
    priority: int  # 1 = highest
    interruptible: bool
    estimated_duration_seconds: int
    handler: Callable[[], Any]
    last_run: Optional[datetime] = None
    run_count: int = 0
    last_error: Optional[str] = None


class SleepTaskScheduler:
    """Priority-based scheduler for sleep-time maintenance tasks."""

    def __init__(
        self,
        config,
        model_pool=None,
        agent_core=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self.agent_core = agent_core
        self._tasks: List[SleepTask] = []

        self._register_default_tasks()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def register_task(self, task: SleepTask) -> None:
        self._tasks.append(task)
        logger.info("Registered sleep task: %s (P%d)", task.task_id, task.priority)

    def _register_default_tasks(self) -> None:
        """Register built-in maintenance tasks."""

        self.register_task(SleepTask(
            task_id="conversation_curation",
            task_type="conversation_curation",
            priority=1,
            interruptible=True,
            estimated_duration_seconds=60,
            handler=self._run_conversation_curation,
        ))

        self.register_task(SleepTask(
            task_id="thought_seed_review",
            task_type="thought_seed_review",
            priority=1,
            interruptible=True,
            estimated_duration_seconds=120,
            handler=self._run_thought_seed_review,
        ))

        self.register_task(SleepTask(
            task_id="initiative_cycle",
            task_type="initiative_cycle",
            priority=2,
            interruptible=True,
            estimated_duration_seconds=180,
            handler=self._run_initiative_cycle,
        ))

        self.register_task(SleepTask(
            task_id="blueprint_validation",
            task_type="blueprint_validation",
            priority=3,
            interruptible=True,
            estimated_duration_seconds=300,
            handler=self._run_blueprint_validation,
        ))

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def get_next_task(self) -> Optional[SleepTask]:
        """Return the highest-priority, least-recently-run task."""
        if not self._tasks:
            return None

        # Sort by (priority ASC, last_run ASC nulls-first)
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        candidates = sorted(
            self._tasks,
            key=lambda t: (t.priority, t.last_run or epoch),
        )
        return candidates[0] if candidates else None

    def execute_task(self, task: SleepTask) -> bool:
        """Execute a task handler. Returns True on success."""
        logger.info("Starting sleep task: %s", task.task_id)
        start = time.monotonic()
        try:
            task.handler()
            elapsed = time.monotonic() - start
            task.last_run = datetime.now(timezone.utc)
            task.run_count += 1
            task.last_error = None
            logger.info("Completed %s in %.1fs (run #%d)", task.task_id, elapsed, task.run_count)
            return True
        except Exception as exc:
            elapsed = time.monotonic() - start
            task.last_run = datetime.now(timezone.utc)
            task.last_error = str(exc)
            logger.error("Task %s failed after %.1fs: %s", task.task_id, elapsed, exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "priority": t.priority,
                "interruptible": t.interruptible,
                "run_count": t.run_count,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "last_error": t.last_error,
            }
            for t in self._tasks
        ]

    # ------------------------------------------------------------------
    # Built-in task handlers
    # ------------------------------------------------------------------

    def _run_conversation_curation(self) -> None:
        """Curate recent session conversations for the knowledge base."""
        from gaia_core.cognition.conversation_curator import ConversationCurator
        from gaia_core.memory.session_manager import SessionManager

        curator = ConversationCurator()
        session_manager = SessionManager(self.config)

        # Curate all active sessions that have enough messages
        curated = 0
        for sid, session in session_manager.sessions.items():
            if session.history:
                if curator.curate(sid, session.history):
                    curated += 1

        logger.info("Conversation curation: %d sessions curated", curated)

    def _run_thought_seed_review(self) -> None:
        """Review unprocessed thought seeds using CPU Lite model."""
        from gaia_core.cognition.thought_seed import review_and_process_seeds

        llm = None
        if self.model_pool is not None:
            llm = self.model_pool.get_model_for_role("lite")

        review_and_process_seeds(config=self.config, llm=llm, auto_act=True)

    def _run_initiative_cycle(self) -> None:
        """Execute one autonomous thought cycle via the initiative engine."""
        from gaia_core.cognition.initiative_engine import InitiativeEngine

        engine = InitiativeEngine(config=self.config, agent_core=self.agent_core)
        engine.execute_turn()

    # Blueprint-to-source mapping for validation
    _BLUEPRINT_SOURCES: Dict[str, List[str]] = {
        "GAIA_SLEEP_CYCLE.md": [
            "gaia_core/cognition/sleep_wake_manager.py",
            "gaia_core/cognition/sleep_cycle_loop.py",
            "gaia_core/api/sleep_endpoints.py",
            "gaia_core/utils/resource_monitor.py",
        ],
        "GAIA_CORE.md": [
            "gaia_core/cognition/agent_core.py",
            "gaia_core/main.py",
        ],
        "GAIA_ORCHESTRATOR.md": [
            "gaia_orchestrator/handoff_manager.py",
            "gaia_orchestrator/gpu_manager.py",
        ],
    }

    def _run_blueprint_validation(self) -> None:
        """Scan blueprints against source files and flag stale content."""

        blueprints_dir = Path("/gaia/GAIA_Project/knowledge/blueprints")
        # Candidate sources preferred over live (they're the latest)
        source_roots = [
            Path("/gaia/GAIA_Project/candidates/gaia-core"),
            Path("/gaia/GAIA_Project/candidates/gaia-orchestrator"),
            Path("/gaia/GAIA_Project/gaia-core"),
            Path("/gaia/GAIA_Project/gaia-orchestrator"),
        ]

        total_mismatches = 0

        for bp_name, source_files in self._BLUEPRINT_SOURCES.items():
            bp_path = blueprints_dir / bp_name
            if not bp_path.exists():
                logger.debug("Blueprint %s not found, skipping", bp_name)
                continue

            bp_text = bp_path.read_text(encoding="utf-8")
            facts = self._extract_facts(source_files, source_roots)
            missing = self._check_facts(facts, bp_text)

            if missing:
                total_mismatches += len(missing)
                logger.warning(
                    "Blueprint %s has %d stale references: %s",
                    bp_name, len(missing), missing,
                )
                self._append_update_notes(bp_path, bp_text, missing)
            else:
                logger.info("Blueprint %s is up-to-date", bp_name)

        logger.info(
            "Blueprint validation complete: %d mismatches across %d blueprints",
            total_mismatches, len(self._BLUEPRINT_SOURCES),
        )

    @staticmethod
    def _extract_facts(
        source_files: List[str],
        source_roots: List[Path],
    ) -> Dict[str, List[str]]:
        """Extract enum members, endpoints, and key constants from source files.

        Returns dict of {category: [fact_string, ...]}.
        """
        facts: Dict[str, List[str]] = {
            "enums": [],
            "endpoints": [],
            "constants": [],
        }

        re_enum_class = re.compile(r"^class\s+(\w+)\(.*Enum.*\):", re.MULTILINE)
        re_enum_member = re.compile(r"^\s+(\w+)\s*=\s*", re.MULTILINE)
        re_endpoint = re.compile(
            r'@router\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
            re.MULTILINE,
        )
        re_constant = re.compile(r"^([A-Z][A-Z_]{2,})\s*=\s*", re.MULTILINE)

        for rel_path in source_files:
            src_path = None
            for root in source_roots:
                candidate = root / rel_path
                if candidate.exists():
                    src_path = candidate
                    break
            if src_path is None:
                continue

            text = src_path.read_text(encoding="utf-8")

            # Extract enum members
            for class_match in re_enum_class.finditer(text):
                class_name = class_match.group(1)
                # Find members in the lines following the class definition
                class_start = class_match.end()
                # Read until next class/def at column 0
                block_end = len(text)
                for m in re.finditer(r"^(?:class |def )", text[class_start:], re.MULTILINE):
                    block_end = class_start + m.start()
                    break
                block = text[class_start:block_end]
                for member_match in re_enum_member.finditer(block):
                    member = member_match.group(1)
                    if not member.startswith("_"):
                        facts["enums"].append(f"{class_name}.{member}")

            # Extract endpoints
            for ep_match in re_endpoint.finditer(text):
                method = ep_match.group(1).upper()
                path = ep_match.group(2)
                facts["endpoints"].append(f"{method} {path}")

            # Extract top-level constants
            for const_match in re_constant.finditer(text):
                facts["constants"].append(const_match.group(1))

        return facts

    @staticmethod
    def _check_facts(
        facts: Dict[str, List[str]],
        bp_text: str,
    ) -> List[str]:
        """Return list of facts not found in blueprint text."""
        missing: List[str] = []
        for category, items in facts.items():
            for item in items:
                # For enum members like GaiaState.ACTIVE, check the member name
                if category == "enums":
                    member = item.split(".")[-1]
                    if member not in bp_text:
                        missing.append(f"enum:{item}")
                elif category == "endpoints":
                    # Check for the path portion (e.g., "/sleep/study-handoff")
                    path = item.split(" ", 1)[1]
                    if path not in bp_text:
                        missing.append(f"endpoint:{item}")
                elif category == "constants":
                    if item not in bp_text:
                        missing.append(f"constant:{item}")
        return missing

    @staticmethod
    def _append_update_notes(
        bp_path: Path,
        bp_text: str,
        missing: List[str],
    ) -> None:
        """Append a timestamped 'Recent Implementation Updates' note to a blueprint."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = "\n## Recent Implementation Updates\n"

        # Check if section already exists
        if "## Recent Implementation Updates" in bp_text:
            # Append to existing section
            note = f"\n### {timestamp}\n\nDetected {len(missing)} item(s) not reflected in blueprint:\n"
        else:
            note = f"{header}\n### {timestamp}\n\nDetected {len(missing)} item(s) not reflected in blueprint:\n"

        for item in missing:
            note += f"- `{item}`\n"
        note += "\n*Auto-detected by blueprint_validation sleep task.*\n"

        with open(bp_path, "a", encoding="utf-8") as f:
            f.write(note)
