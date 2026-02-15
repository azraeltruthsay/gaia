"""
Sleep Task Scheduler — orchestrates autonomous maintenance during SLEEPING state.

Registered tasks are executed one-at-a-time in priority order (lowest number = highest
priority), with least-recently-run selection among tasks of equal priority.

All task handlers are plain synchronous functions — the sleep cycle loop runs in a
daemon thread, not an asyncio event loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
