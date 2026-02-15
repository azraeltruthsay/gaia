"""
Sleep cycle loop — runs as a daemon thread in gaia-core.

Uses gaia-common primitives (IdleMonitor) for idle detection but owns
all sleep/wake orchestration logic.  Replaces the legacy
BackgroundProcessor for the v0.3 microservice architecture.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import httpx

from gaia_common.utils.background.idle_monitor import IdleMonitor
from gaia_core.cognition.sleep_wake_manager import GaiaState, SleepWakeManager

logger = logging.getLogger("GAIA.SleepCycle")


class SleepCycleLoop:
    """Background thread that monitors idle state and drives sleep/wake."""

    POLL_INTERVAL = 10  # seconds between idle checks

    def __init__(self, config, discord_connector=None, model_pool=None, agent_core=None) -> None:
        self.config = config
        self.idle_monitor = IdleMonitor()
        self.sleep_wake_manager = SleepWakeManager(config)
        self.discord_connector = discord_connector
        self.model_pool = model_pool
        self.agent_core = agent_core
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Service URLs for SOA mode
        self._orchestrator_url = os.getenv("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
        self._web_url = os.getenv("WEB_ENDPOINT", "http://gaia-web:6414")

        # Phase 2: Sleep task scheduler
        from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler
        self.sleep_task_scheduler = SleepTaskScheduler(
            config, model_pool=model_pool, agent_core=agent_core,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="SleepCycleLoop")
        self._thread.start()
        logger.info("Sleep cycle loop started")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("Sleep cycle loop stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                idle_minutes = self.idle_monitor.get_idle_minutes()
                state = self.sleep_wake_manager.get_state()

                if state == GaiaState.AWAKE:
                    self._handle_awake(idle_minutes)
                elif state == GaiaState.SLEEPING:
                    self._handle_sleeping()
                elif state == GaiaState.FINISHING_TASK:
                    self._handle_finishing_task()
                elif state == GaiaState.WAKING:
                    self._handle_waking()
                # DROWSY is handled inside initiate_drowsy() — we just wait

            except Exception:
                logger.error("Sleep cycle error", exc_info=True)
                time.sleep(15)
                continue

            time.sleep(self.POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Per-state handlers
    # ------------------------------------------------------------------

    def _handle_awake(self, idle_minutes: float) -> None:
        if self.sleep_wake_manager.should_transition_to_drowsy(idle_minutes):
            logger.info("Idle for %.1f min — entering DROWSY", idle_minutes)
            self._update_presence("Drifting off...")

            success = self.sleep_wake_manager.initiate_drowsy()
            if success:
                self._release_gpu_for_sleep()
                self._update_presence("dreaming...", sleeping=True)
            else:
                # Cancelled or failed — reset to normal idle status
                self._update_presence(None)

    def _handle_sleeping(self) -> None:
        task = self.sleep_task_scheduler.get_next_task()
        if task is None:
            return

        # Register current task so SleepWakeManager can check interruptibility
        self.sleep_wake_manager.current_task = {
            "task_id": task.task_id,
            "interruptible": task.interruptible,
        }
        self._update_presence(f"dreaming: {task.task_type}", sleeping=True)

        self.sleep_task_scheduler.execute_task(task)

        self.sleep_wake_manager.current_task = None

        # After each task, check if a wake signal arrived
        if self.sleep_wake_manager.wake_signal_pending:
            self.sleep_wake_manager.transition_to_waking()

    def _handle_finishing_task(self) -> None:
        # When the current non-interruptible task finishes (current_task is
        # cleared by _handle_sleeping), transition to WAKING.
        if self.sleep_wake_manager.current_task is None:
            self.sleep_wake_manager.transition_to_waking()

    def _handle_waking(self) -> None:
        self._update_presence("Waking up...")
        self._reclaim_gpu_for_wake()
        restored = self.sleep_wake_manager.complete_wake()
        if restored.get("checkpoint_loaded"):
            logger.info("Context restored from checkpoint")
        self._update_presence(None)  # Reset to dynamic idle status

    # ------------------------------------------------------------------
    # GPU release / reclaim via orchestrator
    # ------------------------------------------------------------------

    def _release_gpu_for_sleep(self) -> None:
        """Ask orchestrator to stop Prime and free VRAM. Non-fatal on failure."""
        try:
            resp = httpx.post(
                f"{self._orchestrator_url}/gpu/sleep",
                json={"reason": "sleep_cycle"},
                timeout=60.0,
            )
            if resp.status_code == 200:
                logger.info("GPU released for sleep")
            else:
                logger.warning("GPU sleep request failed: %s", resp.status_code)
        except Exception:
            logger.warning("Orchestrator unreachable — sleeping without GPU release", exc_info=True)

    def _reclaim_gpu_for_wake(self) -> None:
        """Ask orchestrator to start Prime and restore model pool. Non-fatal on failure."""
        try:
            resp = httpx.post(
                f"{self._orchestrator_url}/gpu/wake",
                json={},
                timeout=180.0,  # Prime boot ~37s + health check
            )
            if resp.status_code == 200:
                logger.info("GPU reclaimed on wake")
            else:
                logger.warning("GPU wake failed: %s — staying CPU-only", resp.status_code)
        except Exception:
            logger.warning("Orchestrator unreachable — waking without GPU", exc_info=True)

    # ------------------------------------------------------------------
    # Discord presence helper
    # ------------------------------------------------------------------

    def _update_presence(self, status_text: Optional[str], sleeping: bool = False) -> None:
        """Update Discord presence.  *None* resets to the dynamic idle status.

        When *sleeping* is True, sets the Discord dot to yellow (idle).
        """
        if self.discord_connector:
            # In-process connector available (monolith / rescue mode)
            if status_text is None:
                self.discord_connector.set_idle()
            elif sleeping:
                self.discord_connector.update_presence(status_text, status_override="idle")
            else:
                self.discord_connector.update_presence(status_text)
        else:
            # SOA mode: call gaia-web /presence endpoint
            try:
                payload: dict = {"activity": status_text or "over the studio"}
                if sleeping:
                    payload["status"] = "idle"
                httpx.post(f"{self._web_url}/presence", json=payload, timeout=5.0)
            except Exception:
                logger.debug("Presence update via gaia-web failed", exc_info=True)
