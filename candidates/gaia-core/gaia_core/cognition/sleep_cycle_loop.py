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
from gaia_core.cognition.sleep_wake_manager import (
    GaiaState,
    SleepWakeManager,
    _TransientPhase,
)

logger = logging.getLogger("GAIA.SleepCycle")


class SleepCycleLoop:
    """Background thread that monitors idle state and drives sleep/wake."""

    POLL_INTERVAL_ACTIVE = 10  # seconds between idle checks when ACTIVE
    POLL_INTERVAL_ASLEEP = 2   # seconds when ASLEEP — react fast to wake signals
    DISTRACTED_RECHECK_INTERVAL = 300  # 5 min between distracted rechecks

    def __init__(self, config, discord_connector=None, model_pool=None, agent_core=None) -> None:
        self.config = config
        self.idle_monitor = IdleMonitor()
        self.sleep_wake_manager = SleepWakeManager(config)
        self.discord_connector = discord_connector
        self.model_pool = model_pool
        self.agent_core = agent_core
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_distracted_recheck = 0.0

        # Service URLs for SOA mode
        self._orchestrator_url = os.getenv("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
        self._web_url = os.getenv("WEB_ENDPOINT", "http://gaia-web:6414")

        # Phase 2: Sleep task scheduler
        from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler
        self.sleep_task_scheduler = SleepTaskScheduler(
            config, model_pool=model_pool, agent_core=agent_core,
        )

        # Resource monitor for distracted detection
        self._resource_monitor = None
        try:
            from gaia_core.utils.resource_monitor import ResourceMonitor
            self._resource_monitor = ResourceMonitor.get_instance()
        except Exception:
            logger.debug("ResourceMonitor not available — distracted detection disabled")

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

    def initiate_shutdown(self) -> None:
        """Graceful shutdown: transition to OFFLINE and stop the loop."""
        self.sleep_wake_manager.initiate_offline()
        self._update_presence(None, offline=True)
        self.stop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                state = self.sleep_wake_manager.get_state()

                # OFFLINE means we're done
                if state == GaiaState.OFFLINE:
                    break

                idle_minutes = self.idle_monitor.get_idle_minutes()

                if state == GaiaState.ACTIVE:
                    self._handle_active(idle_minutes)
                elif state == GaiaState.ASLEEP:
                    self._handle_asleep()
                elif state == GaiaState.DREAMING:
                    self._handle_dreaming()
                elif state == GaiaState.DISTRACTED:
                    self._handle_distracted()
                # DROWSY is handled inside initiate_drowsy() — we just wait

            except Exception:
                logger.error("Sleep cycle error", exc_info=True)
                time.sleep(15)
                continue

            # Poll faster when asleep to react quickly to wake signals
            if state in (GaiaState.ASLEEP, GaiaState.DISTRACTED):
                time.sleep(self.POLL_INTERVAL_ASLEEP)
            else:
                time.sleep(self.POLL_INTERVAL_ACTIVE)

    # ------------------------------------------------------------------
    # Per-state handlers
    # ------------------------------------------------------------------

    def _handle_active(self, idle_minutes: float) -> None:
        if self.sleep_wake_manager.should_transition_to_drowsy(idle_minutes):
            logger.info("Idle for %.1f min — entering DROWSY", idle_minutes)
            self._update_presence("drifting off...")

            success = self.sleep_wake_manager.initiate_drowsy()
            if success:
                self._release_gpu_for_sleep()
                self._update_presence("sleeping...", sleeping=True)
            else:
                # Cancelled or failed — reset to normal idle status
                self._update_presence(None)

    def _handle_asleep(self) -> None:
        # Check transient phases first
        phase = self.sleep_wake_manager._phase

        if phase == _TransientPhase.FINISHING_TASK:
            # When the current non-interruptible task finishes, transition to WAKING
            if self.sleep_wake_manager.current_task is None:
                self.sleep_wake_manager.transition_to_waking()
            return

        if phase == _TransientPhase.WAKING:
            self._update_presence("waking up...")
            self._reclaim_gpu_for_wake()
            restored = self.sleep_wake_manager.complete_wake()
            if restored.get("checkpoint_loaded"):
                logger.info("Context restored from checkpoint")
            self._update_presence(None)  # Reset to dynamic idle status
            return

        # Check for distracted state (sustained load)
        if self._resource_monitor and self._resource_monitor.is_distracted():
            self.sleep_wake_manager.enter_distracted()
            self._update_presence("occupied...", status_override="dnd")
            return

        # Normal ASLEEP: run sleep tasks
        task = self.sleep_task_scheduler.get_next_task()
        if task is None:
            return

        # Register current task so SleepWakeManager can check interruptibility
        self.sleep_wake_manager.current_task = {
            "task_id": task.task_id,
            "interruptible": task.interruptible,
        }
        self._update_presence(f"sleeping: {task.task_type}", sleeping=True)

        self.sleep_task_scheduler.execute_task(task)

        self.sleep_wake_manager.current_task = None

        # After each task, check if a wake signal arrived
        if self.sleep_wake_manager.wake_signal_pending:
            self.sleep_wake_manager.transition_to_waking()

    def _handle_dreaming(self) -> None:
        """DREAMING is driven by orchestrator API calls — nothing to do here."""
        self._update_presence("studying...", status_override="dnd")

    def _handle_distracted(self) -> None:
        """Periodically recheck if system load has dropped."""
        now = time.monotonic()
        if now - self._last_distracted_recheck < self.DISTRACTED_RECHECK_INTERVAL:
            return

        self._last_distracted_recheck = now

        if self._resource_monitor and self._resource_monitor.check_and_clear_distracted():
            self.sleep_wake_manager.exit_distracted()
            self._update_presence("sleeping...", sleeping=True)

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

    def _update_presence(
        self,
        status_text: Optional[str],
        sleeping: bool = False,
        offline: bool = False,
        status_override: Optional[str] = None,
    ) -> None:
        """Update Discord presence.

        *None* resets to the dynamic idle status.
        *sleeping* sets the Discord dot to yellow (idle).
        *offline* sets the Discord status to invisible.
        *status_override* allows explicit status ("dnd", "idle", etc.).
        """
        if self.discord_connector:
            # In-process connector available (monolith / rescue mode)
            if offline:
                self.discord_connector.update_presence(None, status_override="invisible")
            elif status_text is None:
                self.discord_connector.set_idle()
            elif sleeping:
                self.discord_connector.update_presence(status_text, status_override="idle")
            elif status_override:
                self.discord_connector.update_presence(status_text, status_override=status_override)
            else:
                self.discord_connector.update_presence(status_text)
        else:
            # SOA mode: call gaia-web /presence endpoint
            try:
                payload: dict = {"activity": status_text or "over the studio"}
                if offline:
                    payload["status"] = "invisible"
                elif sleeping:
                    payload["status"] = "idle"
                elif status_override:
                    payload["status"] = status_override
                httpx.post(f"{self._web_url}/presence", json=payload, timeout=5.0)
            except Exception:
                logger.debug("Presence update via gaia-web failed", exc_info=True)
