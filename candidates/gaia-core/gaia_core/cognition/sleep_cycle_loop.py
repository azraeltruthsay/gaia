"""
Sleep cycle loop — runs as a daemon thread in gaia-core.

Uses gaia-common primitives (IdleMonitor) for idle detection but owns
all sleep/wake orchestration logic.  Replaces the legacy
BackgroundProcessor for the v0.3 microservice architecture.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from gaia_common.utils.background.idle_monitor import IdleMonitor
from gaia_core.cognition.sleep_wake_manager import GaiaState, SleepWakeManager

logger = logging.getLogger("GAIA.SleepCycle")


class SleepCycleLoop:
    """Background thread that monitors idle state and drives sleep/wake."""

    POLL_INTERVAL = 10  # seconds between idle checks

    def __init__(self, config, discord_connector=None) -> None:
        self.config = config
        self.idle_monitor = IdleMonitor()
        self.sleep_wake_manager = SleepWakeManager(config)
        self.discord_connector = discord_connector
        self._thread: Optional[threading.Thread] = None
        self._running = False

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
                self._update_presence("Sleeping")
            else:
                # Cancelled or failed — reset to normal idle status
                self._update_presence(None)

    def _handle_sleeping(self) -> None:
        # Phase 2 will add SleepTaskScheduler task execution here.
        # For now we just poll — wake signals are received via the
        # /sleep/wake HTTP endpoint and handled by SleepWakeManager.
        pass

    def _handle_finishing_task(self) -> None:
        # When the current non-interruptible task finishes, the task
        # executor (Phase 2) will call transition_to_waking().
        # For Phase 1 there are no tasks, so transition immediately.
        if self.sleep_wake_manager.current_task is None:
            self.sleep_wake_manager.transition_to_waking()

    def _handle_waking(self) -> None:
        self._update_presence("Waking up...")
        restored = self.sleep_wake_manager.complete_wake()
        if restored.get("checkpoint_loaded"):
            logger.info("Context restored from checkpoint")
        self._update_presence(None)  # Reset to dynamic idle status

    # ------------------------------------------------------------------
    # Discord presence helper
    # ------------------------------------------------------------------

    def _update_presence(self, status_text: Optional[str]) -> None:
        """Update Discord presence.  *None* resets to the dynamic idle status."""
        if not self.discord_connector:
            return
        if status_text is None:
            self.discord_connector.set_idle()
        else:
            self.discord_connector.update_presence(status_text)
