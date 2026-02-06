"""
background/idle_monitor.py

Handles idle time and overnight period detection for GAIA background processing.
"""

import time
import logging
from pathlib import Path
from app.commands.self_analysis_trigger import run_self_analysis
from app.status_tracker import get_idle_duration

logger = logging.getLogger("GAIA.IdleMonitor")

class IdleMonitor:
    """
    Monitors system activity to determine whether GAIA is idle.
    Used to trigger background tasks like summarization and memory cleanup.
    """

    def __init__(self, idle_seconds=600):
        self.last_active_time = time.time()
        self.idle_threshold = idle_seconds

    def mark_active(self):
        """Update the last active time to now."""
        self.last_active_time = time.time()
        logger.debug("⏱️ GAIA marked as active.")

    def is_system_idle(self) -> bool:
        """
        Check if the system has been idle long enough to run background tasks.

        Returns:
            bool: True if system is idle
        """
        idle_time = time.time() - self.last_active_time
        if idle_time > self.idle_threshold:
            logger.debug(f"🌙 System idle for {idle_time:.1f} seconds.")
            return True
        return False


    def idle_check(self, ai_manager):
        """
        Run background task if idle long enough and summary not present.
        """
        idle_time = get_idle_duration()
        if idle_time > self.idle_threshold and not summary_exists():
            logger.info("🤖 Initiating self-analysis during idle period.")
            run_self_analysis(ai_manager)

def summary_exists():
    return (
        Path("/app/knowledge/system_reference/code_summaries/code_summary.md").exists()
        and Path("/app/knowledge/system_reference/code_summaries/function_map.md").exists()
    )