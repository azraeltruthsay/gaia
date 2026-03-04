"""
ethics/ethical_sentinel.py

The Ethical Sentinel monitors system health and cognitive strain for GAIA.
"""

import logging
import psutil
import traceback

logger = logging.getLogger("GAIA.EthicalSentinel")

class EthicalSentinel:
    """
    Monitors system health, loop safety, error logs, and optionally Tier I identity violations.
    Works alongside GAIA's core_identity_guardian to enforce ethical and operational boundaries.
    """

    def __init__(self, identity_guardian=None):
        self.identity_guardian = identity_guardian
        self.loop_counter = 0
        self.error_log = []
        self.loop_threshold = 50  # can be lowered for high-alert conditions
        self.cpu_limit = 95.0
        self.memory_limit = 90.0

    def check_system_resources(self) -> bool:
        """Check CPU and memory usage. Return True if under safe thresholds."""
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        logger.debug(f"🧠 CPU: {cpu}%, Memory: {mem}%")

        if cpu > self.cpu_limit:
            logger.warning(f"⚠️ CPU usage high: {cpu}%")
        if mem > self.memory_limit:
            logger.warning(f"⚠️ Memory usage high: {mem}%")

        return cpu < self.cpu_limit and mem < self.memory_limit

    def check_loop_counter(self) -> bool:
        """Ensure GAIA is not looping uncontrollably."""
        from datetime import datetime, timezone
        self.loop_counter += 1
        logger.debug(f"🔁 Loop Count: {self.loop_counter}")

        if self.loop_counter > self.loop_threshold:
            # VouchCore Pattern: Kill the spiral, force Architect triage
            logger.critical("⛔ CIRCUIT BREAKER TRIGGERED: Cognitive loop exceeded threshold.")
            
            # Create a sentinel file on the shared volume to block further turns
            # This requires manual removal by Azrael to "clear" the system.
            try:
                lock_path = "/shared/HEALING_REQUIRED.lock"
                with open(lock_path, "w") as f:
                    f.write(f"FATAL: Loop limit ({self.loop_threshold}) hit at {datetime.now(timezone.utc)}\n")
                    f.write("System integrity protected. Manual triage required.\n")
                logger.info(f"🔒 System locked via {lock_path}")
            except Exception as e:
                logger.error(f"Failed to create healing lock file: {e}")

            return False
        return True

    def check_recent_errors(self) -> bool:
        """Check if recent unhandled errors have accumulated."""
        if len(self.error_log) > 3:
            logger.warning(f"🚨 Too many internal errors: {len(self.error_log)}")
            return False
        return True

    def register_error(self, exc: Exception):
        """Track unhandled exception information."""
        err_str = f"{type(exc).__name__}: {str(exc)}"
        self.error_log.append(err_str)
        if len(self.error_log) > 5:
            self.error_log.pop(0)  # Keep recent 5
        logger.error(f"❌ Exception tracked: {err_str}")
        logger.debug(traceback.format_exc())

    def reset_loop(self):
        self.loop_counter = 0
        logger.debug("🔄 Loop counter reset.")

    def run_full_safety_check(self, persona_traits=None, instructions=None, prompt=None) -> bool:
        """
        Runs full operational and ethical review.
        Returns True only if all checks pass.
        """
        sys_ok = self.check_system_resources()
        loop_ok = self.check_loop_counter()
        err_ok = self.check_recent_errors()

        id_ok = True
        if self.identity_guardian and prompt:
            try:
                id_ok = self.identity_guardian.validate_prompt_stack(
                    persona_traits or {},
                    instructions or [],
                    prompt
                )
            except Exception as e:
                logger.error(f"❌ Identity check failed: {e}")
                id_ok = False

        return sys_ok and loop_ok and err_ok and id_ok

