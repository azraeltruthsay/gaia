from datetime import datetime, timezone
"""
ethics/ethical_sentinel.py

The Ethical Sentinel monitors system health and cognitive strain for GAIA.

Phase 5i — CPR Loop (Cognitive Pulse Resuscitation):
  Instead of a binary lock-on-loop, the sentinel implements a tiered
  escalation ladder that attempts self-recovery before requiring manual
  Architect intervention:
    Tier 1 (Breath):      KV cache reset for the session
    Tier 2 (Diagnosis):   Nano analyzes the loop trace, injects loop_diagnosis
    Tier 3 (Intubation):  HEALING_REQUIRED.lock (manual triage)
"""

import json
import logging
import os
import psutil
import traceback
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.EthicalSentinel")

_NANO_ENDPOINT = os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080")


class RecoverySignal:
    """Signal from the CPR escalation ladder back to agent_core.run_turn()."""

    def __init__(self, tier: int, action: str, diagnosis: str = "", success: bool = False):
        self.tier = tier          # 1=Breath, 2=Diagnosis, 3=Intubation
        self.action = action      # kv_reset | diagnose | lock
        self.diagnosis = diagnosis  # Nano's analysis (Tier 2+)
        self.success = success    # Did this tier resolve the issue?

    def __repr__(self):
        return f"RecoverySignal(tier={self.tier}, action={self.action}, success={self.success})"


class EthicalSentinel:
    """
    Monitors system health, loop safety, error logs, and optionally Tier I identity violations.
    Works alongside GAIA's core_identity_guardian to enforce ethical and operational boundaries.

    CPR Loop Escalation:
      - loop_counter hits tier_1_threshold (15) -> Tier 1: KV cache reset
      - loop_counter hits tier_2_threshold (30) -> Tier 2: Nano diagnosis
      - loop_counter hits tier_3_threshold (50) -> Tier 3: HEALING_REQUIRED.lock
    """

    def __init__(self, identity_guardian=None):
        self.identity_guardian = identity_guardian
        self.loop_counter = 0
        self.error_log = []
        self.cpu_limit = 95.0
        self.memory_limit = 90.0

        # CPR Loop thresholds
        self.tier_1_threshold = 15   # Breath: KV cache reset
        self.tier_2_threshold = 30   # Diagnosis: Nano loop analysis
        self.tier_3_threshold = 50   # Intubation: HEALING_REQUIRED.lock

        # Track which tiers have fired to avoid re-triggering
        self._tier_1_fired = False
        self._tier_2_fired = False
        self._tier_3_fired = False

        # Last recovery signal for run_turn() to consume
        self.last_recovery_signal: Optional[RecoverySignal] = None

        # Loop trace buffer for Tier 2 diagnosis
        self._loop_trace: list = []

    def check_system_resources(self) -> bool:
        """Check CPU and memory usage. Return True if under safe thresholds."""
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        logger.debug(f"CPU: {cpu}%, Memory: {mem}%")

        if cpu > self.cpu_limit:
            logger.warning(f"CPU usage high: {cpu}%")
        if mem > self.memory_limit:
            logger.warning(f"Memory usage high: {mem}%")

        return cpu < self.cpu_limit and mem < self.memory_limit

    def record_loop_trace(self, entry: str):
        """Record a loop trace entry for Tier 2 diagnosis."""
        self._loop_trace.append(entry)
        if len(self._loop_trace) > 20:
            self._loop_trace.pop(0)

    def check_loop_counter(self) -> bool:
        """CPR Loop: tiered escalation instead of binary lock.

        Returns True if the system can continue, False only at Tier 3
        (HEALING_REQUIRED.lock). Recovery signals for Tiers 1-2 are
        stored in self.last_recovery_signal for run_turn() to consume.
        """
        self.loop_counter += 1
        logger.debug(f"Loop Count: {self.loop_counter}")

        # ── Tier 1: Breath (KV Cache Reset) ──
        if self.loop_counter >= self.tier_1_threshold and not self._tier_1_fired:
            self._tier_1_fired = True
            logger.warning(
                "CPR Tier 1 (Breath): Loop counter at %d — triggering KV cache reset",
                self.loop_counter,
            )
            self.last_recovery_signal = RecoverySignal(
                tier=1, action="kv_reset", success=True
            )
            # Don't block — let the system continue after the reset
            return True

        # ── Tier 2: Diagnosis (Nano Loop Analysis) ──
        if self.loop_counter >= self.tier_2_threshold and not self._tier_2_fired:
            self._tier_2_fired = True
            logger.warning(
                "CPR Tier 2 (Diagnosis): Loop counter at %d — requesting Nano analysis",
                self.loop_counter,
            )
            diagnosis = self._nano_diagnose_loop()
            self.last_recovery_signal = RecoverySignal(
                tier=2, action="diagnose", diagnosis=diagnosis, success=bool(diagnosis)
            )
            # Don't block — inject diagnosis and let the system try again
            return True

        # ── Tier 3: Intubation (HEALING_REQUIRED.lock) ──
        if self.loop_counter >= self.tier_3_threshold and not self._tier_3_fired:
            self._tier_3_fired = True
            logger.critical(
                "CPR Tier 3 (Intubation): Loop counter at %d — creating HEALING_REQUIRED.lock",
                self.loop_counter,
            )
            try:
                lock_path = "/shared/HEALING_REQUIRED.lock"
                with open(lock_path, "w") as f:
                    f.write(f"FATAL: Loop limit ({self.tier_3_threshold}) hit at {datetime.now(timezone.utc)}\n")
                    f.write("CPR Tiers 1-2 failed to break the loop.\n")
                    f.write("System integrity protected. Manual triage required.\n")
                    if self._loop_trace:
                        f.write(f"\nLoop trace (last {len(self._loop_trace)} entries):\n")
                        for entry in self._loop_trace[-10:]:
                            f.write(f"  - {entry}\n")
                logger.info(f"System locked via {lock_path}")
            except Exception as e:
                logger.error(f"Failed to create healing lock file: {e}")

            self.last_recovery_signal = RecoverySignal(
                tier=3, action="lock", success=False
            )
            return False

        return True

    def _nano_diagnose_loop(self) -> str:
        """Use Nano to analyze the loop trace and produce a diagnosis.

        Returns a short diagnosis string, or empty string on failure.
        """
        trace_text = "\n".join(self._loop_trace[-10:]) if self._loop_trace else "(no trace recorded)"
        prompt = (
            "You are a cognitive diagnostician. GAIA is stuck in a reasoning loop. "
            "Analyze the loop trace below and provide a ONE-LINE diagnosis of why "
            "the loop is occurring and a ONE-LINE suggested fix.\n\n"
            "Format:\n"
            "DIAGNOSIS: <why the loop is happening>\n"
            "FIX: <what to change to break the loop>\n\n"
            f"Loop trace:\n{trace_text}"
        )
        try:
            payload = json.dumps({
                "messages": [
                    {"role": "system", "content": "Cognitive diagnostician. Be concise."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 64,
                "temperature": 0.0,
            }).encode()
            req = Request(
                f"{_NANO_ENDPOINT}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            # Strip think tags
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            logger.info("Nano loop diagnosis: %s", answer[:200])
            return answer[:200]
        except Exception as e:
            logger.debug("Nano loop diagnosis failed: %s", e)
            return ""

    def consume_recovery_signal(self) -> Optional[RecoverySignal]:
        """Consume and return the pending recovery signal (if any).

        Called by agent_core.run_turn() after check_loop_counter().
        Returns the signal and clears it, so each signal is consumed once.
        """
        signal = self.last_recovery_signal
        self.last_recovery_signal = None
        return signal

    def check_recent_errors(self) -> bool:
        """Check if recent unhandled errors have accumulated."""
        if len(self.error_log) > 3:
            logger.warning(f"Too many internal errors: {len(self.error_log)}")
            return False
        return True

    def register_error(self, exc: Exception):
        """Track unhandled exception information."""
        err_str = f"{type(exc).__name__}: {str(exc)}"
        self.error_log.append(err_str)
        if len(self.error_log) > 5:
            self.error_log.pop(0)  # Keep recent 5
        logger.error(f"Exception tracked: {err_str}")
        logger.debug(traceback.format_exc())

    def reset_loop(self):
        self.loop_counter = 0
        self._tier_1_fired = False
        self._tier_2_fired = False
        self._tier_3_fired = False
        self._loop_trace.clear()
        self.last_recovery_signal = None
        logger.debug("Loop counter and CPR tiers reset.")

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
                logger.error(f"Identity check failed: {e}")
                id_ok = False

        return sys_ok and loop_ok and err_ok and id_ok

