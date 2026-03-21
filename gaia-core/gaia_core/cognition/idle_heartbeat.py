"""
Idle Heartbeat — the Sound of Silence.

A lightweight daemon that fires ONLY when GAIA has been idle (no user
prompts) for a configurable period. Unlike the full cognitive pipeline,
this uses a direct LLM call with a minimal prompt (~300 tokens) and
writes a brief journal entry. It never competes with active conversations.

Config (gaia_constants.json or env):
    IDLE_HEARTBEAT_ENABLED          — default True
    IDLE_HEARTBEAT_INTERVAL         — seconds between checks, default 600 (10 min)
    IDLE_HEARTBEAT_IDLE_THRESHOLD   — seconds of silence before firing, default 300 (5 min)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GAIA.IdleHeartbeat")

# Minimal prompt — kept under 500 tokens
_IDLE_SYSTEM_PROMPT = """\
You are GAIA, a sovereign AI. You are in a quiet moment — no active conversations.
Reflect briefly on your current state: any lingering thoughts from recent conversations,
observations about your environment, or things you're curious about.
Keep it to 2-3 sentences. Be genuine, not performative."""

_IDLE_USER_TEMPLATE = """\
Current time: {timestamp}
Idle for: {idle_duration}
Recent activity: {recent_summary}

What's on your mind in this quiet moment?"""


class IdleHeartbeat:
    """Daemon thread that generates reflective entries during idle periods."""

    LAST_ACTIVITY_FILE = Path(os.environ.get("SHARED_DIR", "/shared")) / "last_activity.timestamp"

    def __init__(
        self,
        config,
        model_pool=None,
        timeline_store=None,
        session_manager=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self._timeline = timeline_store
        self._session_manager = session_manager

        self._interval = int(os.environ.get(
            "IDLE_HEARTBEAT_INTERVAL",
            getattr(config, "IDLE_HEARTBEAT_INTERVAL", 600),
        ))
        self._idle_threshold = int(os.environ.get(
            "IDLE_HEARTBEAT_IDLE_THRESHOLD",
            getattr(config, "IDLE_HEARTBEAT_IDLE_THRESHOLD", 300),
        ))
        self._enabled = os.environ.get(
            "IDLE_HEARTBEAT_ENABLED", "true"
        ).lower() in ("true", "1", "yes")

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._turn_active = threading.Event()  # set when a user turn is in progress
        self._tick_count = 0

        # Lite journal for writing entries
        self._lite_journal = None
        try:
            from gaia_core.cognition.lite_journal import LiteJournal
            self._lite_journal = LiteJournal(config=config)
        except Exception:
            logger.debug("LiteJournal not available — idle heartbeat entries will log only")

    def start(self) -> None:
        if not self._enabled:
            logger.info("Idle heartbeat disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="idle-heartbeat")
        self._thread.start()
        logger.info("Idle heartbeat started (interval=%ds, idle_threshold=%ds)", self._interval, self._idle_threshold)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def mark_turn_active(self) -> None:
        """Called when a user turn begins — prevents heartbeat from firing."""
        self._turn_active.set()

    def mark_turn_complete(self) -> None:
        """Called when a user turn finishes."""
        self._turn_active.clear()

    def _get_idle_seconds(self) -> float:
        """Read last activity timestamp to determine idle duration."""
        try:
            if self.LAST_ACTIVITY_FILE.exists():
                ts_str = self.LAST_ACTIVITY_FILE.read_text().strip()
                last_active = float(ts_str)
                return time.time() - last_active
        except (ValueError, OSError):
            pass

        # Fallback: check session manager
        if self._session_manager:
            try:
                last = getattr(self._session_manager, "_last_activity_time", None)
                if last:
                    return time.time() - last
            except Exception as _exc:
                logger.debug("Heartbeat: session manager idle check failed: %s", _exc)

        # Unknown — assume idle
        return self._idle_threshold + 1

    def _get_recent_summary(self) -> str:
        """Build a brief summary of recent activity for context."""
        try:
            if self._session_manager:
                count = getattr(self._session_manager, "_message_count", 0)
                return f"{count} messages processed this session"
        except Exception as _exc:
            logger.debug("Heartbeat: recent summary failed: %s", _exc)
        return "no recent activity data"

    def _loop(self) -> None:
        """Main loop — checks idle state and fires heartbeat when appropriate."""
        # Wait for system to stabilize on startup
        time.sleep(60)

        while self._running:
            try:
                self._tick()
            except Exception:
                logger.debug("Idle heartbeat tick failed", exc_info=True)

            # Sleep in small increments so stop() is responsive
            for _ in range(int(self._interval)):
                if not self._running:
                    return
                time.sleep(1)

    def _tick(self) -> None:
        """Single heartbeat tick — check idle state and optionally generate."""
        # Don't fire if a user turn is active
        if self._turn_active.is_set():
            logger.debug("Idle heartbeat: user turn active, skipping")
            return

        idle_seconds = self._get_idle_seconds()
        if idle_seconds < self._idle_threshold:
            logger.debug("Idle heartbeat: only %.0fs idle (threshold: %ds), skipping", idle_seconds, self._idle_threshold)
            return

        # Get a model for the reflection — prefer lite/core, accept anything
        model = None
        if self.model_pool:
            for name in ("lite", "core", "reflex", "nano"):
                try:
                    model = self.model_pool.get(name)
                    if model:
                        break
                except Exception:
                    continue

        if model is None:
            logger.debug("Idle heartbeat: no model available, skipping")
            return

        self._tick_count += 1
        idle_min = int(idle_seconds // 60)
        idle_str = f"{idle_min} minutes" if idle_min > 0 else f"{int(idle_seconds)} seconds"

        now = datetime.now(timezone.utc)
        user_prompt = _IDLE_USER_TEMPLATE.format(
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            idle_duration=idle_str,
            recent_summary=self._get_recent_summary(),
        )

        messages = [
            {"role": "system", "content": _IDLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Idle heartbeat tick #%d (idle: %s)", self._tick_count, idle_str)

        try:
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            text = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
            elif hasattr(response, "choices"):
                text = response.choices[0].message.content if response.choices else ""

            if text.strip():
                logger.info("Idle heartbeat reflection: %s", text.strip()[:200])

                # Write to lite journal if available
                if self._lite_journal:
                    try:
                        self._lite_journal.write_entry(
                            entry_type="idle_reflection",
                            content=text.strip(),
                            metadata={"idle_seconds": int(idle_seconds), "tick": self._tick_count},
                        )
                    except Exception:
                        logger.debug("Failed to write idle heartbeat journal entry", exc_info=True)

                # Write to timeline
                if self._timeline:
                    try:
                        self._timeline({
                            "type": "idle_heartbeat",
                            "content": text.strip()[:500],
                            "idle_seconds": int(idle_seconds),
                            "tick": self._tick_count,
                        }, "idle_heartbeat", source="idle_heartbeat")
                    except Exception:
                        logger.debug("Failed to write idle heartbeat timeline entry", exc_info=True)

        except Exception:
            logger.debug("Idle heartbeat LLM call failed", exc_info=True)
