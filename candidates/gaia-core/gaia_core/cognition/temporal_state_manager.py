"""
Temporal State Manager — KV cache state baking, storage, and restoration for Lite.

Manages segmented KV cache snapshots that capture Lite's cognitive state at specific
points in time.  These snapshots can later be loaded back into Lite to simulate
past states, enabling Prime to interview a "past version" of the system.

Phase 1: Bake + Store + Load + Rotate
Phase 2: Interview protocol (Prime ↔ past-Lite via state swapping)

Technical approach:
  - llama-cpp-python exposes save_state() / load_state() on the Llama class
  - save_state() returns a LlamaState object containing the full KV cache + RNG
  - We serialize this via pickle to a binary file (.bin) with a JSON sidecar
  - Lite runs on CPU, so KV cache lives in RAM (no VRAM contention with Prime)
  - State files persist on the Docker volume (/shared/temporal_states/)

Thread safety:
  - _LITE_LOCK protects concurrent Lite access (shared with lite_journal.py)
  - Bake operations acquire the lock around all Lite calls
  - This prevents corruption from concurrent intent-detection on the web thread
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.TemporalState")

# Module-level lock for serializing all Lite model access.
# Shared between TemporalStateManager and LiteJournal.
_LITE_LOCK = threading.Lock()

# Bake conditioning prompt
_BAKE_SYSTEM_PROMPT = """\
You are GAIA's temporal self-awareness process. You are being asked to process \
and internalize the following context about your current state. This context \
will be preserved as a cognitive snapshot that your future self can revisit.

Read the context carefully and respond with a brief first-person reflection \
(3-5 sentences) summarizing:
- What you understand about your current situation
- What patterns or changes you notice
- What feels important to remember

This reflection becomes part of your temporal memory."""

_BAKE_USER_TEMPLATE = """\
[Temporal State Bake — {timestamp}]

## Current Time
{semantic_time}

## Wake/Sleep Cycle
{wake_cycle}

## Recent Activity
{timeline_summary}

## Active Conversation Context
{conversation_context}

## World State
{world_state}

## My Journal (Lite.md)
{journal_content}"""


class TemporalStateManager:
    """Manages Lite KV cache state snapshots for temporal self-awareness."""

    def __init__(
        self,
        config,
        model_pool=None,
        timeline_store=None,
        session_manager=None,
        lite_journal=None,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self._timeline = timeline_store
        self._session_manager = session_manager
        self._lite_journal = lite_journal

        shared_dir = getattr(config, "SHARED_DIR", "/shared")
        self.state_dir = Path(shared_dir) / "temporal_states"
        self.max_files = getattr(config, "TEMPORAL_STATE_MAX_FILES", 5)
        self.max_bytes = getattr(config, "TEMPORAL_STATE_MAX_BYTES", 10_737_418_240)
        self.bake_context_tokens = getattr(
            config, "TEMPORAL_STATE_BAKE_CONTEXT_TOKENS", 6000
        )

        # Ensure directory exists
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create temporal state dir: %s", exc)

    # ------------------------------------------------------------------
    # Core State Management
    # ------------------------------------------------------------------

    def bake_state(self) -> Optional[Path]:
        """Reconstruct temporal context, process through Lite, save KV state.

        Acquires _LITE_LOCK around all Lite operations.
        Returns the path to the saved .bin file, or None on failure.
        """
        llm = None
        if self.model_pool is not None:
            try:
                llm = self.model_pool.get_model_for_role("lite")
            except Exception:
                logger.warning("TemporalState: could not get Lite model", exc_info=True)

        if llm is None:
            logger.warning("TemporalState: no Lite model available, skipping bake")
            return None

        start_ms = time.monotonic()

        with _LITE_LOCK:
            try:
                # 1. Build conditioning context
                messages = self._build_bake_context()

                # 2. Process through Lite (fills KV cache)
                result = llm.create_chat_completion(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=256,
                    stream=False,
                )
                reflection = result["choices"][0]["message"]["content"].strip()
                logger.debug("Bake reflection: %s", reflection[:100])

                # 3. Save KV state
                elapsed_ms = int((time.monotonic() - start_ms) * 1000)
                metadata = self._build_metadata(elapsed_ms)
                state_path = self._save_lite_state(llm, metadata)

                # 4. Rotate old states
                self.cleanup_old_states()

                return state_path
            except Exception:
                logger.error("TemporalState: bake failed", exc_info=True)
                return None

    def load_state(self, state_id: str) -> bool:
        """Load a previously baked state into Lite's KV cache.

        Returns True if successfully loaded, False otherwise.
        """
        state_path = self.state_dir / f"{state_id}.bin"
        if not state_path.exists():
            logger.warning("TemporalState: state not found: %s", state_id)
            return False

        llm = None
        if self.model_pool is not None:
            try:
                llm = self.model_pool.get_model_for_role("lite")
            except Exception:
                pass

        if llm is None:
            return False

        with _LITE_LOCK:
            return self._load_lite_state(llm, state_path)

    def restore_current(self) -> bool:
        """Restore Lite to the most recently baked state.

        Used after an interview session to 'flip back' to present.
        """
        states = self.list_states()
        if not states:
            logger.warning("TemporalState: no states available to restore")
            return False

        # Most recent state (sorted by timestamp, newest last)
        latest = states[-1]
        return self.load_state(latest["state_id"])

    def list_states(self) -> List[Dict[str, Any]]:
        """List all available states with metadata, sorted by timestamp."""
        if not self.state_dir.exists():
            return []

        states = []
        for bin_path in sorted(self.state_dir.glob("lite_state_*.bin")):
            meta_path = bin_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

            states.append({
                "state_id": bin_path.stem,
                "path": str(bin_path),
                "size_bytes": bin_path.stat().st_size if bin_path.exists() else 0,
                "timestamp": meta.get("timestamp", ""),
                "gaia_state": meta.get("gaia_state", ""),
                "heartbeat_tick": meta.get("heartbeat_tick", 0),
            })

        return states

    def get_state_metadata(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Get full metadata for a specific state."""
        meta_path = self.state_dir / f"{state_id}.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def cleanup_old_states(self) -> int:
        """Enforce rotation policy: max files and max total bytes.

        Returns number of states deleted.
        """
        bins = sorted(self.state_dir.glob("lite_state_*.bin"), key=lambda p: p.name)
        deleted = 0

        # Enforce file count limit
        while len(bins) > self.max_files:
            oldest = bins.pop(0)
            self._delete_state_files(oldest)
            deleted += 1

        # Enforce total size limit
        total = sum(p.stat().st_size for p in bins if p.exists())
        while total > self.max_bytes and bins:
            oldest = bins.pop(0)
            if oldest.exists():
                total -= oldest.stat().st_size
            self._delete_state_files(oldest)
            deleted += 1

        if deleted:
            logger.info("TemporalState: cleaned up %d old states", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Context Reconstruction
    # ------------------------------------------------------------------

    def _build_bake_context(self) -> List[Dict[str, str]]:
        """Build the message list used to condition Lite before state capture."""
        now = datetime.now(timezone.utc)
        semantic_time = now.strftime("%A %Y-%m-%d, %H:%M UTC")

        wake_cycle = self._reconstruct_wake_cycle()
        timeline_summary = self._reconstruct_timeline_context()
        conversation_context = self._reconstruct_conversation_context()
        world_state = self._reconstruct_world_state()
        journal_content = self._reconstruct_journal_content()

        user_content = _BAKE_USER_TEMPLATE.format(
            timestamp=now.isoformat(),
            semantic_time=semantic_time,
            wake_cycle=wake_cycle or "No wake cycle data available.",
            timeline_summary=timeline_summary or "No recent events.",
            conversation_context=conversation_context or "No active conversations.",
            world_state=world_state or "No world state data available.",
            journal_content=journal_content or "No journal entries yet.",
        )

        return [
            {"role": "system", "content": _BAKE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _reconstruct_wake_cycle(self) -> str:
        """Summarize current wake/sleep cycle from timeline."""
        if self._timeline is None:
            return ""

        try:
            state_changes = self._timeline.events_by_type("state_change", limit=10)
            if not state_changes:
                return "No state changes recorded."

            lines = []
            for event in state_changes[:5]:
                fr = event.data.get("from", "?")
                to = event.data.get("to", "?")
                reason = event.data.get("reason", "")
                ts_short = event.ts[:19] if event.ts else "?"
                line = f"  {ts_short}: {fr} → {to}"
                if reason:
                    line += f" ({reason})"
                lines.append(line)
            return "\n".join(lines)
        except Exception:
            return ""

    def _reconstruct_timeline_context(self) -> str:
        """Summarize recent timeline events."""
        if self._timeline is None:
            return ""

        try:
            events = self._timeline.recent_events(limit=10)
            if not events:
                return ""

            lines = []
            for e in events:
                ts_short = e.ts[:19] if e.ts else "?"
                summary = self._event_summary(e.data)
                lines.append(f"  [{ts_short}] {e.event}: {summary}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _reconstruct_conversation_context(self) -> str:
        """Pull recent messages from the most active session."""
        if self._session_manager is None:
            return ""

        try:
            # Find sessions with recent activity
            sessions = getattr(self._session_manager, "sessions", {})
            if not sessions:
                return ""

            # Pick the session with the most recent message
            best_session = None
            best_ts = None
            for sid, session in sessions.items():
                if sid.startswith("gaia_"):
                    continue  # Skip internal sessions
                last_ts = session.last_message_timestamp() if hasattr(session, "last_message_timestamp") else None
                if last_ts is not None and (best_ts is None or last_ts > best_ts):
                    best_ts = last_ts
                    best_session = sid

            if not best_session:
                return ""

            history = self._session_manager.get_history(best_session)
            if not history:
                return ""

            # Take last 10 messages
            recent = history[-10:]
            lines = [f"Session: {best_session} ({len(history)} total messages)"]
            for msg in recent:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"  [{role}]: {content}")
            return "\n".join(lines)
        except Exception:
            logger.debug("Conversation context reconstruction failed", exc_info=True)
            return ""

    def _reconstruct_world_state(self) -> str:
        """Get current world state snapshot."""
        try:
            from gaia_core.utils.world_state import format_world_state_snapshot
            return format_world_state_snapshot(max_lines=6)
        except Exception:
            return ""

    def _reconstruct_journal_content(self) -> str:
        """Load recent Lite.md journal entries."""
        if self._lite_journal is None:
            return ""
        try:
            entries = self._lite_journal.load_recent_entries(n=3)
            return "\n".join(entries) if entries else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # State File I/O
    # ------------------------------------------------------------------

    def _save_lite_state(self, llm, metadata: Dict[str, Any]) -> Path:
        """Save Lite's current KV cache state to disk.

        Uses pickle for the LlamaState object (contains numpy arrays + bytes).
        Writes atomically via tmp file + rename.
        """
        now = datetime.now(timezone.utc)
        ts_safe = now.strftime("%Y-%m-%dT%H-%M-%SZ")
        state_id = f"lite_state_{ts_safe}"

        bin_path = self.state_dir / f"{state_id}.bin"
        meta_path = self.state_dir / f"{state_id}.json"
        tmp_path = self.state_dir / f"{state_id}.bin.tmp"

        # Save KV state
        state_data = llm.save_state()

        # Write binary blob atomically
        with open(tmp_path, "wb") as f:
            pickle.dump(state_data, f)
        os.rename(str(tmp_path), str(bin_path))

        # Update metadata with actual size
        metadata["state_id"] = state_id
        metadata["timestamp"] = now.isoformat()
        metadata["state_size_bytes"] = bin_path.stat().st_size

        # Write metadata sidecar
        meta_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        logger.info(
            "Temporal state saved: %s (%d bytes)",
            state_id, metadata["state_size_bytes"],
        )
        return bin_path

    def save_current_state_memory(self, llm) -> Any:
        """Save Lite's current KV state to a Python object (not disk).

        Used by TemporalInterviewer to preserve the current state before
        swapping to a past state for interview.  Lighter than bake_state()
        — no disk I/O, no context reconstruction.

        Caller must already hold _LITE_LOCK.
        """
        return llm.save_state()

    def restore_state_memory(self, llm, state_data) -> None:
        """Restore a previously saved in-memory state.

        Caller must already hold _LITE_LOCK.
        """
        llm.load_state(state_data)

    def _load_lite_state(self, llm, state_path: Path) -> bool:
        """Load a saved state into Lite's KV cache.

        Caller must already hold _LITE_LOCK (the public load_state()
        acquires the lock; call this internal method directly when the
        lock is already held, e.g. from TemporalInterviewer).

        On failure, renames the corrupt file and returns False.
        """
        try:
            with open(state_path, "rb") as f:
                state_data = pickle.load(f)

            llm.load_state(state_data)
            logger.info("Temporal state loaded: %s", state_path.stem)
            return True
        except Exception as exc:
            logger.error(
                "Failed to load temporal state %s: %s", state_path.name, exc,
                exc_info=True,
            )
            # Rename corrupt file
            try:
                corrupt_path = state_path.with_suffix(".bin.corrupt")
                os.rename(str(state_path), str(corrupt_path))
                logger.warning("Corrupt state renamed to %s", corrupt_path)
            except OSError:
                pass
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, bake_duration_ms: int) -> Dict[str, Any]:
        """Build metadata dict for the state sidecar."""
        meta: Dict[str, Any] = {
            "bake_duration_ms": bake_duration_ms,
        }

        # GAIA state
        try:
            from gaia_core.cognition.sleep_wake_manager import SleepWakeManager
            if self._timeline is not None:
                last_state = self._timeline.last_event_of_type("state_change")
                if last_state:
                    meta["gaia_state"] = last_state.data.get("to", "unknown")
        except Exception:
            meta["gaia_state"] = "unknown"

        # Active sessions
        if self._session_manager is not None:
            try:
                sessions = getattr(self._session_manager, "sessions", {})
                meta["active_sessions"] = [
                    sid for sid in sessions
                    if not sid.startswith("gaia_")
                ]
            except Exception:
                pass

        # Journal entries
        if self._lite_journal is not None:
            try:
                meta["journal_entries_included"] = min(
                    3, self._lite_journal.get_entry_count()
                )
            except Exception:
                pass

        return meta

    def _delete_state_files(self, bin_path: Path) -> None:
        """Delete a state file and its JSON sidecar."""
        try:
            bin_path.unlink(missing_ok=True)
            bin_path.with_suffix(".json").unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete state %s: %s", bin_path.name, exc)

    @staticmethod
    def _event_summary(data: Dict[str, Any]) -> str:
        """One-line summary of a timeline event's data."""
        if not data:
            return ""
        parts = []
        for key in ("from", "to", "session_id", "method", "seeds_found", "success"):
            if key in data:
                parts.append(f"{key}={data[key]}")
        if parts:
            return ", ".join(parts[:3])
        first_key = next(iter(data), None)
        return f"{first_key}={data[first_key]}" if first_key else ""
