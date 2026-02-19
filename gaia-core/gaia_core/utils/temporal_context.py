"""
Temporal Context Builder — assembles a rich temporal snapshot for prompt injection.

Produces a concise text block (~100-150 tokens) injected into the system prompt
so GAIA has awareness of time, her wake cycle, session state, and code evolution.

Example output:

    [Temporal Context]
    Tuesday 2026-02-18, 22:41 UTC (evening)
    Awake for 2h 15m. Last sleep: 45m (20:26–21:11 UTC).
    This conversation: 45m old, 12 messages. Last message 3m ago.
    Since waking: 3 conversations, 1 sleep task completed.
    Code: 2 services have pending candidate changes (gaia-core, gaia-web).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("GAIA.TemporalContext")

_TIME_OF_DAY = [
    (0, "early morning"),
    (6, "morning"),
    (12, "afternoon"),
    (17, "evening"),
    (21, "night"),
]


def build_temporal_context(
    timeline_store: Optional[Any] = None,
    sleep_manager_status: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    session_created_at: Optional[datetime] = None,
    session_message_count: int = 0,
    last_message_ts: Optional[datetime] = None,
    code_evolution_path: str = "/shared/self_model/code_evolution.md",
) -> str:
    """Build the complete temporal context block for prompt injection.

    All parameters are optional — the function degrades gracefully,
    omitting sections it cannot populate.

    Returns a formatted string, or "" if no temporal data is available.
    """
    now = datetime.now(timezone.utc)
    sections = []

    # 1. Semantic time (always available)
    sections.append(_semantic_time(now))

    # 2. Wake cycle summary (from timeline store)
    if timeline_store is not None:
        try:
            wake_line = _wake_cycle_summary(timeline_store)
            if wake_line:
                sections.append(wake_line)
        except Exception:
            logger.debug("Wake cycle summary failed", exc_info=True)

    # 3. Session summary
    if session_id or session_created_at or session_message_count > 0:
        try:
            session_line = _session_summary(
                session_id, session_created_at,
                session_message_count, last_message_ts,
            )
            if session_line:
                sections.append(session_line)
        except Exception:
            logger.debug("Session summary failed", exc_info=True)

    # 4. Activity summary (from timeline store)
    if timeline_store is not None:
        try:
            activity_line = _activity_summary(timeline_store)
            if activity_line:
                sections.append(activity_line)
        except Exception:
            logger.debug("Activity summary failed", exc_info=True)

    # 5. State summary (from sleep manager)
    if sleep_manager_status:
        try:
            state_line = _state_summary(sleep_manager_status)
            if state_line:
                sections.append(state_line)
        except Exception:
            logger.debug("State summary failed", exc_info=True)

    # 6. Code evolution one-liner (from snapshot file)
    try:
        code_line = _code_evolution_summary(code_evolution_path)
        if code_line:
            sections.append(code_line)
    except Exception:
        logger.debug("Code evolution summary failed", exc_info=True)

    if not sections:
        return ""

    return "[Temporal Context]\n" + "\n".join(sections)


# ── Sub-functions ────────────────────────────────────────────────────────


def _semantic_time(dt: datetime) -> str:
    """'Tuesday 2026-02-18, 22:41 UTC (evening)'"""
    day_name = dt.strftime("%A")
    date_str = dt.strftime("%Y-%m-%d, %H:%M UTC")
    hour = dt.hour
    period = "night"
    for threshold, label in _TIME_OF_DAY:
        if hour >= threshold:
            period = label
    return f"{day_name} {date_str} ({period})"


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable: '2h 15m', '45m', '3m', '<1m'."""
    if seconds < 60:
        return "<1m"
    minutes = int(seconds / 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining = minutes % 60
    if remaining == 0:
        return f"{hours}h"
    return f"{hours}h {remaining}m"


def _wake_cycle_summary(timeline_store: Any) -> str:
    """'Awake for 2h 15m. Last sleep: 45m (20:26–21:11 UTC).'"""
    now = datetime.now(timezone.utc)

    # Find last wake event (state_change to "active")
    state_changes = timeline_store.events_by_type("state_change", limit=20)

    last_wake = None
    last_sleep_start = None
    last_sleep_end = None

    for event in state_changes:
        to_state = event.data.get("to", "").lower()
        if to_state == "active" and last_wake is None:
            last_wake = event.timestamp
        elif to_state in ("asleep", "drowsy") and last_sleep_start is None:
            if last_wake is not None:
                # This is the sleep that preceded the current wake
                last_sleep_start = event.timestamp
        elif to_state == "active" and last_sleep_start is not None and last_sleep_end is None:
            last_sleep_end = event.timestamp

    parts = []
    if last_wake:
        awake_secs = (now - last_wake).total_seconds()
        parts.append(f"Awake for {_format_duration(awake_secs)}.")

    if last_sleep_start and last_sleep_end:
        sleep_secs = (last_sleep_end - last_sleep_start).total_seconds()
        t1 = last_sleep_start.strftime("%H:%M")
        t2 = last_sleep_end.strftime("%H:%M")
        parts.append(f"Last sleep: {_format_duration(sleep_secs)} ({t1}–{t2} UTC).")
    elif last_sleep_start:
        t1 = last_sleep_start.strftime("%H:%M")
        parts.append(f"Last sleep started at {t1} UTC.")

    return " ".join(parts)


def _session_summary(
    session_id: Optional[str],
    session_created_at: Optional[datetime],
    message_count: int,
    last_message_ts: Optional[datetime],
) -> str:
    """'This conversation: 45m old, 12 messages. Last message 3m ago.'"""
    now = datetime.now(timezone.utc)
    parts = []

    if session_created_at:
        age_secs = (now - session_created_at).total_seconds()
        parts.append(f"{_format_duration(age_secs)} old")

    if message_count > 0:
        parts.append(f"{message_count} messages")

    if last_message_ts:
        gap_secs = (now - last_message_ts).total_seconds()
        parts.append(f"last message {_format_duration(gap_secs)} ago")

    if not parts:
        return ""

    return f"This conversation: {', '.join(parts)}."


def _activity_summary(timeline_store: Any) -> str:
    """'Since waking: 3 conversations, 1 sleep task completed.'"""
    # Find last wake time
    state_changes = timeline_store.events_by_type("state_change", limit=20)
    last_wake = None
    for event in state_changes:
        if event.data.get("to", "").lower() == "active":
            last_wake = event.timestamp
            break

    if not last_wake:
        return ""

    # Count events since wake
    from datetime import timedelta
    events = timeline_store.events_since(last_wake, limit=500)

    session_count = sum(1 for e in events if e.event == "session_start")
    task_count = sum(
        1 for e in events
        if e.event == "task_exec" and e.data.get("success", False)
    )
    msg_count = sum(1 for e in events if e.event == "message")

    parts = []
    if session_count:
        parts.append(f"{session_count} conversation{'s' if session_count != 1 else ''}")
    if task_count:
        parts.append(f"{task_count} sleep task{'s' if task_count != 1 else ''} completed")
    if msg_count and not session_count:
        parts.append(f"{msg_count} messages processed")

    if not parts:
        return ""

    return f"Since waking: {', '.join(parts)}."


def _state_summary(status: Dict[str, Any]) -> str:
    """'State: ACTIVE for 2h 15m.'"""
    state = status.get("state", "")
    seconds = status.get("seconds_in_state", 0)
    if not state:
        return ""
    return f"State: {state.upper()} for {_format_duration(seconds)}."


def _code_evolution_summary(path: str) -> str:
    """'Code: 2 services have pending candidate changes (gaia-core, gaia-web).'

    Reads the first few lines of the code evolution snapshot file.
    """
    try:
        snapshot = Path(path)
        if not snapshot.exists():
            return ""
        content = snapshot.read_text(encoding="utf-8")
        # Extract the "Pending Candidate Changes" section
        if "All candidates match production" in content:
            return "Code: all candidates match production."

        # Count service lines (lines starting with "- **")
        services = []
        in_section = False
        for line in content.splitlines():
            if "## Pending Candidate Changes" in line:
                in_section = True
                continue
            if in_section and line.startswith("- **"):
                # Extract service name: "- **gaia-core**: ..."
                name = line.split("**")[1] if "**" in line else ""
                if name:
                    services.append(name)
            elif in_section and line.startswith("##"):
                break

        if services:
            names = ", ".join(services)
            return f"Code: {len(services)} service{'s' if len(services) != 1 else ''} with pending candidate changes ({names})."
        return ""
    except OSError:
        return ""
