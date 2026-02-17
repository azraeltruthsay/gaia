import os, json, time, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects gracefully."""
    def default(self, obj):
        # Handle common non-serializable types
        try:
            # Try standard serialization first
            return super().default(obj)
        except TypeError:
            # Return a safe string representation
            return f"<non-serializable: {type(obj).__name__}>"


def write(
    entry: dict,
    session_id: str = "default",
    source: Optional[str] = None,
    destination_context: Optional[Dict[str, Any]] = None,
    ts_dir: Optional[Path] = None
):
    """
    Append a JSONL line with timestamp + model thought to the current
    session's thought-stream file.

    Args:
        entry: The thought entry to log
        session_id: Session identifier (e.g., "discord_dm_12345" for DMs)
        source: Source of the interaction (cli, discord_channel, discord_dm, web, api)
        destination_context: Additional destination metadata (is_dm, user_id, channel_id, etc.)
        ts_dir: Optional Path to the thoughtstream directory. If None, uses a default.
    """
    if ts_dir is None:
        ts_dir = Path('/tmp/gaia/thoughtstreams') # Fallback to a safe temp directory

    stamp = datetime.now(timezone.utc).isoformat()
    # Ensure target dir exists (race-safe best-effort)
    try:
        ts_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.getLogger("GAIA.ThoughtStream").warning(f"Failed to create thoughtstream directory {ts_dir}: {e}")
        return # Cannot write if directory cannot be created

    path = ts_dir / f"{session_id}_{time.strftime('%Y%m%d')}.jsonl"
    entry["ts_utc"] = stamp

    # Add source and destination context if provided
    if source:
        entry["source"] = source
    if destination_context:
        entry["destination_context"] = destination_context
        # Extract key flags for easy querying
        if destination_context.get("is_dm"):
            entry["is_dm"] = True
        if destination_context.get("user_id"):
            entry["user_id"] = destination_context["user_id"]

    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, cls=SafeJSONEncoder) + "\n")
        logging.getLogger("GAIA.ThoughtStream").debug("Wrote TS entry: %s", entry.get("type"))
    except Exception:
        logging.getLogger("GAIA.ThoughtStream").exception("Failed to write thoughtstream entry")


def write_dm_thought(
    entry: dict,
    user_id: str,
    author_name: Optional[str] = None,
    ts_dir: Optional[Path] = None
):
    """
    Convenience function for writing DM-specific thought entries.
    Automatically sets the session_id and destination_context for DMs.

    Args:
        entry: The thought entry to log
        user_id: Discord user ID for the DM
        author_name: Display name of the user (optional)
        ts_dir: Optional Path to the thoughtstream directory.
    """
    session_id = f"discord_dm_{user_id}"
    destination_context = {
        "is_dm": True,
        "user_id": user_id,
        "author_name": author_name,
        "source": "discord_dm"
    }
    write(entry, session_id=session_id, source="discord_dm", destination_context=destination_context, ts_dir=ts_dir)