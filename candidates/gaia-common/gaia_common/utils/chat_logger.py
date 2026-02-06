import logging
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects gracefully."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return f"<non-serializable: {type(obj).__name__}>"

def setup_chat_logger():
    """Sets up a dedicated logger for chat history."""
    log_dir = "logs/chat_history"
    os.makedirs(log_dir, exist_ok=True)

    # Use a unique filename for each session
    log_file = os.path.join(log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("GAIA.ChatHistory")
    logger.setLevel(logging.INFO)

    # Prevent chat logs from appearing in the main console log
    logger.propagate = False

    # Add a file handler only if one doesn't exist
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize the logger when the module is loaded
chat_history_logger = setup_chat_logger()


def log_chat_entry(
    user_input: str,
    assistant_output: str,
    source: str = "cli",
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Logs a user and assistant turn to the dedicated session file.

    Args:
        user_input: The user's message
        assistant_output: GAIA's response
        source: Source of the interaction (cli, discord_channel, discord_dm, web, api)
        session_id: Session identifier for tracking conversations
        metadata: Additional context (is_dm, user_id, channel_id, etc.)
    """
    meta = metadata or {}
    is_dm = meta.get("is_dm", False)

    # Build context prefix for non-CLI sources
    context_parts = []
    if source != "cli":
        context_parts.append(f"[{source}]")
    if session_id:
        context_parts.append(f"[session:{session_id}]")
    if is_dm:
        context_parts.append("[DM]")
    if meta.get("user_id"):
        context_parts.append(f"[user:{meta['user_id']}]")

    context_prefix = " ".join(context_parts)
    if context_prefix:
        context_prefix = f"{context_prefix} "

    if user_input:
        chat_history_logger.info(f"{context_prefix}User > {user_input}")
    if assistant_output:
        chat_history_logger.info(f"{context_prefix}GAIA > {assistant_output}\n" + "-"*20)


def log_chat_entry_structured(
    user_input: str,
    assistant_output: str,
    source: str = "cli",
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Logs a structured JSON entry for detailed analysis.
    Writes to a separate JSONL file for machine parsing.

    Args:
        user_input: The user's message
        assistant_output: GAIA's response
        source: Source of the interaction
        session_id: Session identifier
        metadata: Additional context
    """
    log_dir = "logs/chat_history"
    os.makedirs(log_dir, exist_ok=True)
    jsonl_file = os.path.join(log_dir, f"structured_{datetime.now().strftime('%Y%m%d')}.jsonl")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": source,
        "session_id": session_id,
        "is_dm": metadata.get("is_dm", False) if metadata else False,
        "user_input": user_input,
        "assistant_output": assistant_output,
        "metadata": metadata or {}
    }

    try:
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, cls=SafeJSONEncoder) + "\n")
    except Exception as e:
        logging.getLogger("GAIA.ChatHistory").warning(f"Failed to write structured log: {e}")
