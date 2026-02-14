"""
Auto-append notable Discord conversations to the knowledge examples file.

Hooks into SessionManager.summarize_and_archive() to evaluate each archived
conversation for "notability" using simple heuristics (no LLM call).
Qualifying conversations are appended to conversation_examples.md, which
is included in the flatten pipeline and auto-syncs to NotebookLM.
"""

import logging
import os
import re
import tempfile
import threading
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.ConversationCurator")

# Greeting/filler patterns — messages matching these don't count as substantive
_FILLER_PATTERNS = re.compile(
    r"^("
    r"hello|hi|hey|howdy|yo|sup|"
    r"thanks|thank you|ok|okay|"
    r"cool|nice|great|bye|goodbye|"
    r"yes|no|yeah|yep|nope|sure|alright|"
    r"good morning|good evening|good afternoon|"
    r"how are you|what'?s up"
    r")[.!?\s]*$",
    re.IGNORECASE,
)

_SIZE_CAP_BYTES = 50 * 1024  # 50 KB
_HEADER = "# GAIA Conversation Examples\n\nReal Discord conversations demonstrating GAIA capabilities.\n"
_write_lock = threading.Lock()


class ConversationCurator:
    """Evaluates archived conversations and appends notable ones to the examples file."""

    def __init__(self, output_dir: Optional[str] = None):
        base = output_dir or os.getenv("KNOWLEDGE_DIR", "/knowledge")
        self.output_path = os.path.join(base, "conversation_examples.md")

    def curate(self, session_id: str, messages: List[Dict]) -> bool:
        """Evaluate and optionally append a conversation. Returns True if curated."""
        # Skip test/smoke sessions
        if session_id.startswith("smoke-test-") or session_id.startswith("test-"):
            logger.debug(f"Curator: skipping test session '{session_id}'")
            return False

        if not self.is_notable(messages):
            logger.debug(f"Curator: session '{session_id}' not notable, skipping")
            return False

        formatted = self._format_conversation(session_id, messages)
        self._append_to_file(formatted)
        logger.info(f"Curator: appended notable conversation from '{session_id}'")
        return True

    def is_notable(self, messages: List[Dict]) -> bool:
        """Heuristic check — all conditions must pass."""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if len(user_msgs) < 4:
            return False
        if len(messages) < 8:
            return False

        # Average message length
        total_len = sum(len(m.get("content", "")) for m in messages)
        if total_len / max(len(messages), 1) < 50:
            return False

        # Filler ratio among user messages
        filler_count = sum(
            1 for m in user_msgs if _FILLER_PATTERNS.match(m.get("content", "").strip())
        )
        if filler_count / max(len(user_msgs), 1) >= 0.6:
            return False

        return True

    def _detect_channel_type(self, session_id: str) -> str:
        return "DM" if "dm" in session_id.lower() else "Channel"

    def _format_conversation(self, session_id: str, messages: List[Dict]) -> str:
        channel_type = self._detect_channel_type(session_id)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        lines = [f"\n---\n\n## Discord {channel_type} Conversation — {date_str}\n"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue
            header = "### User" if role == "user" else "### GAIA"
            lines.append(f"\n{header}\n{content}\n")

        return "\n".join(lines)

    def _append_to_file(self, formatted: str) -> None:
        with _write_lock:
            # Create file with header if missing
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8") as f:
                    f.write(_HEADER)

            # Check size cap before appending
            current_size = os.path.getsize(self.output_path)
            new_bytes = len(formatted.encode("utf-8"))

            if current_size + new_bytes > _SIZE_CAP_BYTES:
                self._trim_oldest(new_bytes)

            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(formatted)

    def _trim_oldest(self, needed_bytes: int) -> None:
        """Remove oldest entries until there's room for needed_bytes."""
        with open(self.output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on --- delimiters (each conversation block)
        parts = content.split("\n---\n")
        if len(parts) <= 1:
            return  # Only header, nothing to trim

        header = parts[0]
        entries = parts[1:]

        # Remove oldest entries (front of list) until under cap
        while entries:
            candidate = header + "\n---\n".join([""] + entries)
            if len(candidate.encode("utf-8")) + needed_bytes <= _SIZE_CAP_BYTES:
                break
            entries.pop(0)
            logger.info("Curator: trimmed oldest conversation entry to stay under size cap")

        trimmed = header + "\n---\n".join([""] + entries) if entries else header + "\n"

        # Atomic rewrite via tempfile
        dir_name = os.path.dirname(self.output_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(trimmed)
            os.replace(tmp_path, self.output_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
