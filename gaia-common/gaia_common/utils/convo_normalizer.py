"""
Conversation Normalizer — Convert any chat export to a standard transcript format.

Adapted from MemPalace (github.com/milla-jovovich/mempalace).

Supported formats:
    - Plain text with > markers (pass through)
    - Claude.ai JSON export
    - Claude Code JSONL sessions
    - ChatGPT conversations.json
    - Slack JSON export
    - Discord JSON export (DiscordChatExporter format)
    - Plain text (pass through for paragraph chunking)

Output format:
    > user message
    assistant response

    > next user message
    next assistant response

No API key. No internet. Everything local. No external dependencies.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple


def normalize(filepath: str) -> str:
    """Load a file and normalize to transcript format.

    Plain text files pass through unchanged.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        raise IOError(f"Could not read {filepath}: {e}")

    if not content.strip():
        return content

    # Already has > markers — pass through
    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return content

    # Try JSON normalization
    ext = Path(filepath).suffix.lower()
    if ext in (".json", ".jsonl") or content.strip()[:1] in ("{", "["):
        normalized = _try_normalize_json(content)
        if normalized:
            return normalized

    return content


def normalize_text(content: str, format_hint: str = "auto") -> str:
    """Normalize from a string instead of a file.

    Args:
        content: The raw text/JSON content
        format_hint: One of "auto", "claude_ai", "claude_code", "chatgpt",
                     "slack", "discord", "plain"
    """
    if not content.strip():
        return content

    if format_hint == "plain" or (format_hint == "auto" and content.strip()[:1] not in ("{", "[")):
        lines = content.split("\n")
        if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
            return content
        return content

    normalized = _try_normalize_json(content)
    return normalized or content


# ── JSON format detection ─────────────────────────────────────────────────

def _try_normalize_json(content: str) -> Optional[str]:
    """Try all known JSON chat schemas."""
    # JSONL first (Claude Code)
    normalized = _try_claude_code_jsonl(content)
    if normalized:
        return normalized

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    for parser in (_try_claude_ai_json, _try_chatgpt_json, _try_discord_json, _try_slack_json):
        normalized = parser(data)
        if normalized:
            return normalized

    return None


# ── Format parsers ────────────────────────────────────────────────────────

def _try_claude_code_jsonl(content: str) -> Optional[str]:
    """Claude Code JSONL sessions."""
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages: List[Tuple[str, str]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("type", "")
        message = entry.get("message", {})
        if msg_type == "human":
            text = _extract_content(message.get("content", ""))
            if text:
                messages.append(("user", text))
        elif msg_type == "assistant":
            text = _extract_content(message.get("content", ""))
            if text:
                messages.append(("assistant", text))
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_claude_ai_json(data) -> Optional[str]:
    """Claude.ai JSON export: [{"role": "user", "content": "..."}]"""
    if isinstance(data, dict):
        data = data.get("messages", data.get("chat_messages", []))
    if not isinstance(data, list):
        return None
    messages: List[Tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        text = _extract_content(item.get("content", ""))
        if role in ("user", "human") and text:
            messages.append(("user", text))
        elif role in ("assistant", "ai") and text:
            messages.append(("assistant", text))
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_chatgpt_json(data) -> Optional[str]:
    """ChatGPT conversations.json with mapping tree."""
    if not isinstance(data, dict) or "mapping" not in data:
        return None
    mapping = data["mapping"]
    messages: List[Tuple[str, str]] = []
    root_id = None
    fallback_root = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            if node.get("message") is None:
                root_id = node_id
                break
            elif fallback_root is None:
                fallback_root = node_id
    if not root_id:
        root_id = fallback_root
    if root_id:
        current_id = root_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id, {})
            msg = node.get("message")
            if msg:
                role = msg.get("author", {}).get("role", "")
                content = msg.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text = " ".join(str(p) for p in parts if isinstance(p, str) and p).strip()
                if role == "user" and text:
                    messages.append(("user", text))
                elif role == "assistant" and text:
                    messages.append(("assistant", text))
            children = node.get("children", [])
            current_id = children[0] if children else None
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_discord_json(data) -> Optional[str]:
    """Discord JSON export (DiscordChatExporter or similar).

    Expected format: {"messages": [{"author": {"name": "..."}, "content": "..."}]}
    or just a list of message objects.
    """
    msgs = data if isinstance(data, list) else data.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None

    # Check if this looks like Discord (has author.name or author.id)
    sample = msgs[0] if msgs else {}
    if not isinstance(sample, dict):
        return None
    author = sample.get("author", {})
    if not isinstance(author, dict) or not (author.get("name") or author.get("id")):
        return None

    messages: List[Tuple[str, str]] = []
    seen_authors = {}
    for item in msgs:
        if not isinstance(item, dict):
            continue
        author = item.get("author", {})
        author_id = author.get("id", author.get("name", ""))
        text = item.get("content", "").strip()
        if not text or not author_id:
            continue
        # Skip bot system messages
        if item.get("type") not in (None, "Default", 0, "default", ""):
            if item.get("type") not in ("Reply", 19):  # Replies are fine
                continue

        # First unique author = user, second = assistant (GAIA)
        if author_id not in seen_authors:
            if not seen_authors:
                seen_authors[author_id] = "user"
            else:
                seen_authors[author_id] = "assistant"

        role = seen_authors.get(author_id, "user")
        messages.append((role, text))

    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_slack_json(data) -> Optional[str]:
    """Slack channel export: [{"type": "message", "user": "...", "text": "..."}]"""
    if not isinstance(data, list):
        return None
    messages: List[Tuple[str, str]] = []
    seen_users = {}
    last_role = None
    for item in data:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        user_id = item.get("user", item.get("username", ""))
        text = item.get("text", "").strip()
        if not text or not user_id:
            continue
        if user_id not in seen_users:
            if not seen_users:
                seen_users[user_id] = "user"
            elif last_role == "user":
                seen_users[user_id] = "assistant"
            else:
                seen_users[user_id] = "user"
        last_role = seen_users[user_id]
        messages.append((seen_users[user_id], text))
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


# ── Utilities ─────────────────────────────────────────────────────────────

def _extract_content(content) -> str:
    """Pull text from content — handles str, list of blocks, or dict."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    if isinstance(content, dict):
        return content.get("text", "").strip()
    return ""


def _messages_to_transcript(messages: List[Tuple[str, str]]) -> str:
    """Convert [(role, text), ...] to transcript format with > markers."""
    lines = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            lines.append(f"> {text}")
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                lines.append(messages[i + 1][1])
                i += 2
            else:
                i += 1
        else:
            lines.append(text)
            i += 1
        lines.append("")
    return "\n".join(lines)
