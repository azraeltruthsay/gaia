#!/usr/bin/env python3
"""5C — Claude Code Continuous Conversation Collection

Extracts conversational content from Claude Code session logs.
Produces two outputs:
  1. azrael_voice.md    — All of Azrael's messages, chronological
  2. conversations.md   — Full dialogue turns (user + assistant text only, no tool calls)

Usage:
    python3 scripts/extract_5c.py [--output-dir OUTPUT_DIR]
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = Path.home() / ".claude" / "history.jsonl"
OUTPUT_DIR = Path("/gaia/GAIA_Project/knowledge/5c")

# All directories to scan for session files (current + backed-up old OS sessions)
SESSIONS_DIRS = [
    Path.home() / ".claude" / "projects" / "-gaia-GAIA-Project",
    Path("/gaia/GAIA_Project/knowledge/5c/raw_sessions"),  # Backed up from old OS
]

# Patterns that indicate non-conversational content
NOISE_PATTERNS = [
    r"^\[Request interrupted",
    r"^<system-reminder>",
    r"^\s*$",
]

# Tool-call heavy responses we want to skip
SKIP_PREFIXES = [
    "Let me read",
    "Let me check",
    "Let me look",
    "Let me search",
    "Let me verify",
    "Let me examine",
    "Let me find",
    "Let me grep",
    "Let me see what",
    "Now I'll read",
    "Now I'll check",
    "Now let me read",
    "Now let me check",
    "I'll read",
    "Reading ",
]

MIN_TEXT_LEN = 15  # Skip very short fragments

# ANSI escape sequence pattern
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')
# Control characters (except newline, tab, carriage return)
CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# XML-like noise tags from Claude Code internals
INTERNAL_TAG_RE = re.compile(r'<(?:local-command-stdout|local-command-stderr|system-reminder|antml:thinking|antml:function_calls|antml:invoke|antml:parameter)>.*?</(?:local-command-stdout|local-command-stderr|system-reminder|antml:thinking|antml:function_calls|antml:invoke|antml:parameter)>', re.DOTALL)
# Also catch self-closing and unclosed internal tags
INTERNAL_TAG_LOOSE_RE = re.compile(r'</?(?:local-command-stdout|local-command-stderr|system-reminder|antml:thinking)[^>]*>')


def is_noise(text: str) -> bool:
    """Check if a message is noise (system, interrupt, etc.)."""
    for pat in NOISE_PATTERNS:
        if re.match(pat, text.strip()):
            return True
    return False


def is_tool_narration(text: str) -> bool:
    """Check if assistant text is just narrating tool use (not conversational)."""
    stripped = text.strip()
    for prefix in SKIP_PREFIXES:
        if stripped.startswith(prefix) and len(stripped) < 120:
            return True
    return False


def clean_text(text: str) -> str:
    """Remove ANSI escapes, control chars, and internal tags from text."""
    text = ANSI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = INTERNAL_TAG_RE.sub("", text)
    text = INTERNAL_TAG_LOOSE_RE.sub("", text)
    # Bare ANSI codes without ESC prefix (e.g. [1m, [22m)
    text = re.sub(r'\[\d+m', '', text)
    # Remove <local-command-*> tags that may appear in user messages
    text = re.sub(r'</?local-command-\w+>', '', text)
    return text.strip()


def extract_text_from_content(content) -> str:
    """Extract text blocks from message content, skipping tool_use/tool_result/thinking."""
    if isinstance(content, str):
        return clean_text(content)
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "")
            t = clean_text(t)
            if t and len(t) >= MIN_TEXT_LEN:
                texts.append(t)
    return "\n\n".join(texts)


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to readable form."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts_str[:19] if ts_str else "unknown"


def parse_history():
    """Parse history.jsonl for Azrael's messages."""
    if not HISTORY_FILE.exists():
        print(f"History file not found: {HISTORY_FILE}", file=sys.stderr)
        return []

    messages = []
    with open(HISTORY_FILE) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            text = d.get("display", "").strip()
            if not text or is_noise(text) or len(text) < MIN_TEXT_LEN:
                continue

            ts = d.get("timestamp")
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                ts_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            else:
                ts_str = str(ts)

            session_id = d.get("sessionId", "unknown")
            messages.append({
                "timestamp": ts_str,
                "session_id": session_id,
                "text": text,
            })

    return messages


def parse_session(filepath: Path):
    """Parse a session .jsonl for dialogue turns."""
    turns = []
    with open(filepath) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            msg_type = d.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            msg = d.get("message", {})
            content = msg.get("content", [])
            text = extract_text_from_content(content)

            if not text or is_noise(text):
                continue

            # For assistant messages, skip pure tool narration
            if msg_type == "assistant" and is_tool_narration(text):
                continue

            ts = d.get("timestamp", "")
            slug = d.get("slug", "")

            turns.append({
                "role": "Azrael" if msg_type == "user" else "Claude",
                "timestamp": format_timestamp(ts),
                "text": text,
                "slug": slug,
            })

    return turns


def get_session_date(filepath: Path) -> str:
    """Get the earliest timestamp from a session file."""
    with open(filepath) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                ts = d.get("timestamp", "")
                if ts:
                    return format_timestamp(ts)
            except (json.JSONDecodeError, KeyError):
                continue
    return "unknown"


def write_azrael_voice(messages, output_dir: Path):
    """Write Azrael's messages as a chronological collection."""
    outfile = output_dir / "azrael_voice.md"
    with open(outfile, "w") as f:
        f.write("# Azrael's Voice — Claude Code Message History\n\n")
        f.write("> Every message Azrael sent through Claude Code, chronological.\n")
        f.write(f"> **Messages**: {len(messages)}\n")
        if messages:
            f.write(f"> **Earliest**: {messages[0]['timestamp']}\n")
            f.write(f"> **Latest**: {messages[-1]['timestamp']}\n")
        f.write("\n---\n\n")

        current_date = None
        for msg in messages:
            date_part = msg["timestamp"][:10]
            if date_part != current_date:
                current_date = date_part
                f.write(f"\n## {current_date}\n\n")

            f.write(f"**[{msg['timestamp']}]**\n\n")
            f.write(f"{msg['text']}\n\n")
            f.write("---\n\n")

    print(f"  Wrote {len(messages)} messages to {outfile}")


def write_conversations(sessions, output_dir: Path):
    """Write full conversation turns grouped by session."""
    outfile = output_dir / "conversations.md"

    total_turns = sum(len(turns) for _, turns in sessions)
    total_sessions = len(sessions)

    with open(outfile, "w") as f:
        f.write("# 5C — Claude Code Continuous Conversation Collection\n\n")
        f.write("> Full dialogue between Azrael and Claude, extracted from session logs.\n")
        f.write("> Tool calls, code blocks, and system messages filtered out.\n")
        f.write(f"> **Sessions**: {total_sessions} | **Dialogue turns**: {total_turns}\n")
        f.write("\n---\n\n")

        for session_path, turns in sessions:
            if not turns:
                continue

            session_name = session_path.stem[:8]
            slug = turns[0].get("slug", "") if turns else ""
            session_date = turns[0]["timestamp"][:10] if turns else "unknown"
            display_name = slug or session_name

            f.write(f"\n## Session: {display_name} ({session_date})\n\n")

            for turn in turns:
                role = turn["role"]
                text = turn["text"]

                # Truncate very long assistant responses (keep first 800 chars)
                if role == "Claude" and len(text) > 1200:
                    text = text[:800] + "\n\n*[... continued ...]*"

                if role == "Azrael":
                    f.write(f"**Azrael** [{turn['timestamp']}]:\n")
                else:
                    f.write(f"*Claude* [{turn['timestamp']}]:\n")

                f.write(f"\n{text}\n\n")

            f.write("---\n\n")

    print(f"  Wrote {total_sessions} sessions, {total_turns} turns to {outfile}")


def main():
    output_dir = OUTPUT_DIR
    if len(sys.argv) > 2 and sys.argv[1] == "--output-dir":
        output_dir = Path(sys.argv[2])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: Azrael's voice
    print("Extracting Azrael's messages from history.jsonl...")
    messages = parse_history()
    write_azrael_voice(messages, output_dir)

    # Part 2: Full conversations — scan all session directories, dedupe by filename
    print("Extracting conversations from session files...")
    seen_names = set()
    session_files = []
    for sdir in SESSIONS_DIRS:
        if not sdir.exists():
            print(f"  Skipping (not found): {sdir}")
            continue
        for sf in sdir.glob("*.jsonl"):
            if sf.name not in seen_names:
                seen_names.add(sf.name)
                session_files.append(sf)
            # If duplicate, prefer the larger file (more complete)
            else:
                existing = [f for f in session_files if f.name == sf.name][0]
                if sf.stat().st_size > existing.stat().st_size:
                    session_files.remove(existing)
                    session_files.append(sf)
    session_files.sort(key=lambda p: os.path.getmtime(p))
    print(f"  Found {len(session_files)} unique session files across {len(SESSIONS_DIRS)} directories")

    sessions = []
    for i, sf in enumerate(session_files):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(session_files)}...")
        try:
            turns = parse_session(sf)
            if turns:  # Only include sessions with actual dialogue
                sessions.append((sf, turns))
        except Exception as e:
            print(f"  Warning: Failed to parse {sf.name}: {e}", file=sys.stderr)

    write_conversations(sessions, output_dir)

    # Summary
    print(f"\nDone! Output in {output_dir}/")
    print(f"  azrael_voice.md  — {len(messages)} messages")
    print(f"  conversations.md — {len(sessions)} sessions with dialogue")


if __name__ == "__main__":
    main()
