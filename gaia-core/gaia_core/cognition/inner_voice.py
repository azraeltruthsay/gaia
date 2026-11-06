"""Her inner voice in the chat: what she did, and what she thought (Azrael, 2026-11-05).

Two switches per conversation, each on, off or auto:

- thoughts: a line saying what she did to answer (looked something up, recalled
  another conversation, weighed a correction, answered as Prime or Core).
  Auto shows it when she did more than just answer.
- thinking: her reasoning. On, Prime thinks on every turn (20-40 s more on the
  GPU). Auto, she thinks when asked to focus or think hard, as before, and
  what she thought is shown.

Azrael wants auto to be her own choice. Asked "does this need thought?",
neither mind could tell (2026-11-05: both answered alike for "Thanks!" and a
train puzzle), so auto is a rule for now and the choice waits on a better
Core (2qus).

Shown to the person, never kept in her history: a thought she reads back as
something she said becomes a script.
"""
import os
import re
from typing import Iterable, List, Optional

MODES = ("on", "off", "auto")
SWITCHES = {"thoughts": "inner_voice_thoughts", "thinking": "inner_voice_thinking"}
_COMMAND_RE = re.compile(r"^\s*[!/](thoughts|thinking)(?:\s+(on|off|auto))?\s*[.!]?\s*$", re.IGNORECASE)
_MAX_THINKING_CHARS = 1500


def default(switch: str) -> str:
    mode = os.environ.get(f"INNER_VOICE_{switch.upper()}", "auto").lower()
    return mode if mode in MODES else "auto"


def mode(session_manager, session_id: str, switch: str) -> str:
    try:
        value = session_manager.get_session_meta(session_id, SWITCHES[switch])
    except Exception:
        value = None
    return value if value in MODES else default(switch)


def parse_command(text: str) -> Optional[tuple]:
    """("thinking", "on") for "!thinking on"; ("thinking", None) asks what it is."""
    m = _COMMAND_RE.match(text or "")
    return (m.group(1).lower(), (m.group(2) or "").lower() or None) if m else None


_EXPLAIN = {
    ("thoughts", "on"): "I'll say what I did to answer, every turn.",
    ("thoughts", "off"): "I'll keep what I did to myself.",
    ("thoughts", "auto"): "I'll say what I did when I did more than just answer.",
    ("thinking", "on"): "I'll think every turn before I answer, and show you. Replies will take longer.",
    ("thinking", "off"): "I'll keep my thinking to myself.",
    ("thinking", "auto"): "I'll think when you ask me to focus or think hard, and show you what I thought.",
}


def handle_command(session_manager, session_id: str, text: str) -> Optional[str]:
    """Set or report a switch. None if the text isn't a switch command."""
    parsed = parse_command(text)
    if not parsed:
        return None
    switch, new = parsed
    if new:
        session_manager.set_session_meta(session_id, SWITCHES[switch], new)
    current = new or mode(session_manager, session_id, switch)
    prefix = f"{switch.capitalize()}: {current}." if new else f"{switch.capitalize()} is {current}."
    return f"{prefix} {_EXPLAIN[(switch, current)]} (Say !{switch} on, off or auto to change it.)"


def wants_deep_focus(thinking_mode: str, asked: bool) -> bool:
    if thinking_mode == "on":
        return True
    return asked and thinking_mode in ("auto", "off")


def show_thinking(thinking_mode: str, thinking: str) -> bool:
    return thinking_mode != "off" and bool((thinking or "").strip())


def render_thinking(thinking: str) -> str:
    text = (thinking or "").strip()
    if len(text) > _MAX_THINKING_CHARS:
        text = text[:_MAX_THINKING_CHARS].rsplit(" ", 1)[0] + " …"
    quoted = "\n".join("> " + line if line.strip() else ">" for line in text.splitlines())
    return f"*[Thinking]*\n{quoted}"


_SOURCE_NAMES = {
    "web": "searched the web", "mempalace": "checked my memory palace", "journal": "read my journals",
    "kg": "checked my knowledge graph", "vector": "searched my knowledge base",
    "memory": "remembered an earlier lookup", "system": "checked my own state",
}


def trace(*, model: str = "", deep_focus: bool = False, pushback: bool = False,
          conv_recall: bool = False, followup: bool = False, memory_answer: bool = False,
          sources: Iterable[str] = (), tool_ran: bool = False, flags: Iterable[str] = ()) -> List[str]:
    """What she did this turn, in her words. The first item is always which mind answered."""
    items = [f"answered as {model.capitalize()}" if model else "answered"]
    if deep_focus:
        items.append("thought it through first")
    if pushback:
        items.append("weighed your correction against what I could find")
    if conv_recall:
        items.append("recalled our earlier conversations")
    elif followup:
        items.append("followed up on what we were just talking about")
    if memory_answer:
        items.append("remembered an earlier lookup")
    for s in dict.fromkeys(sources):
        name = _SOURCE_NAMES.get(s)
        if name and name not in items:
            items.append(name)
    if tool_ran:
        items.append("used a tool")
    if any(flags):
        items.append("flagged parts of my answer as uncertain")
    return items


def show_trace(thoughts_mode: str, items: List[str]) -> bool:
    if thoughts_mode == "on":
        return bool(items)
    return thoughts_mode == "auto" and len(items) > 1


def render_trace(items: List[str]) -> str:
    return "*[What I did]* " + " · ".join(items)
