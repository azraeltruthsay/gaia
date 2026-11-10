"""Gear switch in the chat: "!gear1", "!gear2", "!gear" (Azrael, 2026-11-10).

Sets the gear she rests in (where she wakes, and comes back to after a voice
session, sleep or training) and shifts there now when she's in an everyday
gear. One setting for the whole system, unlike the inner-voice switches,
which are per conversation. The orchestrator owns it
(/lifecycle/resting_gear); this only parses the command and words the reply.

  !gear1  Core on the GPU is her voice; Prime thinks from the CPU.
  !gear2  Prime on the GPU is her voice; Core listens from the CPU.
  !gear   Says which one she rests in.
"""
import json
import os
import re
import urllib.request
from typing import Optional

_COMMAND_RE = re.compile(r"^\s*[!/]gear\s*([12])?\s*[.!]?\s*$", re.IGNORECASE)
_ORCHESTRATOR = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")

_SAYS = {
    "1": "Gear 1: Core on the GPU is my voice, Prime thinks from the CPU.",
    "2": "Gear 2: Prime on the GPU is my voice, Core listens from the CPU.",
}


def parse_command(text: str) -> Optional[tuple]:
    """("gear", "1") for "!gear1" / "!gear 1"; ("gear", None) asks which one."""
    m = _COMMAND_RE.match(text or "")
    return ("gear", m.group(1)) if m else None


def _call(method: str, path: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        f"{_ORCHESTRATOR}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def handle_command(text: str, set_by: str = "chat") -> Optional[str]:
    """Set or report the resting gear. None if the text isn't a gear command."""
    parsed = parse_command(text)
    if not parsed:
        return None
    _, gear = parsed
    try:
        if gear is None:
            gear = _call("GET", "/lifecycle/resting_gear").get("gear", "?")
            return f"I rest in gear {gear}. {_SAYS.get(gear, '')} (Say !gear1 or !gear2 to change it.)"
        r = _call("POST", "/lifecycle/resting_gear", {"gear": gear, "shift_now": True, "set_by": set_by})
    except Exception as e:
        return f"I couldn't reach the gearbox to change gears ({e.__class__.__name__})."
    moving = " Shifting now." if r.get("shifting_to") else ""
    return f"{_SAYS[gear]} I'll rest here from now on.{moving}"
