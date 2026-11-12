"""CodeMind's Validate step (a0aw.3): replay the offending turn on
gaia-core-candidate and check the live symptom, not just a unit test.

Candidate tests (a0aw.2) and a regression test (a0aw.8) prove a function
behaves correctly in isolation. Neither proves her diagnosis was even the
right place — "her lead can be wrong and still pass the tests" (the comment
that used to gate every issue-sourced fix straight to the architect queue).
This resends the real request that showed the symptom straight through
gaia-core-candidate's real cognitive pipeline and checks its own log for
whether the symptom still fires.

Scoped deliberately narrow: PLAYBOOK_CHECKS is keyed by playbook CLASS (one
of her own ts-*/SKILL.md names) and a GENERIC check for that class — never a
specific bug's substring, file, or function name. New playbook classes get a
new entry here when they exist; this module never guesses at one.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

from gaia_common.utils import turn_review
from gaia_common.utils.turn_review import ROUTING_FIRED

logger = logging.getLogger("GAIA.CodeMind.Replay")

DEFAULT_CANDIDATE_URL = "http://gaia-core-candidate:6415/process_packet"
DEFAULT_CANDIDATE_LOG = Path("/logs/gaia-core-candidate.log")

# playbook class -> its generic, bug-agnostic symptom check. Reuses
# turn_review's own compiled regex (the canonical source for this log line)
# rather than duplicating the literal text here.
PLAYBOOK_CHECKS: Dict[str, dict] = {
    "ts-misrouted-request": {"expect_absent": [ROUTING_FIRED.pattern]},
}


def check_for_playbook(playbook: str) -> Optional[dict]:
    """The {"expect_absent"/"expect_present": [...]} kwargs for replay_turn,
    or None if this playbook has no registered check (an unrecognized or
    not-yet-supported class — callers must skip replay cleanly, not guess)."""
    return PLAYBOOK_CHECKS.get((playbook or "").strip()) or None


def _build_packet(original_prompt: str) -> dict:
    return {
        "version": "v0.4",
        "header": {
            "session_id": f"codemind-replay-{uuid.uuid4().hex[:12]}",
            "packet_id": str(uuid.uuid4()),
            "persona": {"persona_id": "gaia", "role": "assistant"},
        },
        "content": {"original_prompt": original_prompt},
        "governance": {
            "security_scan": {"ran": True, "passed": True, "injection_score": 0.0},
            "context_pool": False,
        },
    }


def _send_packet(url: str, packet: dict, timeout: float) -> None:
    """POST the packet and drain the (streaming) response body. The reply
    content is irrelevant here — only that the real pipeline ran. Never
    raises: an unreachable/erroring candidate is the caller's problem to
    treat as inconclusive, not a replay crash."""
    body = json.dumps(packet).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001 - connectivity/timeout, not our bug to raise
        logger.info("CodeMind replay: candidate call did not complete cleanly: %s", e)


def replay_turn(original_prompt: str, *, expect_absent: Sequence[str] = (),
                 expect_present: Sequence[str] = (),
                 candidate_url: str = DEFAULT_CANDIDATE_URL,
                 candidate_log: Path = DEFAULT_CANDIDATE_LOG,
                 poll_timeout_s: float = 20.0, request_timeout_s: float = 60.0) -> dict:
    """Send `original_prompt` as a fresh CognitionPacket to gaia-core-candidate,
    then check its own log, from just before the send, for the given regexes.

    Returns {ok, symptom_seen, detail}:
      ok=True,  symptom_seen=True  -> a disallowed line fired (bug reproduced)
      ok=True,  symptom_seen=False -> none fired, within the window checked
      ok=False, symptom_seen=None  -> inconclusive (no check given, or the
                                       candidate never produced a log line in
                                       the window at all) — never a verdict.
    """
    if not original_prompt or not (expect_absent or expect_present):
        return {"ok": False, "symptom_seen": None, "detail": "no original_prompt or no check given"}

    since = datetime.now(timezone.utc)
    _send_packet(candidate_url, _build_packet(original_prompt), request_timeout_s)

    absent_rx = [re.compile(p) for p in expect_absent]
    present_rx = [re.compile(p) for p in expect_present]
    deadline = time.monotonic() + poll_timeout_s
    saw_any_line = False
    while True:
        lines = turn_review.read_log_lines(turn_review.log_paths(candidate_log), since=since)
        saw_any_line = saw_any_line or bool(lines)
        messages = [m for (_ref, _ts, _lvl, _logger, m) in lines]
        hit_absent = next((m for m in messages for rx in absent_rx if rx.search(m)), None)
        if hit_absent:
            return {"ok": True, "symptom_seen": True, "detail": f"saw: {hit_absent[:200]}"}
        if present_rx and any(rx.search(m) for m in messages for rx in present_rx):
            return {"ok": True, "symptom_seen": False, "detail": "expected line seen, nothing disallowed"}
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)

    if present_rx:
        return {"ok": False, "symptom_seen": None,
                "detail": "timed out waiting for the expected line" if saw_any_line
                else "candidate produced no log output in the window"}
    if not saw_any_line:
        return {"ok": False, "symptom_seen": None, "detail": "candidate produced no log output in the window"}
    return {"ok": True, "symptom_seen": False, "detail": "no disallowed line seen within the poll window"}
