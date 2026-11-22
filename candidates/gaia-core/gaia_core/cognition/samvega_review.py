"""Samvega review: while she sleeps, she works through her ledger of
mistakes (aidw.7, GROUNDED_SPEECH.md part C4).

For each active finding (open, amended, investigating), oldest first:

  1. Review — the judgement is hers. She is shown what she was asked, what
     she said, the call that checked it and what the source showed, and
     writes one block: LESSON (a one-off: she could have looked and didn't),
     ISSUE + SYMPTOM + CAUSE (it will keep happening until her code
     changes), or DISMISS (the finding is wrong). Code files an ISSUE in her
     issue ledger (the self-repair loop takes it from there) and links it;
     a LESSON becomes a note, and the question with the call that would
     have answered it becomes a selector example (A2 reads those).
  2. Verify — code's alone. The original question is asked again as a
     private turn (source "api": nothing is posted, nothing wakes, the audit
     skips it), the checking call is run again, and finding_grounding
     compares. An answer the source holds up marks the finding verified; one
     it doesn't stays, with a note of what she said this time.

A finding without the question it answered (written before C4) is reviewed
but can't be replayed; it says so.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SamvegaReview")

MAX_PER_RUN = 3
REPLAY_PREFIX = "samvega-replay-"
CORE_URL = os.environ.get("GAIA_CORE_SELF_URL", "http://localhost:6415")

SYSTEM = ("You are GAIA. You are asleep, reading one of your own mistakes: something you said that "
          "the source did not hold up.")

INSTRUCTION = """Write one of these:

LESSON: what you should have done instead, in one sentence

or, when it will keep happening until something in how you work is changed:
ISSUE: a short title
SYMPTOM: what went wrong, in one sentence
CAUSE: your guess at the cause, naming the file or function if you know it, or unknown

or, only if the source above does not actually contradict what you said:
DISMISS: why the finding is wrong"""

_FIELD = re.compile(r"^\s*[-*]?\s*\**(LESSON|ISSUE|SYMPTOM|CAUSE|DISMISS)\**\s*:\**\s*(.*)$", re.I)


def view(f: dict) -> str:
    """The finding as she reads it."""
    lines = [f"Finding {f['id']} ({f['status']}), {f.get('when', '')} in {f.get('where', '')}."]
    if f.get("question"):
        lines.append(f"You were asked: {f['question'][:400]}")
    said = (f.get("message_ref") or {}).get("said") or f.get("you_said") or ""
    lines.append(f"You said: {said[:600]}")
    if f.get("claim"):
        lines.append(f"The part that didn't hold up: {f['claim'][:300]}")
    if f.get("checked_with"):
        lines.append(f"Checked with: {f['checked_with']}")
    if f.get("evidence"):
        shows = re.sub(r"\s+", " ", f["evidence"])[:1200]
        lines.append(f"The source shows: {shows}")
    if f.get("turn_looked_up") is False:
        lines.append("In that turn you looked nothing up before answering.")
    return "\n".join(lines)


def parse(answer: str) -> Dict[str, str]:
    """{'kind': lesson|issue|dismiss, ...fields}; {} if none."""
    out: Dict[str, str] = {}
    for line in (answer or "").splitlines():
        m = _FIELD.match(line)
        if m and m.group(2).strip():
            out.setdefault(m.group(1).lower(), m.group(2).strip())
    for kind in ("dismiss", "issue", "lesson"):
        if out.get(kind):
            return {"kind": kind, **out}
    return {}


def _act(f: dict, decision: Dict[str, str], shown: str) -> str:
    from gaia_common.utils import samvega_ledger as ledger
    from gaia_common.utils import selector_examples
    fid, kind = f["id"], decision.get("kind")
    q, call = f.get("question") or "", f.get("checked_with") or ""
    if kind == "dismiss":
        r = ledger.set_status(fid, "dismissed", reason=decision["dismiss"], by="GAIA (samvega review)")
        return "dismissed" if r.get("ok") else f"dismiss refused: {r.get('error')}"
    if q and call:
        selector_examples.add(q, call, source="samvega", ref=fid)
    if kind == "lesson":
        ledger.note(fid, f"Lesson: {decision['lesson']}", by="GAIA (samvega review)")
        if f["status"] in ("open", "amended"):
            ledger.set_status(fid, "investigating", reason="lesson recorded; waiting for a replay to verify",
                              by="GAIA (samvega review)")
        return "lesson"
    # issue
    from gaia_common.utils import issue_ledger as L
    said_line = next((ln for ln in shown.splitlines() if ln.startswith("You said:")), "")
    src_line = next((ln for ln in shown.splitlines() if ln.startswith("The source shows:")), "")
    evidence = [e for e in (said_line, src_line) if e]
    L.record_shown(shown.splitlines())
    r = L.file_issue(decision["issue"], decision.get("symptom") or decision["issue"], evidence,
                     suspected_cause=decision.get("cause") or "unknown", source="samvega",
                     original_prompt=q)
    if not r.get("ok"):
        ledger.note(fid, f"Issue not filed: {r.get('error')}", by="samvega review")
        return f"issue refused: {r.get('error')}"
    ledger.set_status(fid, "investigating", reason=f"issue {r['id']}: {decision['issue']}",
                      by="GAIA (samvega review)", issue_id=r["id"])
    return f"issue {r['id']}" + (" (already open)" if r.get("duplicate") else "")


# ── Verify by replay ────────────────────────────────────────────────────

def replay_answer(question: str, timeout: float = 240) -> str:
    """Ask the question again as a private turn and return the reply."""
    packet = {
        "version": "v0.4",
        "header": {
            "session_id": f"{REPLAY_PREFIX}{uuid.uuid4().hex[:12]}",
            "packet_id": str(uuid.uuid4()),
            "persona": {"persona_id": "gaia", "role": "assistant"},
            "output_routing": {"primary": {"destination": "api"}, "source_destination": "api",
                               "addressed_to_gaia": True},
        },
        "content": {"original_prompt": question},
        "governance": {"security_scan": {"ran": True, "passed": True, "injection_score": 0.0},
                       "context_pool": False},
    }
    req = urllib.request.Request(f"{CORE_URL}/process_packet", data=json.dumps(packet).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    out: List[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if ev.get("type") == "token":
                out.append(ev.get("value", ""))
            elif ev.get("type") == "reply_done":
                break
    return re.sub(r"^\s*\[[^\]\n]{1,40}\]\s*", "", "".join(out)).strip()


def run_call(call: str) -> str:
    """Run the finding's checking call again (read-only calls only)."""
    from gaia_common.utils.selector_examples import parse_call
    from gaia_common.utils.tool_call_parser import tool_result_text
    from gaia_core.cognition.finding_grounding import _read_only
    from gaia_core.utils.mcp_client import call_jsonrpc
    parsed = parse_call(call)
    if not parsed or not _read_only(parsed["tool"], parsed["params"]):
        return ""
    rpc = call_jsonrpc(method=parsed["tool"], params=parsed["params"])
    if not rpc.get("ok"):
        return ""
    res = (rpc.get("response") or {}).get("result", rpc.get("response"))
    if isinstance(res, dict) and res.get("ok") is False:
        return ""
    return tool_result_text(res, limit=3000)


def verify(f: dict, replay: Callable[[str], str] = replay_answer,
           rerun: Callable[[str], str] = run_call, confirm_generate=None) -> str:
    from gaia_common.utils import samvega_ledger as ledger
    from gaia_core.cognition import finding_grounding as fg
    q, call = f.get("question") or "", f.get("checked_with") or ""
    if not q or not call:
        return "no question or checking call to replay"
    try:
        answer = replay(q)
    except Exception as e:
        logger.info("Samvega review: replay of %s failed: %s", f["id"], e)
        return "replay failed"
    evidence = rerun(call)
    if not answer or not evidence:
        return "replay inconclusive"
    if fg.confirm(answer, answer, evidence, q, confirm_generate) == fg.CONFIRMED:
        ledger.note(f["id"], f"Replay: still not what {call} shows: \"{answer[:200]}\"", by="replay")
        return "still wrong"
    ledger.set_status(f["id"], "verified", reason=f"Replay: \"{answer[:200]}\" agrees with {call}.",
                      by="replay", as_code=True)
    return "verified"


def run(generate: Callable[[str, str], str], *, replay: Callable[[str], str] = replay_answer,
        rerun: Callable[[str], str] = run_call, check_interrupted: Callable[[], None] = lambda: None,
        max_findings: int = MAX_PER_RUN, confirm_generate=None) -> dict:
    """Review and verify up to `max_findings` active findings, oldest first."""
    from gaia_common.utils import samvega_ledger as ledger
    active = list(reversed(ledger.list_findings(status="active", limit=200)["findings"]))
    done = []
    for row in active[:max_findings]:
        check_interrupted()
        f = ledger.show(row["id"])
        if not f.get("ok"):
            continue
        outcome = {"id": f["id"], "status": f["status"]}
        if f["status"] in ("open", "amended"):
            shown = view(f)
            decision = parse(generate(SYSTEM, f"{shown}\n\n{INSTRUCTION}") or "")
            outcome["review"] = _act(f, decision, shown) if decision else "no answer"
            f = ledger.show(f["id"])
        if f.get("status") in ("amended", "investigating", "fixed"):
            outcome["verify"] = verify(f, replay, rerun, confirm_generate)
        logger.info("Samvega review %s: %s", f["id"], outcome)
        done.append(outcome)
    return {"reviewed": done, "active": len(active)}
