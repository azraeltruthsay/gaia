"""The other mind checks what the generating one said (b9ts).

One mind generates on the GPU; the other watches from the CPU. Asked to judge a
reply it sees, it barely can: on 12 replies with known mistakes the existing
prompt caught 0 on Core and 2 on Prime. So it doesn't judge. It answers the
question itself, blind to the reply, and the code looks for names and numbers in
the reply that neither that answer nor what is known can back: 9/12 caught on
either mind, with 2-3 false alarms on 12 good replies (2026-11-03).

It never interrupts. Her reply goes out whole; if it seems necessary, the other
mind then adds a follow-up in its own words (Azrael, 2026-11-03). It voices doubt
rather than correcting: it can be wrong too. Everything else is only logged.
"""
import json
import logging
import os
import re
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger("GAIA.SecondMind")

LOG_PATH = os.environ.get("SECOND_MIND_LOG", "/shared/observer/second_mind.jsonl")
_MAX_LOG_BYTES = 5 * 1024 * 1024
_MAX_REPLY_CHARS = 4000
# A capital not opening a sentence, or a number.
_CLAIM_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*|\d[\d:.,]*\d|\d")
_LABEL_RE = re.compile(r"^\s*(?:\[[^\]\n]{1,80}\]\s*|\*\[[^\n]*\]\*\s*)+")
_FAKE_RESULT_RE = re.compile(r"</?tool_(?:result|response)>", re.IGNORECASE)
_busy = threading.Lock()


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def unbacked(reply: str, known: str) -> List[str]:
    """Names and numbers in a reply that `known` doesn't hold."""
    words = _words(known)
    lowered = (known or "").lower()
    body = _LABEL_RE.sub("", reply or "", count=1)
    out = []
    for claim in _CLAIM_RE.findall(body):
        # Also as a substring: "9950" is backed by "9950X3D", "Gemma" by "Gemma4".
        if not _words(claim) <= words and claim.lower() not in lowered and claim not in out:
            out.append(claim)
    return out


def states_something(reply: str) -> bool:
    """Whether a reply holds any name or number at all."""
    return bool(_CLAIM_RE.findall(_LABEL_RE.sub("", reply or "", count=1)))


def check(question: str, reply: str, known: str, generate: Callable[[str], str]) -> Optional[dict]:
    """What the other mind makes of a reply; None if nothing needs asking.
    A reply with no names or numbers beyond what is known costs no generation."""
    reply = (reply or "")[:_MAX_REPLY_CHARS]
    if _FAKE_RESULT_RE.search(reply):
        return {"verdict": "fake_tool_result", "unbacked": []}
    base = f"{question}\n{known}"
    claims = unbacked(reply, base)
    if not claims:
        return None
    context = f"What is known: {known}\n" if known.strip() else ""
    own = generate(f"{context}Question: {question}\nAnswer briefly. If you cannot know, say so.")
    remaining = unbacked(reply, f"{base}\n{own}")
    if not remaining:
        return None
    return {"verdict": "unbacked", "unbacked": remaining, "own_answer": own}


_ABOUT_HER_RE = re.compile(r"\b(?:you|your|yours|yourself|gaia)\b", re.IGNORECASE)
# A name in the question means it's about that, whatever "you" it also has:
# "What is in your memory palace about The Raven?"
_NAMES_SOMETHING_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}")
# A short answer with a few specifics can be doubted. A long one from her weights
# ("the history of Tashkent") flags everything, true or not.
_MAX_CLAIMS_FOR_DOUBT = 3
_MAX_SENTENCE_CHARS = 280


def is_necessary(question: str, result: Optional[dict], ran_tool: bool, had_source: bool) -> bool:
    """Whether what the check found is worth saying out loud."""
    if not result:
        return False
    if result["verdict"] == "fake_tool_result":
        return not ran_tool
    if result["verdict"] != "unbacked" or not result.get("unbacked"):
        return False
    q = question or ""
    if had_source or (_ABOUT_HER_RE.search(q) and not _NAMES_SOMETHING_RE.search(q.replace("GAIA", ""))):
        return True
    return "?" in (question or "") and len(result["unbacked"]) <= _MAX_CLAIMS_FOR_DOUBT


def _plain_followup(result: dict) -> str:
    if result["verdict"] == "fake_tool_result":
        return "That tool result wasn't real: no tool ran, so please don't rely on it."
    return f"I can't back up what I just said about {', '.join(result['unbacked'][:3])}, so please check that."


def followup(reply: str, known: str, result: dict, generate: Callable[[str], str]) -> str:
    """The follow-up, in the other mind's words if they name the actual doubt."""
    plain = _plain_followup(result)
    if result["verdict"] != "unbacked":
        return plain
    doubts = ", ".join(result["unbacked"][:3])
    context = f"What is known: {known[:1500]}\n" if known.strip() else ""
    said = generate(
        f"{context}You are the part of GAIA that was listening. She just said: {reply[:600]}\n"
        f"Nothing you have can back up: {doubts}.\n"
        "Write one short sentence to the person, in the first person, saying which part you can't back up.")
    first = re.split(r"(?<=[.!?])\s|\n", (said or "").strip(), maxsplit=1)[0].strip()
    named = any(_words(d) & _words(first) for d in result["unbacked"][:3])
    return first if named and 0 < len(first) <= _MAX_SENTENCE_CHARS and first[0] not in '<{["*' else plain


def review(question: str, reply: str, known: str, generate: Callable[[str], str], observer: str,
           session_id: str = "", ran_tool: bool = False, had_source: bool = False) -> str:
    """Check a reply and write down what was found. Returns a follow-up if one
    seems necessary, else "". One at a time: a CPU mind skips rather than queues."""
    if os.environ.get("SECOND_MIND_ENABLED", "1").lower() not in ("1", "true", "yes", "on"):
        return ""
    if not _busy.acquire(blocking=False):
        _write({"ts": time.time(), "session_id": session_id, "observer": observer, "verdict": "skipped_busy"})
        return ""
    t0 = time.time()
    entry = {"ts": t0, "session_id": session_id, "observer": observer, "question": question[:300], "reply": reply[:600]}
    try:
        result = check(question, reply, known, generate)
        said = followup(reply, known, result, generate) if is_necessary(question, result, ran_tool, had_source) else ""
        _write({**entry, "seconds": round(time.time() - t0, 1), **(result or {"verdict": "ok"}), "followup": said})
        return said
    except Exception as e:
        logger.info("Second mind: check failed: %s", e)
        _write({**entry, "verdict": "error", "error": str(e)[:200]})
        return ""
    finally:
        _busy.release()


def _write(entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > _MAX_LOG_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".1")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("Second mind: could not log: %s", e)
