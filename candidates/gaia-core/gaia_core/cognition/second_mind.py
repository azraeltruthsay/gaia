"""The other mind checks what the generating one said (b9ts).

One mind generates on the GPU; the other watches from the CPU. Asked to judge a
reply it sees, it barely can: on 12 replies with known mistakes the existing
prompt caught 0 on Core and 2 on Prime. So it doesn't judge. It answers the
question itself, blind to the reply, and the code looks for names and numbers in
the reply that neither that answer nor what is known can back: 9/12 caught on
either mind, with 2-3 false alarms on 12 good replies (2026-11-03).

For now it only writes down what it would have said, for the architect to read.
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
    body = _LABEL_RE.sub("", reply or "", count=1)
    out = []
    for claim in _CLAIM_RE.findall(body):
        if not _words(claim) <= words and claim not in out:
            out.append(claim)
    return out


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


def _write(entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > _MAX_LOG_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".1")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("Second mind: could not log: %s", e)


def watch(question: str, reply: str, known: str, generate: Callable[[str], str], observer: str, session_id: str = "") -> None:
    """Check a reply in the background; the turn never waits. If the last check
    is still running this one is skipped: a CPU mind can't fall behind by more
    than a reply, and it must never delay the one talking."""
    if os.environ.get("SECOND_MIND_ENABLED", "1").lower() not in ("1", "true", "yes", "on"):
        return
    if not _busy.acquire(blocking=False):
        _write({"ts": time.time(), "session_id": session_id, "observer": observer, "verdict": "skipped_busy"})
        return

    def run():
        t0 = time.time()
        try:
            result = check(question, reply, known, generate)
            _write({"ts": t0, "session_id": session_id, "observer": observer, "question": question[:300],
                    "reply": reply[:600], "seconds": round(time.time() - t0, 1),
                    **(result or {"verdict": "ok"})})
        except Exception as e:
            logger.info("Second mind: check failed: %s", e)
            _write({"ts": t0, "session_id": session_id, "observer": observer, "verdict": "error", "error": str(e)[:200]})
        finally:
            _busy.release()

    threading.Thread(target=run, name="second-mind", daemon=True).start()
