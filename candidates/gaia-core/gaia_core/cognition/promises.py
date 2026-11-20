"""She keeps a promise to look something up (wtfr, Azrael 2026-11-19).

Asked the IBC code year in the Grotto and pushed back on, she said "I'll look
it up and get back to you", and nothing came: a turn ends with its reply, and
nothing remembered what she said she'd do. Now, when her reply promises a
lookup, the lookup runs after the turn: the question is searched, the page is
read, she answers from the page (checked against it the same way as any
answer: her other mind tries if hers isn't on the page), and the answer is
posted in the same conversation. If she can't find a source, she says that,
which keeps the promise honestly too.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger("GAIA.Promises")

Generate = Callable[[List[dict]], str]

_PROMISE_RE = re.compile(
    r"\b(?:I'?ll|I\s+will|let\s+me|I'?m\s+going\s+to)\s+(?:go\s+|quickly\s+|just\s+|now\s+)?"
    r"(?:look\s+(?:it|that|this|them)\s+up|look\s+(?:\w+\s+)?(?:up|into|for)|"
    r"check(?:\s+\w+){0,3}|search(?:\s+for)?|find\s+(?:out|it|that)|dig\s+into|research|"
    r"double[- ]check|re-?verify|re-?check|re-?calculate|re-?do|re-?work|verify|confirm)"
    r"|\bget\s+back\s+to\s+you\b|\bgive\s+me\s+a\s+(?:sec|second|moment|minute)\b",
    re.IGNORECASE)
# "Can you check and get back to me?" -- "I can check." agrees to it
_ASKED_TO_LOOK_RE = re.compile(
    r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:check|look|search|find|verify|confirm)|"
    r"\bget\s+back\s+to\s+me\b|\blook\s+it\s+up\b", re.IGNORECASE)
_AGREES_RE = re.compile(r"^\s*(?:sure|yes|ok(?:ay)?|of\s+course|will\s+do|I\s+can)\b", re.IGNORECASE)
# offers aren't promises: "I can search for it. Want me to?" waits for a yes
_OFFER_RE = re.compile(r"\b(?:want|would\s+you\s+like|should|shall)\s+(?:me\s+to|I)\b|\?\s*$", re.IGNORECASE)

QUESTION_SYSTEM = ("You promised to look something up for the person you're talking with. From the "
                   "conversation, write the question you promised to answer, as one standalone "
                   "question with the names and details it needs. Reply with the question only.")
ANSWER = ("You told {who} you'd look this up and get back to them. You've now read the page below. "
          "Answer the question from the page: take facts, figures and section numbers as the page "
          "gives them, and say which page you used ({url}). If the page doesn't settle it, say what "
          "it does show and what's still open. Start naturally, as a follow-up (\"I looked it up: ...\").\n\n"
          "The page ({url}):\n{page}\n\nThe question: {question}")
NOT_FOUND = ("I looked it up, as promised, but couldn't find a source I trust that answers it: "
             "{question} I'd rather not guess. If you know where it's published, point me at it.")


def _log_path() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "promises.jsonl"


def made_promise(reply: str, asked: str = "") -> bool:
    """Her reply says she'll look something up (not just offers to), or agrees
    when asked to ("Can you check and get back to me?" -- "I can check.")."""
    text = (reply or "").strip()
    if asked and _ASKED_TO_LOOK_RE.search(asked) and _AGREES_RE.match(text) and not text.rstrip().endswith("?"):
        return True
    m = _PROMISE_RE.search(text)
    if not m:
        return False
    # the sentence holding the promise, not an offer awaiting a yes
    start = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
    end_dot = text.find(".", m.end())
    sentence = text[start: end_dot if end_dot != -1 else len(text)]
    tail = text[m.end():]
    if _OFFER_RE.search(sentence) or re.match(r"[^.!\n]*\?", tail):
        return False
    # "... Want me to look it up?" closing the reply: still asking
    return not re.search(r"\b(?:want|would\s+you\s+like)\s+me\s+to\b[^.!]*\?\s*$", text, re.IGNORECASE)


def _is_question(q: str) -> bool:
    return q.rstrip().endswith("?") or bool(re.match(
        r"^\s*(?:what|which|who|when|where|why|how|is|are|does|do|can|should|calculate|find|determine)\b", q, re.I))


def promised_question(history: List[dict], generate: Optional[Generate]) -> str:
    """The question she promised to answer, standalone. A calculation asked
    earlier in the conversation is taken as written; a model's restatement is
    used only if it is a question (one restated "I was wrong, the flow is
    0.576 cfs" and that was searched, 2026-11-19)."""
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    turns = [m for m in history if m.get("role") in ("user", "assistant")][-8:]
    asked = [re.sub(r"^\w[\w ]{0,30}:\s*", "", m.get("content") or "").strip() for m in turns
             if m.get("role") == "user"]
    # the question nearest the promise; a calculation is taken as written
    nearest = next((a for a in reversed(asked) if is_worked_problem(a) or ("?" in a and len(a.split()) > 6)), "")
    if nearest and is_worked_problem(nearest):
        return nearest
    if generate is not None:
        convo = "\n".join(f"{m['role']}: {(m.get('content') or '')[:600]}" for m in turns)
        try:
            q = (generate([{"role": "system", "content": QUESTION_SYSTEM},
                           {"role": "user", "content": convo}]) or "").strip().splitlines()
            q = q[0].strip(" \"'") if q else ""
            if len(q) > 10 and _is_question(q):
                return q
        except Exception:
            logger.debug("promised question: generation failed", exc_info=True)
    # the question itself, not the pushback after it ("Are you sure?")
    return nearest


def keep(session_id: str, history: List[dict], who: str, same: Optional[Generate],
         other: Optional[Generate], send: Callable[[str, str], bool], record: Callable[[str], None],
         ground=None, still_open: Optional[Callable[[], bool]] = None) -> Optional[str]:
    """Do the lookup she promised and post what she found. Returns the text."""
    from gaia_core.cognition import answer_checks as ac
    if ground is None:
        from gaia_core.cognition.knowledge_router import ground_query

        def ground(q):
            return ground_query(q, intent="other", max_total_ms=30000).results
    question = promised_question(history, other or same)
    if not question:
        logger.info("promises: couldn't tell what she promised to look up in %s", session_id)
        return None
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    if is_worked_problem(question):
        # a calculation she promised to re-check: computed, not searched
        text, verdict = ac.check_computation(question, "", same or other, other)
        if verdict == "rewritten":
            text, _ = ac.check_arithmetic(question, text, None)
            if still_open is not None and not still_open():
                return None
            return _post(session_id, question, "I re-checked the calculation: " + text, "computed", send, record)
    results = ground(question) or []
    pages = ac.web_pages(results)
    text, verdict = None, "not_found"
    if pages:
        url, page = pages[0]
        sources = " ".join(t for _, t in pages)
        for mind in (same, other):
            if mind is None:
                continue
            try:
                draft = (mind([{"role": "user", "content": ANSWER.format(
                    who=who or "them", url=url, page=page[:6000], question=question)}]) or "").strip()
            except Exception:
                logger.debug("promises: answer failed", exc_info=True)
                draft = ""
            if draft and ac._backed_share(draft, sources) >= 0.5:
                text, verdict = draft if url in draft else f"{draft.rstrip()}\n\n(Source: {url})", "answered"
                break
    if text is None:
        # "couldn't find it" only for a question a person asked; a restated
        # question that isn't one ("Which tool do you need reviewed? The URL
        # doesn't name the tool.") was posted as a non-answer (2026-11-20)
        asked = " ".join((m.get("content") or "").lower() for m in history if m.get("role") == "user")
        if not _key_words_in(question, asked):
            logger.info("promises: nothing found for a question nobody asked (%r); not posting", question[:80])
            return None
        text = NOT_FOUND.format(question=question.rstrip(" ?.") + "?")
    if still_open is not None and not still_open():
        logger.info("promises: she already followed up in %s; not posting", session_id)
        return None
    try:
        text, _ = ac.check_arithmetic(question, text, None)
    except Exception:
        pass
    return _post(session_id, question, text, verdict, send, record)


def _key_words_in(question: str, asked: str) -> bool:
    words = [w for w in re.findall(r"[a-z]{4,}", question.lower())
             if w not in {"what", "which", "where", "when", "does", "have", "that", "this", "with", "from", "about"}]
    return bool(words) and sum(1 for w in words if w in asked) / len(words) >= 0.5


def _post(session_id, question, text, verdict, send, record) -> str:
    delivered = send(session_id, text)
    if delivered:
        record(text)
    logger.info("promises: kept in %s (%s, delivered=%s): %r", session_id, verdict, delivered, question[:80])
    try:
        with _log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "session_id": session_id, "question": question,
                                "verdict": verdict, "delivered": delivered, "text": text},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass
    return text


def start(*args, **kwargs) -> threading.Thread:
    def run():
        try:
            keep(*args, **kwargs)
        except Exception:
            logger.warning("promises: keeping a promise failed", exc_info=True)
    t = threading.Thread(target=run, name="keep-promise", daemon=True)
    t.start()
    return t


def still_open_check(session_manager, session_id: str, promise_text: str) -> Callable[[], bool]:
    """True while her last message in the conversation is still the promise:
    if she has said anything since (answered it another way), it's not."""
    key = (promise_text or "").strip()[:60]

    def check() -> bool:
        try:
            mine = [m for m in session_manager.get_history(session_id) or [] if m.get("role") == "assistant"]
        except Exception:
            return True
        if not mine:
            return True
        return key in (mine[-1].get("content") or "")
    return check
