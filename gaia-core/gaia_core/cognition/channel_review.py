"""She looks back over a channel without meaning to answer anything (Azrael, 2026-11-19).

After a quiet spell (the same moment the missed-message sweep runs), she reads
the channel two ways, each line with its speaker and time:

  - the latest messages, whoever wrote them;
  - her own last messages, so a busy channel can't push what she said out of
    view.

and asks herself, in her inner voice, the questions a person asks scrolling
back: Did I say I'd follow up and not do it? Was I wrong about something,
because someone disputed it or something said later shows it? (Did I miss a
question meant for me? is the missed-message sweep's, which runs alongside.)

What she decides is checked by code before anything is said:
  - an open promise (her message said she'd look something up, and no
    follow-up came) is kept by promises.keep, whether or not she noticed it;
  - a message she thinks was wrong is re-checked against something other
    than herself: a calculation computed in code, or a page she reads. Only
    when that disagrees with what she said does she post a correction, quoting
    herself, and the finding is filed for Samvega (p3ta). If the check backs
    her, nothing is said.

Each of her messages is re-checked at most once. Everything is logged to
/shared/channel_review.jsonl.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.ChannelReview")

Generate = Callable[[List[dict]], str]

VIEW_N = 10
COMPUTED_PER_LOOK = 2         # answers to calculations re-checked per look back
MAX_TRIES = 3                 # inconclusive re-checks of one message before it's left
PROMISE_MAX_AGE_S = 4 * 3600
_KEPT_RE = re.compile(r"^\s*(?:\[\w+\]\s*)?(?:I looked it up|I re-checked|Looking back at what I said)", re.I)


def _shared() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared"))


def _state_path() -> Path:
    return _shared() / "channel_review.json"


def _load_state() -> Dict:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_state(state: Dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, p)


def _log(entry: Dict) -> None:
    try:
        with (_shared() / "channel_review.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **entry}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _epoch(ts) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _when(ts) -> str:
    e = _epoch(ts)
    if e is None:
        return "?"
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(e, ZoneInfo(os.environ.get("GAIA_USER_TZ", "America/Los_Angeles")))
    except Exception:
        dt = datetime.fromtimestamp(e)
    return dt.strftime("%-I:%M %p")


def _line(m: Dict, n: Optional[int] = None) -> str:
    text = (m.get("content") or "").strip().replace("\n", " ")
    who = "Me" if m.get("role") == "assistant" else ""
    head = f"[{n}] " if n is not None else ""
    return f"{head}{_when(m.get('timestamp'))} {who + ': ' if who else ''}{text[:300]}"


def views(history: List[Dict], n: int = VIEW_N):
    """(latest messages, my last messages as (index, message))."""
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    mine = [(i, m) for i, m in enumerate(history) if m.get("role") == "assistant"][-n:]
    return msgs[-n:], mine


def _prev_user(history: List[Dict], i: int) -> str:
    """What her message at i answered: the nearest question before it."""
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    # only what was said since her previous message: an older question was
    # answered by an older message
    start = max((j for j in range(i) if history[j].get("role") == "assistant"), default=-1) + 1
    before = [m.get("content") or "" for m in history[max(start, i - 6):i] if m.get("role") == "user"]
    strip = [re.sub(r"^[^:\n]{1,64}:\s", "", b, count=1).strip() for b in before]
    for b in reversed(strip):
        if is_worked_problem(b) or ("?" in b and len(b.split()) > 5):
            return b
    return strip[-1] if strip else ""


def open_promises(history: List[Dict], now: Optional[float] = None) -> List[int]:
    """Indexes of her messages that promised a follow-up nobody got."""
    from gaia_core.cognition.promises import made_promise
    now = now or time.time()
    out = []
    for i, m in enumerate(history):
        if m.get("role") != "assistant":
            continue
        e = _epoch(m.get("timestamp"))
        if e is None or now - e > PROMISE_MAX_AGE_S:
            continue
        asked = next((h.get("content") or "" for h in reversed(history[:i]) if h.get("role") == "user"), "")
        if not made_promise(m.get("content") or "", asked):
            continue
        later = history[i + 1:]
        if any(h.get("role") == "assistant" and ((h.get("meta") or {}).get("via") == "promise"
                                                 or _KEPT_RE.match(h.get("content") or "")) for h in later):
            continue
        out.append(i)
    return out


def recheck(question: str, said: str, same: Optional[Generate], other: Optional[Generate],
            ground=None):
    """(what she should have said, or None; whether the check concluded).
    Checked against something other than herself. Not concluded: the
    calculations disagreed or there was nothing to read, so it can be tried
    again later."""
    from gaia_core.cognition import answer_checks as ac
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    if not question:
        return None, True
    if is_worked_problem(question):
        text, verdict = ac.check_computation(question, said, same, other)
        if verdict == "rewritten":
            return ac.check_arithmetic(question, text, None)[0], True
        return None, verdict == "ok"
    if ground is None:
        from gaia_core.cognition.knowledge_router import ground_query

        def ground(q):
            return ground_query(q, intent="other", max_total_ms=30000).results
    results = ground(question) or []
    if not ac.web_pages(results):
        return None, False
    text, verdict = ac.check_page(question, said, results, same, other, engineering=False)
    return (text if verdict == "rewritten" else None), verdict in ("rewritten", "kept")


def _own_follow_up(m: Dict) -> bool:
    """Something she posted on her own (a kept promise, a look back, a post):
    it answered no one, so there is no question to check it against. One look
    back re-checked her own correction against an older question (2026-11-20)."""
    meta = m.get("meta") or {}
    return bool(meta.get("unprompted") or meta.get("via") in ("promise", "review")
                or _KEPT_RE.match(m.get("content") or "")
                or (m.get("content") or "").lstrip().startswith("(Withdrawn"))


def _already_corrected(history: List[Dict], i: int) -> bool:
    """She edited it, or her own next message is a follow-up on it (a kept
    promise, a re-check, an earlier look back), or a promise to follow up
    that then was kept ("Let me check" ... "I re-checked the calculation")."""
    from gaia_core.cognition.promises import made_promise
    if history[i].get("revisions"):
        return True
    for h in history[i + 1:]:
        if h.get("role") != "assistant":
            continue
        if (h.get("meta") or {}).get("via") in ("promise", "review") or _KEPT_RE.match(h.get("content") or ""):
            return True
        if not made_promise(h.get("content") or ""):
            return False
    return False


REVIEW_THOUGHT = (
    "I'm looking back over the chat; nobody asked me to. The latest messages:\n{latest}\n\n"
    "My own last messages, numbered:\n{mine}\n\n"
    "Was I wrong about something I said: did someone dispute it, or does something said later, or "
    "what I know now, show it was wrong? If one of my numbered messages needs re-checking, which one? "
    "If everything I said holds up, that's fine.")


def _same_result(said: str, fixed: str) -> bool:
    """A "correction" that repeats every number she gave isn't one (her
    7,200 psi "corrected" to 7,200 psi). 5.76 cfs -> 0.589 cfs is one."""
    def nums(t):
        return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", t or "")]
    a, b = nums(said), nums(fixed)
    if not a or not b:
        return False
    return all(any(abs(x - y) <= max(0.015 * abs(y), 0.01) for y in b) for x in a)


def _recheck_one(session_id, history, i, msg, same, other, ground, send, record, handled, tries, done) -> None:
    mid = msg.get("id") or str(i)
    question = _prev_user(history, i)
    mine_text = re.sub(r"^\s*\[\w+\]\s*", "", msg.get("content") or "").strip()
    fixed, concluded = recheck(question, mine_text, same, other, ground)
    if fixed and _same_result(mine_text, fixed):
        logger.info("channel review: the re-check came to what she said; nothing to correct")
        fixed = None
    # an inconclusive check doesn't use up the message: tried again on a
    # later look back, up to MAX_TRIES (the 20 mR/hr answer's first check
    # disagreed with itself and it was never looked at again)
    tries[mid] = tries.get(mid, 0) + 1
    if concluded or tries[mid] >= MAX_TRIES:
        handled.add(mid)
    if fixed:
        quote = mine_text[:120] + ("…" if len(mine_text) > 120 else "")
        text = f"Looking back at what I said earlier (\"{quote}\"), I was wrong. {fixed}"
        if send(session_id, text):
            record(text)
        _file_finding(session_id, question, mine_text, fixed)
        try:
            from gaia_core.cognition import learning_gaps
            learning_gaps.note(question, "corrected", session_id, fixed[:300])
        except Exception:
            pass
    done["recheck"] = {"message": mine_text[:300], "question": question, "corrected": fixed,
                       "concluded": concluded, "try": tries[mid]}
    _log({"session_id": session_id, "kind": "recheck", **done["recheck"]})


def review(session_id: str, history: List[Dict], generate: Optional[Generate], same: Optional[Generate],
           other: Optional[Generate], send: Callable[[str, str], bool], record: Callable[[str], None],
           ground=None, now: Optional[float] = None) -> Dict:
    """One look back. Returns what was done: {"promise": ..., "recheck": ...}."""
    from gaia_core.cognition import promises
    state = _load_state()
    mine_state = state.get(session_id, {})
    if isinstance(mine_state, list):          # the first format: handled ids only
        mine_state = {"done": mine_state, "tries": {}}
    handled = set(mine_state.get("done", []))
    tries = dict(mine_state.get("tries", {}))
    done: Dict = {"promise": None, "recheck": None}

    # 1. A follow-up she promised and didn't give: kept, by code.
    for i in open_promises(history, now):
        mid = history[i].get("id") or str(i)
        if mid in handled:
            continue
        handled.add(mid)
        who = re.match(r"^\s*([^:\n]{1,40}):", next((h.get("content") or "" for h in reversed(history[:i])
                                                     if h.get("role") == "user"), ""))
        done["promise"] = promises.keep(session_id, history[:i + 1], who.group(1) if who else "",
                                        same, other, send, record, ground=ground)
        _log({"session_id": session_id, "kind": "open_promise", "message": history[i].get("content", "")[:300],
              "kept": done["promise"]})
        break

    # 2. Was I wrong? Every answer of hers to a calculation is re-checked
    # by code, whether or not she notices it (she let "20 mR/hr" stand);
    # then she picks among the rest, and code checks her pick.
    from gaia_core.cognition.nlu.intent_detection import is_worked_problem
    latest, mine = views(history)
    candidates = [(i, m) for i, m in mine if (m.get("id") or str(i)) not in handled
                  and not _own_follow_up(m) and not _already_corrected(history, i)]
    computed = [(i, m) for i, m in candidates if is_worked_problem(_prev_user(history, i))]
    # fewest tries first, newest first among those; up to COMPUTED_PER_LOOK
    computed.sort(key=lambda im: (tries.get(im[1].get("id") or str(im[0]), 0), -im[0]))
    for i, msg in computed[:COMPUTED_PER_LOOK]:
        _recheck_one(session_id, history, i, msg, same, other, ground, send, record, handled, tries, done)
        candidates = [(j, m) for j, m in candidates if j != i]
    if generate is not None and candidates and not done.get("recheck"):
        from gaia_core.cognition.inner_monologue import ask_myself
        numbered = "\n".join(_line(m, n) for n, (_, m) in enumerate(candidates, 1))
        thought = REVIEW_THOUGHT.format(latest="\n".join(_line(m) for m in latest), mine=numbered)
        said = ask_myself(thought, generate, about="looking back over the chat", session_id=session_id,
                          label="RECHECK", form="the number of your message to re-check",
                          my_form="the number of my message to re-check")["say"] or ""
        m = re.match(r"\D*?(\d{1,2})\b", said)
        if m and 1 <= int(m.group(1)) <= len(candidates):
            i, msg = candidates[int(m.group(1)) - 1]
            _recheck_one(session_id, history, i, msg, same, other, ground, send, record, handled, tries, done)
    state[session_id] = {"done": sorted(handled)[-200:],
                         "tries": {k: v for k, v in tries.items() if k not in handled}}
    _save_state(state)
    return done


def _file_finding(session_id: str, question: str, said: str, fixed: str) -> None:
    """Being wrong becomes something to learn from (p3ta)."""
    try:
        from gaia_core.cognition.samvega import (SamvegaArtifact, SamvegaTrigger, compute_samvega_weight,
                                                 save_samvega_artifact)
        save_samvega_artifact(SamvegaArtifact(
            timestamp=datetime.utcnow().isoformat() + "Z", session_id=session_id,
            trigger=SamvegaTrigger.USER_CORRECTION.value,
            original_output_summary=said[:500], what_went_wrong=f"Re-checked on looking back: {question[:300]}",
            root_cause="found on review (computed or read from a page)", corrected_understanding=fixed[:800],
            weight=compute_samvega_weight(SamvegaTrigger.USER_CORRECTION.value)))
    except Exception:
        logger.debug("samvega finding not filed", exc_info=True)


def start(*args, **kwargs) -> threading.Thread:
    def run():
        try:
            out = review(*args, **kwargs)
            logger.info("channel review: %s", {k: bool(v) for k, v in out.items()})
        except Exception:
            logger.warning("channel review failed", exc_info=True)
    t = threading.Thread(target=run, name="channel-review", daemon=True)
    t.start()
    return t
