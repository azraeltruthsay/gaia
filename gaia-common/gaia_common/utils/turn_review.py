"""Turn-level view of gaia-core's log over a time window (a0aw.6).

The Notice step of the self-repair loop. `introspect(action=logs)` gives raw
lines and only the tail; asked to review the last hour, GAIA got the 50 lines of
the turn she was in, called it a past misroute, and invented a second finding.
This reads the window turn by turn: what was asked, the intent, whether tool
routing fired and what it chose, which tools ran, which mind answered, and the
head of the reply (from chat history). It states what the log says; noticing
what went wrong is hers.

The turn the request is made from (the newest, still without a reply) is left out.

Below the turns go the Symptoms of her active troubleshooting playbooks (ycx7):
what her playbooks say to look for, read from her own SKILL.md files, so a
playbook she updates changes her next review. Criteria, not verdicts — no code
here decides that a turn went wrong.
Every returned line can be cited as issue evidence (see issue_ledger.record_shown).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)\+00:00 \[[\w-]+\] (\w+):([\w.]+):(.*)$")
START = re.compile(r"Processing packet (\S+): '(.*?)(?:\.\.\.)?' from (\S+)\s*$")
SESSION = (re.compile(r"Final prompt assembled for session '([^']+)'"),
           re.compile(r"session_id='([^']+)'"))
INTENT_DONE = re.compile(r"Intent detection: done intent=(\S+) complexity=(\S+) routing=(\S+)")
DOWNGRADE = re.compile(r"LLM intent '(\w+)' .*downgrading to '(\w+)'")
ROUTING_FIRED = re.compile(r"Tool routing triggered for intent: (\S+)")
SELECTED = re.compile(r"Tool selected: (\S+) \(confidence=([\d.]+)\)")
NO_TOOL = re.compile(r"Tool selector determined no tool needed")
INVALID_TOOL = re.compile(r"Tool selector picked an invalid tool: (.{0,40})")
REVIEW_REJECT = re.compile(r"Tool review: approved=False")
EXECUTING = re.compile(r"Tool routing: executing (\S+)")
DISPATCH = re.compile(r"Dispatching domain tool '(\w+)'(?: via action '(\w+)')?")
INLINE_CALL = re.compile(r"Tool call detected: (\w+)\(action=(\w+)")
SUPPRESSED = re.compile(r"Tool call '([^']+)' (suppressed|dropped)")
TOOL_ANSWER = re.compile(r"answering from routed (.+?) result \(tool-answer shape\)")
GROUNDING = re.compile(r"KnowledgeRouter: calling ground_query")
MIND = re.compile(r"\[DELIBERATION\] \S+ \| model=(\S+).*?elapsed=(\d+)ms")
TOOL_FREE = re.compile(r"tool-free intent '(\w+)'")
MODEL_SELECT = re.compile(r"selected_model_name=(\w+)")
# 'nano' is an alias pool key for Core since nano was retired (Sovereign Duality)
_MIND_ALIASES = {"nano": "core"}

CHAT_ENTRY = re.compile(r"^### (USER|ASSISTANT) \((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\)\s*$", re.M)

_LINE_CAP = 400  # some lines carry a whole packet; never hold more than this


def _ts(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_window_bound(value, now: Optional[datetime] = None) -> Optional[datetime]:
    """Minutes back (number or numeric string) or an ISO time (UTC if naive)."""
    if value in (None, ""):
        return None
    now = now or datetime.now(timezone.utc)
    try:
        return now - timedelta(minutes=float(value))
    except (TypeError, ValueError):
        pass
    text = str(value).strip().replace("Z", "+00:00").replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def read_log_lines(paths: List[Path], since: Optional[datetime]) -> List[tuple]:
    """(lineno, ts, level, logger, message) for each timestamped line, oldest
    first, from rotated files (path.1 …) and the live one. Lines are capped."""
    out = []
    for path in paths:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for n, raw in enumerate(fh, 1):
                m = LOG_LINE.match(raw[:_LINE_CAP].rstrip("\n"))
                if not m:
                    continue
                ts = _ts(m.group(1))
                if ts is None or (since and ts < since):
                    continue
                out.append((f"{path.name}:{n}", ts, m.group(2), m.group(3), m.group(4)))
    out.sort(key=lambda r: r[1])
    return out


def log_paths(live: Path, rotations: int = 2) -> List[Path]:
    return [live.with_name(f"{live.name}.{i}") for i in range(rotations, 0, -1)] + [live]


def _add(items: list, value: str) -> None:
    if value not in items:
        items.append(value)


def _warning_key(msg: str) -> str:
    """A warning without its per-call specifics (addresses, numbers), so the
    same warning on every turn is recognisably the same."""
    msg = re.sub(r"0x[0-9a-f]+|\d+(\.\d+)?", "#", msg)
    return " ".join(msg.split())[:120]


def split_turns(lines: List[tuple]) -> List[dict]:
    turns: List[dict] = []
    cur: Optional[dict] = None
    for ref, ts, level, logger, msg in lines:
        m = START.search(msg)
        if m:
            cur = {"ref": ref, "ts": ts, "packet": m.group(1), "asked": m.group(2), "source": m.group(3),
                   "session": None, "intent": None, "routing": None, "selected": [], "executed": [],
                   "notes": [], "mind": None, "elapsed_ms": None, "warnings": [], "_lines": []}
            turns.append(cur)
            continue
        if cur is None:
            continue
        cur["_lines"].append(f"{ref} {level} {logger}: {msg}"[:300])
        if cur["session"] is None:
            for pat in SESSION:
                s = pat.search(msg)
                if s and s.group(1) not in ("default",):
                    cur["session"] = s.group(1)
                    break
        if (s := INTENT_DONE.search(msg)):
            cur["intent"] = s.group(1)
        elif (s := DOWNGRADE.search(msg)):
            cur["notes"].append(f"intent {s.group(1)} downgraded to {s.group(2)}")
        elif (s := ROUTING_FIRED.search(msg)):
            cur["routing"] = f"fired (intent {s.group(1)})"
        elif (s := SELECTED.search(msg)):
            _add(cur["selected"], f"{s.group(1)} ({s.group(2)})")
        elif NO_TOOL.search(msg):
            _add(cur["selected"], "no tool needed")
        elif (s := INVALID_TOOL.search(msg)):
            _add(cur["selected"], f"invalid tool {s.group(1).strip()}")
        elif REVIEW_REJECT.search(msg):
            _add(cur["notes"], "tool review rejected the selection")
        elif (s := EXECUTING.search(msg)):
            cur["executed"].append(s.group(1))
        elif (s := DISPATCH.search(msg)):
            name = s.group(1) + (f"({s.group(2)})" if s.group(2) else "")
            if not any(e.startswith(s.group(1)) for e in cur["executed"]):
                cur["executed"].append(name)
        elif (s := INLINE_CALL.search(msg)):
            _add(cur["executed"], f"{s.group(1)}({s.group(2)}) called mid-reply")
        elif (s := SUPPRESSED.search(msg)):
            _add(cur["notes"], f"tool call {s.group(1)} {s.group(2)}")
        elif (s := TOOL_ANSWER.search(msg)):
            _add(cur["notes"], f"answered from routed {s.group(1)} result")
        elif GROUNDING.search(msg):
            _add(cur["notes"], "knowledge grounding")
        elif (s := MIND.search(msg)) and not cur["elapsed_ms"]:
            cur["mind"], cur["elapsed_ms"] = _MIND_ALIASES.get(s.group(1), s.group(1)), int(s.group(2))
        elif (s := MODEL_SELECT.search(msg)) and cur["mind"] is None:
            cur["mind"] = _MIND_ALIASES.get(s.group(1), s.group(1))
        elif (s := TOOL_FREE.search(msg)):
            _add(cur["notes"], f"tool-free intent {s.group(1)}")
        if level in ("WARNING", "ERROR"):
            _add(cur["warnings"], f"{level}: {_warning_key(msg)}")
    return turns


def _chat_entries(path: Path) -> List[tuple]:
    text = path.read_text(encoding="utf-8", errors="replace")
    marks = list(CHAT_ENTRY.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if body.endswith("---"):
            body = body[:-3].strip()
        out.append((m.group(1), _ts(m.group(2).replace(" ", "T")), body))
    return out


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def attach_chat(turns: List[dict], history_dir: Path, since: Optional[datetime] = None) -> None:
    """Full question, reply head and session from chat history. A turn's USER
    entry is the one whose text begins with what the log quoted, written within
    two minutes of the turn starting; its reply is the next ASSISTANT entry.
    Fast-path turns log no session, so every history file touched in the
    window is searched."""
    if not history_dir.is_dir() or not turns:
        return
    floor = (since or turns[0]["ts"]) - timedelta(minutes=5)
    files: Dict[str, list] = {}
    for path in history_dir.glob("*.md"):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) >= floor:
                files[path.stem] = _chat_entries(path)
        except OSError:
            continue
    used = set()
    for t in turns:
        prefix = _norm(t["asked"])[:40]
        lo, hi = t["ts"] - timedelta(seconds=2), t["ts"] + timedelta(minutes=2)
        sessions = [t["session"]] if t.get("session") in files else list(files)
        best = None
        for sess in sessions:
            entries = files[sess]
            for i, (role, ts, body) in enumerate(entries):
                if role != "USER" or ts is None or not (lo <= ts <= hi) or (sess, i) in used:
                    continue
                if prefix and not _norm(body).startswith(prefix):
                    continue
                if best is None or ts < best[2]:
                    best = (sess, i, ts)
        if not best:
            continue
        sess, i, _ = best
        used.add((sess, i))
        t["session"] = t.get("session") or sess
        t["asked_full"] = files[sess][i][2]
        for role, ts, body in files[sess][i + 1:]:
            if role == "USER":
                break
            if role == "ASSISTANT":
                t["reply"] = body
                break


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render_turn(t: dict, recurring: frozenset = frozenset()) -> List[str]:
    asked = _clip(t.get("asked_full") or t["asked"], 110)
    head = f"[{t['ref']}] {t['ts']:%H:%M:%S} {t['source']}"
    if t.get("session"):
        head += f" session {t['session']}"
    lines = [head, f"  asked: \"{asked}\""]
    facts = [f"intent {t['intent'] or '?'}"]
    facts.append(f"tool routing {t['routing']}" if t["routing"] else "tool routing did not fire")
    if t["selected"]:
        facts.append("selector: " + ", ".join(t["selected"]))
    if t["executed"]:
        facts.append("ran: " + ", ".join(t["executed"]))
    if t["mind"]:
        facts.append(f"answered by {t['mind']}" + (f" in {t['elapsed_ms'] / 1000:.1f}s" if t["elapsed_ms"] else ""))
    lines.append("  " + " | ".join(facts))
    if t["notes"]:
        lines.append("  also: " + "; ".join(t["notes"]))
    own = [w for w in t["warnings"] if w not in recurring]
    if own:
        lines.append(f"  only this turn: {_clip(own[0], 130)}" + (f" (+{len(own) - 1} more)" if len(own) > 1 else ""))
    lines.append(f"  reply: \"{_clip(t['reply'], 140)}\"" if t.get("reply") else "  reply: (none recorded)")
    return lines


_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def playbook_symptoms(skills_dir: Path, budget: int = 1400) -> List[tuple]:
    """(name, [symptom, ...]) for each active troubleshooting playbook, from
    the '## Symptoms' section of its SKILL.md. Bullets are joined to one line;
    playbooks are cut once the budget is spent."""
    out: List[tuple] = []
    if not skills_dir or not Path(skills_dir).is_dir():
        return out
    used = 0
    for path in sorted(Path(skills_dir).glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = _FRONT.match(text)
        front = m.group(1) if m else ""
        if not re.search(r"^domain:\s*troubleshooting\s*$", front, re.M):
            continue
        if re.search(r"^status:\s*(retired|draft|inactive)\s*$", front, re.M):
            continue
        # A playbook says where its symptoms can be seen; only those visible in
        # turns belong here (gear evidence lives in /consciousness/gear, and
        # listing it made her claim a gear mismatch with no turn to show).
        seen_in = re.search(r"^seen_in:\s*\[?([^\]\n]*)", front, re.M)
        if not seen_in or "turns" not in seen_in.group(1):
            continue
        name = (re.search(r"^name:\s*(\S+)", front, re.M) or [None, path.parent.name])[1]
        sec = re.search(r"^## Symptoms\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
        if not sec:
            continue
        bullets: List[str] = []
        for line in sec.group(1).splitlines():
            if line.lstrip().startswith("- "):
                bullets.append(line.lstrip()[2:].strip())
            elif line.strip() and bullets:
                bullets[-1] += " " + line.strip()
        bullets = [b.replace("*", "") for b in bullets if b]
        cost = len(name) + sum(len(b) + 4 for b in bullets)
        if not bullets or (out and used + cost > budget):
            continue
        out.append((name, bullets))
        used += cost
    return out


def review_turns(log_file: Path, history_dir: Path, since: Optional[datetime] = None,
                 until: Optional[datetime] = None, limit: int = 12, search: str = "",
                 session: str = "", now: Optional[datetime] = None, text_budget: int = 3600,
                 include_reviews: bool = False, skills_dir: Optional[Path] = None,
                 oldest_first: bool = False) -> dict:
    """oldest_first: show the first turns after `since` instead of the last ones,
    so a reader working through a long window in order (the self_review sleep
    task, a0aw.4) misses none; `turns[-1]["ts"]` is where to continue."""
    now = now or datetime.now(timezone.utc)
    since = since or now - timedelta(hours=1)
    turns = split_turns(read_log_lines(log_paths(log_file), since))
    attach_chat(turns, history_dir, since)

    excluded = None
    if turns and not turns[-1].get("reply") and (now - turns[-1]["ts"]) < timedelta(minutes=10):
        excluded = turns.pop()  # the turn asking for this review

    # Her earlier looks at herself are left out unless asked for: their replies
    # are past conclusions ("nothing wrong"), and with them in view she repeated
    # them instead of reading the turns (a0aw.6, 2026-11-08).
    # The whole conversation counts: a follow-up in a review session ("which
    # turns exactly?") made no tool call, and its reply fed her old answer back.
    reviews = 0
    if not include_reviews:
        def _is_review(t):
            return any(e.startswith("introspect") for e in t["executed"])
        review_sessions = {t.get("session") for t in turns if _is_review(t) and t.get("session")}
        if excluded is not None and excluded.get("session"):
            review_sessions.add(excluded["session"])
        kept = [t for t in turns if not _is_review(t) and t.get("session") not in review_sessions]
        reviews, turns = len(turns) - len(kept), kept

    if until:
        turns = [t for t in turns if t["ts"] <= until]
    if session:
        turns = [t for t in turns if t.get("session") == session]
    if search:
        s = search.lower()
        turns = [t for t in turns if s in (t.get("asked_full") or t["asked"]).lower()]
    total = len(turns)
    keep = max(1, min(int(limit or 12), 25))
    turns = turns[:keep] if oldest_first else turns[-keep:]

    counts: Dict[str, int] = {}
    for t in turns:
        for w in t["warnings"]:
            counts[w] = counts.get(w, 0) + 1
    # Warnings seen on more than one turn are background (gaia-core logs routine calls at WARNING).
    recurring = frozenset(w for w, c in counts.items() if c > 1)

    window = f"{since:%Y-%m-%d %H:%M} to {(until or now):%Y-%m-%d %H:%M} UTC"
    criteria = playbook_symptoms(skills_dir) if skills_dir else []
    criteria_lines = []
    if criteria:
        criteria_lines.append("What your troubleshooting playbooks say to look for:")
        for name, bullets in criteria:
            criteria_lines += [f"  {name}:"] + [f"   - {b}" for b in bullets]
        text_budget -= sum(len(x) + 1 for x in criteria_lines)
    rendered = [render_turn(t, recurring) for t in turns]
    while len(rendered) > 1 and sum(len("\n".join(r)) + 1 for r in rendered) > text_budget:
        if oldest_first:
            rendered.pop()
            turns = turns[:-1]
        else:
            rendered.pop(0)
            turns = turns[1:]
    header = f"{total} turn(s) in gaia-core's log, {window}"
    if total > len(turns):
        which = "first" if oldest_first else "last"
        header += f"; the {which} {len(turns)} shown (narrow with since/until/search for the others)"
    if excluded:
        header += "; this turn left out"
    if reviews:
        header += f"; {reviews} turn(s) from earlier reviews of yourself left out"
    body = [header]
    if recurring:
        body.append(f"({len(recurring)} warning kinds recur across turns and are left out; introspect logs shows them)")
    body += [ln for r in rendered for ln in r]
    body += criteria_lines
    return {
        "ok": True,
        "text": "\n".join(body),
        "count": total,
        "playbooks": [name for name, _ in criteria],
        "turns": [{"ref": t["ref"], "packet": t["packet"], "session": t.get("session"),
                   "ts": t["ts"].isoformat(), "asked_full": t.get("asked_full") or t.get("asked", ""),
                   "rendered": "\n".join(r)} for t, r in zip(turns, rendered)],
        "_raw_lines": [ln for t in turns for ln in t["_lines"]],
    }
