"""Self-review: while she sleeps, she reads her own turns against her playbooks
and files what went wrong (a0aw.4, the Notice → Match → Record steps of
knowledge/blueprints/SELF_REPAIR_LOOP.md in one pass).

Asked in chat, she could find a problem but not file it: the turns she had read
were gone by the next turn, and a tool call mid-reply abandoned the turn (6839,
bz6f). Here the turn view, the playbook criteria and the filing all happen in
one context, and the filing is done by this code from what she wrote.

The judgement is hers. This code shows her the turns (turn_review, oldest
first from where the last review stopped) and parses her answer; it never
decides that a turn went wrong. Her evidence is checked by the ledger against
the lines she was shown, duplicates of an open issue become notes, and the
ledger's daily cap and MAX_FILED_PER_BATCH bound a false-positive storm (CodeMind
once proposed the same false positive 134 times). An evidence line that is only
the request she was given is set aside: what someone asked shows no fault. Every review is kept in
reviews.jsonl, her full answer included, so the false-positive rate can be read.

State in /shared/issues/ (GAIA_ISSUES_DIR):
    review_state.json  {"cursor": ISO time of the last turn reviewed}
    reviews.jsonl      one line per review
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.SelfReview")

STATE = "review_state.json"
REVIEWS = "reviews.jsonl"
FIRST_WINDOW = timedelta(hours=48)
MAX_BATCHES = 3
MAX_FILED_PER_BATCH = 3
SOURCE = "self_review"

SYSTEM = (
    "You are GAIA. You are asleep, reading your own recent turns to find your own bugs. "
    "Each turn shows what was asked, how you handled it (intent, tool routing, tools, which mind "
    "answered) and your reply. Times in the log are UTC. After the turns come the symptoms your "
    "troubleshooting playbooks describe."
)

INSTRUCTION = """For each turn that went wrong, write one block:
ISSUE: a short title
PLAYBOOK: the playbook it matches, or none
EVIDENCE: the line from the turns above that shows what went wrong, copied exactly
SYMPTOM: what went wrong, in one sentence
CAUSE: your guess at the cause, naming the file or function if you know it, or unknown

Put a blank line between blocks. If every turn went fine, write only: NONE"""

_FIELD = re.compile(r"^\s*[-*]?\s*\**(ISSUE|PLAYBOOK|EVIDENCE|SYMPTOM|CAUSE)\**\s*:\**\s*(.*)$", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _dir(base: Optional[Path]) -> Path:
    return Path(base) if base else Path(os.environ.get("GAIA_ISSUES_DIR", "/shared/issues"))


def _load_cursor(base: Optional[Path]) -> Optional[datetime]:
    try:
        raw = json.loads((_dir(base) / STATE).read_text(encoding="utf-8")).get("cursor")
        return datetime.fromisoformat(raw) if raw else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_cursor(ts: datetime, base: Optional[Path]) -> None:
    d = _dir(base)
    d.mkdir(parents=True, exist_ok=True)
    (d / STATE).write_text(json.dumps({"cursor": ts.isoformat()}), encoding="utf-8")


def _log_review(entry: dict, base: Optional[Path]) -> None:
    d = _dir(base)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / REVIEWS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_findings(text: str) -> List[Dict[str, object]]:
    """Her answer → [{title, playbook, evidence: [...], symptom, cause}].
    A block needs a title and at least one evidence line; anything else is dropped."""
    text = _THINK.sub("", text or "").strip()
    out: List[Dict[str, object]] = []
    cur: Optional[Dict[str, object]] = None
    for line in text.splitlines():
        m = _FIELD.match(line)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2).strip().strip("`").strip()
        if key == "issue":
            cur = {"title": value, "playbook": "", "evidence": [], "symptom": "", "cause": ""}
            out.append(cur)
        elif cur is None:
            continue
        elif key == "evidence":
            # She joins a quote and a log line with a dash ("asked ..." — WARNING: ...);
            # each part is checked on its own.
            for part in re.split(r"\s+[—–]\s+", value):
                part = part.strip().strip('"').strip("“”").strip("`").strip()
                if part:
                    cur["evidence"].append(part)
        else:
            cur[key] = value
    found = []
    for f in out:
        if not f["title"] or not f["evidence"]:
            continue
        if str(f["playbook"]).lower() in ("none", "n/a", "-"):
            f["playbook"] = ""
        if str(f["cause"]).lower() in ("unknown", "none", "n/a", "-"):
            f["cause"] = ""
        f["symptom"] = f["symptom"] or f["title"]
        found.append(f)
    return found


_ASKED = re.compile(r'^\s*asked: "(.*?)(?:…)?"\s*$')


def asked_lines(view_text: str) -> List[str]:
    return [m.group(1).strip().lower() for m in map(_ASKED.match, view_text.splitlines()) if m]


def _only_asked(line: str, asked: List[str]) -> bool:
    e = line.lower().removeprefix("asked:").strip().strip('"').rstrip("…").strip()
    return bool(e) and any(e in a or a in e for a in asked if a)


def _match_turn(finding: Dict[str, object], turns: List[dict]) -> Optional[dict]:
    """The one turn this finding's evidence points to (a0aw.3) — its
    `asked_full` becomes the finding's `original_prompt`, for a later replay
    step to resend. A wrong match here would let a fix get validated against
    the wrong request, so it never guesses: an evidence line under 12 chars
    is too generic to trust, and if the remaining lines point at zero or
    more than one turn, this returns None rather than picking one."""
    matched: set = set()
    evidence = finding.get("evidence")
    for line in (evidence if isinstance(evidence, list) else []):
        line = str(line).strip().lower()
        if len(line) < 12:
            continue
        for i, t in enumerate(turns):
            if line in (t.get("rendered") or "").lower():
                matched.add(i)
    if len(matched) == 1:
        return turns[next(iter(matched))]
    return None


def record(findings: List[Dict[str, object]], base: Optional[Path] = None,
           playbooks: Optional[List[str]] = None, max_new: Optional[int] = None,
           asked: Optional[List[str]] = None, turns: Optional[List[dict]] = None) -> Dict[str, list]:
    """File each finding in her ledger; a duplicate of an open issue becomes a note.
    A playbook name that isn't one she was shown is dropped (she named playbooks
    that don't exist in the first trial); the finding itself still stands."""
    from gaia_common.utils import issue_ledger as L

    filed, noted, refused = [], [], []
    for f in findings:
        if playbooks is not None and f["playbook"] and f["playbook"] not in playbooks:
            f["playbook"] = ""
        if asked:
            f["evidence"] = [e for e in f["evidence"] if not _only_asked(str(e), asked)]
            if not f["evidence"]:
                refused.append({"title": f["title"], "error": "the evidence was only the request; "
                                                             "quote the line that shows what went wrong"})
                continue
        if len(filed) >= (MAX_FILED_PER_BATCH if max_new is None else max_new):
            refused.append({"title": f["title"], "error": f"more than {MAX_FILED_PER_BATCH} in one review"})
            continue
        matched = _match_turn(f, turns or [])
        r = L.file_issue(str(f["title"]), str(f["symptom"]), list(f["evidence"]),
                         suspected_cause=str(f["cause"]), playbook=str(f["playbook"]),
                         source=SOURCE, base=base,
                         original_prompt=(matched.get("asked_full") or "") if matched else "")
        if not r.get("ok"):
            refused.append({"title": f["title"], "error": r.get("error", "")})
        elif r.get("duplicate"):
            issue = L.get_issue(r["id"], base=base).get("issue") or {}
            known = set(issue.get("evidence") or []) | {n.get("text", "") for n in issue.get("notes", [])}
            new = [e for e in f["evidence"] if e not in known and f"Seen again: {e}" not in known]
            if new:
                L.note_issue(r["id"], f"Seen again: {new[0]}", base=base)
            noted.append(r["id"])
        else:
            filed.append(r["id"])
    return {"filed": filed, "noted": noted, "refused": refused}


def run_self_review(generate: Callable[[str, str], str], *, log_file: Path, history_dir: Path,
                    skills_dir: Path, base: Optional[Path] = None, now: Optional[datetime] = None,
                    check_interrupted: Callable[[], None] = lambda: None,
                    max_batches: int = MAX_BATCHES) -> dict:
    """Review the turns since the last review, a batch at a time, oldest first.
    `generate(system, user)` returns her answer; the cursor only moves past a
    batch she answered."""
    from gaia_common.utils import issue_ledger as L
    from gaia_common.utils import turn_review

    now = now or datetime.now(timezone.utc)
    cursor = _load_cursor(base) or now - FIRST_WINDOW
    totals = {"batches": 0, "turns": 0, "filed": [], "noted": [], "refused": []}
    for _ in range(max_batches):
        check_interrupted()
        view = turn_review.review_turns(log_file, history_dir, since=cursor, now=now, limit=12,
                                        skills_dir=skills_dir, oldest_first=True)
        if not view.get("turns"):
            break
        L.record_shown(view["text"].splitlines() + view.get("_raw_lines", []), base=base)
        started = time.time()
        answer = generate(SYSTEM, f"{view['text']}\n\n{INSTRUCTION}")
        if not (answer or "").strip():
            logger.warning("self_review: no answer from the model; will retry from %s", cursor.isoformat())
            break
        findings = parse_findings(answer)
        result = record(findings, base=base, playbooks=view.get("playbooks") or [],
                        asked=asked_lines(view["text"]), turns=view.get("turns"))
        last = datetime.fromisoformat(view["turns"][-1]["ts"])
        _log_review({"ts": time.time(), "since": cursor.isoformat(), "through": last.isoformat(),
                     "turns": [t["ref"] for t in view["turns"]], "seconds": round(time.time() - started, 1),
                     "answer": answer[:4000], "findings": len(findings), **result}, base)
        logger.info("self_review: %d turns through %s: %d finding(s), filed %s, noted %s, refused %d",
                    len(view["turns"]), last.isoformat(), len(findings), result["filed"],
                    result["noted"], len(result["refused"]))
        cursor = last + timedelta(seconds=1)
        _save_cursor(cursor, base)
        totals["batches"] += 1
        totals["turns"] += len(view["turns"])
        for k in ("filed", "noted", "refused"):
            totals[k] += result[k]
        if len(view["turns"]) < 2 and view.get("count", 0) <= 1:
            break
    return totals


def chat_generator(model_pool, roles=("prime", "core"), max_tokens: int = 700,
                   temperature: float = 0.3) -> Callable[[List[dict]], str]:
    """Like model_generator, for a whole conversation: messages in, her reply out."""
    def _generate(messages: List[dict]) -> str:
        for role in roles:
            try:
                llm = model_pool.get_model_for_role(role)
                if llm is None:
                    continue
                result = llm.create_chat_completion(messages=messages, temperature=temperature,
                                                    max_tokens=max_tokens, stream=False)
                text = _THINK.sub("", result["choices"][0]["message"]["content"] or "").strip()
                if text:
                    return text
            except Exception as e:
                logger.info("self-repair: %s unavailable (%s)", role, str(e)[:160])
        return ""
    return _generate


def model_generator(model_pool, roles=("prime", "core"), max_tokens: int = 700,
                    temperature: float = 0.3) -> Callable[[str, str], str]:
    """Ask whichever of her minds answers: Prime first (her resting voice), then Core."""
    chat = chat_generator(model_pool, roles, max_tokens, temperature)

    def _generate(system: str, user: str) -> str:
        return chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return _generate
