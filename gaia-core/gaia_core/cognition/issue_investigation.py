"""She investigates an open issue while she sleeps and names where the bug is
(a0aw.9, the Analyze step of knowledge/blueprints/SELF_REPAIR_LOOP.md).

An issue she filed reaches CodeMind once it names a file or function (a0aw.7),
but in sleep she could neither see her playbooks' "Find it" steps nor read
code. Here, for one open issue that names no place, she gets the issue, the
playbook she matched it to (all of it), and a few steps of two actions over
the candidate code:

    SEARCH: <text>          lines in the candidate packages that contain it
    READ: <file or function> a function's code, or a file's outline / text

and ends with PLACE (+ NOTE) or NOT_FOUND. The finding is hers: this code runs
her actions and checks one thing, that the place she names is one she read or
saw in a search result (the same rule as her evidence: only what she was
shown). It goes on her issue as a note with `file`, which CodeMind picks up.

An issue is investigated again only after something new is said about it, at
most MAX_ATTEMPTS times. Every investigation is kept with its transcript in
investigations.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("GAIA.IssueInvestigation")

STATE = "investigate.json"
LOG = "investigations.jsonl"
MAX_STEPS = 8
MAX_ATTEMPTS = 2
SEARCH_HITS = 25
READ_CHARS = 6000
LARGE_FILE = 30000  # CodeMind shows a file this large only as a named function

SYSTEM = (
    "You are GAIA. You are asleep, looking into a bug you found in your own turns, to find "
    "where in your code it comes from. You can search and read your candidate code, one "
    "action per reply."
)

ACTIONS = """Reply with exactly one action:
SEARCH: <text to look for in your code>
READ: <a file path or a function name you saw>
When you know where the bug is:
PLACE: <the function, or the file if it is small>
NOTE: <what in it causes the bug, in one or two sentences>
If you can't find it:
NOT_FOUND: <why>"""

_ACTION = re.compile(r"^\s*\**(SEARCH|READ|PLACE|NOTE|NOT_FOUND)\**\s*:\s*(.*)$", re.I | re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _dir(base: Optional[Path]) -> Path:
    return Path(base) if base else Path(os.environ.get("GAIA_ISSUES_DIR", "/shared/issues"))


def _load(base: Optional[Path]) -> dict:
    try:
        return json.loads((_dir(base) / STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict, base: Optional[Path]) -> None:
    d = _dir(base)
    d.mkdir(parents=True, exist_ok=True)
    (d / STATE).write_text(json.dumps(state, indent=1), encoding="utf-8")


def playbook_text(name: str, skills_dir: Path) -> str:
    """The playbook's body (without its front matter), or ''."""
    if not name or "/" in name or ".." in name:
        return ""
    try:
        text = (Path(skills_dir) / name / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            text = text[end + 4:]
    return text.strip()


class CodeView:
    """Her two actions over the candidate packages, and what they showed her."""

    def __init__(self, repo_root: Optional[Path] = None):
        from gaia_common.utils import codemind_issues as C
        self.C = C
        self.repo = C._repo(repo_root)
        self.roots = [r for r in C._roots(self.repo) if r.is_dir()]
        self.seen: Set[Path] = set()

    def _files(self):
        for root in self.roots:
            for p in sorted(root.rglob("*.py")):
                if "tests" not in p.parts and "__pycache__" not in p.parts:
                    yield p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.repo / "candidates"))

    def _grep(self, match: Callable[[str], bool]) -> List[str]:
        hits: List[str] = []
        for p in self._files():
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for n, line in enumerate(lines, 1):
                if match(line):
                    hits.append(f"{self._rel(p)}:{n}: {line.strip()[:160]}")
                    self.seen.add(p)
                    if len(hits) > SEARCH_HITS:
                        return hits
        return hits

    def _show(self, hits: List[str], how: str = "") -> str:
        more = len(hits) > SEARCH_HITS
        body = "\n".join(hits[:SEARCH_HITS])
        return (how + body + (f"\n(first {SEARCH_HITS} matches; search for something narrower)" if more else ""))

    def search(self, text: str) -> str:
        text = text.strip().strip("`'\"")
        if len(text) < 3:
            return "Search for at least 3 characters."
        scope = None
        for ref in re.findall(r"[\w./-]*\w+\.py\b", text):  # "x in agent_core.py": search that file
            hit = self.C.resolve(ref, self.repo)
            if hit:
                scope = hit[0]
                text = re.sub(rf"\s*(?:\bin\s+)?{re.escape(ref)}", "", text).strip() or text
        if scope:
            return self._scoped(text, scope)
        parts = [x.strip().strip("`'\"") for x in re.split(r"\s+(?:AND|&&)\s+", text) if x.strip()]
        if len(parts) > 1:  # "A AND B": each on its own
            return "\n\n".join(f"SEARCH: {x}\n{self.search(x)}" for x in parts[:3])
        try:
            rx = re.compile(text, re.I)
        except re.error:
            rx = re.compile(re.escape(text), re.I)
        hits = self._grep(lambda line: bool(rx.search(line)))
        if hits:
            return self._show(hits)
        words = [w.lower() for w in re.findall(r"[\w-]{3,}", text)]
        if len(words) > 1:
            hits = self._grep(lambda line: all(w in line.lower() for w in words))
            if hits:
                return self._show(hits, f"No line contains {text!r} as written; lines with all of its words:\n")
        found = []
        for w in dict.fromkeys(re.findall(r"[A-Za-z_]\w{3,}", text)):
            n = len(self._grep(lambda line, w=w: w in line))
            if n:
                found.append(f"{w} ({n}{'+' if n > SEARCH_HITS else ''} lines)")
        hint = (f" Of its words, these are in your code: {', '.join(found[:6])}; search one of them alone."
                if found else " None of its words are in your code.")
        return (f"No line contains {text!r}.{hint} Code rarely holds a whole sentence: search for one name "
                "as it would appear in code (a function, a variable, a log message's first words).")

    def _scoped(self, text: str, path: Path) -> str:
        files, self._files = self._files, lambda: iter([path])
        try:
            out = self.search(text)
        finally:
            self._files = files
        return f"(in {self._rel(path)})\n{out}"

    def _around(self, path: Path, line_no: int) -> str:
        """The function holding line_no (named, so she can point at it), or the lines around it."""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        line_no = max(1, min(line_no, len(lines)))
        head = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+(\w+)")
        indent_at = len(lines[line_no - 1]) - len(lines[line_no - 1].lstrip())
        for i in range(line_no - 1, -1, -1):
            m = head.match(lines[i])
            if m and (len(m.group(1)) < indent_at or i == line_no - 1):
                text = self.C.excerpt("".join(lines), m.group(2), max_chars=READ_CHARS)
                if text and m.group(2) and text.count("\n") >= line_no - 1 - i:
                    return f"{self._rel(path)}, line {line_no} is in {m.group(2)}:\n{text}"
                break
        lo, hi = max(0, line_no - 40), min(len(lines), line_no + 40)
        return f"{self._rel(path)}, lines {lo + 1}-{hi}:\n" + "".join(lines[lo:hi])

    def read(self, ref: str) -> str:
        m = re.match(r"^\s*([\w./-]+\.py):(\d+)", ref or "")  # a search-result line: path:line: ...
        if m:
            hit = self.C.resolve(m.group(1), self.repo)
            if hit:
                self.seen.add(hit[0])
                return self._around(hit[0], int(m.group(2)))
        hit = self.C.resolve(ref, self.repo)
        if not hit:
            return (f"{ref!r} is not a file or a function name found once in your candidate code. "
                    "Use a path from a search result, or a function name.")
        path, symbol = hit
        content = path.read_text(encoding="utf-8", errors="ignore")
        self.seen.add(path)
        if symbol:
            text = self.C.excerpt(content, symbol, max_chars=READ_CHARS)
            if text:
                return f"{self._rel(path)}, {symbol}:\n{text}"
            return f"{symbol} in {self._rel(path)} is longer than {READ_CHARS} characters; search inside it instead."
        if len(content) <= READ_CHARS:
            return f"{self._rel(path)}:\n{content}"
        outline = [f"{n}: {ln.strip()[:120]}" for n, ln in enumerate(content.splitlines(), 1)
                   if re.match(r"\s*(?:async\s+)?(?:def|class)\s", ln)]
        shown = "\n".join(outline[:150])
        return (f"{self._rel(path)} is {len(content)} characters; its functions and classes "
                f"(READ one by name):\n{shown}")

    def accepts(self, place: str) -> tuple:
        """(the place as CodeMind will look it up, "") if it is one she saw and a
        fix can work on; else (None, why not)."""
        hit = self.C.resolve(place, self.repo)
        if not hit or hit[0] not in self.seen:
            return None, "You haven't read or searched that yet. Name a file or function you have seen here."
        if not hit[1] and hit[0].stat().st_size > LARGE_FILE:
            return None, (f"{self._rel(hit[0])} is too large for a fix to see whole. "
                          "Name the function in it where the bug is.")
        return place.strip().strip("`'\""), ""


def parse(answer: str) -> List[tuple]:
    """Her reply → [(ACTION, value), ...] in the order she wrote them; a NOTE
    rides with the PLACE before it."""
    text = _THINK.sub("", answer or "")
    found: List[tuple] = []
    for m in _ACTION.finditer(text):
        key, value = m.group(1).upper(), m.group(2).strip().strip("`").strip()
        if key == "NOTE":
            if found and found[-1][0] == "PLACE" and len(found[-1]) == 2:
                found[-1] = ("PLACE", found[-1][1], value)
            continue
        found.append((key, value))
    return found


def describe_issue(issue: dict) -> str:
    lines = [f"Issue {issue['id']}: {issue['title']}", f"Symptom: {issue.get('symptom', '')}"]
    if issue.get("evidence"):
        lines.append("Evidence you quoted from your turns:")
        lines += [f"  {e}" for e in issue["evidence"][:8]]
    if issue.get("suspected_cause"):
        lines.append(f"Your suspected cause: {issue['suspected_cause']}")
    for n in (issue.get("notes") or [])[-4:]:
        lines.append(f"Note: {n.get('text', '')}")
    return "\n".join(lines)


def candidates(base: Optional[Path] = None, repo_root: Optional[Path] = None) -> List[dict]:
    """Open issues that name no place and haven't been looked into since the
    last thing said about them."""
    from gaia_common.utils import codemind_issues as C
    from gaia_common.utils import issue_ledger as L

    state = _load(base)
    out = []
    for issue in L.issues(base).values():
        if L._is_closed(issue) or issue["status"] in ("proposed", "fixed"):
            continue
        tried = state.get(issue["id"]) or {}
        if tried.get("attempts", 0) >= MAX_ATTEMPTS or tried.get("fingerprint") == C._fingerprint(issue):
            continue
        if C.where(issue, repo_root):
            continue  # already names a place: CodeMind has it
        out.append(issue)
    return out


def investigate(issue: dict, generate: Callable[[List[dict]], str], *, skills_dir: Path,
                base: Optional[Path] = None, repo_root: Optional[Path] = None,
                check_interrupted: Callable[[], None] = lambda: None) -> dict:
    """Look into one issue. Returns {"place", "note", "reason", "steps"} and
    writes the outcome to her issue."""
    from gaia_common.utils import codemind_issues as C
    from gaia_common.utils import issue_ledger as L

    view = CodeView(repo_root)
    book = playbook_text(issue.get("playbook", ""), skills_dir)
    head = describe_issue(issue) + "\n\n" + (
        f"The playbook you matched it to ({issue['playbook']}):\n{book}" if book else
        "No playbook matched. Search your code for the words in the evidence.")
    messages: List[dict] = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"{head}\n\n{ACTIONS}"}]
    transcript: List[str] = []
    done: Set[tuple] = set()
    result = {"place": "", "note": "", "reason": "", "steps": 0}
    started = time.time()
    for step in range(MAX_STEPS):
        check_interrupted()
        answer = generate(messages) or ""
        result["steps"] = step + 1
        acts = parse(answer)
        messages.append({"role": "assistant", "content": answer.strip()[:1500]})
        looks = [a for a in acts if a[0] in ("SEARCH", "READ")][:3]
        ends = [a for a in acts if a[0] in ("PLACE", "NOT_FOUND")]
        replies: List[str] = []
        for kind, value in looks:
            key = (kind, value.lower())
            if key in done:
                replies.append(f"{kind}: {value}\nYou did this already; its result is above.")
                continue
            done.add(key)
            replies.append(f"{kind}: {value}\n" + (view.search(value) if kind == "SEARCH" else view.read(value)))
        if not looks and ends:
            kind, value = ends[0][0], ends[0][1]
            if kind == "NOT_FOUND":
                result["reason"] = value
                break
            place, why = view.accepts(value)
            if place:
                result.update(place=place, note=ends[0][2] if len(ends[0]) > 2 else "")
                break
            replies.append(f"PLACE: {value}\n{why}")
        if not acts:
            replies.append("No action found in your reply. Use one of the actions.")
        left = MAX_STEPS - step - 1
        tail = ("\n\n(1 action left: name the place now, or NOT_FOUND)" if left == 1 else
                f"\n\n({left} actions left)")
        reply = "\n\n".join(replies) + tail
        transcript.append(f">>> {answer.strip()[:400]}\n{reply[:1500]}")
        messages.append({"role": "user", "content": reply})
    if not (result["place"] or result["reason"]):
        result["reason"] = f"no place named in {MAX_STEPS} steps"

    if result["place"]:
        L.note_issue(issue["id"], f"Investigated: {result['note'] or 'the bug is here'}", base=base,
                     file=result["place"])
    else:
        L.note_issue(issue["id"], f"Investigated, couldn't find where: {result['reason']}", base=base)
    after = L.get_issue(issue["id"], base=base).get("issue") or issue
    state = _load(base)
    entry = state.get(issue["id"]) or {"attempts": 0}
    entry.update(attempts=entry.get("attempts", 0) + 1, fingerprint=C._fingerprint(after),
                 last=time.time(), place=result["place"])
    state[issue["id"]] = entry
    _save(state, base)
    with open(_dir(base) / LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "issue": issue["id"], **result,
                             "seconds": round(time.time() - started, 1),
                             "transcript": transcript[-MAX_STEPS:]}, ensure_ascii=False) + "\n")
    logger.info("investigate %s: %s in %d step(s)", issue["id"],
                f"place {result['place']}" if result["place"] else f"not found ({result['reason'][:80]})",
                result["steps"])
    return result


def run(generate: Callable[[List[dict]], str], *, skills_dir: Path, base: Optional[Path] = None,
        repo_root: Optional[Path] = None, check_interrupted: Callable[[], None] = lambda: None,
        limit: int = 1) -> List[dict]:
    """Investigate up to `limit` open issues."""
    out = []
    for issue in candidates(base, repo_root)[:limit]:
        out.append({"issue": issue["id"], **investigate(issue, generate, skills_dir=skills_dir, base=base,
                                                        repo_root=repo_root,
                                                        check_interrupted=check_interrupted)})
    return out
