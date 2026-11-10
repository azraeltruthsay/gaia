"""Her issues become CodeMind detections (a0aw.7, the Record → Propose link of
knowledge/blueprints/SELF_REPAIR_LOOP.md).

CodeMind's queue was fed by thought seeds, canaries and the observer, never by
the issues she files, so nothing she found could reach a fix. Here an open
issue goes to CodeMind once she has said where the bug is: a file or a
function, given as `file` on an issue note/status, or named in her suspected
cause or notes. This code only looks up what she named (a path, or a unique
`def`/`class` in the candidate packages); it never decides where a bug is.

An issue is tried again only after she adds something (a new note or file), at
most MAX_ATTEMPTS times. The outcome goes back to her issue: 'proposed' with
what the candidate tests said when a fix was kept, a note saying why not
otherwise.

State in /shared/issues/codemind.json: {issue_id: {"attempts", "fingerprint", "last"}}.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STATE = "codemind.json"
MAX_ATTEMPTS = 2
SOURCE = "codemind_issue"
def _repo(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root) if repo_root else Path(os.environ.get("GAIA_REPO_ROOT", "/gaia/GAIA_Project"))

# where a named file or function may be: the candidate packages CodeMind can test
PACKAGES = (("gaia-core", "gaia_core"), ("gaia-common", "gaia_common"))

_PATH = re.compile(r"[\w./-]*\w+\.py\b")
_DOTTED = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_TICKED = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
_CALLED = re.compile(r"\b([A-Za-z_]\w*_\w*)\(\)")


def _roots(repo_root: Path) -> List[Path]:
    return [repo_root / "candidates" / svc / pkg for svc, pkg in PACKAGES]


def _resolve_path(ref: str, repo_root: Path) -> Optional[Path]:
    ref = ref.strip().lstrip("./")
    rel = ref.split("candidates/", 1)[-1]
    for svc, pkg in PACKAGES:
        for cand in (repo_root / "candidates" / rel, repo_root / "candidates" / svc / rel):
            if cand.is_file() and cand.suffix == ".py" and (repo_root / "candidates" / svc / pkg) in cand.parents:
                return cand
    if "/" not in ref:  # a bare file name: only if it names one file
        hits = [p for root in _roots(repo_root) if root.is_dir() for p in root.rglob(ref)
                if "tests" not in p.parts and "__pycache__" not in p.parts]
        if len(hits) == 1:
            return hits[0]
    return None


def _resolve_symbol(name: str, repo_root: Path) -> Optional[Path]:
    rx = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", re.M)
    hits = []
    for root in _roots(repo_root):
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "tests" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                if rx.search(p.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(p)
            except OSError:
                continue
            if len(hits) > 1:
                return None
    return hits[0] if hits else None


def resolve(ref: str, repo_root: Optional[Path] = None) -> Optional[Tuple[Path, str]]:
    """(candidate file, function name or "") for a path or a function she named."""
    repo_root = _repo(repo_root)
    ref = (ref or "").strip().strip("`'\"")
    if not ref:
        return None
    m = re.match(r"^(.+?\.py)\s*(?:::|:|#|,)\s*([A-Za-z_][\w.]*?)(?:\(\))?$", ref)
    if m:  # "agent_core.py:_should_use_tool_routing": that function, in that file
        path = _resolve_path(m.group(1), repo_root)
        name = m.group(2).split(".")[-1]
        if path and re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b",
                              path.read_text(encoding="utf-8", errors="ignore"), re.M):
            return path, name
        return None
    if ref.endswith(".py") or "/" in ref:
        path = _resolve_path(ref, repo_root)
        return (path, "") if path else None
    parts = [x for x in re.split(r"[.:]", ref.removesuffix("()")) if x]
    for name in reversed(parts):  # AgentCore._should_use_tool_routing → the method first
        path = _resolve_symbol(name, repo_root)
        if path:
            return path, name
    return None


def refs_in(text: str) -> List[str]:
    """What in her words could name a place: paths, `Class.method`, `ticked` names, calls()."""
    text = text or ""
    found = _PATH.findall(text) + [m.group(0) for m in _DOTTED.finditer(text) if not m.group(0).endswith(".py")]
    found += _TICKED.findall(text) + _CALLED.findall(text)
    return list(dict.fromkeys(f for f in found if len(f) > 3))


def where(issue: dict, repo_root: Optional[Path] = None) -> Optional[Tuple[Path, str]]:
    """The place she named for this issue: an explicit `file` first, then her
    newest notes, then her suspected cause."""
    repo_root = _repo(repo_root)
    if issue.get("file"):
        hit = resolve(issue["file"], repo_root)
        if hit:
            return hit
    texts = [n.get("text", "") for n in reversed(issue.get("notes") or [])] + [issue.get("suspected_cause", "")]
    for text in texts:
        for ref in refs_in(text):
            hit = resolve(ref, repo_root)
            if hit:
                return hit
    return None


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


def _fingerprint(issue: dict) -> str:
    """What she has said about it: her notes (not CodeMind's) and the file she named."""
    hers = [n for n in issue.get("notes") or [] if not _by_codemind(n.get("text", ""))]
    return f"{len(hers)}|{issue.get('file', '')}"


def _by_codemind(text: str) -> bool:
    return text.startswith(("CodeMind", "[proposed] CodeMind"))


def describe(issue: dict, symbol: str = "") -> str:
    lines = [f"{issue['title']}. {issue.get('symptom', '')}".strip()]
    if issue.get("suspected_cause"):
        lines.append(f"Suspected cause: {issue['suspected_cause']}")
    if symbol:
        lines.append(f"Where: {symbol}")
    if issue.get("evidence"):
        lines.append("Evidence (from her own turns):")
        lines += [f"  {e}" for e in issue["evidence"][:6]]
    for n in (issue.get("notes") or [])[-4:]:
        if not _by_codemind(str(n.get("text", ""))):
            lines.append(f"Her note: {n.get('text', '')}")
    return "\n".join(lines)


def detections(base: Optional[Path] = None, repo_root: Optional[Path] = None, limit: int = 3) -> List[dict]:
    """Detections for her open issues that name a place and haven't been tried
    since she last added something."""
    repo_root = _repo(repo_root)
    from gaia_common.utils import issue_ledger as L

    state = _load(base)
    out = []
    for issue in L.issues(base).values():
        if L._is_closed(issue) or issue["status"] in ("proposed", "fixed"):
            continue
        tried = state.get(issue["id"]) or {}
        if tried.get("attempts", 0) >= MAX_ATTEMPTS or tried.get("fingerprint") == _fingerprint(issue):
            continue
        hit = where(issue, repo_root)
        if not hit:
            continue
        path, symbol = hit
        out.append({
            "source": SOURCE, "issue_type": "self_reported", "file_path": str(path),
            "description": describe(issue, symbol), "severity": "warn",
            "metadata": {"issue_id": issue["id"], "symbol": symbol, "fingerprint": _fingerprint(issue)},
        })
        if len(out) >= limit:
            break
    return out


def record_attempt(detection: dict, kept: bool, why: str, base: Optional[Path] = None) -> None:
    """Remember the attempt and tell her issue how it went."""
    from gaia_common.utils import issue_ledger as L

    meta = detection.get("metadata") or {}
    iid = meta.get("issue_id")
    if not iid:
        return
    state = _load(base)
    entry = state.get(iid) or {"attempts": 0}
    entry.update(attempts=entry.get("attempts", 0) + 1, fingerprint=meta.get("fingerprint", ""),
                 last=time.time(), kept=kept, why=why[:300])
    state[iid] = entry
    _save(state, base)
    where_ = os.path.relpath(detection["file_path"], _repo()) if detection.get("file_path") else "?"
    if kept:
        L.set_status(iid, "proposed", note=f"CodeMind fix in candidates ({where_}): {why}", base=base, source=SOURCE)
    else:
        left = MAX_ATTEMPTS - entry["attempts"]
        L.note_issue(iid, f"CodeMind tried {where_} and kept nothing: {why}. "
                          + (f"Add a note with more of what you found and it tries again ({left} left)."
                             if left > 0 else "No tries left."), base=base)


def excerpt(content: str, symbol: str, max_chars: int = 12000) -> Optional[str]:
    """The text of `def symbol` (or class) in content, for a file too large to
    show whole. A patch's SEARCH text copied from it matches the whole file too."""
    if not symbol:
        return None
    lines = content.splitlines(keepends=True)
    head = re.compile(rf"^(\s*)(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    for i, line in enumerate(lines):
        m = head.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        start = i
        while start > 0 and lines[start - 1].strip().startswith("@"):
            start -= 1
        end = i + 1
        while end < len(lines):
            ln = lines[end]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent and not ln.lstrip().startswith((")", "]", "}")):
                break
            end += 1
        text = "".join(lines[start:end])
        return text if len(text) <= max_chars else None
    return None
