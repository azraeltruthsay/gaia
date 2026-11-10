"""GAIA's issue ledger: where she records the bugs she finds in herself (a0aw.1).

The Record step of the self-repair loop (knowledge/blueprints/SELF_REPAIR_LOOP.md).
Beads is the system of record, but the beads database lives on the host and no
container runs `bd`. So she writes here, an append-only JSONL of events in
/shared/issues/, and a host bridge (scripts/gaia_issue_bridge.py) turns each issue
into a bead labelled `gaia-found` and writes the bead id and status back to
beads.json, which she reads.

    ledger.jsonl  — hers. One event per line: file / note / status.
    beads.json    — the bridge's. {"issues": {gi-id: {"bead", "bead_status", ...}},
                    "ledger_offset": <events already mirrored>}

An issue needs evidence, and the evidence is checked (a0aw.6): every line
introspect shows her (turns, logs) is kept for two hours in shown.jsonl, and
an issue she files herself is refused unless at least one evidence line matches
something she was shown. The loop's first probe had her inventing a finding that
was in nothing she had been given. Each line's result is kept with the issue. Filing an issue that matches an open one returns
the open one instead of a duplicate (CodeMind once proposed the same false
positive 134 times).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

ISSUES_DIR = Path(os.environ.get("GAIA_ISSUES_DIR", "/shared/issues"))
LEDGER = "ledger.jsonl"
BEADS_MAP = "beads.json"
SHOWN = "shown.jsonl"
SHOWN_TTL_S = 2 * 3600
_SHOWN_MAX_LINES = 20000

STATUSES = ("open", "investigating", "proposed", "fixed", "wontfix")
MAX_NEW_PER_DAY = 10
DUPLICATE_SIMILARITY = 0.6
_MAX_FIELD = 2000
_MAX_EVIDENCE_LINES = 20

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset("a an the and or of to in on for is was it she her i my me that this with as at by from be".split())


def _dir(base: Optional[Path] = None) -> Path:
    # read at call time: a test (or a caller) that sets GAIA_ISSUES_DIR after
    # import must not write into the live ledger
    return Path(base) if base else Path(os.environ.get("GAIA_ISSUES_DIR", str(ISSUES_DIR)))


def _clip(text, limit: int = _MAX_FIELD) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _events(base: Optional[Path] = None) -> List[dict]:
    path = _dir(base) / LEDGER
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append(event: dict, base: Optional[Path] = None) -> None:
    d = _dir(base)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _norm(text) -> str:
    text = str(text).lower().strip().strip('"\'`').strip()
    return " ".join(text.split())


def record_shown(lines, base: Optional[Path] = None) -> None:
    """Remember what introspect showed her, so her evidence can be checked."""
    lines = [str(ln)[:600] for ln in (lines or []) if str(ln).strip()]
    if not lines:
        return
    d = _dir(base)
    try:
        d.mkdir(parents=True, exist_ok=True)
        path = d / SHOWN
        now = time.time()
        keep = []
        if path.is_file() and path.stat().st_size > 4 * 1024 * 1024:
            for raw in path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(raw).get("ts", 0) >= now - SHOWN_TTL_S:
                        keep.append(raw)
                except json.JSONDecodeError:
                    continue
            path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": now, "lines": lines}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _shown_lines(base: Optional[Path] = None) -> List[str]:
    path = _dir(base) / SHOWN
    if not path.is_file():
        return []
    cutoff = time.time() - SHOWN_TTL_S
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("ts", 0) >= cutoff:
            out.extend(_norm(ln) for ln in entry.get("lines", []))
    return out[-_SHOWN_MAX_LINES:]


def verify_evidence(evidence: List[str], base: Optional[Path] = None) -> List[bool]:
    """For each evidence line: does it match something introspect showed her in
    the last two hours? A quote inside a shown line, a shown line inside the
    quote, or (for a paraphrased quote) 80% of its words in one shown line."""
    shown = _shown_lines(base)
    out = []
    for ev in evidence:
        e = _norm(ev)
        ok = False
        if len(e) >= 8 and shown:
            toks = _tokens(e)
            for line in shown:
                if e in line or (len(line) >= 20 and line in e):
                    ok = True
                    break
                if len(toks) >= 4 and len(toks & _tokens(line)) >= 0.8 * len(toks):
                    ok = True
                    break
        out.append(ok)
    return out


def _beads_map(base: Optional[Path] = None) -> dict:
    path = _dir(base) / BEADS_MAP
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def issues(base: Optional[Path] = None) -> Dict[str, dict]:
    """Fold the event log into the current state of each issue, joined with
    what the bridge knows (bead id, bead status)."""
    state: Dict[str, dict] = {}
    for ev in _events(base):
        iid = ev.get("id")
        kind = ev.get("event")
        if kind == "file" and iid:
            state[iid] = {
                "id": iid,
                "title": ev.get("title", ""),
                "symptom": ev.get("symptom", ""),
                "evidence": ev.get("evidence", []),
                "evidence_verified": ev.get("evidence_verified", []),
                "suspected_cause": ev.get("suspected_cause", ""),
                "playbook": ev.get("playbook", ""),
                "source": ev.get("source", ""),
                "status": "open",
                "filed": ev.get("ts"),
                "notes": [],
                "file": "",
            }
        elif iid in state and kind == "note":
            state[iid]["notes"].append({"ts": ev.get("ts"), "text": ev.get("text", "")})
            if ev.get("file"):
                state[iid]["file"] = ev["file"]
        elif iid in state and kind == "status":
            if ev.get("file"):
                state[iid]["file"] = ev["file"]
            state[iid]["status"] = ev.get("status", state[iid]["status"])
            if ev.get("note"):
                state[iid]["notes"].append({"ts": ev.get("ts"), "text": f"[{ev['status']}] {ev['note']}"})
    bridged = _beads_map(base).get("issues", {})
    for iid, issue in state.items():
        b = bridged.get(iid) or {}
        issue["bead"] = b.get("bead")
        issue["bead_status"] = b.get("bead_status")
    return state


def _is_closed(issue: dict) -> bool:
    return issue["status"] in ("fixed", "wontfix") or issue.get("bead_status") == "closed"


def file_issue(title: str, symptom: str, evidence, suspected_cause: str = "",
               playbook: str = "", source: str = "self", base: Optional[Path] = None) -> dict:
    """Record a bug she found. `evidence` is the log lines or reply text she
    actually saw (a string or a list of lines); at least one is required."""
    title = _clip(title, 120)
    symptom = _clip(symptom)
    if isinstance(evidence, str):
        evidence = [ln for ln in evidence.splitlines() if ln.strip()]
    evidence = [_clip(e, 500) for e in (evidence or []) if str(e).strip()][:_MAX_EVIDENCE_LINES]
    if not title or not symptom:
        return {"ok": False, "error": "title and symptom are required"}
    if not evidence:
        return {"ok": False, "error": "evidence is required: quote the log lines or reply text you saw "
                                      "(introspect logs first); an issue with no evidence is a guess"}

    verified = verify_evidence(evidence, base)
    reported = (source or "").lower().startswith("user")
    if not reported and not any(verified):
        return {"ok": False, "error": "none of the evidence matches anything introspect showed you in the last "
                                      "two hours. Look first (introspect turns or logs), then quote the lines "
                                      "you saw. If a person reported this to you, file it with source 'user'."}

    current = issues(base)
    probe = f"{title} {symptom}"
    for issue in current.values():
        if _is_closed(issue):
            continue
        if _similarity(probe, f"{issue['title']} {issue['symptom']}") >= DUPLICATE_SIMILARITY:
            return {"ok": True, "duplicate": True, "id": issue["id"], "bead": issue.get("bead"),
                    "message": f"Already filed as {issue['id']} ('{issue['title']}'). "
                               "Add what you found as a note on it instead."}

    day_ago = time.time() - 86400
    recent = sum(1 for i in current.values() if (i.get("filed") or 0) >= day_ago)
    if recent >= MAX_NEW_PER_DAY:
        return {"ok": False, "error": f"{recent} issues filed in the last day (limit {MAX_NEW_PER_DAY}); "
                                      "add notes to open ones, or wait"}

    ts = time.time()
    iid = "gi-" + hashlib.sha1(f"{title}|{symptom}|{ts}".encode()).hexdigest()[:6]
    _append({"event": "file", "id": iid, "ts": ts, "title": title, "symptom": symptom,
             "evidence": evidence, "evidence_verified": verified, "suspected_cause": _clip(suspected_cause),
             "playbook": _clip(playbook, 80), "source": _clip(source, 40)}, base)
    unverified = len(verified) - sum(verified)
    note = f" {unverified} evidence line(s) did not match what you were shown and are marked unverified." if unverified else ""
    return {"ok": True, "id": iid, "evidence_verified": verified, "message": f"Filed {iid}.{note} It becomes a bead (label gaia-found) "
                                              "when the bridge next runs; `issues` shows its bead id."}


def note_issue(issue_id: str, text: str, base: Optional[Path] = None, file: str = "") -> dict:
    """Add what she learned. `file` names the file or function where she thinks
    the bug is (a0aw.7): an open issue that names one goes to CodeMind."""
    text = _clip(text)
    if not text:
        return {"ok": False, "error": "text is required"}
    if issue_id not in issues(base):
        return {"ok": False, "error": f"no issue {issue_id}"}
    event = {"event": "note", "id": issue_id, "ts": time.time(), "text": text}
    if file:
        event["file"] = _clip(file, 300)
    _append(event, base)
    return {"ok": True, "id": issue_id}


def set_status(issue_id: str, status: str, note: str = "", base: Optional[Path] = None,
               file: str = "", source: str = "") -> dict:
    """Move an issue along: investigating → proposed → fixed (or wontfix).
    'fixed' is her report; the bead stays open until the fix is promoted and
    someone who can promote closes it."""
    status = (status or "").strip().lower()
    if status not in STATUSES:
        return {"ok": False, "error": f"status must be one of {', '.join(STATUSES)}"}
    if issue_id not in issues(base):
        return {"ok": False, "error": f"no issue {issue_id}"}
    event = {"event": "status", "id": issue_id, "ts": time.time(), "status": status, "note": _clip(note)}
    if file:
        event["file"] = _clip(file, 300)
    if source:
        event["source"] = _clip(source, 40)
    _append(event, base)
    return {"ok": True, "id": issue_id, "status": status}


def list_issues(status: str = "", include_closed: bool = False, base: Optional[Path] = None) -> dict:
    out = []
    for issue in issues(base).values():
        if status and issue["status"] != status:
            continue
        if not status and not include_closed and _is_closed(issue):
            continue
        out.append({k: issue[k] for k in ("id", "title", "status", "bead", "bead_status", "playbook")}
                   | {"notes": len(issue["notes"])})
    out.sort(key=lambda i: i["id"])
    return {"ok": True, "count": len(out), "issues": out}


def get_issue(issue_id: str, base: Optional[Path] = None) -> dict:
    issue = issues(base).get(issue_id)
    if not issue:
        return {"ok": False, "error": f"no issue {issue_id}"}
    return {"ok": True, "issue": issue}
