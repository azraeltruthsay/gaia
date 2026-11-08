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

An issue needs evidence (the log lines or reply text she saw), not only a
suspicion: the loop's first probe showed her inventing a finding that was not in
anything she had been shown. Filing an issue that matches an open one returns
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

STATUSES = ("open", "investigating", "proposed", "fixed", "wontfix")
MAX_NEW_PER_DAY = 10
DUPLICATE_SIMILARITY = 0.6
_MAX_FIELD = 2000
_MAX_EVIDENCE_LINES = 20

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset("a an the and or of to in on for is was it she her i my me that this with as at by from be".split())


def _dir(base: Optional[Path] = None) -> Path:
    return Path(base) if base else ISSUES_DIR


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
                "suspected_cause": ev.get("suspected_cause", ""),
                "playbook": ev.get("playbook", ""),
                "source": ev.get("source", ""),
                "status": "open",
                "filed": ev.get("ts"),
                "notes": [],
            }
        elif iid in state and kind == "note":
            state[iid]["notes"].append({"ts": ev.get("ts"), "text": ev.get("text", "")})
        elif iid in state and kind == "status":
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
             "evidence": evidence, "suspected_cause": _clip(suspected_cause),
             "playbook": _clip(playbook, 80), "source": _clip(source, 40)}, base)
    return {"ok": True, "id": iid, "message": f"Filed {iid}. It becomes a bead (label gaia-found) "
                                              "when the bridge next runs; `issues` shows its bead id."}


def note_issue(issue_id: str, text: str, base: Optional[Path] = None) -> dict:
    text = _clip(text)
    if not text:
        return {"ok": False, "error": "text is required"}
    if issue_id not in issues(base):
        return {"ok": False, "error": f"no issue {issue_id}"}
    _append({"event": "note", "id": issue_id, "ts": time.time(), "text": text}, base)
    return {"ok": True, "id": issue_id}


def set_status(issue_id: str, status: str, note: str = "", base: Optional[Path] = None) -> dict:
    """Move an issue along: investigating → proposed → fixed (or wontfix).
    'fixed' is her report; the bead stays open until the fix is promoted and
    someone who can promote closes it."""
    status = (status or "").strip().lower()
    if status not in STATUSES:
        return {"ok": False, "error": f"status must be one of {', '.join(STATUSES)}"}
    if issue_id not in issues(base):
        return {"ok": False, "error": f"no issue {issue_id}"}
    _append({"event": "status", "id": issue_id, "ts": time.time(), "status": status,
             "note": _clip(note)}, base)
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
