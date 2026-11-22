"""Samvega as a ledger of mistakes (aidw.5, GROUNDED_SPEECH.md part C2).

Samvega artifacts were a pile: 2,224 of them by 2026-11-21, 2,221 marked
"reviewed" by a weight bump, none ever acted on. Here each one has an id
and a status, tracked the way beads track our work:

  open → amended → investigating → fixed → verified
       ↘ dismissed (with a reason)
  noted   — a finding nothing confirmed (judgment, unverified): kept, not acted on
  legacy  — written before the ledger (no status field); read as such, never rewritten

Who sets what:
  open / noted   the code that writes the finding (cross_tier_audit)
  amended        C3, after she corrects the message in place
  investigating, fixed, dismissed
                 her, through samvega(action=status)
  verified       only code (C4's replay of the original question); the
                 tool refuses it

The artifact files stay where they are (/knowledge/samvega/samvega_*.json)
and the fields are additive, so the sleep-cycle introspection and KV fold,
which read and rewrite whole files, keep working.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("GAIA.SamvegaLedger")

STATUSES = ("open", "amended", "investigating", "fixed", "verified", "dismissed", "noted", "legacy")
ACTIVE = ("open", "amended", "investigating")
# Transitions she may make with the tool. verified is code's alone.
_HER_MOVES = {
    "open": {"investigating", "fixed", "dismissed"},
    "amended": {"investigating", "fixed", "dismissed"},
    "investigating": {"fixed", "dismissed", "open"},
    "fixed": {"investigating"},
    "noted": {"open", "dismissed"},
    "legacy": {"open", "dismissed"},
    "dismissed": {"open"},
    "verified": set(),
}


def samvega_dir() -> Path:
    return Path(os.environ.get("SAMVEGA_DIR") or
                Path(os.environ.get("KNOWLEDGE_DIR", "/knowledge")) / "samvega")


def short_id(stem: str) -> str:
    """'samvega_20261121_211651196478' -> 'sv-1a2b3c4d' (stable)."""
    return "sv-" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_of(d: dict) -> str:
    return d.get("status") or "legacy"


def _files(include_archive: bool = False) -> Iterable[Path]:
    root = samvega_dir()
    if root.exists():
        yield from sorted(root.glob("samvega_*.json"), reverse=True)
        if include_archive and (root / "archive").exists():
            yield from sorted((root / "archive").glob("samvega_*.json"), reverse=True)


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(p: Path, d: dict) -> None:
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def find(ref: str) -> Optional[Tuple[Path, dict]]:
    """A finding by short id (sv-…) or file stem."""
    ref = (ref or "").strip()
    if not ref:
        return None
    for p in _files(include_archive=True):
        if p.stem == ref or short_id(p.stem) == ref or p.name == ref:
            d = _load(p)
            if d is not None:
                return p, d
    return None


def summary(p: Path, d: dict, full: bool = False) -> dict:
    """A finding as she reads it."""
    out = {
        "id": short_id(p.stem),
        "status": _status_of(d),
        "when": (d.get("timestamp") or "")[:16].replace("T", " "),
        "where": (d.get("message_ref") or {}).get("where") or _where(d.get("session_id", "")),
        "claim": d.get("claim") or "",
        # A confirmed finding is what she said against what the source shows;
        # the old auditor's prose (what_went_wrong) only when there's nothing better.
        "what_went_wrong": _grounded_line(d) or (d.get("what_went_wrong") or "")[:300],
        "grounding": d.get("grounding") or "",
        "weight": d.get("weight", 0),
    }
    if d.get("issue_id"):
        out["issue_id"] = d["issue_id"]
    if full:
        out.update({
            "you_said": (d.get("original_output_summary") or "")[:800],
            "evidence": (d.get("evidence") or "")[:1500],
            "checked_with": d.get("grounding_call") or "",
            "findings": d.get("findings") or [],
            "message_ref": d.get("message_ref") or {},
            "history": d.get("history") or [],
            "trigger": d.get("trigger") or "",
        })
    return out


def _grounded_line(d: dict) -> str:
    if d.get("grounding") != "confirmed" or not d.get("claim"):
        return ""
    src = d.get("grounding_call") or "the evidence"
    shows = re.sub(r"\s+", " ", d.get("evidence") or "").strip()[:300]
    return f'You said: "{d["claim"][:200]}". {src} shows: {shows}' if shows else \
        f'You said: "{d["claim"][:200]}", which {src} does not hold.'


def _where(session_id: str) -> str:
    s = session_id or ""
    if s.startswith("discord_dm_"):
        return "a Discord DM"
    if s.startswith("discord_channel_") or s.startswith("discord_"):
        return "a Discord channel"
    if s.startswith("web_"):
        return "the web chat"
    return s or "unknown"


def message_ref(session_id: str, said: str, sent_at: str = "") -> dict:
    """Where the mistaken message is, enough for C3 to find and edit it:
    platform, channel or user, when, and its opening words."""
    s = session_id or ""
    ref = {"session_id": s, "sent_at": sent_at or _now(), "said": (said or "")[:300], "where": _where(s)}
    m = re.match(r"discord_dm_(\d+)$", s)
    if m:
        ref.update(platform="discord", user_id=m.group(1))
    else:
        m = re.match(r"discord_(?:channel_)?(\d+)$", s)
        if m:
            ref.update(platform="discord", channel_id=m.group(1))
        elif s.startswith("web_"):
            ref.update(platform="web")
    return ref


def initial_fields(grounding: str, session_id: str = "", said: str = "") -> dict:
    """Ledger fields for a new finding: open when evidence confirmed it,
    noted otherwise."""
    status = "open" if grounding == "confirmed" else "noted"
    return {"status": status, "issue_id": "",
            "message_ref": message_ref(session_id, said) if session_id else {},
            "history": [{"at": _now(), "status": status, "by": "audit",
                         "note": "confirmed against evidence" if status == "open" else "not confirmed"}]}


def stamp(path, fields: dict) -> bool:
    """Merge fields into a finding written a moment ago (by path)."""
    try:
        p = Path(path)
        d = _load(p)
        if d is None:
            return False
        for k, v in fields.items():
            if k == "history" and d.get("history"):
                d["history"] = list(d["history"]) + list(v)
            else:
                d[k] = v
        _write(p, d)
        return True
    except Exception:
        logger.debug("Samvega stamp failed for %s", path, exc_info=True)
        return False


def list_findings(status: str = "active", limit: int = 10, include_archive: bool = False) -> dict:
    """Findings by status ("active" = open/amended/investigating; "all"),
    newest first, with counts of every status."""
    want = ACTIVE if status in ("", "active") else None if status == "all" else (status,)
    counts: Dict[str, int] = {}
    rows = []
    for p in _files(include_archive):
        d = _load(p)
        if d is None:
            continue
        st = _status_of(d)
        counts[st] = counts.get(st, 0) + 1
        if (want is None or st in want) and len(rows) < limit:
            rows.append(summary(p, d))
    return {"ok": True, "status": status or "active", "counts": counts, "findings": rows,
            "summary": f"{sum(counts.get(s, 0) for s in ACTIVE)} active "
                       f"({', '.join(f'{counts.get(s, 0)} {s}' for s in ACTIVE)}); "
                       f"{counts.get('legacy', 0)} from before the ledger."}


def show(ref: str) -> dict:
    hit = find(ref)
    if not hit:
        return {"ok": False, "error": f"No finding '{ref}'."}
    return {"ok": True, **summary(*hit, full=True)}


def note(ref: str, text: str, by: str = "GAIA") -> dict:
    hit = find(ref)
    if not hit:
        return {"ok": False, "error": f"No finding '{ref}'."}
    if not (text or "").strip():
        return {"ok": False, "error": "A note needs text."}
    p, d = hit
    d.setdefault("history", []).append({"at": _now(), "status": _status_of(d), "by": by, "note": text[:1000]})
    _write(p, d)
    return {"ok": True, "id": short_id(p.stem), "status": _status_of(d)}


def set_status(ref: str, status: str, reason: str = "", by: str = "GAIA",
               as_code: bool = False, **fields) -> dict:
    """Move a finding. She (as_code=False) may make the moves in _HER_MOVES;
    code may set any status. dismissed needs a reason; fixed needs a note of
    what changed."""
    status = (status or "").strip().lower()
    if status not in STATUSES or status == "legacy":
        return {"ok": False, "error": f"Status must be one of: {', '.join(s for s in STATUSES if s != 'legacy')}."}
    hit = find(ref)
    if not hit:
        return {"ok": False, "error": f"No finding '{ref}'."}
    p, d = hit
    cur = _status_of(d)
    if not as_code:
        if status == "verified":
            return {"ok": False, "error": "verified is set when replaying the original question no longer "
                                          "produces the mistake; mark it fixed and the check will follow."}
        if status not in _HER_MOVES.get(cur, set()):
            return {"ok": False, "error": f"Can't move a {cur} finding to {status}.",
                    "allowed": sorted(_HER_MOVES.get(cur, set()))}
        if status in ("dismissed", "fixed") and not (reason or "").strip():
            return {"ok": False, "error": f"Say why it is {status} (reason=...)."}
    d["status"] = status
    for k, v in fields.items():
        if k in ("issue_id", "message_ref", "amended_text") and v is not None:
            d[k] = v
    d.setdefault("history", []).append({"at": _now(), "status": status, "by": by, "note": (reason or "")[:1000]})
    _write(p, d)
    logger.info("Samvega %s: %s -> %s (%s)", short_id(p.stem), cur, status, by)
    return {"ok": True, "summary": f"{short_id(p.stem)} is now {status} (it was {cur}).",
            "id": short_id(p.stem), "status": status, "was": cur}
