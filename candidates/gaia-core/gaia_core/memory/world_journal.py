"""World Journal — in-world (Heimric) journal context.

Wraps gaia_core.memory.journal with `context='heimr'` and adds a timeline
index at /shared/journals/heimr/timeline.json for fast in-world-time
navigation.

Why a separate timeline index?
  The journal entries themselves are sorted by real-world write time on
  disk. For "what happened between Rogue's End and now?" we need an
  ordered view by *in-world* date. The timeline index gives O(log N)
  bisect lookups and lets us record events that don't necessarily have
  a full journal entry yet (e.g. the DM mentions something happened on
  Erom 7 but we haven't written it up).

Storage layout:
  /shared/journals/heimr/timeline.json
    {
      "current_date": "CE 159, Thame 1",
      "events": [
        {
          "id": "evt-001",
          "in_world_date": "CE 157, Sedjem 12",
          "day_count": -1003,           # canonical sort key
          "summary": "Rupert lands in Rogue's End",
          "tags": ["arrival", "rogues-end"],
          "entry_ids": ["heimr-CE0157-Sedjem-12-01"],
          "real_date": "2026-04-30T01:00:00+00:00"
        },
        ...
      ]
    }
  /shared/journals/heimr/<era><year>/<MM-month>/heimr-<...>-NN.md  (entries)
"""
from __future__ import annotations

import json
import logging
import threading
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from gaia_core.memory.heimric_calendar import HeimricDate, parse as parse_heimric, try_parse
from gaia_core.memory import journal as _j

logger = logging.getLogger("GAIA.WorldJournal")

CONTEXT = "heimr"
HEIMR_ROOT = _j.JOURNAL_ROOT / CONTEXT
TIMELINE_FILE = HEIMR_ROOT / "timeline.json"

_lock = threading.RLock()


# ── Timeline index ──────────────────────────────────────────────────────


def _load_timeline() -> Dict[str, Any]:
    if not TIMELINE_FILE.exists():
        return {"current_date": None, "events": []}
    try:
        return json.loads(TIMELINE_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Timeline read failed; returning empty")
        return {"current_date": None, "events": []}


def _save_timeline(data: Dict[str, Any]) -> None:
    HEIMR_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = TIMELINE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(TIMELINE_FILE)


def _next_event_id(events: List[Dict[str, Any]]) -> str:
    n = 1
    used = {e.get("id") for e in events}
    while f"evt-{n:03d}" in used:
        n += 1
    return f"evt-{n:03d}"


def _sorted_index_insert(events: List[Dict[str, Any]], event: Dict[str, Any]) -> int:
    """Insert into the events list, kept sorted by day_count ascending."""
    keys = [e["day_count"] for e in events]
    pos = bisect_right(keys, event["day_count"])
    events.insert(pos, event)
    return pos


# ── "Now" anchor ────────────────────────────────────────────────────────


def current_date() -> Optional[HeimricDate]:
    """The campaign's current in-world 'now', or None if unset."""
    data = _load_timeline()
    return try_parse(data.get("current_date") or "")


def set_current_date(date: str | HeimricDate) -> HeimricDate:
    """Anchor the campaign's current in-world date.

    Accepts a HeimricDate or a parseable string ('CE 159, Thame 1').
    """
    hd = date if isinstance(date, HeimricDate) else parse_heimric(date)
    with _lock:
        data = _load_timeline()
        data["current_date"] = hd.format()
        _save_timeline(data)
    logger.info("World journal: current_date set to %s", hd.format(long=True))
    return hd


# ── Entry writing ───────────────────────────────────────────────────────


def add_entry(
    in_world_date: str,
    body: str,
    *,
    significance: int = 0,
    tags: Optional[List[str]] = None,
    inspired_by: Optional[List[str]] = None,
) -> _j.JournalEntry:
    """Write a Heimric-context journal entry timestamped to an in-world date."""
    hd = parse_heimric(in_world_date)
    iw_str = hd.format()
    return _j.write_entry(
        body=body,
        significance=significance,
        tags=list(tags or []),
        inspired_by=list(inspired_by or []),
        context=CONTEXT,
        in_world_date=iw_str,
    )


def annotate_entry(entry_id: str, note: str, *, reason: Optional[str] = None,
                   source: str = "manual",
                   inspired_by: Optional[str] = None) -> bool:
    """Append an annotation to a Heimric-context entry."""
    return _j.annotate_entry(
        entry_id, note, reason=reason, source=source,
        inspired_by=inspired_by, context=CONTEXT,
    )


# ── Timeline events ─────────────────────────────────────────────────────


def add_event(
    in_world_date: str,
    summary: str,
    *,
    tags: Optional[List[str]] = None,
    entry_ids: Optional[List[str]] = None,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a timeline event at an in-world date.

    Events are an ordered index keyed by in-world date. They can optionally
    point to one or more journal entry IDs, but standalone events are
    fine — useful for marking 'something happened on this date but we
    haven't written it up yet'.
    """
    hd = parse_heimric(in_world_date)
    with _lock:
        data = _load_timeline()
        events = data.get("events", [])
        eid = event_id or _next_event_id(events)
        event: Dict[str, Any] = {
            "id": eid,
            "in_world_date": hd.format(),
            "day_count": hd.day_count,
            "summary": summary.strip(),
            "tags": list(tags or []),
            "entry_ids": list(entry_ids or []),
            "real_date": datetime.now(timezone.utc).isoformat(),
        }
        _sorted_index_insert(events, event)
        data["events"] = events
        _save_timeline(data)
    logger.info("World journal: timeline event %s (%s) — %s",
                eid, hd.format(), summary[:80])
    return event


def attach_entry_to_event(event_id: str, entry_id: str) -> bool:
    """Link a journal entry id to an existing timeline event."""
    with _lock:
        data = _load_timeline()
        for e in data.get("events", []):
            if e.get("id") == event_id:
                ids = list(e.get("entry_ids") or [])
                if entry_id not in ids:
                    ids.append(entry_id)
                    e["entry_ids"] = ids
                    _save_timeline(data)
                return True
        return False


def list_events(
    *,
    since: str | HeimricDate | None = None,
    until: str | HeimricDate | None = None,
    tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return timeline events ordered by in-world date.

    `since`/`until` accept either HeimricDate or parseable strings; both
    are inclusive on the in-world axis.
    """
    data = _load_timeline()
    events = data.get("events", [])
    lo = -10**12
    hi = 10**12
    if since is not None:
        s = since if isinstance(since, HeimricDate) else parse_heimric(since)
        lo = s.day_count
    if until is not None:
        u = until if isinstance(until, HeimricDate) else parse_heimric(until)
        hi = u.day_count
    out = [e for e in events if lo <= e.get("day_count", 0) <= hi]
    if tag:
        tl = tag.lower()
        out = [e for e in out if tl in {t.lower() for t in (e.get("tags") or [])}]
    return out


def find_gap(
    start_date: str | HeimricDate,
    end_date: str | HeimricDate,
) -> Dict[str, Any]:
    """Inspect what's known about the in-world span between two dates.

    Returns a dict with:
      • span: {'from', 'to', 'days'}
      • events: timeline events in the span, in order
      • entry_ids: union of journal-entry IDs referenced by those events
      • missing_summary: dates within the span that have no event/entry

    Useful for "we know X happened at Rogue's End and Y is happening now —
    what's recorded between them?" — if missing_summary is non-empty,
    that's the temporal gap to fill in.
    """
    s = start_date if isinstance(start_date, HeimricDate) else parse_heimric(start_date)
    e = end_date if isinstance(end_date, HeimricDate) else parse_heimric(end_date)
    if e.day_count < s.day_count:
        s, e = e, s
    events = list_events(since=s, until=e)
    entry_ids: List[str] = []
    seen: set = set()
    for ev in events:
        for eid in ev.get("entry_ids") or []:
            if eid not in seen:
                seen.add(eid)
                entry_ids.append(eid)
    days = e.day_count - s.day_count
    # Days within the span with no event — capped to a sane preview list
    event_days = {ev["day_count"] for ev in events}
    missing_summary: List[str] = []
    if days > 0 and not events:
        missing_summary.append(
            f"No timeline events recorded between {s.format()} and {e.format()} "
            f"({days} in-world days)."
        )
    elif days > 0:
        # Coarse: report any contiguous gap > 7 days inside the span
        markers = [s.day_count] + sorted(event_days) + [e.day_count]
        for i in range(len(markers) - 1):
            gap = markers[i + 1] - markers[i]
            if gap > 7:
                a = HeimricDate(day_count=markers[i]).format()
                b = HeimricDate(day_count=markers[i + 1]).format()
                missing_summary.append(f"{gap} days unrecorded between {a} and {b}.")
    return {
        "span": {"from": s.format(), "to": e.format(), "days": days},
        "events": events,
        "entry_ids": entry_ids,
        "missing_summary": missing_summary,
    }


# ── Search delegation ───────────────────────────────────────────────────


def search_entries(query: str, k: int = 5):
    """Keyword search scoped to the heimr context."""
    return _j.search_entries(query, k=k, context=CONTEXT)


def list_entries(limit: Optional[int] = None) -> List[_j.JournalEntry]:
    """List heimr-context entries, newest write-time first."""
    return _j.list_entries(limit=limit, context=CONTEXT)
