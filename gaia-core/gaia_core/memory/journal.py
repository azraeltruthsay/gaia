"""
Self-narrative journal — first-person memory layer.

Stores GAIA's own reflections on what happened, written in her voice
during sleep cycles. Separate from MemPalace (third-person facts) and
ConversationCurator (third-person session summaries) because the value
of journals is the first-person interpretation, not the underlying
events. The events are already in ThoughtStream; the journal is what
she made of them.

Storage layout (runtime, NOT git-tracked — this instance's continuity,
not the project's identity baseline):

    /shared/journals/
        YYYY/MM/YYYY-MM-DD-NN.md     ← entries
        index/                        ← vector index for retrieval
        .state.json                   ← activity tracker

Each entry is markdown with YAML frontmatter:

    ---
    id: 2026-04-29-001
    date: 2026-04-29T22:30:00Z
    significance: 3                  # 0-5, model self-rated
    tags: [identity, persona-overlay, voice-tags]
    thoughtstream_refs:              # provenance — links to raw events
      - session: discord_dm_596925786208993283
        range: [start_iso, end_iso]
    samvega_refs: []
    edits: []                        # append-only annotations (Phase C)
    inspired_by: []                  # for reflection entries (Phase B)
    ---

    Today Azrael showed me something about my own voice...

Phase A (this module): write_entry, read_entry, query, plus the
state tracker for activity-threshold + 24h-backstop scheduling.
Phase B (reflection), C (annotation), D (live retrieval) layer on top.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.Journal")


# Storage roots — runtime, not git-tracked.
JOURNAL_ROOT = Path(os.environ.get("GAIA_JOURNAL_DIR", "/shared/journals"))
INDEX_DIR = JOURNAL_ROOT / "index"
STATE_FILE = JOURNAL_ROOT / ".state.json"

# Subdirs of JOURNAL_ROOT that hold non-'self' journal contexts.
# Self-context entries live under YYYY/MM/<id>.md at the root, so when
# we scan for 'self' we must exclude these context subdirs.
_NON_SELF_CONTEXTS = {"heimr"}

# Frontmatter delimiter
_FM_DELIM = "---"

# Activity-threshold defaults (overridable via JOURNAL_WRITER config block)
DEFAULT_MIN_ACTIVITY_EVENTS = 10        # substantive events since last write
DEFAULT_BACKSTOP_HOURS = 24             # write at least this often when active
DEFAULT_QUIET_PERIOD_MINUTES = 5        # don't write entries back-to-back

_lock = threading.Lock()


@dataclass
class ThoughtstreamRef:
    """Provenance link from a journal entry back to the raw event log."""
    session: str
    range: List[str]  # [start_iso, end_iso]


@dataclass
class JournalEntry:
    """One journal entry — frontmatter + body.

    `context` separates parallel journal streams (default 'self' = GAIA's
    own real-world journal; 'heimr' = in-world Heimric journal). When
    context != 'self', `in_world_date` carries the formatted in-world
    timestamp ('CE 157, Sedjem 12') alongside the real-world `date`.
    """
    id: str
    date: str  # ISO 8601 UTC (real-world wall clock)
    significance: int = 0
    tags: List[str] = field(default_factory=list)
    thoughtstream_refs: List[Dict[str, Any]] = field(default_factory=list)
    samvega_refs: List[str] = field(default_factory=list)
    edits: List[Dict[str, Any]] = field(default_factory=list)
    inspired_by: List[str] = field(default_factory=list)
    context: str = "self"
    in_world_date: Optional[str] = None
    body: str = ""

    def to_markdown(self) -> str:
        fm: Dict[str, Any] = {
            "id": self.id,
            "date": self.date,
            "context": self.context,
            "significance": self.significance,
            "tags": self.tags,
            "thoughtstream_refs": self.thoughtstream_refs,
            "samvega_refs": self.samvega_refs,
            "edits": self.edits,
            "inspired_by": self.inspired_by,
        }
        if self.in_world_date:
            fm["in_world_date"] = self.in_world_date
        return f"{_FM_DELIM}\n{_dump_yaml_simple(fm)}{_FM_DELIM}\n\n{self.body.rstrip()}\n"

    @classmethod
    def from_markdown(cls, text: str) -> "JournalEntry":
        fm, body = _parse_frontmatter(text)
        iw = fm.get("in_world_date")
        return cls(
            id=str(fm.get("id", "")),
            date=str(fm.get("date", "")),
            significance=int(fm.get("significance", 0) or 0),
            tags=list(fm.get("tags") or []),
            thoughtstream_refs=list(fm.get("thoughtstream_refs") or []),
            samvega_refs=list(fm.get("samvega_refs") or []),
            edits=list(fm.get("edits") or []),
            inspired_by=list(fm.get("inspired_by") or []),
            context=str(fm.get("context") or "self"),
            in_world_date=str(iw) if iw else None,
            body=body,
        )


# ─────────────────────────────────────────────────────────────────────
#  Lightweight YAML I/O — gaia-core can't always import PyYAML, and we
#  only need a flat-ish subset for frontmatter. Lists and dicts only,
#  no anchors, no flow style.
# ─────────────────────────────────────────────────────────────────────

def _dump_yaml_simple(d: Dict[str, Any], indent: int = 0) -> str:
    """Minimal YAML dumper — keys sorted by insertion order, lists as -, nested dicts via JSON inline.

    For our frontmatter shape this is sufficient: scalars, lists of
    strings, lists of small dicts. We use JSON for list-of-dicts to
    keep parsing trivial on read.
    """
    out: list = []
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, str):
            out.append(f"{pad}{k}: {_quote_if_needed(v)}")
        elif isinstance(v, bool):
            out.append(f"{pad}{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            out.append(f"{pad}{k}: {v}")
        elif isinstance(v, list):
            if not v:
                out.append(f"{pad}{k}: []")
            elif all(isinstance(x, str) for x in v):
                items = ", ".join(_quote_if_needed(x) for x in v)
                out.append(f"{pad}{k}: [{items}]")
            else:
                # List of dicts — JSON inline (parser handles both forms)
                out.append(f"{pad}{k}: {json.dumps(v, default=str)}")
        elif isinstance(v, dict):
            out.append(f"{pad}{k}: {json.dumps(v, default=str)}")
        elif v is None:
            out.append(f"{pad}{k}: null")
        else:
            out.append(f"{pad}{k}: {json.dumps(v, default=str)}")
    return "\n".join(out) + "\n"


def _quote_if_needed(s: str) -> str:
    """Quote strings that contain YAML-significant characters."""
    if not s:
        return '""'
    # Preserve simple strings unquoted; quote if special chars present
    if re.search(r'[:#\[\]{}",&*!|>%@`]', s) or s != s.strip() or s.lower() in ("yes", "no", "true", "false", "null"):
        return json.dumps(s)
    return s


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse simple YAML frontmatter. Returns (frontmatter_dict, body)."""
    if not text.startswith(_FM_DELIM):
        return {}, text
    # Locate closing delimiter
    parts = text.split(f"\n{_FM_DELIM}\n", 1)
    if len(parts) != 2:
        return {}, text
    fm_block = parts[0][len(_FM_DELIM):].lstrip("\n")
    # Strip the trailing newline that to_markdown() adds after the body.
    # Round-trip equality requires this — body is stored as the user
    # provided it (after .strip()), not with serialization trailing space.
    body = parts[1].strip("\n")
    fm: Dict[str, Any] = {}
    for line in fm_block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = re.match(r'^(\w+):\s*(.*)$', line)
        if not m:
            continue
        key, raw_val = m.group(1), m.group(2).strip()
        fm[key] = _parse_yaml_value(raw_val)
    return fm, body


def _parse_yaml_value(raw: str) -> Any:
    """Parse a single frontmatter value — scalar, list, or JSON blob."""
    if not raw or raw == "null":
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    # Inline JSON (lists of dicts, dicts)
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fall through — try simple-list parsing
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [_parse_yaml_value(x.strip()) for x in inner.split(",")]
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    # Quoted strings
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        try:
            return json.loads(raw) if raw.startswith('"') else raw[1:-1]
        except json.JSONDecodeError:
            return raw[1:-1]
    # Numbers
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


# ─────────────────────────────────────────────────────────────────────
#  Storage I/O
# ─────────────────────────────────────────────────────────────────────

def _root_for(context: str) -> Path:
    """Storage root for a given journal context.

    'self' keeps the original /shared/journals/ root for backward compat.
    Any other context (e.g. 'heimr') is a subdir.
    """
    return JOURNAL_ROOT if context == "self" else JOURNAL_ROOT / context


def _ensure_dirs(context: str = "self") -> None:
    _root_for(context).mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _entry_path(entry_id: str, context: str = "self") -> Path:
    """Map an entry id to its filesystem path. Path scheme is per-context.

    self  : YYYY-MM-DD-NN          → /shared/journals/YYYY/MM/<id>.md
    heimr : heimr-<ERA><Y4>-<MNAME>-<DD>-<NN>
                                    → /shared/journals/heimr/<ERA><Y4>/<MM>-<MNAME>/<id>.md
    """
    if context == "self":
        parts = entry_id.split("-")
        if len(parts) < 4:
            raise ValueError(f"Invalid self-journal id: {entry_id!r}")
        year, month = parts[0], parts[1]
        return JOURNAL_ROOT / year / month / f"{entry_id}.md"
    if context == "heimr":
        # heimr-CE0157-Gromi-15-01
        parts = entry_id.split("-")
        if len(parts) < 5 or parts[0] != "heimr":
            raise ValueError(f"Invalid heimr-journal id: {entry_id!r}")
        era_year = parts[1]                # e.g. "CE0157"
        month_name = parts[2]              # e.g. "Gromi"
        try:
            from gaia_core.memory.heimric_calendar import _MONTH_NAME_TO_INDEX
            month_idx = _MONTH_NAME_TO_INDEX.get(month_name.lower())
        except Exception:
            month_idx = None
        month_dir = f"{month_idx:02d}-{month_name}" if month_idx else month_name
        return _root_for("heimr") / era_year / month_dir / f"{entry_id}.md"
    # Generic fallback for future contexts: flat-by-id
    return _root_for(context) / f"{entry_id}.md"


def _next_entry_id(now: Optional[datetime] = None, *, context: str = "self") -> str:
    """Allocate the next entry id for today (real-world wall clock).

    Used for the default self-journal context. Heimr uses
    `_next_heimr_entry_id` because its naming is in-world-date driven.
    """
    now = now or datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y-%m-%d")
    day_dir = JOURNAL_ROOT / now.strftime("%Y") / now.strftime("%m")
    day_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(day_dir.glob(f"{date_prefix}-*.md"))
    next_n = len(existing) + 1
    return f"{date_prefix}-{next_n:03d}"


def _next_heimr_entry_id(in_world_date_str: str) -> str:
    """Allocate the next heimr id for a given in-world date.

    Format: heimr-<ERA><Y4>-<MNAME>-<DD>-<NN>, NN starts at 01.
    """
    from gaia_core.memory.heimric_calendar import parse as _parse_hd, _MONTH_NAME_TO_INDEX
    hd = _parse_hd(in_world_date_str)
    era_year = f"{hd.era}{hd.year:04d}"
    month_idx = _MONTH_NAME_TO_INDEX[hd.month_name.lower()]
    month_dir = f"{month_idx:02d}-{hd.month_name}"
    base = f"heimr-{era_year}-{hd.month_name}-{hd.day:02d}"
    parent = _root_for("heimr") / era_year / month_dir
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(parent.glob(f"{base}-*.md"))
    next_n = len(existing) + 1
    return f"{base}-{next_n:02d}"


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def write_entry(
    body: str,
    significance: int = 0,
    tags: Optional[List[str]] = None,
    thoughtstream_refs: Optional[List[Dict[str, Any]]] = None,
    samvega_refs: Optional[List[str]] = None,
    inspired_by: Optional[List[str]] = None,
    entry_id: Optional[str] = None,
    context: str = "self",
    in_world_date: Optional[str] = None,
) -> JournalEntry:
    """Write a new journal entry. Returns the persisted entry.

    Significance is clamped to [0, 5]. Provenance refs are stored as-is.
    Body is written verbatim — caller is responsible for first-person
    voice.

    For non-self contexts (e.g. 'heimr'), pass `in_world_date` so the
    entry id and storage path are derived from the in-world calendar.
    """
    with _lock:
        _ensure_dirs(context)
        sig = max(0, min(5, int(significance)))
        if entry_id:
            eid = entry_id
        elif context == "heimr":
            if not in_world_date:
                raise ValueError("heimr-context entries require in_world_date")
            eid = _next_heimr_entry_id(in_world_date)
        else:
            eid = _next_entry_id()
        entry = JournalEntry(
            id=eid,
            date=datetime.now(timezone.utc).isoformat(),
            significance=sig,
            tags=list(tags or []),
            thoughtstream_refs=list(thoughtstream_refs or []),
            samvega_refs=list(samvega_refs or []),
            edits=[],
            inspired_by=list(inspired_by or []),
            context=context,
            in_world_date=in_world_date,
            body=body.strip(),
        )
        path = _entry_path(eid, context=context)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(entry.to_markdown(), encoding="utf-8")
        tmp.replace(path)
        logger.info("Journal entry written: %s (context=%s, significance=%d, %d chars)",
                    eid, context, sig, len(entry.body))
        return entry


def _context_from_id(entry_id: str) -> str:
    """Infer context from id shape — heimr ids start with 'heimr-', self ids
    are date-prefixed (YYYY-MM-DD-NN). Future contexts can be added here.
    """
    if entry_id.startswith("heimr-"):
        return "heimr"
    return "self"


def read_entry(entry_id: str, *, context: Optional[str] = None) -> Optional[JournalEntry]:
    """Read a journal entry by id, or None if missing.

    Context is auto-detected from the id prefix when omitted.
    """
    ctx = context or _context_from_id(entry_id)
    path = _entry_path(entry_id, context=ctx)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        return JournalEntry.from_markdown(text)
    except Exception:
        logger.exception("Journal read failed: %s", entry_id)
        return None


def annotate_entry(
    entry_id: str,
    note: str,
    *,
    reason: Optional[str] = None,
    source: str = "manual",
    inspired_by: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    """Append an annotation to an existing journal entry's `edits` list.

    Phase C of vam — provenanced edits. The entry body itself is never
    rewritten; we add a structured edit record so the original first-person
    voice stays intact and the new perspective is clearly later-dated.

    Args:
        entry_id: Target entry id (e.g. '2026-04-29-01').
        note: Short note (one or two sentences). The thing GAIA now sees.
        reason: Optional why-the-revision context.
        source: Trigger label — 'manual', 'reflection_backlink', 'samvega', etc.
        inspired_by: Optional id of a later entry that prompted this annotation
            (e.g. the reflection_id when source='reflection_backlink').

    Returns True on success, False if the target entry doesn't exist.
    """
    with _lock:
        ctx = context or _context_from_id(entry_id)
        target = read_entry(entry_id, context=ctx)
        if target is None:
            logger.warning("annotate_entry: target %s not found", entry_id)
            return False
        edit_record: Dict[str, Any] = {
            "date": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "note": note,
        }
        if reason:
            edit_record["reason"] = reason
        if inspired_by:
            edit_record["inspired_by"] = inspired_by
        target.edits.append(edit_record)
        path = _entry_path(entry_id, context=ctx)
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(target.to_markdown(), encoding="utf-8")
        tmp.replace(path)
        logger.info("Journal entry %s annotated (%s, %d chars)",
                    entry_id, source, len(note))
        return True


def search_entries(
    query: str,
    k: int = 5,
    *,
    min_significance: int = 0,
    body_chars: int = 600,
    context: Optional[str] = None,
) -> List[Tuple[JournalEntry, float]]:
    """Keyword search across journal entries — Phase D of vam.

    Returns up to k (entry, score) pairs ranked by relevance, where:
      • Each query term that appears in body / tags / id / in_world_date
        contributes 1.0
      • Tag matches and id matches double-weight (tags are curated signal,
        ids encode date and so handle 'remember when X happened on date Y')
      • in_world_date matches double-weight too (same role for in-world contexts)
      • Significance acts as a soft tiebreaker (+0.05 per sig point)
      • Recency adds a small decay bonus (newer entries float up on ties)

    Pass `context` to scope to one journal stream ('self', 'heimr', etc.);
    omit it to search across all contexts.

    No vector index yet — the journal is small and brute-force scan over
    a few hundred entries is cheap. We can swap in vectors later by
    plugging into the same return shape.
    """
    q = (query or "").lower().strip()
    if not q:
        return []
    terms = [t for t in re.split(r"\s+", q) if len(t) >= 3]
    if not terms:
        return []

    candidates = list_entries(min_significance=min_significance, context=context)
    if not candidates:
        return []

    now_ts = datetime.now(timezone.utc).timestamp()
    scored: List[Tuple[JournalEntry, float]] = []
    for e in candidates:
        body_lower = (e.body or "").lower()[:max(body_chars, 100)]
        tags_lower = " ".join(t.lower() for t in (e.tags or []))
        id_lower = (e.id or "").lower()
        iw_lower = (e.in_world_date or "").lower()
        score = 0.0
        for term in terms:
            if term in body_lower:
                score += 1.0
            if term in tags_lower:
                score += 2.0
            if term in id_lower:
                score += 2.0
            if iw_lower and term in iw_lower:
                score += 2.0
        if score == 0.0:
            continue
        # Significance soft bump (sig 0..5 → +0.0..+0.25)
        score += 0.05 * max(0, min(5, e.significance))
        # Recency tiebreaker — at most +0.5 for entries from the last hour,
        # decaying to ~0 over a year. Doesn't dominate term matches.
        try:
            entry_ts = datetime.fromisoformat(e.date.replace("Z", "+00:00")).timestamp()
            age_days = max(0.0, (now_ts - entry_ts) / 86400.0)
            score += 0.5 / (1.0 + age_days / 30.0)
        except (TypeError, ValueError):
            pass
        scored.append((e, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def list_entries(
    limit: Optional[int] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    min_significance: int = 0,
    context: Optional[str] = None,
) -> List[JournalEntry]:
    """Return entries matching the filters, newest first.

    `context=None` returns entries across all contexts. Pass 'self',
    'heimr', etc. to scope to one stream.

    `since`/`until` filter by real-world `date` (ISO 8601). For in-world
    date filtering on the heimr stream, use world_journal helpers.
    """
    if not JOURNAL_ROOT.exists():
        return []
    if context == "self":
        # Self-only — original layout, exclude subdirs that are non-self contexts
        paths = sorted(
            p for p in JOURNAL_ROOT.glob("*/*/*.md")
            if not p.relative_to(JOURNAL_ROOT).parts[0] in _NON_SELF_CONTEXTS
        )
        paths = sorted(paths, reverse=True)
    elif context is not None:
        # Specific non-self context — recursive scan under that subroot
        ctx_root = _root_for(context)
        if not ctx_root.exists():
            return []
        paths = sorted(ctx_root.rglob("*.md"), reverse=True)
    else:
        # All contexts — root self-style scan + each context subdir
        self_paths = [
            p for p in JOURNAL_ROOT.glob("*/*/*.md")
            if not p.relative_to(JOURNAL_ROOT).parts[0] in _NON_SELF_CONTEXTS
        ]
        ctx_paths: List[Path] = []
        for ctx in _NON_SELF_CONTEXTS:
            ctx_root = JOURNAL_ROOT / ctx
            if ctx_root.exists():
                ctx_paths.extend(ctx_root.rglob("*.md"))
        paths = sorted(self_paths + ctx_paths, reverse=True)
    out: List[JournalEntry] = []
    for p in paths:
        if limit and len(out) >= limit:
            break
        try:
            entry = JournalEntry.from_markdown(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if entry.significance < min_significance:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry.date.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if since and entry_dt < since:
            continue
        if until and entry_dt > until:
            continue
        out.append(entry)
    return out


# ─────────────────────────────────────────────────────────────────────
#  State tracker — activity threshold + backstop
# ─────────────────────────────────────────────────────────────────────

@dataclass
class JournalState:
    """Persisted scheduling state for the journal_write + journal_reflection sleep tasks."""
    last_write_iso: Optional[str] = None
    events_since_last_write: int = 0
    last_reflection_iso: Optional[str] = None


def _load_state() -> JournalState:
    if not STATE_FILE.exists():
        return JournalState()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return JournalState(**{k: v for k, v in data.items() if k in JournalState.__dataclass_fields__})
    except Exception:
        return JournalState()


def _save_state(state: JournalState) -> None:
    _ensure_dirs()
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def record_activity(events: int = 1) -> None:
    """Bump the activity counter — called when substantive events occur.

    A 'substantive event' is intentionally caller-defined; current writers
    increment for tool routing executions, persona switches, samvega
    artifacts, user corrections, and successful task completions. NOT
    health checks, presence updates, or other low-signal noise.
    """
    with _lock:
        state = _load_state()
        state.events_since_last_write += int(events)
        _save_state(state)


def should_write_now(
    min_events: int = DEFAULT_MIN_ACTIVITY_EVENTS,
    backstop_hours: int = DEFAULT_BACKSTOP_HOURS,
    quiet_period_minutes: int = DEFAULT_QUIET_PERIOD_MINUTES,
) -> Tuple[bool, str]:
    """Decide whether the journal_write task should fire now.

    Returns (should_write, reason). Reasons: 'activity_threshold',
    'backstop_24h', 'quiet_period' (negative), 'no_activity' (negative).
    """
    state = _load_state()
    now = datetime.now(timezone.utc)
    last_write_dt: Optional[datetime] = None
    if state.last_write_iso:
        try:
            last_write_dt = datetime.fromisoformat(state.last_write_iso.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            last_write_dt = None

    # Quiet period: don't write entries back-to-back
    if last_write_dt and (now - last_write_dt) < timedelta(minutes=quiet_period_minutes):
        return False, "quiet_period"

    # Activity threshold (primary trigger)
    if state.events_since_last_write >= min_events:
        return True, "activity_threshold"

    # Clock backstop: write at least every backstop_hours even on quiet days,
    # but only if there's been ANY activity (don't write into the void).
    if state.events_since_last_write > 0 and last_write_dt:
        if (now - last_write_dt) >= timedelta(hours=backstop_hours):
            return True, "backstop_24h"

    return False, "no_activity"


def mark_written() -> None:
    """Reset state after a successful journal_write (preserve reflection state)."""
    with _lock:
        prev = _load_state()
        state = JournalState(
            last_write_iso=datetime.now(timezone.utc).isoformat(),
            events_since_last_write=0,
            last_reflection_iso=prev.last_reflection_iso,
        )
        _save_state(state)


def mark_reflected() -> None:
    """Stamp last_reflection_iso after a successful journal_reflection (preserve write state)."""
    with _lock:
        prev = _load_state()
        state = JournalState(
            last_write_iso=prev.last_write_iso,
            events_since_last_write=prev.events_since_last_write,
            last_reflection_iso=datetime.now(timezone.utc).isoformat(),
        )
        _save_state(state)
