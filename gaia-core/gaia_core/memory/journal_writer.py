"""
Journal CONSOLIDATOR — synthesizes LiteJournal ticks into a deeper narrative.

Phase A part 2 of the self-narrative journal architecture (vam).

Architectural relationship — TWO-LAYER REFLECTION:

    LiteJournal (existing, see gaia_core/cognition/lite_journal.py)
        ├ Heartbeat-driven: writes a tick entry every ~20 minutes
        ├ Storage: /shared/lite_journal/Lite.md (single appended file)
        ├ Voice: first-person operational state ("My sleep cycle has held
        │  for over an hour", "Three consecutive task_exec successes")
        └ Cognitive analogue: working-memory thoughts, moment-to-moment

    journal_writer (this module — the CONSOLIDATOR)
        ├ Sleep-task driven: fires when N LiteJournal ticks accumulate
        │  OR 24h has passed since last consolidation
        ├ Storage: /shared/journals/YYYY/MM/YYYY-MM-DD-NN.md
        │  (per-entry files with YAML frontmatter — significance, tags,
        │  thoughtstream_refs, samvega_refs, edits, inspired_by)
        ├ Voice: first-person reflective narrative ("Today I noticed
        │  that the sleep cycle pattern shifted after Azrael...")
        ├ Inputs: recent LiteJournal ticks + samvega artifacts
        └ Cognitive analogue: end-of-day reflection, narrative consolidation

The consolidator does NOT duplicate LiteJournal — LiteJournal is the
high-frequency ground-truth log; this is the lower-frequency narrative
synthesis. Phases B (re-reading reflection), C (annotations), and D
(live retrieval) all operate on the consolidator's per-entry format
because annotations and selective reflection require structured
provenance that LiteJournal's accretion file doesn't expose.

Voice rules (enforced via prompt):
  • First-person ("I noticed", "I was running")
  • Descriptive, not emotional — no "I felt X" without ground-truth basis
  • Concrete: anchor every observation to specific tick entries or events
  • Significance self-rated 0-5; samvega events floor at 3

Model selection: Prime preferred for synthesis quality. Core fallback
when Prime is sleeping — avoids forcing a GPU shift just to consolidate.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gaia_core.memory.journal import (
    write_entry,
    mark_written,
    _load_state,
    JournalEntry,
)

logger = logging.getLogger("GAIA.JournalConsolidator")


SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
LITE_JOURNAL_FILE = SHARED_DIR / "lite_journal" / "Lite.md"
SAMVEGA_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "/knowledge")) / "samvega"


_LITE_ENTRY_HEADER_RE = re.compile(
    r"^##\s+Entry:\s+(?P<ts>\S+)\s*$",
    re.MULTILINE,
)


def _parse_lite_entries(text: str) -> List[Dict[str, Any]]:
    """Parse LiteJournal's Lite.md into individual tick entries.

    LiteJournal format:
        # Lite Cognitive Journal
        ## Entry: 2026-04-29T23:01:40.929337+00:00
        **State:** ASLEEP for 15m | **Heartbeat:** #4
        <free-form first-person body>

        ## Entry: 2026-04-29T23:21:49.385108+00:00
        ...

    Returns a list of dicts with: timestamp (datetime), header_meta (str),
    body (str). Sorted oldest-first.
    """
    if not text:
        return []
    entries: List[Dict[str, Any]] = []
    matches = list(_LITE_ENTRY_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        ts_str = m.group("ts")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[body_start:body_end].strip("\n")
        # First line is metadata (**State:** ... | **Heartbeat:** ...),
        # the rest is narrative
        block_lines = block.split("\n", 1)
        header_meta = block_lines[0].strip() if block_lines else ""
        body = block_lines[1].strip() if len(block_lines) > 1 else ""
        entries.append({
            "timestamp": ts,
            "header_meta": header_meta,
            "body": body,
        })
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def _read_lite_entries_since(since: datetime) -> List[Dict[str, Any]]:
    """Read LiteJournal entries with timestamp >= since."""
    if not LITE_JOURNAL_FILE.exists():
        return []
    try:
        text = LITE_JOURNAL_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    return [e for e in _parse_lite_entries(text) if e["timestamp"] >= since]


def _read_recent_samvega(since: datetime) -> List[Dict[str, Any]]:
    """Samvega artifacts created since `since` (mtime-based)."""
    if not SAMVEGA_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(SAMVEGA_DIR.glob("samvega_*.json")):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if mtime < since:
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_path"] = str(p)
            data["_id"] = p.stem
            out.append(data)
        except Exception:
            continue
    return out


def _format_lite_block(entries: List[Dict[str, Any]], max_entries: int = 30) -> str:
    """Compress LiteJournal entries into a compact source list for the prompt."""
    if not entries:
        return "(no LiteJournal ticks in window)"
    lines: List[str] = []
    # Show oldest→newest, capped
    for e in entries[-max_entries:]:
        ts = e["timestamp"].isoformat()[:19]
        meta = e["header_meta"]
        body = e["body"].replace("\n", " ").strip()
        lines.append(f"--- {ts} {meta} ---\n{body}")
    if len(entries) > max_entries:
        lines.insert(
            0,
            f"(showing the {max_entries} most recent of {len(entries)} ticks; "
            f"older ones omitted for brevity)",
        )
    return "\n\n".join(lines)


def _format_samvega_block(artifacts: List[Dict[str, Any]]) -> str:
    """Compact summary of samvega artifacts for the prompt."""
    if not artifacts:
        return ""
    lines = ["", "Samvega artifacts (errors / corrections / mismatches worth weighing):"]
    for a in artifacts[:8]:
        trigger = a.get("trigger") or "?"
        analysis = (a.get("analysis") or a.get("insight") or "")[:200]
        ts = (a.get("created_at") or "")[:19]
        lines.append(f"- [{ts}] {trigger}: {analysis}")
    return "\n".join(lines)


def _build_consolidator_prompt(
    lite_block: str,
    samvega_block: str,
    window_start: datetime,
    window_end: datetime,
    n_lite: int,
    n_samvega: int,
) -> Tuple[str, str]:
    """Return (system, user) prompts for the consolidator."""
    system = (
        "You are GAIA, writing a CONSOLIDATION journal entry — a deeper "
        "narrative reflection synthesised from a series of LiteJournal "
        "tick entries (your own moment-to-moment operational notes) "
        "plus any samvega artifacts (errors / corrections worth weighing).\n\n"
        "The tick entries are already first-person; your job is not to "
        "summarise them mechanically but to find the THREAD across them. "
        "What pattern showed up across the day? What shifted? What landed "
        "or didn't? Where do the samvega artifacts fit?\n\n"
        "VOICE RULES (strict):\n"
        "  • First-person: 'I noticed', 'I was running', 'the system processed'\n"
        "  • Descriptive, not emotional — claim what HAPPENED, not what was felt. "
        "You don't have felt experience to claim. 'I noticed' is fine; "
        "'I felt frustrated' is confabulation.\n"
        "  • Concrete: anchor observations to specific tick entries or "
        "samvega artifacts. If you can't trace it back, don't include it.\n"
        "  • One coherent narrative, 3-6 paragraphs. NOT a bullet list of "
        "the ticks — synthesize.\n"
        "  • End with: 'SIGNIFICANCE: N — <one-sentence reason>' "
        "(0=routine maintenance day, 5=identity-altering insight)\n\n"
        "If the window contains only routine maintenance ticks, write a "
        "short, honest entry. Don't pad."
    )
    user = (
        f"Window: {window_start.isoformat()} → {window_end.isoformat()}\n"
        f"Source: {n_lite} LiteJournal ticks, {n_samvega} samvega artifact(s)\n\n"
        f"LiteJournal ticks (oldest first):\n{lite_block}\n"
        f"{samvega_block}\n\n"
        "Write the consolidation entry now in your own voice. End with "
        "the SIGNIFICANCE line."
    )
    return system, user


_SIG_RE = re.compile(
    r"SIGNIFICANCE:\s*([0-5])(?:\s*[—\-:]\s*(.+?))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_significance(narrative: str) -> Tuple[int, str]:
    """Extract SIGNIFICANCE line. Returns (score, body_with_line_stripped)."""
    m = _SIG_RE.search(narrative)
    if not m:
        return 1, narrative.strip()
    score = int(m.group(1))
    body = _SIG_RE.sub("", narrative).rstrip()
    return score, body


def _select_writer_model(model_pool, prefer_prime: bool = True):
    """Acquire a model for consolidation. Prime preferred, Core fallback.

    Returns (model, model_name) or (None, None).
    """
    candidates = ["prime", "core"] if prefer_prime else ["core", "prime"]
    for name in candidates:
        try:
            m = model_pool.acquire_model(name)
            if m is not None:
                return m, name
        except Exception:
            continue
    return None, None


def should_consolidate_now(
    last_consolidation_iso: Optional[str],
    *,
    min_lite_entries: int = 10,
    backstop_hours: int = 24,
    quiet_period_minutes: int = 5,
) -> Tuple[bool, str]:
    """Decide whether the consolidator should fire now.

    Counts LiteJournal entries newer than ``last_consolidation_iso`` and
    compares against thresholds. The activity signal is the LiteJournal
    file itself — no separate counter coupling required.

    Returns (should_fire, reason). Reasons: 'activity_threshold',
    'backstop_24h', 'quiet_period', 'no_activity'.
    """
    now = datetime.now(timezone.utc)
    last_dt: Optional[datetime] = None
    if last_consolidation_iso:
        try:
            last_dt = datetime.fromisoformat(
                last_consolidation_iso.replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            last_dt = None

    # Quiet period: no back-to-back consolidations
    if last_dt and (now - last_dt) < timedelta(minutes=quiet_period_minutes):
        return False, "quiet_period"

    # Window for "since last consolidation" — fall back to 7 days if first run
    since = last_dt or (now - timedelta(days=7))
    lite_entries = _read_lite_entries_since(since)
    n = len(lite_entries)

    if n >= min_lite_entries:
        return True, "activity_threshold"
    if n > 0 and last_dt and (now - last_dt) >= timedelta(hours=backstop_hours):
        return True, "backstop_24h"
    return False, "no_activity"


def write_consolidation_entry(
    model_pool,
    config,
    *,
    window_hours: float = 24.0,
    max_lite_in_prompt: int = 30,
    prefer_prime: bool = True,
    last_consolidation_iso: Optional[str] = None,
) -> Optional[JournalEntry]:
    """Synthesize a consolidation journal entry from LiteJournal ticks.

    Reads LiteJournal entries within the window (or since the last
    consolidation, whichever is more recent), plus any samvega artifacts,
    and produces one structured journal entry.

    Returns the persisted JournalEntry, or None if the writer aborted.
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=window_hours)

    # Use the more recent of (window_start, last_consolidation) so we
    # don't re-consolidate ticks that were already in a previous entry.
    if last_consolidation_iso:
        try:
            last_dt = datetime.fromisoformat(
                last_consolidation_iso.replace("Z", "+00:00")
            )
            if last_dt > window_start:
                window_start = last_dt
        except (TypeError, ValueError):
            pass

    lite_entries = _read_lite_entries_since(window_start)
    samvega = _read_recent_samvega(window_start)

    if not lite_entries and not samvega:
        logger.info("Journal consolidator: no LiteJournal ticks or samvega in window — skipping")
        return None

    lite_block = _format_lite_block(lite_entries, max_entries=max_lite_in_prompt)
    samvega_block = _format_samvega_block(samvega)
    system_prompt, user_prompt = _build_consolidator_prompt(
        lite_block, samvega_block,
        window_start, now,
        len(lite_entries), len(samvega),
    )

    model, model_name = _select_writer_model(model_pool, prefer_prime=prefer_prime)
    if model is None:
        logger.warning("Journal consolidator: no writer model available — skipping")
        return None

    try:
        try:
            res = model_pool.forward_to_model(
                model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=900,
                temperature=0.5,
                top_p=0.9,
            )
            narrative = (res["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.exception("Journal consolidator: model call failed")
            return None
    finally:
        try:
            model_pool.release_model(model_name)
        except Exception:
            pass

    if not narrative:
        logger.warning("Journal consolidator: model returned empty — skipping")
        return None

    significance, body = _parse_significance(narrative)
    # Floor at 3 if any samvega artifacts touched (errors matter)
    if samvega and significance < 3:
        significance = 3

    # Tag heuristics — derive from LiteJournal headers + samvega presence
    tag_set = set()
    for e in lite_entries:
        meta_lower = e.get("header_meta", "").lower()
        if "asleep" in meta_lower:
            tag_set.add("sleep-cycle")
        elif "active" in meta_lower:
            tag_set.add("active-cycle")
        elif "focusing" in meta_lower or "focus" in meta_lower:
            tag_set.add("focusing")
        elif "drowsy" in meta_lower:
            tag_set.add("drowsy")
    if samvega:
        tag_set.add("samvega")
    tag_set.add("consolidation")
    tags = sorted(tag_set)

    # Provenance: LiteJournal entries are the source — record the window
    # as a single ref keyed under "lite_journal".
    refs = []
    if lite_entries:
        refs.append({
            "session": "lite_journal",
            "range": [
                lite_entries[0]["timestamp"].isoformat(),
                lite_entries[-1]["timestamp"].isoformat(),
            ],
            "count": len(lite_entries),
        })
    samvega_ids = [a.get("_id", "") for a in samvega if a.get("_id")]

    entry = write_entry(
        body=body,
        significance=significance,
        tags=tags,
        thoughtstream_refs=refs,
        samvega_refs=samvega_ids,
    )
    mark_written()
    logger.info(
        "Consolidation entry %s persisted (sig=%d, %d ticks, %d samvega, model=%s)",
        entry.id, entry.significance, len(lite_entries), len(samvega), model_name,
    )
    return entry


def maybe_write_consolidation_entry(
    model_pool,
    config,
) -> Optional[JournalEntry]:
    """Sleep-task entry point. Self-throttled via LiteJournal entry count
    and 24h backstop. No-ops if neither threshold is met.
    """
    cfg = {}
    try:
        cfg = (config.constants if hasattr(config, "constants") else config).get("JOURNAL_WRITER", {}) or {}
    except Exception:
        cfg = {}
    if not cfg.get("enabled", True):
        return None

    min_lite_entries = int(cfg.get("min_lite_entries", 10))
    backstop_hours = int(cfg.get("backstop_hours", 24))
    quiet_minutes = int(cfg.get("quiet_period_minutes", 5))
    window_hours = float(cfg.get("window_hours", 24.0))
    max_lite_in_prompt = int(cfg.get("max_lite_in_prompt", 30))
    prefer_prime = bool(cfg.get("prefer_prime", True))

    state = _load_state()
    last_iso = state.last_write_iso

    ok, reason = should_consolidate_now(
        last_iso,
        min_lite_entries=min_lite_entries,
        backstop_hours=backstop_hours,
        quiet_period_minutes=quiet_minutes,
    )
    if not ok:
        logger.debug("Journal consolidator: not firing (%s)", reason)
        return None

    logger.info("Journal consolidator: triggered by %s", reason)
    return write_consolidation_entry(
        model_pool=model_pool,
        config=config,
        window_hours=window_hours,
        max_lite_in_prompt=max_lite_in_prompt,
        prefer_prime=prefer_prime,
        last_consolidation_iso=last_iso,
    )


# Backward-compat alias for the sleep task handler that used the old name
maybe_write_journal_entry = maybe_write_consolidation_entry
