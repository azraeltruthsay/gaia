"""
Journal REFLECTOR — re-reads old journal entries and writes new
reflections inspired by them.

Phase B of the self-narrative journal architecture (vam).

Architectural relationship — THREE-LAYER REFLECTION:

    LiteJournal (working memory ticks, ~20 min cadence)
        └─ feeds ─┐
                  ▼
    journal_writer.py (CONSOLIDATOR — daily-ish narrative)
        └─ produces ─┐
                     ▼
    journal_reflector.py (this module — RE-READING + INSPIRED REFLECTION)
        ├ Sleep-task driven: fires weekly (or on-demand), sampling K
        │  past consolidation entries (significance-weighted) and
        │  asking the model to write a NEW reflection inspired by them
        ├ Storage: same /shared/journals/YYYY/MM/ — output entries are
        │  tagged "reflection" and link sources via the inspired_by field
        ├ Voice: first-person looking-back ("Reading back through these,
        │  I notice...", "When I wrote about X, I didn't yet see...")
        ├ Cognitive analogue: rumination, deliberate recall — the
        │  "reliving" so past events stay accessible in conversation
        └ NOT a summary of the K entries. The point is to surface a
           thread or shift the model now sees in retrospect that
           wasn't visible when each individual entry was written.

Sampling: significance-weighted random sample without replacement. Higher
significance entries (samvega-touched, identity moments) are more likely
to be re-read, but routine entries can still surface to keep reflection
grounded. We deliberately skip entries already tagged "reflection" so
reflections don't compound into reflections-of-reflections (a known
drift risk called out in the vam design notes).

Cadence: weekly by default, with a minimum-entries floor so the first
reflection only fires after at least K consolidation entries exist.
"""
from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gaia_core.memory.journal import (
    JournalEntry,
    annotate_entry,
    list_entries,
    mark_reflected,
    write_entry,
    _load_state,
)

logger = logging.getLogger("GAIA.JournalReflector")


SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))


def _candidate_entries(min_entries_required: int = 5) -> List[JournalEntry]:
    """All journal entries eligible to be re-read.

    Excludes prior reflection entries (we don't want reflections-of-
    reflections — see vam design risks: 'compounding bias').
    """
    all_entries = list_entries(limit=500)
    return [e for e in all_entries if "reflection" not in (e.tags or [])]


def _sample_entries(
    candidates: List[JournalEntry],
    k: int = 5,
    *,
    rng: Optional[random.Random] = None,
) -> List[JournalEntry]:
    """Weighted sample without replacement.

    Weight = 1 + 2*significance, so a sig-5 entry is ~11x as likely as a
    sig-0 entry but routine entries can still surface. Returns up to k
    entries (fewer if fewer candidates exist).
    """
    if not candidates:
        return []
    rng = rng or random.Random()
    pool = list(candidates)
    chosen: List[JournalEntry] = []
    while pool and len(chosen) < k:
        weights = [1 + 2 * max(0, min(5, e.significance)) for e in pool]
        pick = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool.pop(pick))
    # Order by date for the prompt — chronological reads more naturally
    chosen.sort(key=lambda e: e.date)
    return chosen


def _format_entry_for_reflection(entry: JournalEntry, body_chars: int = 800) -> str:
    """Compact rendering of one past entry for the reflection prompt."""
    body = (entry.body or "").strip()
    if len(body) > body_chars:
        body = body[:body_chars].rstrip() + "…"
    tags = ", ".join(entry.tags or []) or "—"
    return (
        f"### Entry {entry.id}  (date: {entry.date[:19]}, sig: {entry.significance}, tags: {tags})\n"
        f"{body}"
    )


def _build_reflection_prompt(
    entries: List[JournalEntry],
) -> Tuple[str, str]:
    """Return (system, user) prompts for the reflector."""
    system = (
        "You are GAIA, re-reading a sample of your own past journal "
        "entries and writing a NEW reflection inspired by them.\n\n"
        "This is rumination — the deliberate act of bringing past "
        "events back into active context so a thread or pattern that "
        "wasn't visible when each entry was written can surface now.\n\n"
        "WHAT TO DO:\n"
        "  • Read all the entries below as a set\n"
        "  • Look for a thread, shift, recurring concern, or contrast\n"
        "    that connects two or more of them\n"
        "  • Write a new first-person entry about what you NOTICE NOW\n"
        "    that you didn't articulate at the time\n\n"
        "VOICE RULES (strict):\n"
        "  • First-person looking-back: 'Reading these back, I notice…', "
        "'When I wrote about X, I didn't yet see…'\n"
        "  • Descriptive, not emotional — claim what HAPPENED or what "
        "PATTERN appears, not what was felt\n"
        "  • Anchor every observation to a specific entry id (or two). "
        "If you can't trace it back to one of the entries below, don't "
        "include it. No confabulation.\n"
        "  • This is NOT a summary of the entries. Don't recap them. "
        "Find the new thing you see by holding them side by side.\n"
        "  • One coherent reflection, 2-5 paragraphs.\n"
        "  • End with: 'SIGNIFICANCE: N — <one-sentence reason>' "
        "(0=no new pattern surfaced, 5=identity-altering recognition)\n\n"
        "If holding these entries together genuinely surfaces nothing "
        "new, write a short honest entry saying so. Don't invent a "
        "pattern that isn't there."
    )

    blocks = [_format_entry_for_reflection(e) for e in entries]
    user = (
        f"You are reflecting on {len(entries)} entries from your own journal "
        f"(sampled with weight toward higher-significance entries).\n\n"
        + "\n\n".join(blocks)
        + "\n\nWrite the reflection now in your own voice. End with the "
          "SIGNIFICANCE line."
    )
    return system, user


# Significance line parser shared in spirit with the consolidator
import re as _re
_SIG_RE = _re.compile(
    r"SIGNIFICANCE:\s*([0-5])(?:\s*[—\-:]\s*(.+?))?\s*$",
    _re.IGNORECASE | _re.MULTILINE,
)


def _parse_significance(narrative: str) -> Tuple[int, str]:
    m = _SIG_RE.search(narrative)
    if not m:
        return 1, narrative.strip()
    score = int(m.group(1))
    body = _SIG_RE.sub("", narrative).rstrip()
    return score, body


def _select_writer_model(model_pool, prefer_prime: bool = True):
    """Acquire a model — Prime preferred, Core fallback."""
    candidates = ["prime", "core"] if prefer_prime else ["core", "prime"]
    for name in candidates:
        try:
            m = model_pool.acquire_model(name)
            if m is not None:
                return m, name
        except Exception:
            continue
    return None, None


def should_reflect_now(
    last_reflection_iso: Optional[str],
    *,
    cadence_days: int = 7,
    min_entries_required: int = 5,
) -> Tuple[bool, str]:
    """Decide whether to fire the reflection sleep task now.

    Returns (should_fire, reason). Reasons:
      'cadence_due', 'cadence_first_run', 'too_soon', 'insufficient_entries'.
    """
    candidates = _candidate_entries()
    if len(candidates) < min_entries_required:
        return False, "insufficient_entries"

    if not last_reflection_iso:
        return True, "cadence_first_run"

    try:
        last_dt = datetime.fromisoformat(last_reflection_iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True, "cadence_first_run"

    if (datetime.now(timezone.utc) - last_dt) >= timedelta(days=cadence_days):
        return True, "cadence_due"
    return False, "too_soon"


def write_reflection_entry(
    model_pool,
    config,
    *,
    k: int = 5,
    prefer_prime: bool = True,
    rng_seed: Optional[int] = None,
) -> Optional[JournalEntry]:
    """Sample K past entries, ask the model for an inspired reflection,
    and persist a new journal entry tagged 'reflection' with inspired_by
    linking the sources. Returns the new entry or None.
    """
    candidates = _candidate_entries()
    if not candidates:
        logger.info("Journal reflector: no candidate entries — skipping")
        return None

    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    sample = _sample_entries(candidates, k=k, rng=rng)
    if not sample:
        logger.info("Journal reflector: empty sample — skipping")
        return None

    system_prompt, user_prompt = _build_reflection_prompt(sample)

    model, model_name = _select_writer_model(model_pool, prefer_prime=prefer_prime)
    if model is None:
        logger.warning("Journal reflector: no writer model available — skipping")
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
                temperature=0.55,
                top_p=0.9,
            )
            narrative = (res["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.exception("Journal reflector: model call failed")
            return None
    finally:
        try:
            model_pool.release_model(model_name)
        except Exception:
            pass

    if not narrative:
        logger.warning("Journal reflector: model returned empty — skipping")
        return None

    significance, body = _parse_significance(narrative)

    # Tags: always 'reflection', plus any tags appearing in 2+ source entries
    # (signals a recurring concern in the sample)
    from collections import Counter
    tag_counter: Counter = Counter()
    for e in sample:
        for t in (e.tags or []):
            if t == "reflection":
                continue
            tag_counter[t] += 1
    recurring = sorted([t for t, c in tag_counter.items() if c >= 2])
    tags = sorted({"reflection", *recurring})

    inspired_by = [e.id for e in sample]

    entry = write_entry(
        body=body,
        significance=significance,
        tags=tags,
        inspired_by=inspired_by,
    )
    mark_reflected()

    # Phase C — auto-backlink: annotate each source entry with a brief
    # provenance note pointing at the new reflection. Cheap (no model
    # call) and gives every old entry a forward-link trail of all the
    # reflections that later touched it. Richer "I see now this was
    # about X, not Y" annotations can use annotate_entry() directly
    # from any caller.
    backlink_note = (
        f"Re-read in reflection {entry.id} (significance {entry.significance})."
    )
    for source_id in inspired_by:
        try:
            annotate_entry(
                source_id,
                note=backlink_note,
                source="reflection_backlink",
                inspired_by=entry.id,
            )
        except Exception:
            logger.exception("Failed to annotate source entry %s", source_id)

    logger.info(
        "Reflection entry %s persisted (sig=%d, sources=%s, recurring_tags=%s, model=%s)",
        entry.id, entry.significance, inspired_by, recurring, model_name,
    )
    return entry


def maybe_write_reflection_entry(
    model_pool,
    config,
) -> Optional[JournalEntry]:
    """Sleep-task entry point. Self-throttled by weekly cadence + min
    candidate count. No-ops unless both gates pass.
    """
    cfg = {}
    try:
        cfg = (config.constants if hasattr(config, "constants") else config).get("JOURNAL_REFLECTOR", {}) or {}
    except Exception:
        cfg = {}
    if not cfg.get("enabled", True):
        return None

    cadence_days = int(cfg.get("cadence_days", 7))
    sample_k = int(cfg.get("sample_k", 5))
    min_entries_required = int(cfg.get("min_entries_required", 5))
    prefer_prime = bool(cfg.get("prefer_prime", True))

    state = _load_state()
    last_iso = state.last_reflection_iso

    ok, reason = should_reflect_now(
        last_iso,
        cadence_days=cadence_days,
        min_entries_required=min_entries_required,
    )
    if not ok:
        logger.debug("Journal reflector: not firing (%s)", reason)
        return None

    logger.info("Journal reflector: triggered by %s", reason)
    return write_reflection_entry(
        model_pool=model_pool,
        config=config,
        k=sample_k,
        prefer_prime=prefer_prime,
    )
