"""KG recency cross-check (GAIA_Project-hkv — World Model Stage 8).

Pre-flight check: before the model generates a response, look in the
KnowledgeGraph for high-relevance recent triples that cover entities
the user just mentioned. If we have something fresher than training
data on a time-sensitive topic, inject it as authoritative reference
ahead of generation.

This is the symmetric counterpart to the consistency_detector:
  - consistency_detector: catches confabulation OUT (post-generation).
  - kg_recency_grounding: prevents staleness IN (pre-generation).

The cross-check is **additive** — it never blocks or refuses; it just
adds a DataField to the packet. If the KG has nothing relevant, the
packet goes through unchanged.

Threshold strategy:
  - `min_relevance_inject` (default 0.30) — facts above this become
    authoritative reference data ("here's what we know").
  - `min_relevance_warn`   (default 0.05) — facts above this but below
    inject threshold are presented as "low confidence — consider
    re-fetching". Below warn → ignore (cache is stale anyway).
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ── Shared KG accessor ──────────────────────────────────────────────
# Lazy singleton — matches the pattern in affect_runtime so both layers
# share a single sqlite handle to the default KG.
_shared_kg = None
_kg_lock = threading.Lock()


def _get_shared_kg():
    """Return a shared KnowledgeGraph. None if init fails (logged once)."""
    global _shared_kg
    if _shared_kg is not None:
        return _shared_kg
    with _kg_lock:
        if _shared_kg is not None:
            return _shared_kg
        try:
            from gaia_common.utils.knowledge_graph import KnowledgeGraph
            _shared_kg = KnowledgeGraph()
        except Exception:
            logger.debug("Shared KG init failed; recency grounding disabled",
                         exc_info=True)
            _shared_kg = None
    return _shared_kg


def reset_for_tests(kg=None) -> None:
    """Replace or clear the cached KG. Tests only."""
    global _shared_kg
    with _kg_lock:
        _shared_kg = kg


# ── Time-sensitivity markers ────────────────────────────────────────
# Words/phrases that signal the user wants CURRENT info (not historical).
# Case-insensitive whole-word match. Keep narrow — false positives push
# us toward over-grounding.
_TIME_MARKERS = {
    "current", "currently", "today", "tonight", "now", "right now",
    "this week", "this month", "this year", "latest", "recent",
    "recently", "newest", "up to date", "as of",
    "still", "anymore", "these days",
}

_TIME_MARKER_RE = re.compile(
    r"\b(" + "|".join(sorted(_TIME_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def is_time_sensitive(prompt: str) -> bool:
    """True if the prompt contains a time-sensitivity marker.

    Examples of True: "who is the current senator from Oregon?",
                      "what's the latest news on the launch?",
                      "is X still alive?"
    Examples of False: "explain quantum entanglement",
                       "what did Marcus Aurelius write?"
    """
    if not prompt:
        return False
    return _TIME_MARKER_RE.search(prompt) is not None


def _format_age(valid_from: Optional[str], now: Optional[datetime] = None) -> str:
    """Human-friendly age label for the reference block."""
    if not valid_from:
        return "unknown date"
    try:
        from datetime import timezone
        vf = datetime.fromisoformat(valid_from)
        if vf.tzinfo is None:
            vf = vf.replace(tzinfo=timezone.utc)
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - vf
        s = delta.total_seconds()
        if s < 3600:
            return f"{int(s // 60)}m ago"
        if s < 86400:
            return f"{int(s // 3600)}h ago"
        days = int(s // 86400)
        if days < 30:
            return f"{days}d ago"
        if days < 365:
            return f"{days // 30}mo ago"
        return f"{days // 365}yr ago"
    except Exception:
        return valid_from[:10]  # fallback: YYYY-MM-DD prefix


def _format_fact_block(facts: list[dict], header: str) -> str:
    """Format a list of {subject, predicate, object, relevance, valid_from,
    source} dicts into a model-facing reference block."""
    lines = [header]
    for f in facts:
        rel = f.get("relevance", 0.0)
        subj = f.get("subject", "?")
        pred = f.get("predicate", "?")
        obj = f.get("object", "?")
        age = _format_age(f.get("valid_from"))
        src = f.get("source") or "kg"
        # Compact source if it's the long web_search format
        if isinstance(src, str) and src.startswith("web_search|"):
            parts = src.split("|", 2)
            src = "web_search"
            if len(parts) >= 2 and parts[1]:
                src = f"web_search:{parts[1][:40]}"
        lines.append(
            f"  - {subj} {pred} {obj}  "
            f"(relevance={rel:.2f}, {age}, source={src})"
        )
    return "\n".join(lines)


def lookup_kg_facts(
    kg,
    phrases: list[str],
    *,
    world: str = "actuality",
    min_relevance: float = 0.05,
    max_per_entity: int = 3,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Look up recency-scored triples in the KG for a list of entity phrases.

    Returns a flat list of triple dicts (subject, predicate, object,
    relevance, valid_from, source, fact_type) sorted by relevance DESC.
    Capped to max_per_entity hits per input phrase to avoid one entity
    swamping the reference block.
    """
    if not phrases or kg is None:
        return []
    seen_ids: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for phrase in phrases:
        try:
            rows = kg.query_entity_with_relevance(
                phrase,
                direction="both",
                world=world,
                min_relevance=min_relevance,
                now=now,
            )
        except Exception as e:
            logger.debug("KG lookup failed for %r: %s", phrase, e)
            continue
        kept = 0
        for r in rows:
            key = (r.get("subject"), r.get("predicate"), r.get("object"))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            out.append(r)
            kept += 1
            if kept >= max_per_entity:
                break
    out.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
    return out


def build_recency_grounding(
    prompt: str,
    kg,
    *,
    extract_phrases=None,
    world: str = "actuality",
    min_relevance_inject: float = 0.30,
    min_relevance_warn: float = 0.05,
    max_facts: int = 5,
    require_time_marker: bool = True,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Build the grounding block for a user prompt, or None if nothing
    relevant in the KG.

    Args:
        prompt: user input text.
        kg: KnowledgeGraph instance.
        extract_phrases: callable(text) -> list[str]. Defaults to
            semantic_probe.extract_candidate_phrases. Pass a custom
            extractor for testing or per-flow control.
        world: KG world to query (default actuality).
        min_relevance_inject: facts above this are authoritative.
        min_relevance_warn:   facts above this (but below inject) are
            flagged "low confidence — consider re-fetching".
        max_facts: total facts to surface across all entities.
        require_time_marker: if True, return None unless prompt has a
            time-sensitivity marker. Set False to ground unconditionally.
        now: optional datetime for testability.

    Returns the formatted reference block string, or None if no useful
    KG facts were found.
    """
    if not prompt or kg is None:
        return None

    if require_time_marker and not is_time_sensitive(prompt):
        return None

    # Phrase extraction — reuse semantic_probe by default. Optional override
    # is for tests and for callers that already have extracted entities.
    if extract_phrases is None:
        try:
            from gaia_core.cognition.semantic_probe import extract_candidate_phrases
            extract_phrases = extract_candidate_phrases
        except Exception as e:
            logger.debug("semantic_probe import failed: %s", e)
            return None

    phrases = extract_phrases(prompt) or []
    if not phrases:
        return None

    facts = lookup_kg_facts(
        kg, phrases, world=world, min_relevance=min_relevance_warn, now=now,
    )
    if not facts:
        return None

    fresh = [f for f in facts if f.get("relevance", 0) >= min_relevance_inject]
    stale = [
        f for f in facts
        if min_relevance_warn <= f.get("relevance", 0) < min_relevance_inject
    ]

    # Apply the max_facts cap proportionally — prefer fresh.
    fresh = fresh[:max_facts]
    remaining = max(0, max_facts - len(fresh))
    stale = stale[:remaining]

    if not fresh and not stale:
        return None

    blocks: list[str] = []
    if fresh:
        blocks.append(_format_fact_block(
            fresh,
            "[KG recent reference — use these as authoritative ground truth:",
        ))
    if stale:
        blocks.append(_format_fact_block(
            stale,
            "[KG stale reference — low confidence, consider a fresh lookup:",
        ))
    blocks.append("]")
    return "\n".join(blocks)


def recency_grounding_for_prompt(
    prompt: str,
    *,
    world: str = "actuality",
    require_time_marker: bool = True,
    max_facts: int = 5,
) -> Optional[str]:
    """Convenience wrapper for agent_core integration.

    Pulls the shared KG via `_get_shared_kg()` and calls
    `build_recency_grounding`. Safe-by-default: returns None on any
    failure rather than raising.
    """
    if not prompt:
        return None
    kg = _get_shared_kg()
    if kg is None:
        return None
    try:
        return build_recency_grounding(
            prompt, kg,
            world=world,
            require_time_marker=require_time_marker,
            max_facts=max_facts,
        )
    except Exception:
        logger.debug("recency_grounding_for_prompt failed", exc_info=True)
        return None
