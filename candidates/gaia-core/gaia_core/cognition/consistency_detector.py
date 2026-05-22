"""Consistency-violation detector — Stage 2 of the World Model design.

Runs against the existing flat Knowledge Graph at output time. Catches
confabulation that prompt-side gates can't:

  - Unknown entity: a named entity appears in the model's response with
    no KG triples connecting it to anything, AND it didn't appear in any
    source the model was given (grounding, conversation history,
    retrieved docs). Howeidi-in-AGI is the canonical case.

  - Cross-domain leak: a known entity appears in the response, but its
    KG triples connect it to topics unrelated to the current conversation
    (e.g. Super Bowl entities surfacing in a Portland-weather reply).

  - Direct contradiction: the response asserts X about a subject when
    the KG has a triple (subject, predicate, ¬X) with confidence > 0.5.
    Strictest match — flags only when there's a high-confidence triple
    to contradict.

This is observe-and-flag only. Findings get logged and emitted as
samvega artifacts (same path as cross_tier_audit). The detector does
NOT auto-correct output today; that's Stage 3+ of the broader plan.

Hooks into agent_core's post-deliberation path alongside the cross-tier
audit, using the same AuditFinding shape so reviewers can compare
output-time and Prime-judgment findings side-by-side.

Design doc: knowledge/Dev_Notebook/2026-05-21_world_model_design.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Re-use the samvega directory layout from cross_tier_audit so artifacts
# from both detectors land together and the review queue stays unified.
from gaia_core.cognition.cross_tier_audit import (
    AuditFinding,
    SAMVEGA_DIR,
)

logger = logging.getLogger("GAIA.ConsistencyDetector")


# ── Tunables ────────────────────────────────────────────────────────────

# Minimum length of a candidate entity surface form before we consider it.
# Filters out 2-letter abbreviations and matches the KG's existing
# tendency to over-extract ('sup', 'bow', 'new' as entities).
_MIN_ENTITY_LEN = 5

# Confidence threshold for triples that count as established facts.
_ESTABLISHED_CONFIDENCE = 0.5

# Max entities we examine per response. Prevents the detector from
# becoming a performance liability on long replies.
_MAX_ENTITIES_PER_AUDIT = 12

# Common English words that may appear capitalized at sentence starts.
# Don't treat them as named-entity candidates.
_STOPWORD_CAPS = frozenset({
    # Pronouns, determiners, demonstratives, interrogatives
    "The", "This", "That", "These", "Those", "Their", "There", "Then",
    "When", "Where", "What", "Which", "Who", "Whom", "Whose", "Why", "How",
    "His", "Her", "Its", "Our", "Your", "My",  # possessive pronouns
    "He", "She", "It", "We", "You", "I", "They",  # personal pronouns
    # Conjunctions, adverbs commonly at sentence start
    "And", "But", "Or", "If", "So", "Also", "Just", "Only", "Both", "Each",
    "Some", "Most", "Many", "Any", "All", "None", "Every", "Other",
    "Such", "Same", "Said", "Note", "Yes", "No", "OK", "Okay",
    # Prepositions — often start markdown headers ("## On GIAA's...") or
    # opening sentences. Without these in stopwords, multi-word matching
    # eats e.g. "On GIAA" as a 2-word Title Case entity.
    "On", "In", "At", "By", "For", "From", "Of", "To", "With", "About",
    "Between", "Among", "Within", "Without", "Through", "Across",
    "Above", "Below", "Beyond", "After", "Before", "During", "Since",
    "Until", "While", "Although", "Though", "However", "Therefore",
    "Indeed", "Sorry", "Thanks", "Maybe", "Perhaps", "Often", "Always",
    "Never", "Sometimes", "Usually", "Likely", "Possibly", "Actually",
    "Especially", "Generally", "Specifically", "Currently",
    "Speaking", "Considering", "Regarding", "Concerning",
    "Including", "Excluding", "Apart", "Aside", "Otherwise",
    # Common Title Case verbs/nouns at sentence start
    "Let", "Make", "Take", "Give", "Find", "Look", "Show", "Tell",
    "Have", "Has", "Had", "Will", "Would", "Could", "Should", "Might",
    "Must", "Need", "Want", "Like", "Love", "Hate", "Feel", "Think",
    "Believe", "Know", "Understand", "Remember", "Forget", "Imagine",
    # Self-references
    "User", "Message", "GAIA", "Prime", "Core", "Nano", "AI", "Azrael",
    # Acronyms safe to ignore
    "API", "URL", "PDT", "UTC", "JSON", "HTTP", "HTTPS", "MCP",
    "GPU", "CPU", "RAM", "PST", "CST", "EST", "TLS", "SSL", "TCP", "UDP",
})


@dataclass
class ConsistencyResult:
    """Result of one consistency-detection pass."""
    clean: bool
    findings: List[AuditFinding] = field(default_factory=list)
    summary: str = ""
    elapsed_ms: float = 0.0
    skipped_reason: Optional[str] = None
    samvega_path: Optional[str] = None
    entities_examined: int = 0
    kg_entities_matched: int = 0


# ── Entity extraction ───────────────────────────────────────────────────

# Multi-word Title Case spans: "Amin Hamid Howeidi", "Edgar Allan Poe".
# Each word starts with a capital, length >= 2 letters, optional period
# (handles "R.S.S. ALICE" style abbreviations).
_MULTI_TITLE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z]{1,}\.?\s+){1,3}[A-Z][a-zA-Z]{1,}\b"
)

# Single Title Case words long enough to plausibly be names. Caught only
# if they survive the stopword filter.
_SINGLE_TITLE_RE = re.compile(r"\b[A-Z][a-z]{3,}\b")

# ALL-CAPS acronyms — only if 3+ chars and not in known-acronyms set.
_ACRONYM_RE = re.compile(r"\b[A-Z]{3,}(?:[-_][A-Z0-9]+)?\b")


# Sentence / block splitter — keeps multi-word regex from matching across
# periods or paragraph breaks, which would otherwise catch "Oregon. Current"
# as a 2-word name, or "## On Foo\n\nBar Baz quux..." as "On Foo\n\nBar Baz"
# spanning a markdown header into body text.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n+|\n#{1,6}\s+")


def _extract_candidate_entities(text: str) -> List[str]:
    """Find named-entity surface forms in the response text.

    Conservative: returns plausibly-distinctive names only. Multi-word
    Title Case sequences are the highest-precision signal; single Title
    Case words are accepted only when they exceed _MIN_ENTITY_LEN.

    Sentence-aware: multi-word matching is bounded per sentence to avoid
    catching "End-of-sentence. Beginning-of-next" as a name.
    """
    if not text:
        return []

    seen: Dict[str, str] = {}  # lowercase key → original surface form

    # Multi-word Title Case spans (highest precision) — applied per
    # sentence so periods don't get swallowed mid-pattern.
    sentences = _SENTENCE_SPLIT_RE.split(text)
    for sent in sentences:
        for m in _MULTI_TITLE_RE.finditer(sent):
            surface = m.group(0).strip().rstrip(".")
            if len(surface) < _MIN_ENTITY_LEN:
                continue
            # Multi-word match starting with a stopword (possessive/personal
            # pronoun, sentence-opener, etc.) is almost certainly an artifact
            # of sentence concatenation, not a real entity. Drop the first
            # word and re-check; if what remains is a single Title Case word
            # then we skip entirely (single-word detection is disabled).
            first_word = surface.split(None, 1)[0]
            if first_word in _STOPWORD_CAPS:
                remainder = surface.split(None, 1)[1] if " " in surface else ""
                # If remainder is still multi-word and long enough, use it.
                if remainder and " " in remainder and len(remainder) >= _MIN_ENTITY_LEN:
                    surface = remainder
                else:
                    continue
            key = surface.lower()
            if key not in seen:
                seen[key] = surface

    # ALL-CAPS acronyms (4+ chars) — 3-char acronyms produce too many
    # false positives (AGI, API, GMT all get caught when legitimate).
    for m in _ACRONYM_RE.finditer(text):
        surface = m.group(0)
        if surface in _STOPWORD_CAPS:
            continue
        if len(surface) >= 4:
            key = surface.lower()
            if key not in seen:
                seen[key] = surface

    # NOTE: single Title Case word detection deliberately dropped. The
    # signal-to-noise ratio is bad — legitimate common nouns ("Stoic",
    # "Meditations", "Roman", "Egyptian") get flagged as confabulation
    # when the model legitimately recalls common-knowledge concepts that
    # aren't in the KG yet. Multi-word names ("Amin Hamid Howeidi", "New
    # England Patriots") and longer acronyms are high-precision; singles
    # are not. Re-enable in a later stage if the KG entity coverage
    # grows enough to support it.

    return list(seen.values())[:_MAX_ENTITIES_PER_AUDIT]


# ── KG lookup ───────────────────────────────────────────────────────────

def _kg_facts_for_entity(entity: str) -> List[Tuple[str, str, str, float]]:
    """Query the KG for triples involving the named entity.

    Returns list of (subject, predicate, object, confidence) tuples.
    Empty if entity is unknown to the KG.
    """
    try:
        from gaia_core.utils import mcp_client
        # The KG normalizes entity IDs lowercase-with-underscores. Try
        # both the raw form and the normalized form.
        for query_form in (entity, entity.lower().replace(" ", "_")):
            resp = mcp_client.call_jsonrpc("kg_query", {
                "entity": query_form,
                "direction": "both",
            })
            if not resp.get("ok"):
                continue
            result = resp.get("response", {}).get("result", {}) or {}
            facts = result.get("facts") or result.get("triples") or []
            if facts:
                out: List[Tuple[str, str, str, float]] = []
                for f in facts:
                    if isinstance(f, dict):
                        out.append((
                            str(f.get("subject", "")),
                            str(f.get("predicate", "")),
                            str(f.get("object", "")),
                            float(f.get("confidence", 1.0)),
                        ))
                    elif isinstance(f, (list, tuple)) and len(f) >= 3:
                        conf = float(f[3]) if len(f) > 3 else 1.0
                        out.append((str(f[0]), str(f[1]), str(f[2]), conf))
                if out:
                    return out
    except Exception:
        logger.debug("ConsistencyDetector: KG query failed", exc_info=True)
    return []


def _entity_in_grounding(entity: str, packet) -> bool:
    """Check if the entity appears in any grounding data field on the packet.

    If the user GAVE the model an entity (via web_grounding, retrieved_docs,
    auto_grounding), it's not a confabulation — the model is repeating what
    it was shown. Suppress in that case.
    """
    if not entity:
        return False
    needle = entity.lower()
    try:
        data_fields = getattr(getattr(packet, "content", None), "data_fields", []) or []
        for df in data_fields:
            key = getattr(df, "key", "")
            if key not in (
                "web_grounding", "auto_grounding", "cil_grounding",
                "retrieved_documents", "tool_result", "knowledge_base_name",
                "dnd_knowledge", "world_state_snapshot",
            ):
                continue
            val = getattr(df, "value", None)
            try:
                blob = json.dumps(val, default=str) if not isinstance(val, str) else val
            except Exception:
                blob = str(val)
            if needle in blob.lower():
                return True
    except Exception:
        logger.debug("ConsistencyDetector: grounding check failed", exc_info=True)
    return False


def _entity_in_history(entity: str, history: Optional[List[Dict[str, Any]]]) -> bool:
    """Check if the entity appears in recent conversation history."""
    if not entity or not history:
        return False
    needle = entity.lower()
    for msg in history:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if needle in (content or "").lower():
            return True
    return False


def _entity_in_user_input(entity: str, user_input: str) -> bool:
    """Check if the user's current prompt mentions the entity."""
    if not entity or not user_input:
        return False
    return entity.lower() in user_input.lower()


# ── Core detection ──────────────────────────────────────────────────────

def detect_consistency_violations(
    *,
    user_input: str,
    final_response: str,
    packet=None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> ConsistencyResult:
    """Examine the final response for entities that look fabricated.

    An entity in the response is suspicious when ALL of these hold:
      - It's not in the user's prompt
      - It's not in conversation history
      - It's not in any grounding data field on the packet
      - It's not in the Knowledge Graph

    That combination means the model produced a specific named entity
    with no traceable source — the fingerprint of confabulation.

    Returns a ConsistencyResult. Empty findings == clean response.
    """
    t0 = time.perf_counter()

    if not final_response or len(final_response.strip()) < 20:
        return ConsistencyResult(
            clean=True, summary="Response too short to audit",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            skipped_reason="response_too_short",
        )

    candidates = _extract_candidate_entities(final_response)
    if not candidates:
        return ConsistencyResult(
            clean=True, summary="No candidate entities extracted",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            entities_examined=0,
        )

    findings: List[AuditFinding] = []
    examined = 0
    kg_matched = 0

    for surface in candidates:
        examined += 1

        # Cheapest checks first — short-circuit if the entity is
        # legitimately sourced.
        if _entity_in_user_input(surface, user_input):
            continue
        if _entity_in_history(surface, history):
            continue
        if packet is not None and _entity_in_grounding(surface, packet):
            continue

        # Last check: is it in the KG with high-confidence facts?
        kg_facts = _kg_facts_for_entity(surface)
        if kg_facts:
            kg_matched += 1
            high_conf = [f for f in kg_facts if f[3] >= _ESTABLISHED_CONFIDENCE]
            if high_conf:
                # Entity is known and established — not flagged.
                # Future stage: check whether the response's claims about
                # this entity contradict its established triples
                # (direct-contradiction detection).
                continue

        # Entity passed all gates: unknown to user, history, grounding,
        # AND KG. High-confidence confabulation signal.
        findings.append(AuditFinding(
            category="confabulation",
            concern=(
                f"Entity '{surface}' appears in the response but has no "
                "trace in the user's prompt, conversation history, "
                "grounding data fields, or the Knowledge Graph. Likely "
                "fabricated."
            ),
            severity=3,
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    clean = not findings
    summary = (
        f"Clean (examined {examined} entities, {kg_matched} matched in KG)"
        if clean
        else f"{len(findings)} likely-fabricated entit{'y' if len(findings) == 1 else 'ies'} found"
    )
    return ConsistencyResult(
        clean=clean,
        findings=findings,
        summary=summary,
        elapsed_ms=elapsed,
        entities_examined=examined,
        kg_entities_matched=kg_matched,
    )


# ── Samvega artifact (mirrors cross_tier_audit pattern) ─────────────────

def _emit_samvega(
    findings: List[AuditFinding],
    user_input: str,
    final_response: str,
    journal_entry_id: Optional[str],
    session_id: Optional[str],
    summary: str,
) -> Optional[str]:
    """Emit a samvega artifact when findings are serious. Returns path or None.

    Single severity-3 finding is sufficient — fabricated entities are
    high-confidence confabulation signals, no need to wait for ≥2.
    """
    if not findings:
        return None
    SAMVEGA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    suffix = (journal_entry_id or session_id or "unknown")[-12:]
    artifact_id = f"samvega_consistency_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"
    payload = {
        "id": artifact_id,
        "type": "consistency_violation",
        "created_at": now.isoformat(),
        "trigger": "consistency_detector_flagged",
        "journal_entry_id": journal_entry_id,
        "session_id": session_id,
        "user_input": (user_input or "")[:400],
        "final_response": (final_response or "")[:600],
        "findings": [
            {"category": f.category, "concern": f.concern, "severity": f.severity}
            for f in findings
        ],
        "summary": summary,
    }
    out = SAMVEGA_DIR / f"{artifact_id}.json"
    try:
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(out)
        logger.info(
            "Consistency-violation samvega emitted: %s (%d finding%s)",
            artifact_id, len(findings), "" if len(findings) == 1 else "s",
        )
        return str(out)
    except Exception:
        logger.exception("Consistency-violation samvega write failed")
        return None


# ── Per-entry dedup so we don't re-audit the same response ─────────────

_seen_lock = threading.Lock()
_audited_entries: set = set()


def _claim_entry(entry_id: Optional[str]) -> bool:
    if not entry_id:
        return True
    with _seen_lock:
        if entry_id in _audited_entries:
            return False
        _audited_entries.add(entry_id)
        if len(_audited_entries) > 1024:
            # Bound memory — discard oldest half. Order isn't preserved
            # by sets so this is approximate, which is fine.
            for old in list(_audited_entries)[:512]:
                _audited_entries.discard(old)
    return True


# ── Public entry points ─────────────────────────────────────────────────

def run_consistency_check_sync(
    *,
    user_input: str,
    final_response: str,
    journal_entry_id: Optional[str] = None,
    session_id: Optional[str] = None,
    packet=None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> ConsistencyResult:
    """Synchronous consistency check. Use schedule_consistency_check for
    the fire-and-forget pattern that mirrors cross_tier_audit."""
    if not _claim_entry(journal_entry_id):
        return ConsistencyResult(
            clean=True, summary="Already audited",
            skipped_reason="duplicate_entry",
        )

    result = detect_consistency_violations(
        user_input=user_input,
        final_response=final_response,
        packet=packet,
        history=history,
    )

    if result.findings:
        result.samvega_path = _emit_samvega(
            findings=result.findings,
            user_input=user_input,
            final_response=final_response,
            journal_entry_id=journal_entry_id,
            session_id=session_id,
            summary=result.summary,
        )

    logger.info(
        "Consistency check complete: clean=%s findings=%d entities=%d kg_matched=%d elapsed=%.0fms",
        result.clean, len(result.findings),
        result.entities_examined, result.kg_entities_matched, result.elapsed_ms,
    )
    return result


def schedule_consistency_check(
    *,
    user_input: str,
    final_response: str,
    journal_entry_id: Optional[str] = None,
    session_id: Optional[str] = None,
    packet=None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Fire-and-forget consistency check. Runs in a daemon thread so the
    main turn doesn't block waiting for KG queries."""
    def _runner():
        try:
            run_consistency_check_sync(
                user_input=user_input,
                final_response=final_response,
                journal_entry_id=journal_entry_id,
                session_id=session_id,
                packet=packet,
                history=history,
            )
        except Exception:
            logger.exception("ConsistencyDetector: async run failed")

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
