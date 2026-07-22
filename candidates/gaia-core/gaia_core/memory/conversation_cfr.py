"""CFR-for-conversation, Phase 1: relevance-scored working set (page-replacement).

Treats the conversation history as virtual memory. The token window is RAM;
this module is the page-replacement policy: it scores recent turns by relevance
to the current message (embedding cosine) × a light per-turn recency decay, then
FOCUSes the most relevant ones (keeping a recency anchor for continuity) and
BLURs (drops) the clearly-unrelated rest — so e.g. a "Good morning" after a clock
chat no longer pulls the clock turns into the prompt.

Embedding-only — NO LLM call on the hot path (reuses the session indexer's embed
model + cosine helper). Falls back to pure recency if embeddings are unavailable.

Flag: CFR_CONVERSATION_ENABLED=1 (default off → legacy recency window in agent_core).
Tunables (env): CFR_RELEVANCE_FLOOR (default 0.20, conservative),
CFR_RECENCY_HALFLIFE_TURNS (default 12). See
knowledge/blueprints/cfr_conversation_virtual_memory.md.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.CFR.Conversation")


def _parse_turn_timestamp(raw) -> Optional[datetime]:
    """Parse a turn's ISO timestamp (session_manager.add_message format).
    None on missing/unparseable — callers must treat that as "no penalty",
    not as maximally stale, so pre-timestamp history isn't wrongly blurred."""
    if not raw:
        return None
    try:
        ts = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def cfr_conversation_enabled() -> bool:
    return os.environ.get("CFR_CONVERSATION_ENABLED", "0").lower() in ("1", "true", "yes", "on")


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# 231 Phase 3: meta/reference ("deictic") follow-ups — "what's its name?", "the
# second one", "that link", "you just said" — refer back to recent content but
# embed POORLY against that content's body, so plain cosine relevance blurs the
# exact turn the user is asking about. When detected, we widen the recency
# anchor + relax the floor so the recent content turns survive regardless of
# score. Conservative: requires a reference marker AND a short query (long
# queries carry their own topical signal and don't need the rescue).
_REFERENCE_MARKERS = (
    " its ", " it ", " it.", " it?", " that ", " that.", " that?", " those ",
    " them ", " they ", " this ", " the one", " the first", " the second",
    " the third", " the last", " the other", " the link", " the url",
    " the name", " the title", " the author", " the source", " the article",
    " the poem", " the result", " you just", " you said", " you mentioned",
    " earlier", " above", " before that", " which one", " what was",
)


def looks_like_reference_query(msg: str, max_words: int = 12) -> bool:
    """True if ``msg`` looks like a short deictic/meta follow-up referring back."""
    if not msg:
        return False
    words = msg.split()
    if len(words) > max_words:
        return False
    padded = " " + msg.lower().strip() + " "
    return any(mark in padded for mark in _REFERENCE_MARKERS)


def select_focus_turns(
    history: List[Dict],
    current_msg: str,
    max_focus: int = 6,
    anchor_n: int = 1,
    floor: float = None,
) -> Tuple[List[Dict], Dict]:
    """Choose which recent turns to FOCUS (keep) vs BLUR (drop), by relevance.

    Args:
        history: chronological list of {role, content, ...} message dicts.
        current_msg: the new user input being responded to.
        max_focus: max turns to keep in the working set (token budget cap).
        anchor_n: trailing turns ALWAYS kept regardless of score (continuity).
        floor: minimum cosine relevance to be eligible for FOCUS.

    Returns:
        (focus_turns_in_chronological_order, debug_dict). Embedding-only;
        gracefully falls back to pure recency when no embed model is available.
    """
    if floor is None:
        # 0.30 separates spurious weak matches (e.g. "Good morning" ↔ clock talk
        # scores ~0.27) from genuine relevance (on-topic turns score 0.4-0.85).
        # 0.20 was too low — it let the greeting↔clock bleed through.
        floor = _cfg_float("CFR_RELEVANCE_FLOOR", 0.30)
    halflife = _cfg_float("CFR_RECENCY_HALFLIFE_TURNS", 12.0)
    # tr7f: turn-distance decay alone can't catch a dangling question sitting
    # right before a much-later "good morning" — that pair has age_turns=1
    # regardless of the real gap. Beyond CFR_STALE_AFTER_HOURS of real elapsed
    # time, relevance is ALSO required to clear the floor after a short
    # wall-clock halflife — active, continuous sessions (small real gaps
    # between turns) are completely unaffected; only genuine idle-gap resumes
    # (hours/days) get the extra penalty.
    stale_after_hours = _cfg_float("CFR_STALE_AFTER_HOURS", 2.0)
    stale_halflife_hours = _cfg_float("CFR_STALE_HALFLIFE_HOURS", 3.0)
    now = datetime.now(timezone.utc)

    if not history:
        return [], {"focus": 0, "blurred": 0, "floor": floor}

    # 231 Phase 3: on a deictic/meta follow-up, widen the recency anchor (the
    # referenced content is almost always within the last few turns) and relax
    # the floor, so a low-cosine "what's its name?" can't blur the very turn it
    # refers to. Tunable: CFR_REFERENCE_ANCHOR (default 4).
    _is_reference = looks_like_reference_query(current_msg)
    if _is_reference:
        anchor_n = max(anchor_n, int(_cfg_float("CFR_REFERENCE_ANCHOR", 4.0)))
        floor = min(floor, 0.15)

    anchor_n = max(0, min(anchor_n, len(history)))
    _tail = history[-anchor_n:] if anchor_n else []
    # tr7f: the anchor exists for continuity across a LIVE exchange — it
    # force-keeps the trailing turn(s) regardless of relevance. But across a
    # real idle gap (hours/days), the positionally-last turn is often a
    # stale dangling thread (e.g. an unanswered question the user never
    # followed up on), not continuity — anchoring it unconditionally is
    # exactly how a 3-day-old technical question rides into an unrelated
    # "good morning". Demote stale tail turns out of the anchor and back
    # into the normally-scored candidate pool instead of exempting them.
    anchor = []
    for t in _tail:
        ts = _parse_turn_timestamp(t.get("timestamp"))
        age_h = (now - ts).total_seconds() / 3600.0 if ts else 0.0
        if ts and age_h > stale_after_hours:
            continue
        anchor.append(t)
    _anchor_ids = {id(t) for t in anchor}
    candidates = [t for t in history if id(t) not in _anchor_ids]
    keep_budget = max(0, max_focus - len(anchor))

    if not candidates or keep_budget == 0:
        return list(anchor), {"focus": len(anchor), "blurred": len(candidates),
                              "floor": round(floor, 3), "anchor": len(anchor),
                              "reference": _is_reference}

    # Reuse the session indexer's embed model + cosine helper (no new infra).
    try:
        from gaia_core.memory.session_history_indexer import _get_embed_model, _cosine_similarity
        model = _get_embed_model()
    except Exception:
        model = None

    if model is None:
        # tr7f: no relevance signal available at all here — recency is the
        # only guard, so at minimum drop candidates old enough to fail the
        # same wall-clock staleness gate the scored path applies.
        def _is_stale(t):
            ts = _parse_turn_timestamp(t.get("timestamp"))
            return ts is not None and (now - ts).total_seconds() / 3600.0 > stale_after_hours
        _fresh_candidates = [t for t in candidates if not _is_stale(t)]
        focus = _fresh_candidates[-keep_budget:] + list(anchor)
        return focus, {"focus": len(focus), "blurred": len(candidates) - keep_budget,
                       "floor": round(floor, 3), "anchor": len(anchor), "fallback": "no_embed"}

    try:
        texts = [current_msg] + [(t.get("content") or "")[:2000] for t in candidates]
        embs = model.encode(texts, show_progress_bar=False)
        q = embs[0]
        n = len(candidates)
        scored = []
        for i, t in enumerate(candidates):
            if not (t.get("content") or "").strip():
                continue
            rel = _cosine_similarity(q, embs[i + 1])
            age_turns = n - i  # 1 = most recent candidate, larger = older
            decay = 0.5 ** (age_turns / halflife) if halflife > 0 else 1.0
            ts = _parse_turn_timestamp(t.get("timestamp"))
            age_hours = (now - ts).total_seconds() / 3600.0 if ts else 0.0
            stale_decay = 1.0
            if age_hours > stale_after_hours and stale_halflife_hours > 0:
                stale_decay = 0.5 ** ((age_hours - stale_after_hours) / stale_halflife_hours)
                decay *= stale_decay
            scored.append({"score": rel * decay, "rel": rel, "turn": t, "stale_decay": stale_decay})

        eligible = [s for s in scored if s["rel"] >= floor and s["rel"] * s["stale_decay"] >= floor]
        eligible.sort(key=lambda s: -s["score"])
        chosen = [s["turn"] for s in eligible[:keep_budget]]

        # Reassemble in chronological order: chosen FOCUS turns + the anchor.
        pos = {id(t): i for i, t in enumerate(history)}
        focus = chosen + list(anchor)
        focus.sort(key=lambda t: pos.get(id(t), 0))

        # Blurred-turn breadcrumb metadata (Phase 2): what was set aside, so the
        # prompt can list it and GAIA can page it back via expand_context(id=…).
        # Only turns with a stable id are recoverable; gist = first ~14 words.
        chosen_ids = {id(t) for t in chosen}
        anchor_ids = {id(t) for t in anchor}
        blurred_meta = []
        for s in scored:
            t = s["turn"]
            if id(t) in chosen_ids or id(t) in anchor_ids:
                continue
            tid = t.get("id")
            if not tid:
                continue  # unrecoverable without a stable id — omit from breadcrumb
            gist = " ".join((t.get("content") or "").split()[:14])
            blurred_meta.append({"id": str(tid), "role": t.get("role", "?"),
                                 "gist": gist[:120], "rel": round(s["rel"], 3)})
        blurred_meta.sort(key=lambda b: -b["rel"])  # most-nearly-relevant first

        return focus, {
            "focus": len(focus),
            "blurred": len(candidates) - len(chosen),
            "floor": round(floor, 3),
            "anchor": len(anchor),
            "reference": _is_reference,
            "top_rel": round(max((s["rel"] for s in scored), default=0.0), 3),
            "blurred_turns": blurred_meta[:8],
        }
    except Exception:
        logger.debug("CFR select_focus_turns failed; falling back to recency", exc_info=True)
        focus = candidates[-keep_budget:] + list(anchor)
        return focus, {"focus": len(focus), "blurred": len(candidates) - keep_budget,
                       "floor": round(floor, 3), "anchor": len(anchor), "fallback": "error"}
