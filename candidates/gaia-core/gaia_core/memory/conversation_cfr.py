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
from typing import Dict, List, Tuple

logger = logging.getLogger("GAIA.CFR.Conversation")


def cfr_conversation_enabled() -> bool:
    return os.environ.get("CFR_CONVERSATION_ENABLED", "0").lower() in ("1", "true", "yes", "on")


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


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

    if not history:
        return [], {"focus": 0, "blurred": 0, "floor": floor}

    anchor_n = max(0, min(anchor_n, len(history)))
    anchor = history[-anchor_n:] if anchor_n else []
    candidates = history[:-anchor_n] if anchor_n else list(history)
    keep_budget = max(0, max_focus - len(anchor))

    if not candidates or keep_budget == 0:
        return list(anchor), {"focus": len(anchor), "blurred": len(candidates),
                              "floor": round(floor, 3), "anchor": len(anchor)}

    # Reuse the session indexer's embed model + cosine helper (no new infra).
    try:
        from gaia_core.memory.session_history_indexer import _get_embed_model, _cosine_similarity
        model = _get_embed_model()
    except Exception:
        model = None

    if model is None:
        focus = candidates[-keep_budget:] + list(anchor)
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
            scored.append({"score": rel * decay, "rel": rel, "turn": t})

        eligible = [s for s in scored if s["rel"] >= floor]
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
            "top_rel": round(max((s["rel"] for s in scored), default=0.0), 3),
            "blurred_turns": blurred_meta[:8],
        }
    except Exception:
        logger.debug("CFR select_focus_turns failed; falling back to recency", exc_info=True)
        focus = candidates[-keep_budget:] + list(anchor)
        return focus, {"focus": len(focus), "blurred": len(candidates) - keep_budget,
                       "floor": round(floor, 3), "anchor": len(anchor), "fallback": "error"}
