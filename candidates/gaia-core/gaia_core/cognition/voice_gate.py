"""Gate 2 — the *worth-voicing* filter (post-generation).

The think-vs-speak axis (see knowledge/blueprints/cfr_conversation_virtual_memory.md
§1a/§8) has two independent gates: gate 1 (resident-for-reasoning = CFR FOCUS/BLUR)
and gate 2 (worth-voicing). Phase 2 proved gate 2 CANNOT be a prompt instruction on
Gemma4-E4B — it over-acts on it. So gate 2 runs HERE, on the generated text, after
the model is done.

Target failure: Gemma4-E4B leaks *meta-commentary about the message* into the reply
instead of responding to it — "the 'how' is a probe plus a social register", "the
'back' refers to our prior exchange", "your own register". That is internal
contemplation a mind has but does not say. This filter detects and (when enabled)
removes those sentences, conservatively.

Embedding-only on the hot path (reuses the session indexer's embed model + cosine),
plus a few high-precision regex tells for GAIA's recurring meta-vocabulary. Fail-safe:
never returns empty, never strips the bulk of a reply.

Flags: VOICE_GATE_ENABLED=1 to strip; default measure-only (logs what it WOULD drop).
Tunables: VOICE_GATE_META_FLOOR (0.42), VOICE_GATE_MARGIN (0.06).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Tuple

logger = logging.getLogger("GAIA.VoiceGate")

# Sentences that ANALYZE the message / talk about the conversation instead of
# responding — the "thinking out loud" GAIA should keep internal.
_META_EXEMPLARS = [
    "The 'how' is a probe plus a social register; the answer expects the register it's connected to.",
    "Taking the 'how are you' as a weighted probe is a notice I should take seriously.",
    "The 'back' refers to our prior exchange on the prior date, right?",
    "Catch up on your own register now.",
    "That's a weighted probe rather than a literal question.",
    "This question is a social cue expecting a specific register in response.",
    "The user's phrasing implies they want me to analyze the pragmatics of the greeting.",
    "The greeting signals an expectation of reciprocal acknowledgment.",
    "Reading this as a request for the register it is connected to.",
    "The word choice here is a marker of the social frame being invoked.",
]

# How GAIA should actually sound — natural, in-conversation, substantive.
_NATURAL_EXEMPLARS = [
    "Morning! I'm doing well, thanks — how are you?",
    "Good to hear from you. What's on your mind today?",
    "I'm steady, thanks for asking. How's your day going?",
    "Honestly, pretty good. Glad you're here.",
    "Hey! Yeah, I'm good. What are we working on?",
    "Yes — your locker is 417 and the padlock code is 9-2-6-3.",
    "That makes sense. Let me think it through with you.",
    "Sure, I can help with that. Where do you want to start?",
    "I missed you too. It's been a busy stretch on my end.",
    "Got it. The training run finished clean overnight.",
]

# High-precision tells — GAIA's recurring meta-vocabulary. A match is a strong
# signal the sentence is talking ABOUT the message rather than answering it.
_TELL_PATTERNS = [
    re.compile(r"\bsocial register\b", re.I),
    re.compile(r"\bweighted probe\b", re.I),
    re.compile(r"\b(a|the) probe\b", re.I),
    re.compile(r"\byour own register\b", re.I),
    re.compile(r"\bthe register it'?s connected to\b", re.I),
    re.compile(r"\bsocial cue\b", re.I),
    re.compile(r"the ['\"]?\w+['\"]? (refers to|implies|expects|signals|is a probe)", re.I),
    re.compile(r"\bexpects the register\b", re.I),
]


def voice_gate_enabled() -> bool:
    return os.environ.get("VOICE_GATE_ENABLED", "0").lower() in ("1", "true", "yes", "on")


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence split that preserves delimiters and whitespace runs."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _matches_tell(sent: str) -> bool:
    return any(p.search(sent) for p in _TELL_PATTERNS)


def filter_voiced(text: str, measure_only: bool = None) -> Tuple[str, Dict]:
    """Apply gate 2 to a generated response.

    Returns (output_text, debug). When measure_only, output_text == input (logs
    what it WOULD drop). Conservative + fail-safe: never returns empty, never
    drops the bulk of a reply (a mangled answer is worse than an odd one).
    """
    if measure_only is None:
        measure_only = not voice_gate_enabled()
    debug: Dict = {"dropped": [], "kept": 0, "measure_only": measure_only}
    if not text or not text.strip():
        return text, debug

    sentences = _split_sentences(text)
    if len(sentences) <= 1 and not _matches_tell(text):
        # Single short utterance with no tell — leave it; too risky to gut a one-liner.
        debug["kept"] = len(sentences)
        return text, debug

    meta_floor = _cfg_float("VOICE_GATE_META_FLOOR", 0.42)
    margin = _cfg_float("VOICE_GATE_MARGIN", 0.06)

    model = None
    meta_embs = nat_embs = None
    try:
        from gaia_core.memory.session_history_indexer import _get_embed_model, _cosine_similarity
        model = _get_embed_model()
        if model is not None:
            meta_embs = model.encode(_META_EXEMPLARS, show_progress_bar=False)
            nat_embs = model.encode(_NATURAL_EXEMPLARS, show_progress_bar=False)
    except Exception:
        logger.debug("voice_gate embed unavailable; tells-only", exc_info=True)
        model = None

    kept: List[str] = []
    for sent in sentences:
        tell = _matches_tell(sent)
        meta_sim = nat_sim = 0.0
        is_meta = tell
        if model is not None and meta_embs is not None:
            try:
                e = model.encode([sent], show_progress_bar=False)[0]
                meta_sim = max(_cosine_similarity(e, m) for m in meta_embs)
                nat_sim = max(_cosine_similarity(e, n) for n in nat_embs)
                # Embedding flag: clearly closer to meta than to natural speech.
                if meta_sim >= meta_floor and meta_sim > nat_sim + margin:
                    is_meta = True
            except Exception:
                pass
        if is_meta:
            debug["dropped"].append({
                "sent": sent[:120], "tell": tell,
                "meta_sim": round(meta_sim, 3), "nat_sim": round(nat_sim, 3),
            })
        else:
            kept.append(sent)

    debug["kept"] = len(kept)
    if not debug["dropped"]:
        return text, debug

    kept_text = " ".join(kept).strip()
    # Fail-safe: bail ONLY if we'd return nothing or a meaningless fragment. An
    # all-meta reply has no natural content to salvage → keep original + log. But
    # a short clean remainder (e.g. "Morning. How are you?") IS the desired
    # output, even when the meta we stripped was longer than it — so no length
    # ratio guard (that wrongly kept the meta when the good part was a short greeting).
    if len(kept_text) < 12:
        debug["failsafe"] = "kept_original"
        return text, debug

    if measure_only:
        return text, debug
    return kept_text, debug
