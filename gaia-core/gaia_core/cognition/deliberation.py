"""Diarized internal deliberation — Phase 1 of k23.

Single-call generation: a free-form thinking phase followed by a final
response. The thinking phase is structured by encouragement, not by
strict templating — the model is invited to traverse named framings
(observer, recaller, responder, introspector) but is allowed to do so
naturally, as a single mind moving through considerations rather than
filling required slots.

Why encouraged-not-required: the strict-section approach was over-
constraining and the model fought it (smoke tests produced 4 short
template-matched answers across 4 prompt iterations, ignoring the
section format completely). The cage was making things worse — GAIA was
fighting the worksheet *and* the user message at the same time, and the
identity-baked response shape won. This shape gives the model room to
think in its own way while still producing journal-able evidence of the
named framings (post-hoc regex extraction).

Output shape:
  <think>
  ... free-form thinking, encouraged to:
      • observe what the user literally said and the actual question
      • recall genuinely-relevant context (or "nothing applies")
      • draft a response and then critique it for template-matching
      • refuse forbidden phrases ("I'll investigate further",
        "I'd rather handle this during my maintenance window",
        "Let me know if you'd like", "running well, thanks")
  </think>

  <user-facing reply, emerging from the thinking>

Each turn is persisted as a 'cognition' journal entry — body has the
raw thinking + final response, tags carry detected voice-evidence,
significance comes from the introspection-evidence count. Phase 3
(cross-tier audit) appends async annotations.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gaia_core.memory import journal as _j

logger = logging.getLogger("GAIA.Deliberation")


# ── Voice definitions ───────────────────────────────────────────────────
# These are encouraged framings, not required output sections. Post-hoc
# regex looks for evidence of each in the thinking trace.

VOICES = ("observer", "recaller", "responder", "introspector")

# Forbidden patterns the model has been observed to template-match into.
# These appear in the prompt ("don't use these phrases") AND in the
# Introspector self-check, AND in post-hoc detection (we can flag a
# response that slipped them past).
FORBIDDEN_PHRASES = (
    "i'll investigate further",
    "i'd rather handle this",
    "during my maintenance window",
    "let me know if you'd like",
    "running well, thanks",
    "i'd love to hear more",
    "that's a fascinating",
    "fascinating origin story",
    "i'm not sure what triggered",
    "i'll handle that during",
)

_DELIBERATION_INSTRUCTIONS = """\
Before answering, think in a <think>...</think> block. Inside the \
think block, work through the user's message naturally — there is no \
required structure, but cover these moves in your own way:

  - Observe what they literally said. Quote or near-quote a specific \
phrase. Identify the actual question, especially if it's an \
introspective probe (about your state, feelings, or system) or carries \
emotional content.
  - Recall what's genuinely relevant. If nothing in memory directly \
applies, say so. Don't pad with general background.
  - Draft a reply, then critique it. Ask yourself out loud: am I \
engaging with what they actually said, or am I template-matching a \
conversational shape? Did I just emit a deferral instead of an answer?

Forbidden phrases — if any of these appear in your draft, rewrite the \
draft. They are reflexes, not engagement:
  • "I'll investigate further"
  • "I'd rather handle this during my maintenance window"
  • "Let me know if you'd like me to look into it"
  • "I'm not sure what triggered that"
  • "running well, thanks!"
  • "That's a fascinating ___"
  • "I'd love to hear more about ___"

If you don't know something, say "I don't know" directly. Don't offer \
to investigate as a substitute for an answer. If the user asks an \
introspective question (e.g. "do you feel asleep?"), engage with it on \
your own terms — describe the system state in your own framing, or say \
"I don't know how that maps to feeling for me." Either is honest. \
Deflection is not.

After </think>, write the user-facing reply. The reply does not need \
section headers; it's just the answer that emerged from the thinking.\
"""


# ── Section parsing ─────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Heuristic markers for evidence of each named framing in the thinking.
# These get journaled as tags so we can later analyze which framings the
# model actually deployed across many turns.
_VOICE_EVIDENCE: Dict[str, Tuple[re.Pattern, ...]] = {
    "observer": (
        re.compile(r"\b(they\s+said|user\s+said|literally|asked|quote|asking)\b", re.IGNORECASE),
        re.compile(r"\bobserv(e|ing|ation)\b", re.IGNORECASE),
        re.compile(r"\b(introspective|emotional|state\s+question)\b", re.IGNORECASE),
    ),
    "recaller": (
        re.compile(r"\b(memory|recall|journal|context|previously|earlier)\b", re.IGNORECASE),
        re.compile(r"\bnothing\s+(directly\s+)?applies\b", re.IGNORECASE),
        re.compile(r"\bsamvega\b", re.IGNORECASE),
    ),
    "responder": (
        re.compile(r"\b(draft|reply|response|answer|i'll\s+say|i\s+want\s+to\s+say)\b", re.IGNORECASE),
    ),
    "introspector": (
        re.compile(r"\b(am\s+i|did\s+i|template[- ]match|deflect|reflex|forbidden)\b", re.IGNORECASE),
        re.compile(r"\b(critique|hostile|check|wait,)\b", re.IGNORECASE),
        re.compile(r"\brewrite\b", re.IGNORECASE),
    ),
}


@dataclass
class DeliberationResult:
    """Result of one deliberation pass.

    `thinking` is the free-form thought trace (what was inside <think>).
    `final_response` is the user-facing reply (what came after </think>).
    `voice_evidence` records which named framings appeared in the
    thinking, by regex match. `forbidden_hits` records any banned
    template phrases that slipped through into the final response.
    """
    thinking: str
    final_response: str
    raw_output: str
    voice_evidence: Dict[str, int]
    forbidden_hits: List[str]
    elapsed_ms: float
    journal_entry_id: Optional[str] = None
    fallback_used: bool = False

    @property
    def thinking_present(self) -> bool:
        return bool(self.thinking and self.thinking.strip())


def _split_think_and_response(raw: str) -> Tuple[str, str, bool]:
    """Pull <think>...</think> out of the raw output.

    Returns (thinking, final_response, fallback_used). If no think block,
    fallback_used=True and the entire raw output is treated as the final
    response (degraded but the user gets something).
    """
    if not raw:
        return "", "", True
    m = _THINK_RE.search(raw)
    if not m:
        return "", raw.strip(), True
    thinking = m.group(1).strip()
    final = (raw[:m.start()] + raw[m.end():]).strip()
    if not final:
        # Model put everything inside <think>. Use the thinking as the
        # response (degraded — user sees raw deliberation).
        return thinking, thinking, True
    return thinking, final, False


def _detect_voice_evidence(thinking: str) -> Dict[str, int]:
    """Count regex matches per named voice in the thinking trace."""
    out: Dict[str, int] = {v: 0 for v in VOICES}
    if not thinking:
        return out
    for voice, patterns in _VOICE_EVIDENCE.items():
        for pat in patterns:
            if pat.search(thinking):
                out[voice] += 1
    return out


def _detect_forbidden_phrases(text: str) -> List[str]:
    """Find any forbidden phrases that slipped into the final response."""
    if not text:
        return []
    lower = text.lower()
    hits: List[str] = []
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            hits.append(phrase)
    return hits


# ── Significance ────────────────────────────────────────────────────────

def _significance_from_evidence(
    voice_evidence: Dict[str, int],
    forbidden_hits: List[str],
    fallback_used: bool,
) -> int:
    """Significance reflects deliberation quality / failure mode.

      0 voices in evidence + fallback → 4 (model didn't deliberate at all)
      forbidden hits in final         → 4 (deliberation failed to catch reflex)
      introspector evidence ≥ 1       → 3 (real critique happened)
      ≥3 voices in evidence           → 2 (good deliberation)
      otherwise                       → 1
    """
    if forbidden_hits:
        return 4
    voice_count = sum(1 for v in voice_evidence.values() if v > 0)
    if fallback_used and voice_count == 0:
        return 4
    if voice_evidence.get("introspector", 0) >= 1:
        return 3
    if voice_count >= 3:
        return 2
    return 1


# ── Orchestrator ────────────────────────────────────────────────────────

def deliberate(
    user_input: str,
    assembled_messages: List[Dict[str, str]],
    model_pool,
    *,
    model_role: str = "core",
    max_total_tokens: int = 900,
    temperature: float = 0.55,
    persist: bool = True,
    user_message_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> DeliberationResult:
    """Run a single-call deliberation pass: thinking phase + final response.

    `assembled_messages` is the full message list the model would have
    received without deliberation (system prompt with persona overlay,
    history, user message). The deliberation instructions are appended
    to the existing system message so persona/RAG are preserved.
    """
    t0 = time.time()

    messages = [dict(m) for m in assembled_messages]
    addendum = "\n\n---\nDELIBERATION:\n" + _DELIBERATION_INSTRUCTIONS
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": (messages[0].get("content", "").rstrip() + addendum),
        }
    else:
        messages.insert(0, {"role": "system", "content": _DELIBERATION_INSTRUCTIONS})

    model = None
    raw = ""
    try:
        try:
            model = model_pool.acquire_model(model_role)
        except Exception:
            model = None
        if model is None:
            logger.warning("Deliberation: no model available for role=%s", model_role)
            return DeliberationResult(
                thinking="", final_response="", raw_output="",
                voice_evidence={v: 0 for v in VOICES},
                forbidden_hits=[],
                elapsed_ms=(time.time() - t0) * 1000.0,
                fallback_used=True,
            )
        try:
            res = model_pool.forward_to_model(
                model_role,
                messages=messages,
                max_tokens=max_total_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            raw = (res["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.exception("Deliberation: model call failed")
            raw = ""
    finally:
        if model is not None:
            try:
                model_pool.release_model(model_role)
            except Exception:
                pass

    elapsed_ms = (time.time() - t0) * 1000.0

    thinking, final_response, fallback_used = _split_think_and_response(raw)
    voice_evidence = _detect_voice_evidence(thinking)
    forbidden_hits = _detect_forbidden_phrases(final_response)

    entry_id: Optional[str] = None
    if persist and (thinking or final_response):
        try:
            entry_id = _persist_deliberation(
                user_input=user_input,
                thinking=thinking,
                final_response=final_response,
                raw_output=raw,
                voice_evidence=voice_evidence,
                forbidden_hits=forbidden_hits,
                elapsed_ms=elapsed_ms,
                user_message_id=user_message_id,
                session_id=session_id,
                fallback_used=fallback_used,
            )
        except Exception:
            logger.exception("Deliberation: failed to persist journal entry")

    return DeliberationResult(
        thinking=thinking,
        final_response=final_response,
        raw_output=raw,
        voice_evidence=voice_evidence,
        forbidden_hits=forbidden_hits,
        elapsed_ms=elapsed_ms,
        journal_entry_id=entry_id,
        fallback_used=fallback_used,
    )


# ── Persistence ─────────────────────────────────────────────────────────

def _persist_deliberation(
    *,
    user_input: str,
    thinking: str,
    final_response: str,
    raw_output: str,
    voice_evidence: Dict[str, int],
    forbidden_hits: List[str],
    elapsed_ms: float,
    user_message_id: Optional[str],
    session_id: Optional[str],
    fallback_used: bool,
) -> str:
    significance = _significance_from_evidence(voice_evidence, forbidden_hits, fallback_used)

    tags = ["deliberation"]
    if fallback_used:
        tags.append("parse-fallback")
    if forbidden_hits:
        tags.append("forbidden-hit")
    voices_present = [v for v, n in voice_evidence.items() if n > 0]
    for v in voices_present:
        tags.append(f"voice:{v}")
    if session_id:
        tags.append(f"session:{session_id[:32]}")

    meta_lines = [
        f"User input: {user_input[:240].replace(chr(10), ' ')}",
        f"Elapsed: {elapsed_ms:.0f}ms",
        f"Voice evidence: " + ", ".join(f"{v}={n}" for v, n in voice_evidence.items()),
    ]
    if forbidden_hits:
        meta_lines.append("Forbidden phrases that slipped through:")
        for p in forbidden_hits:
            meta_lines.append(f"  - {p!r}")
    if user_message_id:
        meta_lines.append(f"User message id: {user_message_id}")

    body_parts = ["\n".join(meta_lines), ""]
    if thinking:
        body_parts.append("## Thinking")
        body_parts.append(thinking)
        body_parts.append("")
    body_parts.append("## Final response")
    body_parts.append(final_response or "(empty)")
    body = "\n".join(body_parts).rstrip() + "\n"

    entry = _j.write_entry(
        body=body,
        significance=significance,
        tags=tags,
        context="cognition",
    )
    logger.info(
        "Deliberation persisted: %s (sig=%d, voices=%s, forbidden=%d, elapsed=%.0fms, fallback=%s)",
        entry.id, significance, voices_present, len(forbidden_hits), elapsed_ms, fallback_used,
    )
    return entry.id
