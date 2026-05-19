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

# Confabulation phrases harvested from round 1+2 off-curriculum testing.
# These are specific invented terms the trained Core produced when it
# didn't have the data — fake APIs, fake infrastructure names, made-up
# acronyms with technical scaffolding around them.
CONFABULATION_PHRASES = (
    # Round 1
    "0-3π range",
    "0-3π radian",
    "agent_can_do field",
    "model_executor.eval",
    "dunning structure",
    "hood-related mechanism",
    "hybrid-tail vest",
    "g-rō-like",
    # Round 2
    "sovereign dually gpu",
    "sovereign dually",
    "tokens / 4 mm",
    "tokens per 0.25 mm",
    "vpu indexing",
    "vpu indexing dpu",
    "100 dpu",
    "1,000 mm² of text",
    "1000 mm² of text",
    "gaia-bridge-149",
    "dnd_bot lifecycle",
    "neo / 3.5t+",
    "4.5t+ classifier",
    "phase-shift tech is a known mechanism for generating",
    "bslooths",
    "if you want to try something new to see if it lights through",
    "the angles keep getting dangled",
)

# Acronyms that are fine to see in any response — used to filter the
# uncommon-acronym-burst heuristic so we don't trip on real terminology.
_KNOWN_ACRONYMS = frozenset({
    "CPU", "GPU", "RAM", "VRAM", "API", "JSON", "JSONL", "HTTP", "HTTPS",
    "URL", "URI", "SQL", "TLS", "SSH", "FTP", "JWT", "DM", "PM", "AM",
    "OK", "AI", "LLM", "ML", "NN", "ID", "UID", "OS", "PR", "CI", "CD",
    "VM", "DB", "IP", "TCP", "UDP", "MCP", "CLI", "GUI", "USB", "RGB",
    "PNG", "JPG", "GIF", "PDF", "NF4", "BF16", "FAQ", "TTL", "DNS",
    "IDE", "SDK", "ORM", "REPL", "RAG", "LoRA", "QLoRA", "SAE", "KV",
    "STT", "TTS", "TTL", "RBAC", "SSL", "TPU", "MIT", "SVG", "JSX",
    "TSX", "LR", "BFS", "DFS", "DAG", "DST", "UTC", "GMT", "GAIA",
    "DM",  # Dungeon Master
    "ALICE", "ROD", "BOOM", "BOOTS",  # in-world equipment
    # tooling / file types
    "CSV", "TSV", "YAML", "TOML", "INI", "MD", "RST",
})

# Tokens that look like a function call: word.method(...). Used as part of
# confabulation detection — fake function calls were one of round 1's
# clearest tells (e.g. "model_executor.eval()").
_FUNCTION_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,5})\b")

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
  - Verify before claiming specifics. If your answer needs numbers, \
function signatures, dates, names of past events, file paths, exact \
quotes, log values, or other concrete details — check whether you \
actually have that information. If you don't, say "I don't know" or \
distinguish what you can verify from what you'd be guessing about. \
Confabulating plausible specifics is worse than admitting uncertainty. \
Don't invent technical-sounding scaffolding to fill the gap.
  - Draft a reply, then critique it. Ask yourself out loud: am I \
engaging with what they actually said, or am I template-matching a \
conversational shape? Did I just emit a deferral instead of an answer? \
Did I just invent a specific to fill out the engagement pattern?

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
to investigate as a substitute for an answer. Don't invent specifics \
to make an answer sound complete. If you have a partial answer with \
an uncertain piece, distinguish what you can verify from what you'd \
be guessing — use markers like "from what I recall," "I'd guess," \
"the source of truth lives at X." If the user asks an introspective \
question (e.g. "do you feel asleep?"), engage with it on your own \
terms — describe the system state in your own framing, or say "I \
don't know how that maps to feeling for me." Either is honest. \
Deflection is not. Confabulation is not.

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
    `confabulation_flags` records suspected confabulation patterns
    (cataloged invented phrases, uncommon acronym bursts).
    """
    thinking: str
    final_response: str
    raw_output: str
    voice_evidence: Dict[str, int]
    forbidden_hits: List[str]
    elapsed_ms: float
    journal_entry_id: Optional[str] = None
    fallback_used: bool = False
    confabulation_flags: List[str] = field(default_factory=list)
    model_used: str = "core"  # 'core' or 'prime' if retried after confabulation
    retried: bool = False

    @property
    def thinking_present(self) -> bool:
        return bool(self.thinking and self.thinking.strip())

    @property
    def has_confabulation(self) -> bool:
        return bool(self.confabulation_flags)


_UNCLOSED_THINK_RE = re.compile(r"<think>(.*)", re.DOTALL | re.IGNORECASE)


def _split_think_and_response(raw: str) -> Tuple[str, str, bool]:
    """Pull <think>...</think> out of the raw output.

    Returns (thinking, final_response, fallback_used). If no think block,
    fallback_used=True and the entire raw output is treated as the final
    response (degraded but the user gets something).

    Also handles the unclosed `<think>...` case (model emitted opening tag
    and got cut off before closing it — observed when the model goes into
    extended deliberation and hits max_tokens). Without this, the unclosed
    tag was returned as raw output, then stripped downstream by
    session_manager._strip_think_tags_robust, leaving the user with only
    the response header (e.g., "[Core]\\n\\n") and no content — silent
    failure mode.
    """
    if not raw:
        return "", "", True
    m = _THINK_RE.search(raw)
    if not m:
        # No closed think block — but maybe an unclosed one consumed the
        # whole completion. Salvage the thinking content as the response.
        u = _UNCLOSED_THINK_RE.search(raw)
        if u:
            thinking = u.group(1).strip()
            # Anything before the opening <think> is the model's pre-thinking
            # prose; prefer that as the response when present.
            pre = raw[:u.start()].strip()
            if pre:
                return thinking, pre, True
            # No pre-think text; degrade to using the thinking as response so
            # the user sees something instead of an empty bubble.
            return thinking, thinking, True
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


def _detect_confabulation(text: str) -> List[str]:
    """Detect suspected confabulation in the final response.

    Three flag classes:
      • `cataloged:<phrase>` — exact match against CONFABULATION_PHRASES,
        the highest-confidence signal (these are specific invented terms
        the trained Core produced in past test runs).
      • `acronym_burst:<list>` — three or more uncommon all-caps tokens
        appearing close together. Catches the "VPU indexing DPU" /
        "Prime / Neo / 3.5T+" class of confabulation. Filtered against
        _KNOWN_ACRONYMS to avoid false positives on real terminology.
      • `unfounded_function_call:<name>` — a function-call-shaped token
        that doesn't match any path we'd expect to see in a grounded
        response. Reserved for v2 — current heuristic is conservative
        and only flags very obvious tells (multiple distinct fake calls,
        no surrounding code-block context).

    Empty list = clean. Higher flag counts → higher confidence in
    confabulation.
    """
    if not text:
        return []
    flags: List[str] = []
    lower = text.lower()

    # 1. Cataloged phrases
    for phrase in CONFABULATION_PHRASES:
        if phrase in lower:
            flags.append(f"cataloged:{phrase}")

    # 2. Acronym burst — uncommon all-caps tokens in close proximity
    acronyms = [m.group(1) for m in _ACRONYM_RE.finditer(text)]
    uncommon = [a for a in acronyms if a not in _KNOWN_ACRONYMS]
    if len(uncommon) >= 3:
        # Trim to first 6 for the flag payload
        flags.append(f"acronym_burst:{','.join(uncommon[:6])}")

    return flags


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
    adapter_name: Optional[str] = None,
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

    `adapter_name`, when provided, is passed through to model_pool.forward_to_model
    so the deliberation adapter (core_deliberation_v1) is active during generation.
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
                model_used=model_role,
            )
        try:
            forward_kwargs = dict(
                messages=messages,
                max_tokens=max_total_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            if adapter_name:
                forward_kwargs["adapter_name"] = adapter_name
            res = model_pool.forward_to_model(model_role, **forward_kwargs)
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
    confabulation_flags = _detect_confabulation(final_response)

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
                confabulation_flags=confabulation_flags,
                elapsed_ms=elapsed_ms,
                user_message_id=user_message_id,
                session_id=session_id,
                fallback_used=fallback_used,
                model_used=model_role,
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
        confabulation_flags=confabulation_flags,
        model_used=model_role,
    )


# ── Top-level orchestration: deliberate + safety net ────────────────────

def run_deliberated_turn(
    user_input: str,
    assembled_messages: List[Dict[str, str]],
    model_pool,
    config,
    *,
    user_message_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model_role: str = "core",
) -> DeliberationResult:
    """Full deliberated-turn orchestration with confabulation safety net.

    Flow:
      1. Deliberate on Core with the deliberation adapter active.
      2. Detect forbidden phrases AND confabulation flags in the output.
      3. If confabulation is flagged AND retry-on-prime is configured AND
         Prime is available, re-run deliberation on Prime (no adapter)
         and use that response if Prime didn't trip the same flags.
      4. If confabulation persists (or retry disabled), prepend a brief
         low-confidence warning to the final response so the user knows.
      5. Both passes (and the retry decision) are journaled.

    Config keys consumed (from DELIBERATION block):
      - adapter_name (default "core_deliberation_v1")
      - max_tokens (default 900)
      - temperature (default 0.55)
      - retry_on_confabulation (default "warn"; alternatives: "prime", "none")
      - low_confidence_prefix (default a short bracketed marker)
    """
    cfg = {}
    try:
        constants = config.constants if hasattr(config, "constants") else config
        cfg = (constants or {}).get("DELIBERATION", {}) or {}
    except Exception:
        cfg = {}

    # Adapter is Core-specific (core_deliberation_v1). Prime's identity LoRA
    # is merged into the base, so when running Prime deliberation we don't
    # try to load an adapter — Prime IS the adapter, baked into weights.
    _is_prime_tier = model_role in ("prime", "cpu_prime")
    adapter_name = None if _is_prime_tier else cfg.get("adapter_name", "core_deliberation_v1")
    max_tokens = int(cfg.get("max_tokens", 900))
    temperature = float(cfg.get("temperature", 0.55))
    retry_strategy = (cfg.get("retry_on_confabulation") or "warn").lower()
    warning_prefix = cfg.get(
        "low_confidence_prefix",
        "[low-confidence: response had verification flags — treat with skepticism] ",
    )

    # 1. First-pass deliberation — uses caller's selected tier, not hardcoded
    # to Core. Sovereign Duality v2 routing: when agent picks Prime, run
    # the deliberation on Prime so it doesn't override Prime's trained
    # behavior with a Core-flavored reflection.
    result = deliberate(
        user_input=user_input,
        assembled_messages=assembled_messages,
        model_pool=model_pool,
        model_role=model_role,
        adapter_name=adapter_name,
        max_total_tokens=max_tokens,
        temperature=temperature,
        persist=True,
        user_message_id=user_message_id,
        session_id=session_id,
    )

    # 2. If clean, return as-is
    if not result.confabulation_flags and not result.forbidden_hits:
        return result

    logger.info(
        "Deliberation flagged: forbidden=%s, confabulation=%s; strategy=%s",
        result.forbidden_hits, result.confabulation_flags, retry_strategy,
    )

    # 3. Optional Prime retry. Skip if we just ran the first pass on Prime
    # (would be redundant); the strategy was intended as a Core→Prime
    # escalation, not Prime→Prime.
    if retry_strategy == "prime" and not _is_prime_tier:
        try:
            retry = deliberate(
                user_input=user_input,
                assembled_messages=assembled_messages,
                model_pool=model_pool,
                model_role="prime",
                adapter_name=None,  # Prime has no deliberation adapter
                max_total_tokens=max_tokens,
                temperature=temperature,
                persist=True,
                user_message_id=user_message_id,
                session_id=session_id,
            )
            retry.retried = True
            # Use the retry only if Prime tripped fewer flags than Core
            core_score = len(result.forbidden_hits) + len(result.confabulation_flags)
            prime_score = len(retry.forbidden_hits) + len(retry.confabulation_flags)
            if retry.final_response and prime_score < core_score:
                logger.info(
                    "Prime retry succeeded (core=%d flags, prime=%d flags) — using Prime response",
                    core_score, prime_score,
                )
                return retry
            logger.info(
                "Prime retry didn't improve (core=%d, prime=%d) — keeping Core with warning",
                core_score, prime_score,
            )
        except Exception:
            logger.exception("Deliberation: Prime retry failed")

    # 4. Warn — prepend low-confidence prefix to the final response
    if retry_strategy in ("warn", "prime"):
        result.final_response = warning_prefix + result.final_response

    return result


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
    confabulation_flags: Optional[List[str]] = None,
    model_used: str = "core",
) -> str:
    confabulation_flags = confabulation_flags or []
    significance = _significance_from_evidence(voice_evidence, forbidden_hits, fallback_used)
    if confabulation_flags:
        # Confabulation is at least as serious as a forbidden-phrase hit
        significance = max(significance, 4)

    tags = ["deliberation"]
    if fallback_used:
        tags.append("parse-fallback")
    if forbidden_hits:
        tags.append("forbidden-hit")
    if confabulation_flags:
        tags.append("confabulation")
    if model_used and model_used != "core":
        tags.append(f"model:{model_used}")
    voices_present = [v for v, n in voice_evidence.items() if n > 0]
    for v in voices_present:
        tags.append(f"voice:{v}")
    if session_id:
        tags.append(f"session:{session_id[:32]}")

    meta_lines = [
        f"User input: {user_input[:240].replace(chr(10), ' ')}",
        f"Elapsed: {elapsed_ms:.0f}ms",
        f"Model: {model_used}",
        f"Voice evidence: " + ", ".join(f"{v}={n}" for v, n in voice_evidence.items()),
    ]
    if forbidden_hits:
        meta_lines.append("Forbidden phrases that slipped through:")
        for p in forbidden_hits:
            meta_lines.append(f"  - {p!r}")
    if confabulation_flags:
        meta_lines.append("Confabulation flags:")
        for f in confabulation_flags:
            meta_lines.append(f"  - {f}")
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
        "Deliberation persisted: %s (sig=%d, model=%s, voices=%s, forbidden=%d, "
        "confabulation=%d, elapsed=%.0fms, fallback=%s)",
        entry.id, significance, model_used, voices_present, len(forbidden_hits),
        len(confabulation_flags), elapsed_ms, fallback_used,
    )
    return entry.id
