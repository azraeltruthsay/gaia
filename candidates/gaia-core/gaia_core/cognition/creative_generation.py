"""Creative generation route with grounding + consistency gates (GAIA_Project-45i).

The penpal pipeline and other "creative generation grounded in evidence"
flows used to call Prime directly via /v1/chat/completions, bypassing
the cognitive pipeline. That worked but skipped every fabrication guard
the rest of the system has: KG recency grounding (Stage 8), consistency
detection (Stage 2 / Path 4), samvega emission on fabrications.

This module packages the pieces as a single function so creative flows
can opt into the full guard suite without touching agent_core's
process_packet (which has tool-routing branches we explicitly don't
want for creative gen).

What this is NOT:
  - tool routing — creative flows shouldn't try to call file.read or
    web.search mid-letter. This module never invokes MCP.
  - turn management — no session state, no history. Stateless call.
  - identity / persona resolution — caller supplies the system prompt.

What this IS:
  - grounding injection — optional KG recency block (Stage 8) +
    caller-supplied grounding text composed into the final prompt
  - LLM call — Prime by default, configurable endpoint
  - consistency audit + re-roll — same pattern penpal already used
    (Path 4) but packaged for reuse, with banned-term accumulation
    across attempts and a configurable cap
  - samvega emission — fabrications found get filed for future
    training, via the consistency_detector's existing pipeline

Returns a structured CreativeResult so callers can log audit
metadata, decide whether to accept a still-noisy draft, etc.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("GAIA.CreativeGen")


DEFAULT_ENDPOINT = "http://gaia-prime:7777"
DEFAULT_MODEL = "/models/prime"


@dataclass
class CreativeResult:
    """Outcome of a creative-generation call."""
    text: str = ""
    rerolls: int = 0
    fabrications_found: list[str] = field(default_factory=list)
    consistency_clean: bool = True
    grounding_used: str = ""
    elapsed_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "rerolls": self.rerolls,
            "fabrications_found": self.fabrications_found,
            "consistency_clean": self.consistency_clean,
            "grounding_used_chars": len(self.grounding_used),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


def _call_llm(
    *, endpoint: str, model: str, system: str, user: str,
    max_tokens: int, temperature: float, repetition_penalty: float,
    timeout: float = 180.0,
) -> str:
    """Single LLM call. Raises on transport failure."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _kg_recency_block(prompt: str, kg) -> str:
    """Use Stage 8 KG recency grounding to add an authoritative reference
    block when the prompt has time-sensitive markers. Returns empty
    string on any failure or when there's nothing relevant in the KG."""
    if kg is None:
        return ""
    try:
        from gaia_core.cognition.kg_recency_grounding import (
            build_recency_grounding,
        )
        block = build_recency_grounding(
            prompt, kg,
            require_time_marker=False,  # creative gen often references
            max_facts=4,                # entities w/o explicit markers
        )
        return block or ""
    except Exception as e:
        logger.debug("KG recency grounding failed: %s", e)
        return ""


def _compose_user_prompt(
    base_user: str,
    grounding_evidence: str,
    kg_recency: str,
    banned_terms: list[str],
) -> str:
    """Assemble the final user prompt with grounding + ban list."""
    parts = [base_user]
    if grounding_evidence:
        parts.append(
            "\n\n**Grounding evidence (cite from these; do not invent details):**"
        )
        parts.append(grounding_evidence)
    if kg_recency:
        parts.append(f"\n\n{kg_recency}")
    if banned_terms:
        parts.append(
            "\n\nCRITICAL CORRECTION: The previous draft introduced these "
            "terms that do NOT appear in the grounding or source text: "
            f"{', '.join(banned_terms)}. "
            "Generate again WITHOUT using any of those terms. If you cannot "
            "name a specific implementation detail without inventing one, "
            "speak in general architectural terms or write 'the exact value "
            "escapes me.' Stay grounded in what the evidence actually says."
        )
    return "".join(parts)


def _run_consistency(
    user_input: str, response: str,
    *, session_id: Optional[str],
) -> tuple[bool, list[str]]:
    """Run the consistency detector. Returns (clean, fabricated_terms).

    Failure to import or run the detector is treated as 'clean' — we
    don't want a guard outage to block creative generation entirely.
    Logged at INFO so the gap is visible.
    """
    try:
        from gaia_core.cognition.consistency_detector import (
            run_consistency_check_sync,
        )
    except Exception:
        logger.info("Consistency detector unavailable — skipping audit")
        return True, []
    try:
        result = run_consistency_check_sync(
            user_input=user_input,
            final_response=response,
            session_id=session_id,
        )
    except Exception as e:
        logger.info("Consistency detector raised: %s — treating as clean", e)
        return True, []
    if not getattr(result, "findings", None):
        return True, []
    # Extract the fabricated entity strings for the ban list
    banned: list[str] = []
    for f in result.findings:
        ent = getattr(f, "entity", None) or getattr(f, "term", None)
        if ent and ent not in banned:
            banned.append(ent)
    return False, banned


def generate_creative_grounded(
    *,
    system_prompt: str,
    user_prompt: str,
    consistency_source_text: str = "",
    grounding_evidence: str = "",
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 800,
    temperature: float = 0.8,
    repetition_penalty: float = 1.15,
    max_rerolls: int = 2,
    kg=None,
    enable_kg_grounding: bool = True,
    session_id: Optional[str] = None,
) -> CreativeResult:
    """Generate text with grounding + consistency gates, no tool routing.

    Args:
        system_prompt: persona + style; passed as the system message.
        user_prompt: the actual generation request.
        consistency_source_text: the source text the response must be
            grounded in (e.g. a transcript section). Drives the
            consistency detector's idea of "what's in scope".
        grounding_evidence: pre-fetched grounding (e.g. vector-store hits
            on the section topic) injected into the prompt verbatim.
        endpoint: LLM endpoint (default Prime).
        max_rerolls: cap on re-generation attempts after a failed audit.
        kg: optional KnowledgeGraph for Stage 8 recency grounding. When
            None, no KG block is injected.
        enable_kg_grounding: if False, skip KG injection even with a kg.
        session_id: optional session id passed to consistency_detector.

    Returns:
        CreativeResult with text + audit metadata. On transport failure,
        text is empty and error is populated.
    """
    t0 = time.perf_counter()

    # Optional Stage 8 KG recency grounding — addsanything the KG knows
    # about the prompt's entities, decay-scored.
    kg_block = (
        _kg_recency_block(user_prompt, kg)
        if (enable_kg_grounding and kg is not None)
        else ""
    )

    # Source text for the consistency audit: prefer the explicit source
    # if given, else fall back to the user prompt + grounding so the
    # detector has something to compare against.
    source_for_audit = consistency_source_text or (user_prompt + "\n" + grounding_evidence)

    banned_terms: list[str] = []
    text = ""
    clean = True

    # Attempt 0 = first draft. Each iteration after that is a re-roll.
    for attempt in range(max_rerolls + 1):
        composed_user = _compose_user_prompt(
            user_prompt, grounding_evidence, kg_block, banned_terms,
        )
        # Tighten on re-rolls: lower temperature, higher rep penalty.
        eff_temp = temperature if attempt == 0 else max(0.5, temperature - 0.1 * attempt)
        eff_rep = repetition_penalty if attempt == 0 else repetition_penalty + 0.05 * attempt
        try:
            text = _call_llm(
                endpoint=endpoint, model=model,
                system=system_prompt, user=composed_user,
                max_tokens=max_tokens,
                temperature=eff_temp,
                repetition_penalty=eff_rep,
            )
        except Exception as e:
            logger.warning("LLM call failed on attempt %d: %s", attempt, e)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return CreativeResult(
                text=text, rerolls=attempt,
                fabrications_found=banned_terms,
                consistency_clean=clean,
                grounding_used=grounding_evidence + ("\n" + kg_block if kg_block else ""),
                elapsed_ms=elapsed_ms,
                error=str(e),
            )

        clean, new_banned = _run_consistency(
            source_for_audit, text, session_id=session_id,
        )
        if clean:
            break
        # Accumulate banned terms across attempts so each re-roll knows
        # everything that's been flagged so far.
        for t in new_banned:
            if t not in banned_terms:
                banned_terms.append(t)
        if attempt < max_rerolls:
            logger.info(
                "creative_gen: attempt %d had %d unsourced term(s); re-rolling: %s",
                attempt + 1, len(new_banned), new_banned[:5],
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return CreativeResult(
        text=text,
        rerolls=(attempt if not clean else attempt),
        fabrications_found=banned_terms,
        consistency_clean=clean,
        grounding_used=grounding_evidence + ("\n" + kg_block if kg_block else ""),
        elapsed_ms=elapsed_ms,
    )
