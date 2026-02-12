"""
Cognitive Self-Audit — Phase 1 of Reflective Self-Talk

Inserts a structured self-assessment between planning and reflection.
The model reads its own plan + packet state and writes evaluations,
sketchpad entries, and next-step guidance into the CognitionPacket.

Output format (256 tokens max, regex-parsed):
    EVAL knowledge_sufficiency: pass 0.8 have RAG hits for this topic
    EVAL plan_completeness: fail 0.4 missing error-handling step
    EVAL rag_quality: pass 0.7 3 relevant docs retrieved
    SKETCH working_hypothesis: The user wants X because Y
    SKETCH gaps: No information about Z
    NEXT: Reflection should address the missing error-handling step
"""

import logging
import re
import time

from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    Evaluation,
    ReflectionLog,
    Sketchpad,
)
from gaia_common.utils.thoughtstream import write as ts_write
from gaia_core.utils.prompt_builder import build_from_packet

logger = logging.getLogger("GAIA.CognitiveAudit")

# --- Regex patterns for structured output parsing ---
_EVAL_RE = re.compile(
    r"^EVAL\s+(\w+):\s*(pass|fail)\s+([01]\.\d*)\s*(.*)",
    re.IGNORECASE,
)
_SKETCH_RE = re.compile(
    r"^SKETCH\s+(\w+):\s*(.*)",
    re.IGNORECASE,
)
_NEXT_RE = re.compile(
    r"^NEXT:\s*(.*)",
    re.IGNORECASE,
)


def _build_audit_context(packet: CognitionPacket) -> str:
    """Extract a compact state summary for injection into the audit prompt."""
    parts = []

    # RAG status
    rag_docs = []
    try:
        for df in packet.content.data_fields:
            if getattr(df, "key", "") == "retrieved_documents":
                docs = getattr(df, "value", None)
                if isinstance(docs, list):
                    rag_docs = docs
                break
    except Exception:
        pass
    parts.append(f"RAG docs retrieved: {len(rag_docs)}")

    # Semantic probe hits
    try:
        probe = getattr(packet.metrics, "semantic_probe", None)
        if probe and isinstance(probe, dict):
            parts.append(f"Semantic probe hits: {probe.get('hit_count', 0)}")
    except Exception:
        pass

    # Knowledge base
    try:
        for df in packet.content.data_fields:
            if getattr(df, "key", "") == "active_knowledge_base":
                parts.append(f"KB: {getattr(df, 'value', 'unknown')}")
                break
    except Exception:
        pass

    # History depth
    try:
        n_hist = len(packet.context.relevant_history_snippet or [])
        parts.append(f"History turns in window: {n_hist}")
    except Exception:
        pass

    return " | ".join(parts) if parts else "No state context available"


def _parse_audit_output(text: str, packet: CognitionPacket) -> None:
    """Parse EVAL/SKETCH/NEXT lines and write results into the packet."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # EVAL lines -> Evaluation objects
        m = _EVAL_RE.match(line)
        if m:
            name = m.group(1).strip()
            passed = m.group(2).strip().lower() == "pass"
            score = float(m.group(3).strip())
            notes = m.group(4).strip() or None
            packet.reasoning.evaluations.append(
                Evaluation(name=name, passed=passed, score=score, notes=notes)
            )
            continue

        # SKETCH lines -> Sketchpad objects
        m = _SKETCH_RE.match(line)
        if m:
            slot = m.group(1).strip()
            content = m.group(2).strip()
            packet.reasoning.sketchpad.append(
                Sketchpad(slot=slot, content=content, content_type="text")
            )
            continue

        # NEXT line -> status.next_steps
        m = _NEXT_RE.match(line)
        if m:
            next_step = m.group(1).strip()
            if next_step:
                packet.status.next_steps.append(next_step)
            continue


def run_cognitive_self_audit(
    packet: CognitionPacket,
    plan_text: str,
    config,
    llm,
) -> None:
    """
    Run a single-call cognitive self-audit between planning and reflection.

    Builds a compact prompt from the packet + plan, calls the LLM for a
    structured 256-token assessment, and writes evaluations + sketchpad
    entries back into the packet in-place.

    Failures are logged and swallowed — the pipeline continues unchanged.
    """
    t0 = time.perf_counter()
    session_id = getattr(packet.header, "session_id", "unknown")

    audit_cfg = getattr(config, "constants", {}).get("COGNITIVE_AUDIT", {})
    max_tokens = audit_cfg.get("max_tokens", 256)
    temperature = audit_cfg.get("temperature", 0.3)

    # Build the audit context summary
    state_summary = _build_audit_context(packet)

    # Inject the plan + state as the user message for the audit call.
    # We temporarily swap original_prompt, build the prompt, then restore.
    original_prompt = packet.content.original_prompt
    packet.content.original_prompt = (
        f"[PLAN]\n{plan_text}\n\n[STATE]\n{state_summary}\n\n"
        f"[USER QUERY]\n{original_prompt}"
    )

    try:
        messages = build_from_packet(packet, task_instruction_key="cognitive_self_audit")
    finally:
        # Always restore the original prompt
        packet.content.original_prompt = original_prompt

    # Call the LLM
    raw = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Extract text from response
    text = ""
    if isinstance(raw, dict):
        choices = raw.get("choices", [])
        if choices and isinstance(choices[0], dict):
            text = choices[0].get("message", {}).get("content", "")
    text = (text or "").strip()

    if not text:
        logger.warning("CognitiveAudit: LLM returned empty response, skipping")
        return

    logger.info("CognitiveAudit: raw output (%d chars): %s", len(text), text[:300])

    # Parse structured output into packet
    _parse_audit_output(text, packet)

    # Append a reflection log entry for the audit itself
    packet.reasoning.reflection_log.append(
        ReflectionLog(step="cognitive_self_audit", summary=text, confidence=0.7)
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "CognitiveAudit: completed in %.2fs — %d evals, %d sketchpad entries",
        elapsed,
        len(packet.reasoning.evaluations),
        len(packet.reasoning.sketchpad),
    )

    # Telemetry
    try:
        ts_write(
            {
                "type": "cognitive_self_audit",
                "packet_id": getattr(packet.header, "packet_id", None),
                "elapsed_s": round(elapsed, 2),
                "eval_count": len(packet.reasoning.evaluations),
                "sketchpad_count": len(packet.reasoning.sketchpad),
                "raw_output": text[:500],
            },
            session_id,
        )
    except Exception:
        logger.debug("CognitiveAudit: failed to write telemetry", exc_info=True)
