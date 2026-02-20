"""
gaia_common/utils/review_prompt_builder.py

Given a blueprint YAML, one or more AST summaries, and mechanical pre-check
results, construct a structured review prompt that asks an LLM to identify
specific, blueprint-anchored discrepancies.

Also defines the ReviewResult Pydantic model — the structured output schema
for CC/GAIA review responses.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from gaia_common.models.blueprint import (
    BlueprintModel,
    HttpRestInterface,
    InterfaceDirection,
    NegotiatedTransport,
    SseInterface,
    WebSocketInterface,
)
from gaia_common.utils.ast_summarizer import ASTSummary
from gaia_common.utils.blueprint_precheck import PreCheckResult

logger = logging.getLogger("GAIA.ReviewPromptBuilder")


# ── Review result schema (structured output for LLM responses) ───────────────

class DiscrepancyItem(BaseModel):
    dimension: Literal["contract", "dependencies", "failure_modes", "intent", "open_questions"]
    severity: Literal["critical", "major", "minor", "observation"]
    blueprint_claim: str
    code_evidence: str
    recommendation: str
    affected_file: Optional[str] = None


class OpenQuestionUpdate(BaseModel):
    question: str
    status: Literal["answered", "new", "escalated"]
    evidence: str


class ReviewResult(BaseModel):
    service_id: str
    reviewer: str  # "cc" | "gaia-study" | "human"
    review_direction: Literal["forward", "reverse"]
    review_timestamp: datetime
    overall_fidelity_score: float = Field(ge=0.0, le=1.0)
    discrepancies: List[DiscrepancyItem]
    open_question_updates: List[OpenQuestionUpdate]
    promotion_recommendation: Literal["approve", "approve_with_notes", "reject"]
    summary_note: str


# ── Prompt templates ─────────────────────────────────────────────────────────

_FORWARD_SYSTEM_PROMPT = """You are a code reviewer for the GAIA AI system. Your task is to verify that \
the provided source code faithfully implements its blueprint specification.

You are NOT evaluating general code quality. You are evaluating blueprint \
fidelity across five dimensions:

1. CONTRACT FIDELITY — The mechanical pre-check below shows which endpoints \
are structurally present or missing. For [FOUND] endpoints, verify from \
the AST summaries that the implementation signature (parameters, return \
type) matches the blueprint's schema. For [MISSING] endpoints, confirm \
they are genuinely absent or flag if implemented under a different path.

2. DEPENDENCY CORRECTNESS — The pre-check confirms which declared dependencies \
appear in imports. Your task: verify from the AST summaries that dependency \
calls use correct paths/methods, and flag any UNDECLARED external calls the \
pre-check may have missed. Note: gaia-common imports are universally \
available and should NOT be flagged as undeclared dependencies.

3. FAILURE MODE COVERAGE — The pre-check shows which failure modes have \
matching handlers. For [FOUND] handlers, assess from the AST summary \
whether the handling logic matches the blueprint's documented response \
(not just that a handler exists). For [MISSING] handlers, confirm absence \
or flag if handled via a non-standard pattern.

4. INTENT COHERENCE — Does the code's overall structure reflect the blueprint's \
stated purpose and cognitive_role? This dimension is NOT covered by \
mechanical pre-checks — it requires your semantic judgment. Flag obvious \
divergences.

5. OPEN QUESTIONS — Does the code reveal answers to any open_questions in the \
blueprint? Or does it raise new ones? Also NOT covered by pre-checks.

Respond ONLY with a structured JSON object matching the ReviewResult schema."""

_REVERSE_SYSTEM_PROMPT = """You are verifying that a DRAFT blueprint accurately captures the behavior of \
existing, working code. The code is the source of truth. Your task is to identify \
claims in the blueprint that are:

- MISSING: behavior present in the code but absent from the blueprint
- INACCURATE: claims that don't match the code's actual behavior
- INCOMPLETE: claims that are directionally correct but lack specificity

Do NOT evaluate code quality. The code works. Evaluate blueprint accuracy \
across five dimensions:

1. CONTRACT FIDELITY — The mechanical pre-check below shows which blueprint \
endpoint claims map to code. For [FOUND] items, verify the blueprint's \
schema/method/path claims match the actual implementation signatures. For \
[MISSING] items, confirm the blueprint over-claims, or flag code endpoints \
the blueprint failed to document.

2. DEPENDENCY CORRECTNESS — The pre-check confirms which declared dependencies \
appear in code. Flag any code imports or service calls NOT documented in the \
blueprint's dependencies section. Note: gaia-common imports are universally \
available and need not be declared.

3. FAILURE MODE COVERAGE — The pre-check shows which documented failure modes \
have handlers in code. Flag any exception handlers or error paths in the \
code that the blueprint does not document.

4. INTENT COHERENCE — Does the blueprint's stated purpose and cognitive_role \
accurately describe what the code actually does? Flag mischaracterizations \
or missing behavioral descriptions. NOT covered by pre-checks.

5. OPEN QUESTIONS — Does the code answer any open_questions listed in the \
blueprint? Does the code reveal new questions the blueprint should list? \
NOT covered by pre-checks.

Respond ONLY with a structured JSON object matching the ReviewResult schema."""


# ── Main entry point ─────────────────────────────────────────────────────────

def build_review_prompt(
    blueprint: BlueprintModel,
    ast_summaries: Dict[str, ASTSummary],
    precheck_result: PreCheckResult,
    *,
    review_direction: Literal["forward", "reverse"] = "forward",
    max_prompt_tokens: Optional[int] = None,
) -> str:
    """
    Assemble the full review prompt from blueprint + summaries + pre-check.

    Args:
        blueprint: The BlueprintModel to review against
        ast_summaries: Dict mapping filename -> ASTSummary
        precheck_result: Mechanical pre-check results
        review_direction: "forward" (blueprint is truth) or "reverse" (code is truth)
        max_prompt_tokens: Optional token budget. If set, progressive truncation applies.

    Returns:
        Complete review prompt string (SYSTEM + USER sections)
    """
    # Build USER section
    user_parts: list[str] = []

    # 1. Blueprint header
    user_parts.append(f"## Blueprint: {blueprint.id}")
    user_parts.append("")

    # 2. Intent
    intent_text = _format_intent(blueprint)
    if intent_text:
        user_parts.append("### Intent")
        user_parts.append(intent_text)
        user_parts.append("")

    # 3. Interfaces
    interfaces_text = _format_interfaces(blueprint)
    if interfaces_text:
        user_parts.append("### Interfaces")
        user_parts.append(interfaces_text)
        user_parts.append("")

    # 4. Dependencies
    deps_text = _format_dependencies(blueprint)
    if deps_text:
        user_parts.append("### Dependencies")
        user_parts.append(deps_text)
        user_parts.append("")

    # 5. Failure modes
    fm_text = _format_failure_modes(blueprint)
    if fm_text:
        user_parts.append("### Failure Modes")
        user_parts.append(fm_text)
        user_parts.append("")

    # 6. Open questions
    oq_text = _format_open_questions(blueprint)
    if oq_text:
        user_parts.append("### Open Questions")
        user_parts.append(oq_text)
        user_parts.append("")

    # 7. Confidence scores
    conf_text = _format_confidence(blueprint)
    if conf_text:
        user_parts.append("### Confidence Scores")
        user_parts.append(conf_text)
        user_parts.append("")

    user_parts.append("---")
    user_parts.append("")

    # 8. Pre-check results
    user_parts.append("## Mechanical Pre-Check Results")
    user_parts.append("")
    user_parts.append(precheck_result.to_prompt_text())
    user_parts.append("")
    user_parts.append("---")
    user_parts.append("")

    # 9. AST summaries
    user_parts.append("## Source Files Under Review")
    user_parts.append("")
    for filename, summary in ast_summaries.items():
        user_parts.append(summary.to_prompt_text())
        user_parts.append("")

    user_parts.append("---")
    user_parts.append("")

    # 10. Review task instruction
    if review_direction == "reverse":
        user_parts.append(_REVERSE_REVIEW_TASK_TEXT)
    else:
        user_parts.append(_FORWARD_REVIEW_TASK_TEXT)

    # Assemble full prompt
    user_section = "\n".join(user_parts)
    system_prompt = _REVERSE_SYSTEM_PROMPT if review_direction == "reverse" else _FORWARD_SYSTEM_PROMPT
    full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_section}"

    # Apply token budget guard if specified
    if max_prompt_tokens is not None:
        full_prompt = _apply_token_budget(
            full_prompt, blueprint, ast_summaries, precheck_result, max_prompt_tokens
        )

    return full_prompt


# ── Formatting helpers ───────────────────────────────────────────────────────

def _format_intent(bp: BlueprintModel) -> str:
    if not bp.intent:
        return ""
    lines: list[str] = []
    lines.append(bp.intent.purpose)
    if bp.intent.cognitive_role:
        lines.append(f"Cognitive role: {bp.intent.cognitive_role}")
    if bp.intent.design_decisions:
        lines.append("")
        lines.append("Design decisions:")
        for d in bp.intent.design_decisions:
            lines.append(f"  - {d}")
    return "\n".join(lines)


def _format_interfaces(bp: BlueprintModel) -> str:
    lines: list[str] = []
    for iface in bp.interfaces:
        transport = iface.transport
        if isinstance(transport, NegotiatedTransport):
            transport = transport.transports[0]

        direction = iface.direction.value
        transport_type = getattr(transport, "type", "unknown")
        if hasattr(transport_type, "value"):
            transport_type = transport_type.value

        path_or_topic = (
            getattr(transport, "path", None)
            or getattr(transport, "topic", None)
            or getattr(transport, "symbol", None)
            or "—"
        )

        method = ""
        if isinstance(transport, HttpRestInterface):
            method = f" {transport.method.upper()}"

        schema_info = ""
        if isinstance(transport, HttpRestInterface):
            parts = []
            if transport.input_schema:
                parts.append(f"in:{transport.input_schema}")
            if transport.output_schema:
                parts.append(f"out:{transport.output_schema}")
            if parts:
                schema_info = f"  [{', '.join(parts)}]"

        lines.append(
            f"  {iface.id} ({direction}) {transport_type}{method} {path_or_topic}{schema_info}"
        )
        lines.append(f"    {iface.description}")

    return "\n".join(lines)


def _format_dependencies(bp: BlueprintModel) -> str:
    lines: list[str] = []
    for dep in bp.dependencies.services:
        req = "required" if dep.required else f"optional (fallback: {dep.fallback or '—'})"
        lines.append(f"  {dep.id} — {dep.role} ({req})")
    for api in bp.dependencies.external_apis:
        req = "required" if api.required else "optional"
        lines.append(f"  [ext] {api.name} — {api.purpose} ({req})")
    return "\n".join(lines)


def _format_failure_modes(bp: BlueprintModel) -> str:
    lines: list[str] = []
    for fm in bp.failure_modes:
        severity = fm.severity.value if hasattr(fm.severity, "value") else fm.severity
        lines.append(f"  [{severity}] {fm.condition}")
        lines.append(f"    Response: {fm.response}")
    return "\n".join(lines)


def _format_open_questions(bp: BlueprintModel) -> str:
    if not bp.intent or not bp.intent.open_questions:
        return ""
    lines: list[str] = []
    for q in bp.intent.open_questions:
        lines.append(f"  - {q}")
    return "\n".join(lines)


def _format_confidence(bp: BlueprintModel) -> str:
    lines: list[str] = []
    for section, level in bp.meta.confidence.model_dump().items():
        lines.append(f"  {section}: {level}")
    return "\n".join(lines)


_FORWARD_REVIEW_TASK_TEXT = """## Review Task

The mechanical pre-check above shows structural completeness — what is present \
or missing at a syntactic level. Your task is to assess SEMANTIC fidelity:

- For items the pre-check marked [FOUND]: does the implementation actually \
fulfill the blueprint's intent, or is it a superficial match?
- For items the pre-check marked [MISSING]: is this genuinely absent, or \
implemented in a way the pre-check couldn't detect?
- For dimensions 4-5 (intent coherence, open questions): apply your own \
judgment — these have no mechanical coverage.

Be specific: cite the blueprint claim and the contradicting (or absent) \
code evidence."""

_REVERSE_REVIEW_TASK_TEXT = """## Review Task

The mechanical pre-check above shows structural mapping between the draft \
blueprint's claims and the actual source code. Your task is to assess \
BLUEPRINT ACCURACY:

- For items the pre-check marked [FOUND]: does the blueprint's claim \
accurately and completely describe the code's behavior?
- For items the pre-check marked [MISSING]: is this a genuine over-claim \
in the blueprint, or is the feature implemented under a different name/path?
- Scan the AST summaries for code behavior NOT represented in the blueprint \
(undocumented endpoints, unmentioned dependencies, unlisted failure modes).
- For dimensions 4-5 (intent coherence, open questions): assess whether the \
blueprint's narrative accurately reflects the code's actual purpose.

Be specific: cite the code evidence and the inaccurate (or absent) \
blueprint claim."""


# ── Token budget guard ───────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Simple heuristic: ~4 chars per token for English text."""
    return len(text) // 4


def _apply_token_budget(
    full_prompt: str,
    blueprint: BlueprintModel,
    ast_summaries: Dict[str, ASTSummary],
    precheck_result: PreCheckResult,
    max_tokens: int,
) -> str:
    """
    Progressive truncation to fit within token budget.

    Priority (drop lowest first):
    1. Open questions
    2. Intent → purpose only (drop design_decisions)
    3. AST summaries → drop error_handlers, http_calls, gaia_imports
    4. Pre-check → summary line only
    5. Interfaces/failure modes → NEVER truncated
    """
    est = _estimate_tokens(full_prompt)
    if est <= max_tokens:
        return full_prompt

    dropped: list[str] = []

    # Level 1: Drop open questions
    if blueprint.intent and blueprint.intent.open_questions:
        full_prompt = full_prompt.replace(
            _format_open_questions(blueprint), "(truncated — open questions removed)"
        )
        dropped.append("open_questions")
        est = _estimate_tokens(full_prompt)
        if est <= max_tokens:
            logger.warning("Token budget: dropped %s, final ~%d tokens", dropped, est)
            return full_prompt

    # Level 2: Reduce intent to purpose only
    intent_full = _format_intent(blueprint)
    if intent_full and blueprint.intent:
        intent_minimal = blueprint.intent.purpose
        full_prompt = full_prompt.replace(intent_full, intent_minimal)
        dropped.append("intent_details")
        est = _estimate_tokens(full_prompt)
        if est <= max_tokens:
            logger.warning("Token budget: dropped %s, final ~%d tokens", dropped, est)
            return full_prompt

    # Level 3: Reduce AST summaries (strip targeted extractions)
    for filename, summary in ast_summaries.items():
        full_text = summary.to_prompt_text()
        # Create a stripped version without error_handlers, http_calls, gaia_imports
        stripped = ASTSummary(
            module_docstring=summary.module_docstring,
            classes=summary.classes,
            functions=summary.functions,
            endpoints=summary.endpoints,
            enums=summary.enums,
            constants=summary.constants,
            gaia_imports=[],
            error_handlers=[],
            http_calls=[],
            filename=summary.filename,
        )
        full_prompt = full_prompt.replace(full_text, stripped.to_prompt_text())
    dropped.append("ast_targeted_extractions")
    est = _estimate_tokens(full_prompt)
    if est <= max_tokens:
        logger.warning("Token budget: dropped %s, final ~%d tokens", dropped, est)
        return full_prompt

    # Level 4: Reduce pre-check to summary line only
    precheck_full = precheck_result.to_prompt_text()
    s = precheck_result.summary
    pct = (s.found / s.total * 100) if s.total else 0
    precheck_minimal = (
        f"Pre-check summary: {s.total} checks | "
        f"{s.found} found | {s.missing} missing | "
        f"{s.diverged} diverged | {pct:.1f}% complete"
    )
    full_prompt = full_prompt.replace(precheck_full, precheck_minimal)
    dropped.append("precheck_details")
    est = _estimate_tokens(full_prompt)

    logger.warning("Token budget: dropped %s, final ~%d tokens", dropped, est)
    return full_prompt
