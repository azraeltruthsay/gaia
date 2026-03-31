"""
Planning Orchestrator — multi-model collaborative planning pipeline.

When a planning task is detected, this orchestrator coordinates:
1. Prime (GPU): Generates plan phases via fragment_write
2. Idle model (Core CPU/GGUF): Reviews sketchpad, provides user feedback, suggests revisions
3. Validation: Reviews assembled plan before delivery

Architecture:
    Prime generates → writes to sketchpad → idle model reviews → user gets progress updates
    → Prime refines based on feedback → final validation → assembled plan delivered

Uses the CognitionPacket.reasoning.sketchpad as the shared workspace.
"""

import logging
from typing import Dict, Any, Generator

logger = logging.getLogger("GAIA.PlanningOrchestrator")


# Phase definitions for structured planning
PLAN_PHASES = [
    {"id": "requirements", "label": "Requirements & Scope", "prompt_suffix": "Define what needs to change, which services are affected, and the acceptance criteria."},
    {"id": "architecture", "label": "Architecture & File Changes", "prompt_suffix": "List specific file paths in candidates/, the functions to modify, and how they connect."},
    {"id": "implementation", "label": "Implementation Details", "prompt_suffix": "Write code examples for the key changes. Use fenced code blocks with language tags."},
    {"id": "testing", "label": "Testing & Rollout", "prompt_suffix": "Define test strategy, Docker restart/rebuild requirements, and promotion order."},
]


def run_planning_pipeline(
    user_request: str,
    prime_model,
    reviewer_model,
    packet,
    config,
    model_pool=None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Collaborative planning pipeline using Prime + idle reviewer.

    Yields streaming events:
        {"type": "token", "value": "..."} — text for the user
        {"type": "status", "value": "..."} — progress updates
        {"type": "phase_complete", "phase": "...", "content": "..."}
        {"type": "review", "value": "..."} — reviewer feedback shown to user
        {"type": "flush"} — flush signal

    Args:
        user_request: The original planning request
        prime_model: GPU Prime model for generation
        reviewer_model: Idle model (Core CPU) for review
        packet: CognitionPacket with sketchpad
        config: GAIA config
        model_pool: ModelPool for adapter management
    """
    from gaia_common.protocols.cognition_packet import Sketchpad

    logger.info("Planning orchestrator started — %d phases", len(PLAN_PHASES))
    yield {"type": "token", "value": "[(i) Planning mode — generating phased implementation plan...]\n\n"}
    yield {"type": "flush"}

    plan_fragments = []
    sketchpad = packet.reasoning.sketchpad

    for i, phase in enumerate(PLAN_PHASES):
        phase_num = i + 1
        total = len(PLAN_PHASES)

        # --- Status update to user ---
        yield {"type": "token", "value": f"**[Phase {phase_num}/{total}: {phase['label']}]**\n"}
        yield {"type": "flush"}

        # --- Prime generates this phase ---
        phase_prompt = _build_phase_prompt(user_request, phase, plan_fragments, i)
        logger.info("Phase %d/%d: %s — generating with Prime", phase_num, total, phase["id"])

        phase_content = _generate_with_model(prime_model, phase_prompt, config, max_tokens=1024)

        if not phase_content or len(phase_content.strip()) < 20:
            logger.warning("Phase %s produced empty/short content, retrying with emphasis", phase["id"])
            phase_prompt += "\n\nIMPORTANT: Provide detailed content with code examples. Do NOT be brief."
            phase_content = _generate_with_model(prime_model, phase_prompt, config, max_tokens=1024)

        # --- Write to sketchpad ---
        sketchpad.append(Sketchpad(
            slot=f"plan_phase_{phase['id']}",
            content=phase_content,
            content_type="markdown",
        ))
        plan_fragments.append({"phase": phase["label"], "content": phase_content})

        # --- Stream phase content to user ---
        yield {"type": "token", "value": phase_content + "\n\n"}
        yield {"type": "flush"}

        # --- Idle model reviews (if available) ---
        if reviewer_model and i < total - 1:  # Don't review the last phase
            review = _review_phase(reviewer_model, phase, phase_content, user_request, config)
            if review and len(review.strip()) > 20:
                yield {"type": "token", "value": f"*💡 Reviewer note: {review.strip()}*\n\n"}
                yield {"type": "flush"}

                # If reviewer suggests a significant revision, let Prime refine
                if any(kw in review.lower() for kw in ["missing", "should also", "don't forget", "important to add"]):
                    logger.info("Reviewer flagged revision needed for phase %s", phase["id"])
                    refinement = _generate_with_model(
                        prime_model,
                        f"The reviewer noted: {review}\n\nRefine this section:\n{phase_content}\n\nAdd what was missing:",
                        config, max_tokens=512
                    )
                    if refinement and len(refinement.strip()) > 20:
                        yield {"type": "token", "value": f"**[Revision]** {refinement.strip()}\n\n"}
                        yield {"type": "flush"}
                        plan_fragments[-1]["content"] += "\n\n" + refinement

    # --- Final validation ---
    yield {"type": "token", "value": "**[Validation]**\n"}
    yield {"type": "flush"}

    assembled = _assemble_plan(plan_fragments)

    if reviewer_model:
        validation = _validate_plan(reviewer_model, assembled, user_request, config)
        if validation:
            yield {"type": "token", "value": f"*{validation.strip()}*\n\n"}
            yield {"type": "flush"}

    # --- Write final assembled plan to sketchpad ---
    sketchpad.append(Sketchpad(
        slot="plan_final",
        content=assembled,
        content_type="markdown",
    ))

    logger.info("Planning orchestrator complete — %d phases, %d chars", len(plan_fragments), len(assembled))
    yield {"type": "flush"}


def _build_phase_prompt(user_request: str, phase: dict, previous_phases: list, phase_idx: int) -> str:
    """Build a focused prompt for one phase of the plan."""
    context = ""
    if previous_phases:
        context = "Previous phases already covered:\n"
        for p in previous_phases:
            context += f"- {p['phase']}: {p['content'][:200]}...\n"
        context += "\n"

    return (
        f"You are writing Phase {phase_idx + 1} of an implementation plan.\n"
        f"User request: {user_request}\n\n"
        f"{context}"
        f"Write the **{phase['label']}** section.\n"
        f"{phase['prompt_suffix']}\n\n"
        f"Use markdown headers, bullet points, and code blocks. Be specific and detailed."
    )


def _generate_with_model(model, prompt: str, config, max_tokens: int = 1024) -> str:
    """Generate text from a model using chat completion."""
    try:
        messages = [
            {"role": "system", "content": "You are GAIA, a sovereign AI architect. Write detailed, specific implementation plans with file paths and code."},
            {"role": "user", "content": prompt},
        ]
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
        )
        if isinstance(result, dict):
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ""
    except Exception as e:
        logger.warning("Generation failed: %s", e)
        return ""


def _review_phase(reviewer_model, phase: dict, content: str, user_request: str, config) -> str:
    """Have the idle model review a plan phase."""
    try:
        prompt = (
            f"Review this implementation plan phase for completeness and correctness.\n"
            f"Original request: {user_request}\n"
            f"Phase: {phase['label']}\n\n"
            f"Content:\n{content}\n\n"
            f"In 1-2 sentences: Is anything missing or incorrect? If it looks good, say 'Looks complete.'"
        )
        messages = [
            {"role": "system", "content": "You are a code reviewer. Be concise — 1-2 sentences only."},
            {"role": "user", "content": prompt},
        ]
        result = reviewer_model.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.2,
        )
        if isinstance(result, dict):
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ""
    except Exception as e:
        logger.debug("Review failed: %s", e)
        return ""


def _validate_plan(reviewer_model, assembled: str, user_request: str, config) -> str:
    """Final validation of the assembled plan."""
    try:
        prompt = (
            f"Validate this implementation plan against the original request.\n"
            f"Request: {user_request}\n\n"
            f"Plan:\n{assembled[:2000]}\n\n"
            f"Check: Does it cover all requested areas? Are file paths specific? Are there code examples?\n"
            f"Respond with: '✅ Plan validated — covers all areas.' or '⚠️ Missing: [what]'"
        )
        messages = [
            {"role": "system", "content": "You are a plan validator. Be concise."},
            {"role": "user", "content": prompt},
        ]
        result = reviewer_model.create_chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.1,
        )
        if isinstance(result, dict):
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ""
    except Exception as e:
        logger.debug("Validation failed: %s", e)
        return ""


def _assemble_plan(fragments: list) -> str:
    """Assemble plan fragments into final document."""
    parts = []
    for frag in fragments:
        parts.append(f"## {frag['phase']}\n\n{frag['content']}")
    return "\n\n".join(parts)
