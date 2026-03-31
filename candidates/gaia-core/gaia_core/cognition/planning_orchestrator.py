"""
Planning Orchestrator — multi-model collaborative planning with RAG + self-exploration.

Pipeline:
1. Context Assembly: Explore codebase, ingest contracts via CFR, AST-summarize source files
2. Prime (GPU): Generates plan phases grounded in real codebase context
3. Idle model (Core CPU/GGUF): Reviews sketchpad, provides user feedback, suggests revisions
4. Validation: Reviews assembled plan before delivery

The assembled context is cached in the CognitionPacket sketchpad and can be
persisted as a KV prefix segment for future planning requests.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Generator

logger = logging.getLogger("GAIA.PlanningOrchestrator")


# Phase definitions for structured planning
PLAN_PHASES = [
    {"id": "requirements", "label": "Requirements & Scope", "prompt_suffix": "Define what needs to change, which services are affected, and the acceptance criteria."},
    {"id": "architecture", "label": "Architecture & File Changes", "prompt_suffix": "List specific file paths in candidates/, the functions to modify, and how they connect. Use the codebase context provided above."},
    {"id": "implementation", "label": "Implementation Details", "prompt_suffix": "Write code examples for the key changes. Match the existing code patterns shown in the codebase context. Use fenced code blocks with language tags."},
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
    Collaborative planning pipeline with RAG context assembly.

    Yields streaming events for real-time user feedback.
    """
    from gaia_common.protocols.cognition_packet import Sketchpad

    logger.info("Planning orchestrator started — %d phases", len(PLAN_PHASES))
    yield {"type": "token", "value": "[(i) Planning mode activated]\n\n"}
    yield {"type": "flush"}

    plan_fragments = []
    sketchpad = packet.reasoning.sketchpad

    # ── Phase 0: Context Assembly (RAG + self-exploration + CFR) ──
    yield {"type": "token", "value": "**[Exploring codebase...]**\n"}
    yield {"type": "flush"}

    codebase_context = _assemble_codebase_context(user_request, config)

    if codebase_context:
        context_summary = f"{len(codebase_context)} chars of codebase context assembled"
        yield {"type": "token", "value": f"*Explored: {context_summary}*\n\n"}
        yield {"type": "flush"}

        # Write context to sketchpad for persistence
        sketchpad.append(Sketchpad(
            slot="planning_context",
            content=codebase_context,
            content_type="markdown",
        ))
    else:
        codebase_context = ""
        yield {"type": "token", "value": "*No additional codebase context found — planning from general knowledge.*\n\n"}
        yield {"type": "flush"}

    # ── Phases 1-4: Generate with context ──
    for i, phase in enumerate(PLAN_PHASES):
        phase_num = i + 1
        total = len(PLAN_PHASES)

        yield {"type": "token", "value": f"**[Phase {phase_num}/{total}: {phase['label']}]**\n"}
        yield {"type": "flush"}

        phase_prompt = _build_phase_prompt(user_request, phase, plan_fragments, i, codebase_context)
        logger.info("Phase %d/%d: %s — generating with Prime", phase_num, total, phase["id"])

        phase_content = _generate_with_model(prime_model, phase_prompt, config, max_tokens=1024)

        if not phase_content or len(phase_content.strip()) < 20:
            logger.warning("Phase %s produced empty/short content, retrying with emphasis", phase["id"])
            phase_prompt += "\n\nIMPORTANT: Provide detailed content with code examples. Do NOT be brief."
            phase_content = _generate_with_model(prime_model, phase_prompt, config, max_tokens=1024)

        sketchpad.append(Sketchpad(
            slot=f"plan_phase_{phase['id']}",
            content=phase_content,
            content_type="markdown",
        ))
        plan_fragments.append({"phase": phase["label"], "content": phase_content})

        yield {"type": "token", "value": phase_content + "\n\n"}
        yield {"type": "flush"}

        # Idle model review (skip last phase)
        if reviewer_model and i < total - 1:
            review = _review_phase(reviewer_model, phase, phase_content, user_request, config)
            if review and len(review.strip()) > 20:
                yield {"type": "token", "value": f"*💡 Reviewer note: {review.strip()}*\n\n"}
                yield {"type": "flush"}

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

    # ── Validation ──
    yield {"type": "token", "value": "**[Validation]**\n"}
    yield {"type": "flush"}

    assembled = _assemble_plan(plan_fragments)

    if reviewer_model:
        validation = _validate_plan(reviewer_model, assembled, user_request, config)
        if validation:
            yield {"type": "token", "value": f"*{validation.strip()}*\n\n"}
            yield {"type": "flush"}

    sketchpad.append(Sketchpad(
        slot="plan_final",
        content=assembled,
        content_type="markdown",
    ))

    logger.info("Planning orchestrator complete — %d phases, %d chars, %d chars context",
                len(plan_fragments), len(assembled), len(codebase_context))
    yield {"type": "flush"}


# ── Context Assembly ──────────────────────────────────────────────────────

def _assemble_codebase_context(user_request: str, config) -> str:
    """
    Explore the actual codebase to build planning context.

    1. Find relevant service contracts
    2. AST-summarize relevant source files
    3. CFR-compress if context too large
    4. Return assembled context string
    """
    context_parts = []
    total_chars = 0
    MAX_CONTEXT_CHARS = 4000  # Budget for context in planning prompt

    # ── 1. Service contracts ──
    # Find contracts directory — varies by mount configuration
    contracts_dir = None
    for p in [Path("/app/contracts/services"),
              Path("/gaia/GAIA_Project/contracts/services"),
              Path("contracts/services")]:
        if p.exists():
            contracts_dir = p
            break

    if contracts_dir.exists():
        # Find contracts matching the user request
        request_lower = user_request.lower()
        relevant_contracts = []

        for yaml_file in sorted(contracts_dir.glob("*.yaml")):
            service_name = yaml_file.stem
            # Match service names mentioned in request or related to common features
            keywords_for_service = {
                "gaia-web": ["web", "dashboard", "frontend", "ui", "upload", "chat", "route"],
                "gaia-mcp": ["mcp", "tool", "file", "read", "write", "execute", "attachment"],
                "gaia-core": ["core", "cognition", "pipeline", "packet", "prompt", "plan"],
                "gaia-common": ["common", "protocol", "packet", "shared", "utility"],
                "gaia-study": ["study", "train", "vector", "embed", "index"],
                "gaia-orchestrator": ["orchestrator", "gpu", "lifecycle", "focus"],
                "gaia-audio": ["audio", "voice", "stt", "tts"],
                "gaia-doctor": ["doctor", "health", "immune", "monitor"],
            }
            service_keywords = keywords_for_service.get(service_name, [service_name.replace("gaia-", "")])
            if any(kw in request_lower for kw in service_keywords):
                relevant_contracts.append(yaml_file)

        for contract_path in relevant_contracts[:4]:  # Max 4 contracts
            try:
                content = contract_path.read_text()
                # Truncate large contracts
                if len(content) > 800:
                    content = content[:800] + "\n... (truncated)"
                context_parts.append(f"### Contract: {contract_path.stem}\n```yaml\n{content}\n```")
                total_chars += len(content)
            except Exception as e:
                logger.debug("Failed to read contract %s: %s", contract_path, e)

    # ── 2. AST summaries of relevant source files ──
    try:
        from gaia_common.utils.ast_summarizer import summarize_file
        # Map request keywords to likely source files
        file_candidates = _find_relevant_source_files(user_request)

        for src_path in file_candidates[:5]:  # Max 5 files
            if total_chars > MAX_CONTEXT_CHARS:
                break
            try:
                source = Path(src_path).read_text()
                summary = summarize_file(source, filename=str(src_path))
                prompt_text = summary.to_prompt_text()
                if prompt_text and len(prompt_text) > 20:
                    context_parts.append(f"### Source: {src_path}\n{prompt_text}")
                    total_chars += len(prompt_text)
            except Exception as e:
                logger.debug("Failed to summarize %s: %s", src_path, e)
    except ImportError:
        logger.debug("AST summarizer not available")

    # ── 3. CFR synthesis for large documents ──
    # If any context part is too large, compress via CFR
    if total_chars > MAX_CONTEXT_CHARS:
        try:
            from gaia_common.utils.cfr_manager import CFRManager
            cfr = CFRManager()
            # Write combined context to temp file, ingest, synthesize
            combined = "\n\n".join(context_parts)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, dir='/tmp') as f:
                f.write(combined)
                tmp_path = f.name

            result = cfr.ingest(tmp_path, doc_id=f"planning_context_{int(time.time())}")
            if result.get("ok"):
                doc_id = result["doc_id"]
                synth = cfr.synthesize(doc_id)
                if synth.get("synthesis"):
                    context_parts = [f"### Codebase Context (CFR synthesized)\n{synth['synthesis']}"]
                    total_chars = len(synth['synthesis'])

            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            logger.debug("CFR synthesis failed: %s", e)

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)


def _find_relevant_source_files(user_request: str) -> List[str]:
    """Find source files relevant to the planning request."""
    request_lower = user_request.lower()
    candidates = []

    # Map keywords to actual GAIA source files
    # Resolve base path — works inside container or on host
    base = Path("/gaia/GAIA_Project") if Path("/gaia/GAIA_Project/candidates").exists() else Path(".")
    c = str(base) + "/candidates"

    file_map = {
        "attachment": [
            f"{c}/gaia-web/gaia_web/routes/files.py",
            f"{c}/gaia-web/static/app.js",
            f"{c}/gaia-mcp/gaia_mcp/tools.py",
            f"{c}/gaia-common/gaia_common/protocols/cognition_packet.py",
        ],
        "upload": [
            f"{c}/gaia-web/gaia_web/routes/files.py",
            f"{c}/gaia-web/static/app.js",
        ],
        "chat": [
            f"{c}/gaia-web/static/app.js",
            f"{c}/gaia-web/static/index.html",
        ],
        "dashboard": [
            f"{c}/gaia-web/static/app.js",
            f"{c}/gaia-web/static/index.html",
            f"{c}/gaia-web/static/style.css",
        ],
        "mcp": [
            f"{c}/gaia-mcp/gaia_mcp/tools.py",
        ],
        "tool": [
            f"{c}/gaia-mcp/gaia_mcp/tools.py",
            f"{c}/gaia-common/gaia_common/utils/tools_registry.py",
        ],
        "packet": [
            f"{c}/gaia-common/gaia_common/protocols/cognition_packet.py",
        ],
        "prompt": [
            f"{c}/gaia-core/gaia_core/utils/prompt_builder.py",
        ],
        "route": [
            f"{c}/gaia-web/gaia_web/routes/system.py",
            f"{c}/gaia-web/gaia_web/routes/hooks.py",
        ],
        "web": [
            f"{c}/gaia-web/gaia_web/main.py",
            f"{c}/gaia-web/static/app.js",
        ],
        "storage": [
            f"{c}/gaia-mcp/gaia_mcp/tools.py",
        ],
    }

    seen = set()
    for keyword, files in file_map.items():
        if keyword in request_lower:
            for f in files:
                if f not in seen and Path(f).exists():
                    candidates.append(f)
                    seen.add(f)

    # Always include cognition_packet for cross-service features
    pkt_path = f"{c}/gaia-common/gaia_common/protocols/cognition_packet.py"
    if pkt_path not in seen and Path(pkt_path).exists():
        candidates.append(pkt_path)

    return candidates


# ── Phase Generation ──────────────────────────────────────────────────────

def _build_phase_prompt(user_request: str, phase: dict, previous_phases: list,
                        phase_idx: int, codebase_context: str = "") -> str:
    """Build a focused prompt for one phase of the plan."""
    context = ""
    if previous_phases:
        context = "Previous phases already covered:\n"
        for p in previous_phases:
            # Include enough context for continuity without bloating
            summary = p['content'][:300] + "..." if len(p['content']) > 300 else p['content']
            context += f"- **{p['phase']}**: {summary}\n"
        context += "\n"

    codebase_section = ""
    if codebase_context:
        codebase_section = (
            f"\n## Actual Codebase Context\n"
            f"Use these REAL file paths and patterns — do not invent paths.\n\n"
            f"{codebase_context}\n\n"
        )

    return (
        f"You are writing Phase {phase_idx + 1} of an implementation plan for GAIA.\n"
        f"User request: {user_request}\n\n"
        f"{context}"
        f"{codebase_section}"
        f"Write the **{phase['label']}** section.\n"
        f"{phase['prompt_suffix']}\n\n"
        f"IMPORTANT: Use the ACTUAL file paths from the codebase context above. "
        f"All changes go to candidates/ first. Use markdown headers, bullet points, and code blocks."
    )


def _generate_with_model(model, prompt: str, config, max_tokens: int = 1024) -> str:
    """Generate text from a model using chat completion."""
    try:
        messages = [
            {"role": "system", "content": (
                "You are GAIA, a sovereign AI architect. Write detailed implementation plans "
                "using the ACTUAL file paths and code patterns from your codebase. "
                "All changes go to candidates/ first. Use Python (FastAPI, Alpine.js) not React/Java."
            )},
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
            f"Content:\n{content[:1500]}\n\n"
            f"In 1-2 sentences: Is anything missing or incorrect? "
            f"Check: Are file paths specific (in candidates/)? Are there code examples? "
            f"If it looks good, say 'Looks complete.'"
        )
        messages = [
            {"role": "system", "content": "You are a code reviewer for the GAIA project. Be concise — 1-2 sentences only."},
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
            f"Check: Does it cover all requested areas? Are file paths in candidates/? "
            f"Are there code examples? Does it use Python/FastAPI/Alpine.js (not React/Java)?\n"
            f"Respond with: '✅ Plan validated — covers all areas.' or '⚠️ Missing: [what]'"
        )
        messages = [
            {"role": "system", "content": "You are a plan validator for the GAIA project. Be concise."},
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
