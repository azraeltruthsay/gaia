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

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional

logger = logging.getLogger("GAIA.PlanningOrchestrator")


# Phase definitions for structured planning
PLAN_PHASES = [
    {"id": "requirements", "label": "Requirements & Scope", "max_tokens": 512,
     "prompt_suffix": "Define what needs to change, which services are affected, and the acceptance criteria."},
    {"id": "architecture", "label": "Architecture & File Changes", "max_tokens": 1024,
     "prompt_suffix": "List specific file paths in candidates/, the functions to modify, and how they connect."},
    {"id": "implementation", "label": "Implementation Details", "max_tokens": 2048,
     "prompt_suffix": (
         "For EACH file that needs to change, write it in this exact format:\n\n"
         "**`candidates/path/to/file.py`**: Brief description of the change\n"
         "```python\n"
         "# The new or modified code for this file\n"
         "```\n\n"
         "This format is required — the executor parses file paths and their adjacent code blocks. "
         "Match existing patterns from the codebase context. Every file MUST have a code block."
     )},
    {"id": "testing", "label": "Testing & Rollout", "max_tokens": 512,
     "prompt_suffix": "Define test strategy, Docker restart/rebuild requirements, and promotion order."},
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

        phase_max_tokens = phase.get("max_tokens", 1024)
        phase_content = _generate_with_model(prime_model, phase_prompt, config, max_tokens=phase_max_tokens)

        if not phase_content or len(phase_content.strip()) < 20:
            logger.warning("Phase %s produced empty/short content, retrying with emphasis", phase["id"])
            phase_prompt += "\n\nIMPORTANT: Provide detailed content with code examples. Do NOT be brief."
            phase_content = _generate_with_model(prime_model, phase_prompt, config, max_tokens=phase_max_tokens)

        # ── Stream Observer: validate phase output ──
        observations = _observe_phase_output(phase_content, codebase_context)
        if observations:
            phase_content = _apply_observations(phase_content, observations)
            for obs in observations:
                yield {"type": "token", "value": f"*🔍 Observer: {obs}*\n"}
            yield {"type": "flush"}

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

    # ── Plan Execution (dry-run) ──
    # Extract actionable file changes and simulate execution
    try:
        from gaia_core.cognition.plan_executor import extract_file_changes, execute_plan_phase

        all_changes = extract_file_changes(assembled)
        if all_changes:
            yield {"type": "token", "value": f"\n**[Execution Preview: {len(all_changes)} file(s) identified]**\n"}
            yield {"type": "flush"}

            for change in all_changes:
                action_icon = "📝" if change["action"] == "modify" else "📄"
                desc = change["description"][:80] if change["description"] else change["action"]
                yield {"type": "token", "value": f"  {action_icon} `{change['file']}` — {desc}\n"}
            yield {"type": "flush"}

            # Dry-run execution: validate without writing
            yield {"type": "token", "value": "\n**[Dry-run validation]**\n"}
            yield {"type": "flush"}

            for event in execute_plan_phase(
                changes=all_changes,
                prime_model=prime_model,
                config=config,
                dry_run=True,
            ):
                yield event

            # Generate reverse-string approval challenge
            try:
                from gaia_common.utils.approval_challenge import create_challenge, format_challenge_prompt
                challenge = create_challenge(
                    action="execute_plan",
                    context=f"{len(all_changes)} file(s) to write to candidates/",
                )
                yield {"type": "token", "value": "\n" + format_challenge_prompt(challenge) + "\n\n"}
                yield {"type": "flush"}
            except Exception as _chal_err:
                logger.warning("Approval challenge failed: %s", _chal_err)
                yield {"type": "token", "value": "\n*Plan ready — approval system unavailable.*\n\n"}
                yield {"type": "flush"}

            # Store changes in sketchpad for later execution
            sketchpad.append(Sketchpad(
                slot="plan_changes",
                content=json.dumps(all_changes, default=str),
                content_type="json",
            ))
        else:
            yield {"type": "token", "value": "\n*No specific file changes extracted from plan. Refine file paths for execution.*\n\n"}
            yield {"type": "flush"}
    except Exception as _exec_err:
        logger.warning("Plan execution preview failed: %s", _exec_err, exc_info=True)
        yield {"type": "token", "value": f"\n*Execution preview skipped: {_exec_err}*\n\n"}
        yield {"type": "flush"}

    logger.info("Planning orchestrator complete — %d phases, %d chars, %d chars context, %d changes",
                len(plan_fragments), len(assembled), len(codebase_context),
                len(all_changes) if 'all_changes' in dir() else 0)
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
    MAX_CONTEXT_CHARS = 8000  # Budget for context — contracts are already compact

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

    # ── 2. Per-file contracts (AST-derived, cached, lightweight) ──
    try:
        from gaia_common.utils.file_contracts import get_contracts_for_planning
        file_candidates = _find_relevant_source_files(user_request)

        if file_candidates:
            contracts_text = get_contracts_for_planning(file_candidates[:8])
            if contracts_text and len(contracts_text) > 20:
                context_parts.append(contracts_text)
                total_chars += len(contracts_text)
                logger.info("File contracts assembled for %d files (%d chars)",
                           len(file_candidates), len(contracts_text))
    except Exception as e:
        logger.debug("File contracts not available: %s", e)

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
    """
    Find source files relevant to the planning request by scanning
    the actual codebase, not from a hardcoded lookup table.

    Strategy:
    1. Identify which services are mentioned in the request
    2. Find key files in those services (main.py, routes, tools, etc.)
    3. Include shared protocol files for cross-service features
    """
    base = Path("/gaia/GAIA_Project") if Path("/gaia/GAIA_Project/candidates").exists() else Path(".")
    candidates_dir = base / "candidates"
    request_lower = user_request.lower()
    results = []
    seen = set()

    # Discover which service directories exist
    service_dirs = []
    if candidates_dir.exists():
        for child in sorted(candidates_dir.iterdir()):
            if child.is_dir() and child.name.startswith("gaia-"):
                service_dirs.append(child)

    # Score each service — direct name match gets priority, contract overlap as secondary
    scored = []
    request_words = set(request_lower.split()) - {
        "a", "the", "to", "for", "and", "in", "of", "create", "detailed",
        "implementation", "plan", "adding", "support",  # generic planning words
    }

    for svc_dir in service_dirs:
        svc_name = svc_dir.name.replace("gaia-", "")
        score = 0

        # Direct name match — strong signal
        if svc_name in request_lower or svc_dir.name in request_lower:
            score += 10

        # Contract keyword overlap — weaker signal, skip generic words
        contract_path = base / "contracts" / "services" / f"{svc_dir.name}.yaml"
        if contract_path.exists():
            try:
                contract_text = contract_path.read_text().lower()
                overlap = sum(1 for w in request_words if len(w) > 3 and w in contract_text)
                score += overlap
            except Exception:
                pass

        if score >= 3:  # Need meaningful overlap, not just generic words
            scored.append((score, svc_dir))

    # Sort by score descending, take top services
    scored.sort(key=lambda x: -x[0])
    for score, svc_dir in scored[:4]:
        _add_key_files(svc_dir, results, seen)

    # Cross-service infrastructure — always include if they exist
    # These are the shared layers that most features touch
    infra_files = [
        candidates_dir / "gaia-common" / "gaia_common" / "protocols" / "cognition_packet.py",
        candidates_dir / "gaia-mcp" / "gaia_mcp" / "tools.py",
    ]
    for infra in infra_files:
        if infra.exists() and str(infra) not in seen:
            results.append(str(infra))
            seen.add(str(infra))

    return results[:12]


def _add_key_files(svc_dir: Path, results: List[str], seen: set, max_per_service: int = 6):
    """Add the most important files from a service directory."""
    # Priority order: limited-count patterns first, then broad globs
    priority_patterns = [
        "**/main.py",           # Entry point (1 file)
        "**/tools.py",          # MCP tools (1 file)
        "static/app.js",        # Frontend (1 file)
        "static/index.html",    # Frontend HTML (1 file)
        "**/models.py",         # Data models (1 file)
        "**/protocols/*.py",    # Shared protocols
        "**/routes/*.py",       # API routes (many files — last so cap applies)
    ]

    added = 0
    for pattern in priority_patterns:
        if added >= max_per_service:
            break
        for match in svc_dir.glob(pattern):
            if str(match) in seen:
                continue
            if any(skip in str(match) for skip in ["__pycache__", ".bak", "test_", "__init__"]):
                continue
            results.append(str(match))
            seen.add(str(match))
            added += 1
            if added >= max_per_service:
                break


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
        # Extract file paths from contracts and shorten to candidates/ relative form
        import re as _re
        raw_paths = _re.findall(r'candidates/gaia-[^\s\n*]+\.(?:py|js|html|css|yaml)', codebase_context)
        if not raw_paths:
            raw_paths = _re.findall(r'\*\*(/[^\*]+\.(?:py|js|html|css|yaml))\*\*', codebase_context)
            # Shorten /gaia/GAIA_Project/candidates/... to candidates/...
            raw_paths = [p.split("candidates/", 1)[-1] if "candidates/" in p else p for p in raw_paths]
            raw_paths = [f"candidates/{p}" if not p.startswith("candidates/") else p for p in raw_paths]

        # Deduplicate
        contract_paths = list(dict.fromkeys(raw_paths))

        path_list = ""
        if contract_paths:
            path_list = (
                "\n**THESE are the real files — use ONLY these paths:**\n"
                "```\n"
            )
            for p in contract_paths[:8]:
                path_list += f"{p}\n"
            path_list += "```\n\n"

        codebase_section = (
            f"\n## Actual Codebase Context\n"
            f"{path_list}"
            f"{codebase_context}\n\n"
        )

    return (
        f"You are writing Phase {phase_idx + 1} of an implementation plan for GAIA.\n"
        f"User request: {user_request}\n\n"
        f"{context}"
        f"{codebase_section}"
        f"Write the **{phase['label']}** section.\n"
        f"{phase['prompt_suffix']}\n\n"
        f"CRITICAL: Only reference files from the codebase context above. "
        f"Do NOT invent file paths. All paths must start with `candidates/gaia-`. "
        f"Use markdown headers, bullet points, and code blocks."
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
            f"Are there code examples? Do the patterns match the codebase context?\n"
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


# ── Stream Observer for Planning ──────────────────────────────────────────

def _observe_phase_output(content: str, codebase_context: str) -> List[str]:
    """
    Observation of phase output — checks against actual codebase context.

    Rather than hardcoding "right" and "wrong" patterns, compares the
    generated content against what was discovered in the codebase context.
    """
    observations = []

    # ── Check 1: Candidate-first workflow (operational rule, not leading) ──
    if "candidates/" not in content and any(
        kw in content.lower() for kw in ["file:", "path:", "modify", "create", "update"]
    ):
        observations.append("No candidates/ prefix found — all changes should go to candidates/ first")

    # ── Check 2: Phase too short for implementation ──
    if len(content.strip()) < 200:
        observations.append(f"Phase content is short ({len(content.strip())} chars) — may need more detail")

    # ── Check 3: Verify paths against codebase context ──
    # If we assembled real file contracts, check if the plan references them
    if codebase_context and "candidates/" in codebase_context:
        import re
        # Extract real paths from context
        real_paths = set(re.findall(r'candidates/gaia-\w+/[\w/._-]+\.\w+', codebase_context))
        # Extract paths from generated content
        gen_paths = set(re.findall(r'candidates/[\w/._-]+\.\w+', content))

        if gen_paths and real_paths:
            ungrounded = gen_paths - real_paths
            if ungrounded and len(ungrounded) > len(gen_paths) * 0.5:
                examples = ", ".join(list(ungrounded)[:3])
                observations.append(f"Some paths not found in codebase context: {examples}")

    return observations


def _apply_observations(content: str, observations: List[str]) -> str:
    """
    Apply corrections based on observations.
    Currently observation-only — no auto-replacement.
    The reviewer model handles corrections via revision suggestions.
    """
    # No auto-replacement — let the reviewer and Prime handle corrections
    # through the natural revision cycle instead of hardcoded string replacement.
    return content
