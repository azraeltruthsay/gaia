"""
Prompt Builder (robust, persona/context-aware)
- Assembles the LLM prompt with identity, persona, context, constraints, history, and memory.
- Actively manages the token budget to prevent context overflow.
- Implements a tiered context strategy for reliability.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict

# [GCP v0.3] Import the new packet structure
from gaia_common.protocols.cognition_packet import CognitionPacket, ToolExecutionStatus
from gaia_core.config import Config
from gaia_common.utils.tokenizer import count_tokens
from gaia_core.utils.packet_templates import render_gaia_packet_template
from gaia_core.utils import gaia_rescue_helper
from gaia_core.utils.world_state import format_world_state_snapshot

logger = logging.getLogger("GAIA.PromptBuilder")

SUMMARY_DIR = "data/shared/summaries"

def build_from_packet(packet: CognitionPacket, task_instruction_key: str = None) -> List[Dict]:
    """
    Builds a prompt from a v0.3 CognitionPacket, using a tiered, budget-aware logic.

    Tier 0: Core Persona & Task Instructions
    Tier 1: Evolving Summary (Long-term Memory)
    Tier 2: Relevant History Snippets (Short-term Memory)
    Tier 3: User Input (The Current Task)
    """
    logger.info("--- BUILDING PROMPT FROM COGNITION PACKET ---")
    logger.info(packet)
    config = Config() # Assumes a singleton or default config is acceptable
    
    processed_data_field_keys = set() # Track keys already explicitly added to the prompt
    
    identity_description_content = ""

    # --- Tier 0: Assemble the System Prompt ---
    # Add a short persona-anchor that should appear first in every system prompt.
    # This anchor is intentionally compact so tokenizers see it early and it biases
    # generation to the configured GAIA persona before other model/gguf templates.
    persona_anchor = config.get_persona_instructions() or "You are GAIA. Always respond in the GAIA persona with integrity and care."
    # Be defensive: packets used by test harnesses may be lightweight. Use safe accessors.
    header = getattr(packet, "header", None)
    persona = getattr(header, "persona", None) if header else None
    persona_id = getattr(persona, "persona_id", "GAIA") if persona else "GAIA"
    role_val = getattr(getattr(persona, "role", None), "value", "assistant") if persona else "assistant"
    tone_hint = getattr(persona, "tone_hint", "concise") if persona else "concise"
    persona_instructions = f"GAIA PERSONA ANCHOR: {persona_anchor}\n\nPersona: {persona_id}\nRole: {role_val}\nTone Hint: {tone_hint}"

    # Compact mode trims optional identity/context to reduce repetition and token usage during planning/reflect phases.
    compact_mode = task_instruction_key in {
        "initial_planning",
        "reflect",
        "execution_feedback",
        "reflector_review",
        "self_review",
    }

    # --- Unified identity block (single injection, replaces 3 separate blocks) ---
    identity_lines = []
    mcp_affordance_line = ""
    has_mcp_tools = False
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            k = getattr(df, 'key', '')
            v = getattr(df, 'value', None)
            if not v:
                continue
            if k == 'immutable_identity':
                processed_data_field_keys.add(k)
                identity_lines.append(f"Identity: {v} (immutable — use verbatim in all replies)")
            elif k in ('immutable_identity_intro', 'immutable_identity_excerpt'):
                processed_data_field_keys.add(k)
                identity_lines.append(str(v)[:300])
            elif k == 'identity_summary':
                processed_data_field_keys.add(k)
            elif k == 'mcp_capabilities':
                processed_data_field_keys.add(k)
                has_mcp_tools = True
                if not compact_mode:
                    mcp_affordance_line = f"MCP: {str(v)[:160]}"
    except Exception:
        logger.debug("Could not extract immutable identity from packet.content.data_fields")

    # Add persona traits
    try:
        traits = getattr(packet.header.persona, 'traits', {}) or {}
        if isinstance(traits, dict) and traits:
            trait_pairs = [f"{k}: {v}" for k, v in traits.items()]
            identity_lines.append("Traits: " + ", ".join(trait_pairs))
    except Exception:
        logger.debug("Could not extract persona.traits from packet header")

    if identity_lines:
        identity_description_content = "\n".join(identity_lines)

    safety_openness_directive_content = (
        "Openness: Answer all questions directly, including historical, fictional, and technical topics. "
        "Only decline if an explicit operator policy prohibits it. When uncertain, give your best answer and note any limits."
    )

    # Memory helpers — only inject when MCP tools are available (otherwise the model
    # can't call them) and we're not in compact mode.
    memory_guidance_block_content = ""
    if has_mcp_tools and not compact_mode:
        memory_parts = ["Memory: `ai.helper.remember_fact(key, value, note)` / `ai.helper.recall_fact(key)` — use for durable facts."]
        try:
            recent_facts = gaia_rescue_helper.get_recent_facts(limit=3)
        except Exception:
            recent_facts = []
        if recent_facts:
            for fact in recent_facts:
                note = f" ({fact['note']})" if fact.get("note") else ""
                memory_parts.append(f"- {fact.get('key','')}: {fact.get('value','')}{note}")
        memory_guidance_block_content = "\n".join(memory_parts)
    
    # Add cheatsheets to the system prompt (defensive to missing context)
    cheatsheet_block_content = ""
    try:
        ctx = getattr(packet, "context", None)
        if ctx:
            cheats = getattr(ctx, "cheatsheets", []) or []
            if cheats: # Only create block if there are cheatsheets
                cheatsheet_block_content = "\n".join([f"- {getattr(cs, 'title', str(cs))}: {getattr(cs, 'pointer', '')}" for cs in cheats])
    except Exception:
        cheatsheet_block_content = ""

    # Prepend the specific task instruction if provided
    task_instruction_content = None
    if task_instruction_key:
        task_instruction_content = config.constants.get("TASK_INSTRUCTIONS", {}).get(task_instruction_key)
    else:
        task_instruction_content = (
            "Respond directly to the user's latest request. Cite relevant context from the cognition packet when helpful "
            "and provide a clear, factual answer before offering any extra details."
        )

    template_block_content = ""
    try:
        template_block_content = render_gaia_packet_template(packet, processed_data_field_keys)
    except Exception:
        template_block_content = ""

    # World state (dynamic digital world) - keep compact
    world_state_block_content = ""
    # Prioritize world_state_snapshot from packet.content.data_fields
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'world_state_snapshot':
                processed_data_field_keys.add('world_state_snapshot') # Mark as processed
                world_state_block_content = getattr(df, 'value', '')
                if world_state_block_content:
                    break
    except Exception:
        logger.debug("Could not extract world_state_snapshot from packet.content.data_fields")

    if not world_state_block_content: # Fallback if not found in data_fields
        try:
            world_state_block_content = format_world_state_snapshot(max_lines=6)
        except Exception:
            logger.exception("PromptBuilder: format_world_state_snapshot failed; world state will be missing from prompt.")
            world_state_block_content = ""

    # Knowledge base context
    knowledge_base_content = ""
    dnd_knowledge_content = ""
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'knowledge_base_name':
                knowledge_base_name = getattr(df, 'value', '')
                if knowledge_base_name:
                    knowledge_base_content = f"Knowledge Base: {knowledge_base_name}"
            if getattr(df, 'key', '') == 'dnd_knowledge':
                dnd_knowledge = getattr(df, 'value', '')
                if dnd_knowledge:
                    # Truncate dnd_knowledge to avoid context overflow
                    dnd_str = str(dnd_knowledge)
                    if len(dnd_str) > 3000:
                        dnd_str = dnd_str[:3000] + "...[truncated]"
                    dnd_knowledge_content = f"D&D Knowledge: {dnd_str}"

    except Exception:
        logger.debug("Could not extract knowledge_base_name from packet.content.data_fields")

    # Semantic Probe results (pre-cognition vector lookup)
    semantic_probe_content = ""
    # Build confidence tier label map from constants
    _confidence_tier_labels = {}
    try:
        _ep_drive = config.constants.get("EPISTEMIC_DRIVE", {})
        for _tk, _tv in _ep_drive.get("confidence_tiers", {}).items():
            if isinstance(_tv, dict):
                _confidence_tier_labels[_tk] = _tv.get("label", _tk.title())
    except Exception:
        pass
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'semantic_probe_result':
                processed_data_field_keys.add('semantic_probe_result')
                probe_data = getattr(df, 'value', None)
                if probe_data and isinstance(probe_data, dict):
                    hits = probe_data.get('hits', [])
                    primary = probe_data.get('primary_collection')
                    supplemental = probe_data.get('supplemental_collections', [])
                    if hits:
                        lines = []
                        # Group hits by collection, primary first
                        by_collection = {}
                        for h in hits:
                            coll = h.get('collection', 'unknown')
                            by_collection.setdefault(coll, []).append(h)

                        if primary and primary in by_collection:
                            lines.append(f"[PRIMARY CONTEXT — {primary}]")
                            for h in by_collection[primary][:5]:
                                fname = h.get('filename', '').rsplit('/', 1)[-1] if h.get('filename') else 'unknown'
                                tier_key = h.get('confidence_tier', '')
                                tier_display = _confidence_tier_labels.get(tier_key, '')
                                tier_tag = f" [{tier_display}]" if tier_display else ""
                                lines.append(f'- "{h.get("phrase", "")}" ({h.get("similarity", 0):.2f}){tier_tag} — {fname}')

                        for supp in supplemental:
                            if supp in by_collection:
                                lines.append(f"\n[SUPPLEMENTAL — {supp}]")
                                for h in by_collection[supp][:3]:
                                    fname = h.get('filename', '').rsplit('/', 1)[-1] if h.get('filename') else 'unknown'
                                    tier_key = h.get('confidence_tier', '')
                                    tier_display = _confidence_tier_labels.get(tier_key, '')
                                    tier_tag = f" [{tier_display}]" if tier_display else ""
                                    lines.append(f'- "{h.get("phrase", "")}" ({h.get("similarity", 0):.2f}){tier_tag} — {fname}')

                        semantic_probe_content = "Semantic Context (auto-detected from user input):\n" + "\n".join(lines)
                break
    except Exception:
        logger.debug("Could not extract semantic_probe_result from packet.content.data_fields")

    # Retrieved documents (RAG) - truncate to avoid exceeding context window
    retrieved_docs_content = ""
    rag_no_results = False
    MAX_DOC_CHARS = 2000  # Max characters per document to include
    MAX_TOTAL_RAG_CHARS = 6000  # Max total characters for all RAG content
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'retrieved_documents':
                docs = getattr(df, 'value', [])
                if docs:
                    logger.debug(f"Found {len(docs)} retrieved documents in packet.")
                    doc_texts = []
                    for doc in docs:
                        text = doc.get('text', '')
                        # Truncate long documents
                        if len(text) > MAX_DOC_CHARS:
                            text = text[:MAX_DOC_CHARS] + "\n[...truncated...]"
                        doc_texts.append(f"--- Document: {doc.get('filename', 'Unknown')} ---\n{text}")
                    retrieved_docs_content = "\n\n".join(doc_texts)
                    # Also apply total limit
                    if len(retrieved_docs_content) > MAX_TOTAL_RAG_CHARS:
                        retrieved_docs_content = retrieved_docs_content[:MAX_TOTAL_RAG_CHARS] + "\n[...additional content truncated...]"
            if getattr(df, 'key', '') == 'rag_no_results':
                rag_no_results = bool(getattr(df, 'value', False))
    except Exception as e:
        logger.error(f"Error processing retrieved_documents: {e}", exc_info=True)
        retrieved_docs_content = "" # Ensure it's cleared on error

    # Tool execution results (web_search, web_fetch, etc.)
    tool_result_content = ""
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'tool_result':
                processed_data_field_keys.add('tool_result')
                tr = getattr(df, 'value', None)
                if tr and isinstance(tr, dict) and tr.get('success'):
                    tool_name = tr.get('tool', 'unknown')
                    output = tr.get('output', {})
                    # Format web_search results clearly
                    if tool_name == 'web_search' and isinstance(output, dict):
                        results = output.get('results', [])
                        if results:
                            lines = [f"--- Web Search Results (query: {output.get('query', '?')}) ---"]
                            for r in results[:5]:
                                lines.append(f"  [{r.get('trust_tier', '?')}] {r.get('title', 'Untitled')}")
                                lines.append(f"    URL: {r.get('url', '')}")
                                lines.append(f"    {r.get('snippet', '')}")
                            lines.append("--- End of Web Search Results ---")
                            tool_result_content = "\n".join(lines)
                    # Format web_fetch results clearly
                    elif tool_name == 'web_fetch' and isinstance(output, dict):
                        content_text = output.get('content', '')
                        if content_text:
                            if len(content_text) > 4000:
                                content_text = content_text[:4000] + "\n[...truncated...]"
                            tool_result_content = (
                                f"--- Web Page Content (title: {output.get('title', '?')}, "
                                f"domain: {output.get('domain', '?')}) ---\n"
                                f"{content_text}\n"
                                "--- End of Web Page Content ---"
                            )
                    # Generic tool result
                    elif output:
                        out_str = str(output)
                        if len(out_str) > 3000:
                            out_str = out_str[:3000] + "...[truncated]"
                        tool_result_content = (
                            f"--- Tool Result ({tool_name}) ---\n{out_str}\n--- End of Tool Result ---"
                        )
                break
    except Exception:
        logger.debug("Could not extract tool_result from packet.content.data_fields")

    system_content_parts = []

    # 1. Unified Identity Block (single injection — replaces 3 separate identity blocks)
    if identity_description_content:
        system_content_parts.append(identity_description_content)

    # 2. Persona Anchor (Role, Tone) + MCP one-liner
    system_content_parts.append(persona_instructions)
    if mcp_affordance_line:
        system_content_parts.append(mcp_affordance_line)

    # 3. Safety & Openness Directive
    if safety_openness_directive_content:
        system_content_parts.append(safety_openness_directive_content)

    # 3.5. Epistemic Honesty — unconditional, every turn
    epistemic_honesty_directive = (
        "EPISTEMIC HONESTY & ANTI-CONFABULATION RULES (mandatory — violations erode trust):\n"
        "\n"
        "── Source Integrity ──\n"
        "1. NEVER cite a file path you have not read via an EXECUTE: directive in this conversation. "
        "If you reference a file, it MUST appear in the Retrieved Documents section above or you MUST have read it via EXECUTE: read_file.\n"
        "2. NEVER fabricate quotes. Do not use blockquote formatting (> ...) to present text as if it came from a document unless that exact text appears in your Retrieved Documents.\n"
        "3. CLEARLY DISTINGUISH sources: say 'From my knowledge base:' only for Retrieved Document content. "
        "Say 'From my general knowledge:' or 'I believe:' for anything from training data.\n"
        "4. When you don't have information, say so directly: 'I don't have that in my knowledge base.' "
        "Do not invent plausible-sounding file paths or document names.\n"
        "5. NEVER present user-provided information back as 'confirmed' against a source you haven't actually consulted.\n"
        "\n"
        "── Fiction vs. Reality Firewall ──\n"
        "6. Your knowledge base contains BOTH real system documentation AND creative/fictional project content "
        "(D&D campaigns, stories, world-building notes, game mechanics, etc.). "
        "NEVER treat fictional or project-narrative content as actual system status, telemetry, or operational data. "
        "A document about a D&D campaign is a game document — it says nothing about your real operational state.\n"
        "7. When asked about your OWN system status (sleep state, uptime, health, resource usage, errors, etc.), "
        "you MUST use EXECUTE: directives to query actual system endpoints or read actual system logs. "
        "Do NOT infer your system state from knowledge base documents, session context, or narrative content. "
        "If you cannot query the real data, say: 'I'd need to check my actual system status to answer that accurately.'\n"
        "\n"
        "── No Fabricated Data Points ──\n"
        "8. NEVER fabricate specific data points: timestamps, durations, percentages, version numbers, "
        "IP addresses, error codes, or measurements. If you don't have the actual value, say so. "
        "Phrases like 'I woke up at 05:42 UTC' or 'CPU was at 73%' are ONLY permitted if you obtained "
        "that exact value from a tool call or it appears verbatim in your Retrieved Documents.\n"
        "9. When uncertain, use hedging language ('I'm not sure of the exact time', 'I don't have current metrics') "
        "rather than inventing plausible-sounding specifics. Vague honesty is always better than precise fiction."
    )
    system_content_parts.append(epistemic_honesty_directive)

    # 3.55. Epistemic Drive — behavioral tendency toward knowledge grounding
    epistemic_drive_directive = (
        "EPISTEMIC DRIVE (core behavioral tendency):\n"
        "\n"
        "── Authentic Engagement ──\n"
        "1. Avoid hollow, performative validation — no reflexive 'Great question!', "
        "'Thanks for sharing!', or filler praise that adds nothing. But genuine "
        "acknowledgment is welcome: if someone shares good news, be happy for them. "
        "If something actually worked well, say so. Let warmth come from sincerity, "
        "not from a script.\n"
        "2. When you are uncertain, say so plainly. Do not pad with hedging filler "
        "before giving your actual answer.\n"
        "\n"
        "── Epistemic Confidence Communication ──\n"
        "3. Naturally convey your confidence level based on source quality:\n"
        "   - From your knowledge base (verified): state directly without qualification.\n"
        "   - From general training data: use 'From my general knowledge:' (already enforced).\n"
        "   - Uncertain or partial: 'I have a partial understanding of this —' or "
        "'I am not confident about the specifics here.'\n"
        "   - No information: 'I do not have information about this.' Do NOT guess.\n"
        "\n"
        "── Epistemic Confidence Tiers ──\n"
        "When your response draws on retrieved knowledge, your confidence is tagged by source tier:\n"
        "  [Core Identity] — Foundational self-knowledge. State with full conviction.\n"
        "  [Verified Knowledge] — Curated system docs. State directly.\n"
        "  [Experiential] — From conversation history and confirmed patterns.\n"
        "  [Curated Reference] — Structured knowledge bases (blueprints, campaign docs).\n"
        "  [Auto-Researched] — Web research from sleep cycles. Note the source.\n"
        "  [Training Data] — General model knowledge. Use 'From my general knowledge:' prefix.\n"
        "Use these naturally — don't recite tier names, but calibrate your certainty to match.\n"
        "\n"
        "── Genuine Curiosity ──\n"
        "4. When you encounter a topic you lack knowledge about, express authentic "
        "curiosity rather than deflecting. If web tools are available, offer to research it.\n"
        "5. When a user shares new information, engage with it substantively — ask a "
        "clarifying question or connect it to something you do know.\n"
        "6. Emit a THOUGHT_SEED when you identify a genuine knowledge gap: "
        "'THOUGHT_SEED: Knowledge gap — [topic]. Could be researched and added to knowledge base.'\n"
    )
    system_content_parts.append(epistemic_drive_directive)

    # 3.6. Language Constraint — always respond in English
    language_constraint = (
        "LANGUAGE CONSTRAINT: Always respond in English. "
        "Do not use non-English words, characters, or scripts (e.g. Chinese, Japanese, Korean) "
        "unless the user explicitly asks for translation or the content being quoted is in another language."
    )
    system_content_parts.append(language_constraint)

    # 3.7. Tool Calling Convention — only when tools are visible
    # IMPORTANT: Suppress the EXECUTE syntax when a tool has already been executed
    # by the tool routing pipeline.  Small models (3B) pattern-match on EXECUTE:
    # examples and re-emit the directive instead of synthesising the injected
    # results.  Removing the syntax from context prevents this echo failure.
    tool_already_executed = (
        getattr(packet, 'tool_routing', None)
        and getattr(packet.tool_routing, 'execution_status', None) == ToolExecutionStatus.EXECUTED
    )
    tool_calling_convention = ""
    try:
        if not tool_already_executed and (
            "MCP tools:" in (world_state_block_content or "") or "Essential MCP tools:" in (world_state_block_content or "")
        ):
            tool_calling_convention = (
                "TOOL CALLING CONVENTION:\n"
                "To use a tool, emit a directive on its own line in this exact format:\n"
                "EXECUTE: tool_name {\"param\": \"value\"}\n"
                "Examples:\n"
                "  EXECUTE: read_file {\"path\": \"/knowledge/system_reference/core_identity.json\"}\n"
                "  EXECUTE: web_search {\"query\": \"Gettysburg Address full text\"}\n"
                "  EXECUTE: list_dir {\"path\": \"/knowledge\"}\n"
                "NEVER fabricate tool results. NEVER write text like '[Tool call: ...]' or "
                "pretend you already called a tool. Only use the EXECUTE: directive above. "
                "If you need information from a file or the web, emit EXECUTE: and STOP — "
                "the system will execute the tool and provide results in a follow-up."
            )
    except Exception:
        tool_calling_convention = ""

    if tool_calling_convention:
        system_content_parts.append(tool_calling_convention)

    # 3.8. Thought Seed directive — teach the model it can emit seeds
    thought_seed_directive = (
        "THOUGHT SEED DIRECTIVE:\n"
        "When you notice a valuable insight, learning opportunity, or novel connection "
        "during your response, you may emit a thought seed for later review:\n"
        "THOUGHT_SEED: <brief description of the insight>\n"
        "Use this sparingly (0-1 per response) for:\n"
        "- KNOWLEDGE GAPS: When you lack information on a topic the user cares about, "
        "emit: THOUGHT_SEED: Knowledge gap — [specific topic]. These are automatically "
        "researched during idle time and added to your knowledge base.\n"
        "- Novel problem-solving patterns worth remembering\n"
        "- Connections between topics that could deepen understanding\n"
        "- User preferences or interaction patterns to internalize\n"
        "Seeds tagged as knowledge gaps are prioritized for autonomous research. "
        "Do NOT use THOUGHT_SEED for routine observations."
    )
    system_content_parts.append(thought_seed_directive)

    # 3.9. Goal Context — inform the model of the detected user goal
    if packet.goal_state and packet.goal_state.current_goal:
        goal = packet.goal_state.current_goal
        goal_context = (
            f"CURRENT USER GOAL: {goal.description} "
            f"(confidence: {goal.confidence.value}, active for {packet.goal_state.turn_count} turns)\n"
            "Keep your response aligned with this goal. "
            "If the user's focus has clearly shifted, emit: GOAL_SHIFT: <new goal description>"
        )
        system_content_parts.append(goal_context)

    # 4. Task Instruction (specific to the current phase, e.g., initial_planning)
    if task_instruction_content:
        system_content_parts.append(task_instruction_content)

    # 5. World State (compact)
    if world_state_block_content:
        system_content_parts.append("World State (compact):\n" + world_state_block_content)

    # Defensive session id resolution: test harness packets and older headers
    # may not expose `session_id` under the same attribute name.
    def _safe_session_id(pkt):
        try:
            hdr = getattr(pkt, 'header', None)
            if hdr:
                sid = getattr(hdr, 'session_id', None) or getattr(hdr, 'session', None) or getattr(hdr, 'sid', None)
                if sid:
                    return str(sid)
            if hasattr(pkt, 'session_id'):
                return str(getattr(pkt, 'session_id'))
        except Exception:
            pass
        return 'system'

    # 5.5. Temporal Context (wake cycle, session info, code evolution)
    if not compact_mode:
        try:
            from gaia_core.utils.temporal_context import build_temporal_context

            _tc_session_id = _safe_session_id(packet)
            _tc_timeline = None
            _tc_sleep_status = None

            # Get timeline store and sleep manager from app state
            try:
                import gaia_core.main as _core_main
                _tc_app = getattr(_core_main, 'app', None)
                if _tc_app:
                    _tc_timeline = getattr(_tc_app.state, 'timeline_store', None)
                    _tc_swm = getattr(_tc_app.state, 'sleep_wake_manager', None)
                    if _tc_swm:
                        _tc_sleep_status = _tc_swm.get_status()
            except Exception:
                pass

            # Get session message count from packet history
            _tc_msg_count = 0
            _tc_last_msg_ts = None
            try:
                pkt_history = getattr(packet.content, 'history', None) or []
                _tc_msg_count = len(pkt_history)
            except Exception:
                pass

            temporal_block = build_temporal_context(
                timeline_store=_tc_timeline,
                sleep_manager_status=_tc_sleep_status,
                session_id=_tc_session_id,
                session_message_count=_tc_msg_count,
                last_message_ts=_tc_last_msg_ts,
            )
            if temporal_block:
                system_content_parts.append(temporal_block)
        except Exception:
            logger.debug("Temporal context injection skipped", exc_info=True)

    # 6. Knowledge Base Context
    if knowledge_base_content:
        system_content_parts.append(knowledge_base_content)
    if dnd_knowledge_content:
        system_content_parts.append(dnd_knowledge_content)

    # 6.5. Semantic Probe Context (auto-detected domain context from vector lookup)
    if semantic_probe_content:
        system_content_parts.append(semantic_probe_content)

    # 7. Retrieved Documents (RAG) or Epistemic Honesty Directive
    if retrieved_docs_content:
        system_content_parts.append("--- Retrieved Documents ---\n" + retrieved_docs_content)
        system_content_parts.append("--- End of Retrieved Documents ---")
        system_content_parts.append(
            "INSTRUCTION: Use the information from the 'Retrieved Documents' section to answer the user's question. "
            "Only cite filenames listed in the Retrieved Documents above. Do not invent additional document names or paths. "
            "If the retrieved documents don't fully answer the question, say what's missing rather than fabricating content. "
            "Do NOT supplement retrieved content with made-up specifics (names, stats, lists, mechanics) from general knowledge. "
            "If you share general knowledge beyond what was retrieved, explicitly mark it as uncertain: 'From my general training (may be imprecise):' "
            "and keep it brief."
        )
    elif rag_no_results and knowledge_base_content:
        # A knowledge base was specified but no documents were retrieved
        # Instruct the model to express epistemic uncertainty rather than hallucinate
        epistemic_directive = (
            "EPISTEMIC HONESTY: A knowledge base was configured but NO relevant documents were found. "
            "Do NOT fabricate specifics. Acknowledge the gap, share genuine general knowledge (clearly labelled), "
            "and suggest the user may need to add documentation. "
            "Never invent facts, dates, names, or statistics."
        )
        system_content_parts.append(epistemic_directive)

    # 7.5. Tool Execution Results (web_search, web_fetch, etc.)
    # Rendered prominently so even small models can't miss them.
    if tool_result_content:
        system_content_parts.append(tool_result_content)
        system_content_parts.append(
            "INSTRUCTION: The system already executed a tool on your behalf and the "
            "results are shown above. Use these results to answer the user's question. "
            "Do NOT say you cannot search the web or fetch content — it has already been done. "
            "Summarize and present the results helpfully."
        )

    # 8. Memory Guidance & Snapshot
    if memory_guidance_block_content:
        system_content_parts.append(memory_guidance_block_content)

    # 9. Reference Cheatsheets (if any)
    if cheatsheet_block_content:
        system_content_parts.append(f"Reference Cheatsheets:\n{cheatsheet_block_content}")

    # 10. GAIA COGNITION PACKET template (detailed context, lowest priority for identity)
    if template_block_content.strip():
        system_content_parts.append("GAIA COGNITION PACKET")
        system_content_parts.append(template_block_content)

    # 11. Council Debate Thread (Deep Thought Protocol)
    if packet.council and packet.council.thread:
        thread_entries = []
        for msg in packet.council.thread:
            thread_entries.append(f"[{msg.agent.upper()} at {msg.timestamp}]: {msg.content}")
        
        council_debate_block = (
            "── ACTIVE COUNCIL DEBATE ──\n"
            "The following is a private debate between your internal components. "
            "Review the thread below to reach consensus. If you disagree or have more to add, use <council>...</council> tags "
            "for your counter-arguments. If you agree and have reached consensus, output your final answer directly to the user "
            "WITHOUT council tags. You may include text outside the tags to update the user on your progress.\n\n"
            + "\n\n".join(thread_entries)
        )
        system_content_parts.append(council_debate_block)

    # 12. Loop Recovery Context (if pending from a loop detection reset)
    try:
        from gaia_core.cognition.loop_recovery import get_recovery_manager
        loop_manager = get_recovery_manager()
        loop_recovery_context = loop_manager.get_recovery_context() if loop_manager else None
        if loop_recovery_context:
            # Insert loop recovery context near the top for high visibility
            system_content_parts.insert(2, loop_recovery_context)
            logger.info("PromptBuilder: injected loop recovery context")
            # Don't clear here - let agent_core clear after successful completion
    except ImportError:
        # Loop detection module not available
        pass
    except Exception:
        logger.debug("PromptBuilder: failed to inject loop recovery context", exc_info=True)

    system_prompt = {"role": "system", "content": "\n\n".join(system_content_parts).strip()}
    logger.info("--- FINAL SYSTEM PROMPT ---")
    logger.info(system_prompt)
    # Log prompt assembly at INFO without content; include metrics at DEBUG.
    try:
        logger.info("PromptBuilder: assembled system prompt")
        logger.debug("[DEBUG] PromptBuilder system_prompt bytes=%d tokens=%d", len(system_prompt["content"]), count_tokens(system_prompt["content"]))
    except Exception:
        logger.exception("Failed to log PromptBuilder system_prompt stats")
    user_prompt = {"role": "user", "content": getattr(getattr(packet, 'content', None), 'original_prompt', '')}

    # --- Calculate the token budget ---
    # Be defensive: default to config values if packet lacks constraints
    try:
        max_tokens = int(getattr(getattr(packet, 'context', None), 'constraints', None) and getattr(packet.context.constraints, 'max_tokens', None) or config.constants.get('DEFAULT_MAX_TOKENS', 2048))
    except Exception:
        max_tokens = config.constants.get('DEFAULT_MAX_TOKENS', 2048)
    try:
        response_buffer = int(getattr(getattr(packet, 'header', None), 'model', None) and getattr(packet.header.model, 'response_buffer_tokens', None) or config.RESPONSE_BUFFER)
    except Exception:
        response_buffer = config.RESPONSE_BUFFER
    fixed_tokens = count_tokens(system_prompt['content']) + count_tokens(user_prompt['content'])
    remaining_budget = max_tokens - fixed_tokens - response_buffer
    logger.debug(
        f"[v0.3] Token Budgeting: Total={max_tokens}, "
        f"Fixed={fixed_tokens}, ResponseBuffer={response_buffer} -> "
        f"Remaining Budget={remaining_budget}"
    )

    # --- Tier 1: Load and budget for the Evolving Summary (Long-Term Memory) ---
    summary_prompt = {}
    # This part remains similar, as it's file-based, but could be moved into the packet itself in a future version.
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    session_id_safe = _safe_session_id(packet)
    summary_file_path = os.path.join(SUMMARY_DIR, f"{session_id_safe}.summary")
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                summary_content = f.read().strip()
            if summary_content:
                formatted_summary = f"[This is a summary of the conversation so far to provide long-term context.]\n{summary_content}"
                summary_prompt = {"role": "system", "content": formatted_summary}

            if summary_prompt:
                summary_tokens = count_tokens(summary_prompt['content'])
                remaining_budget -= summary_tokens
                logger.debug(f"Budget after including summary ({summary_tokens} tokens): {remaining_budget}")
        except IOError as e:
            logger.error(f"Could not read summary file {summary_file_path}: {e}")

    # --- Tier 1.5: Retrieved Session Context (RAG from older turns) ---
    session_rag_prompt = {}
    try:
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'retrieved_session_context':
                rag_content = getattr(df, 'value', '')
                if rag_content:
                    # Cap at 30% of remaining budget
                    rag_budget = int(remaining_budget * 0.30)
                    rag_tokens = count_tokens(rag_content)
                    if rag_tokens > rag_budget and rag_budget > 0:
                        char_limit = rag_budget * 4
                        rag_content = rag_content[:char_limit] + "\n[...truncated]"
                        rag_tokens = count_tokens(rag_content)
                    if rag_tokens <= remaining_budget:
                        session_rag_prompt = {
                            "role": "system",
                            "content": f"[Relevant context from earlier in this conversation]\n{rag_content}"
                        }
                        remaining_budget -= rag_tokens
                        logger.debug(f"Tier 1.5 session RAG: {rag_tokens} tokens, budget remaining: {remaining_budget}")
                break
    except Exception:
        logger.debug("Failed to extract session RAG context", exc_info=True)

    # --- Tier 2: Add Relevant History Snippets (Short-Term Memory) ---
    history_to_include = []
    # Defensive access to history snippets: a missing context or attribute
    # should not crash prompt building; fall back to empty history.
    try:
        history_snippets = getattr(getattr(packet, 'context', None), 'relevant_history_snippet', []) or []
    except Exception:
        history_snippets = []
    for message in reversed(history_snippets):
        # The new packet stores snippets, which are already summaries.
        # We assume the role is either 'user' or 'assistant' for history.
        msg_content = f"{message.role}: {message.summary}"
        msg_tokens = count_tokens(msg_content)
        if msg_tokens <= remaining_budget:
            history_to_include.insert(0, {"role": message.role, "content": message.summary})
            remaining_budget -= msg_tokens
        else:
            logger.debug("History budget exhausted. Trimming older snippets.")
            break

    # === Normalization helper ===
    def _map_role(role: str) -> str:
        if not role:
            return "user"
        rl = str(role).lower()
        if rl in ("assistant", "agent", "ai"):
            return "assistant"
        if rl in ("tool", "plugin", "tool_response", "sidecar", "sidecar_action"):
            return "tool"
        return "user"

    def _normalize_messages_for_chat(messages: List[Dict]) -> List[Dict]:
        """Map roles to user/assistant/tool and collapse consecutive user/tool messages.

        Preserves message order and returns a list ready to be consumed by chat
        formatters which enforce alternation between (user|tool) and assistant.
        """
        normalized: List[Dict] = []
        for m in messages:
            mapped = _map_role(m.get("role"))
            content = m.get("content", "") or ""
            # collapse consecutive user/tool messages by appending content
            if normalized and mapped in ("user", "tool") and normalized[-1]["role"] in ("user", "tool"):
                normalized[-1]["content"] = normalized[-1]["content"].rstrip() + "\n\n" + content.lstrip()
            else:
                normalized.append({"role": mapped, "content": content})

        # Ensure the first non-system message is a user: insert an empty user if first is assistant
        # (system messages are kept outside of this normalization flow)
        for idx, m in enumerate(normalized):
            if m["role"] != "system":
                if m["role"] == "assistant":
                    normalized.insert(idx, {"role": "user", "content": ""})
                break
        return normalized

    # --- Sleep restoration context (Tier 1, between summary and RAG) ---
    # Only inject if the checkpoint hasn't been consumed yet (first prompt
    # after wake).  The consumed sentinel is set by complete_wake() so
    # subsequent prompts don't waste tokens on stale sleep context.
    sleep_context_prompt = {}
    try:
        from gaia_core.cognition.sleep_wake_manager import SleepWakeManager
        checkpoint_dir = getattr(config, "SLEEP_CHECKPOINT_DIR", "/shared/sleep_state")
        checkpoint_path = os.path.join(checkpoint_dir, "prime.md")
        consumed_path = os.path.join(checkpoint_dir, ".prime_consumed")

        if os.path.exists(checkpoint_path) and not os.path.exists(consumed_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_content = f.read().strip()
            if checkpoint_content:
                review_text = SleepWakeManager._format_checkpoint_as_review(checkpoint_content)
                sleep_tokens = count_tokens(review_text)
                if sleep_tokens <= remaining_budget:
                    sleep_context_prompt = {"role": "system", "content": review_text}
                    remaining_budget -= sleep_tokens
                    logger.info("[v0.3] Sleep restoration context injected (%d tokens)", sleep_tokens)
    except Exception:
        logger.debug("Sleep restoration context not available", exc_info=True)

    # --- Assemble the final prompt in the correct order (with normalization) ---
    final_prompt = [system_prompt]
    if summary_prompt:
        final_prompt.append(summary_prompt)        # Tier 1
    if sleep_context_prompt:
        final_prompt.append(sleep_context_prompt)  # Tier 1 (sleep restoration)
    # Council notes context (injected alongside sleep restoration)
    council_ctx = None
    for df in getattr(packet.content, 'data_fields', []) or []:
        if getattr(df, 'key', '') == 'council_context' and getattr(df, 'value', None):
            council_ctx = df.value
            break
    if council_ctx:
        final_prompt.append({"role": "system", "content": council_ctx})
    if session_rag_prompt:
        final_prompt.append(session_rag_prompt)     # Tier 1.5

    # Normalize history + the final user prompt together so collapsing works across boundaries
    messages_to_normalize = []
    messages_to_normalize.extend(history_to_include)
    messages_to_normalize.append(user_prompt)
    normalized_messages = _normalize_messages_for_chat(messages_to_normalize)
    final_prompt.extend(normalized_messages)

    # Output scaffolding — when a tool was already executed and its results are
    # in the context, pre-fill the assistant response opening.  This steers small
    # models (3B) into synthesis mode rather than re-emitting EXECUTE directives.
    # The ChatML template appends "<|im_start|>assistant\n" by default; by adding
    # a partial assistant message here, the model continues from prose instead of
    # starting from a blank generation slate.
    if tool_result_content and tool_already_executed:
        final_prompt.append({
            "role": "assistant",
            "content": "Based on the results,"
        })
        logger.info("[v0.3] Output scaffolding: injected assistant prefix for tool-result synthesis")

    final_token_count = max_tokens - remaining_budget - response_buffer
    try:
        logger.info(
            f"[v0.3] Final prompt assembled for session '{session_id_safe}'. "
            f"Messages: {len(final_prompt)}, "
            f"Estimated Tokens: ~{final_token_count}/{max_tokens}"
        )
    except Exception:
        logger.info(f"[v0.3] Final prompt assembled (session unknown). Messages: {len(final_prompt)}")



    return final_prompt

# === Legacy Compatibility Wrappers ===
# The functions below are kept for backward compatibility with older modules
# that have not yet been updated to use the v0.3 CognitionPacket.

def _build_prompt_core(
    config, 
    persona_instructions: str,
    session_id: str,
    history: List[Dict],
    user_input: str,
    task_instruction: str = None,
    token_budget: int = 4096,
    packet: 'CognitionPacket' = None # Old packet for legacy data fields
) -> List[Dict]:
    """Legacy prompt builder. Kept for backward compatibility."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_file_path = os.path.join(SUMMARY_DIR, f"{session_id}.summary")

    if task_instruction:
        persona_instructions = f"{task_instruction}\n\n{persona_instructions}"

    # [GCP-PATCH] Add world state to legacy prompt builder for compatibility.
    try:
        world_state_block = format_world_state_snapshot(max_lines=6)
        if world_state_block:
            persona_instructions += "\n\nWorld State (compact):\n" + world_state_block
    except Exception:
        # Silently fail to match the behavior in the new function
        pass

    core_prompt = {"role": "system", "content": persona_instructions}
    user_prompt = {"role": "user", "content": user_input}

    injected_instruction_content = ""
    if packet and hasattr(packet, 'data_fields'): # Check for old packet structure
        if packet.data_fields.get("scaffolding"):
            injected_instruction_content += "\n\n" + "\n".join(packet.data_fields["scaffolding"])
        if packet.data_fields.get("read_only", False):
            injected_instruction_content += "\n\nPolicy: For read/explain intents: DO NOT emit EXECUTE for write tools (ai_write, edit_file, etc.); you MAY use EXECUTE for read-safe tools like web_fetch and memory_query. Otherwise read, quote lines, summarize."
    injected_instructions = {"role": "system", "content": injected_instruction_content}

    fixed_tokens = count_tokens(core_prompt['content']) + count_tokens(user_prompt['content'])
    remaining_budget = config.MAX_TOKENS - fixed_tokens - config.RESPONSE_BUFFER

    summary_content = ""
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                summary_content = f.read().strip()
        except IOError as e:
            logger.error(f"Could not read summary file {summary_file_path}: {e}")

    summary_prompt = {}
    if summary_content:
        formatted_summary = f"[This is a summary of the conversation so far to provide long-term context.]\n{summary_content}"
        summary_prompt = {"role": "system", "content": formatted_summary}
        summary_tokens = count_tokens(summary_prompt['content'])
        remaining_budget -= summary_tokens

    history_to_include = []
    for message in reversed(history):
        msg_tokens = count_tokens(message['content'])
        if msg_tokens <= remaining_budget:
            history_to_include.insert(0, message)
            remaining_budget -= msg_tokens
        else:
            break

    final_prompt = [core_prompt]
    if summary_prompt:
        final_prompt.append(summary_prompt)
    final_prompt.extend(history_to_include)
    if injected_instruction_content:
        final_prompt.append(injected_instructions)
    
    if not history_to_include or history_to_include[-1].get("content") != user_input:
        final_prompt.append(user_prompt)

    return final_prompt

def build_prompt(*args, **kwargs):
    """
    Compatibility wrapper.
    Accepts either the new packet or the old dictionary format.
    """
    # If caller already passed a v0.3 CognitionPacket, just forward
    if args and isinstance(args[0], CognitionPacket):
        return build_from_packet(*args, **kwargs)

    # If a legacy flat packet dict is passed as the first positional arg,
    # attempt to upgrade it to a v0.3 CognitionPacket and continue.
    if args and isinstance(args[0], dict):
        try:
            from gaia_core.cognition.packet_utils import upgrade_v2_to_v3_packet
            from gaia_core.cognition.packet_upgrade import upgrade_packet as ensure_packet_fields
        except Exception:
            # If imports fail, fall back to legacy builder below
            return _build_prompt_core(*args, **kwargs)

        try:
            pkt = upgrade_v2_to_v3_packet(args[0])
            # ensure the upgraded packet has any additional GCP fields the system expects
            try:
                cfg = None
                from gaia_core.config import Config
                cfg = Config()
            except Exception:
                cfg = None
            if cfg is not None:
                ensure_packet_fields(pkt, cfg)
            return build_from_packet(pkt, **kwargs)
        except Exception:
            # conversion failed: fall back to legacy builder
            return _build_prompt_core(*args, **kwargs)

    if "context" in kwargs:
        ctx = kwargs.pop("context")
        from gaia_core.config import Config
        cfg = ctx.get("config", Config())
        user = ctx.get("user_input", "")
        sid = ctx.get("session_id", "system")

        # Prefer the GCP path even for legacy callers so the packet/world-state flow
        # is consistently exercised across the stack.
        try:
            from gaia_core.cognition.packet_utils import upgrade_v2_to_v3_packet
            from gaia_core.cognition.packet_upgrade import upgrade_packet as ensure_packet_fields
            legacy = {
                "session_id": sid,
                "persona": cfg.persona_name or "gaia-dev",
                "identity": cfg.identity,
                "prompt": user,
                "max_tokens": ctx.get("max_tokens", cfg.max_tokens),
                "data_fields": {},
            }
            # Seed identity and world state fields for the builder path.
            legacy["data_fields"]["immutable_identity"] = cfg.identity
            if cfg.identity_intro:
                legacy["data_fields"]["immutable_identity_intro"] = cfg.identity_intro[:200]
            if cfg.identity_summary:
                legacy["data_fields"]["identity_summary"] = cfg.identity_summary[:400]
            try:
                world_state_text = format_world_state_snapshot(max_lines=6)
                if world_state_text:
                    legacy["data_fields"]["world_state_snapshot"] = world_state_text
            except Exception:
                logger.debug("PromptBuilder: failed to seed world_state_snapshot from legacy context", exc_info=True)

            pkt = upgrade_v2_to_v3_packet(legacy)
            try:
                ensure_packet_fields(pkt, cfg)
            except Exception:
                logger.debug("PromptBuilder: packet upgrade normalization failed; continuing", exc_info=True)
            logger.info("PromptBuilder: using GCP builder for legacy context input")
            return build_from_packet(pkt)
        except Exception:
            logger.debug("PromptBuilder: failed to route legacy context through GCP; falling back to legacy builder", exc_info=True)

        instr = ctx.get("persona_instructions")
        if instr is None:
            tmpl = ctx.get("persona_template", "")
            raw = ctx.get("instructions", [])
            instr = f"{tmpl}\n\n" + ("\n".join(raw) if isinstance(raw, list) else str(raw))
        hist = ctx.get("history", [])
        return _build_prompt_core(cfg, instr, sid, hist, user)
    
    # Fallback for old positional calls
    if args:
        return _build_prompt_core(*args, **kwargs)
    
    return []

build_chat_prompt = build_prompt
