import logging
import regex as re
import ast
import uuid
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from gaia_core.memory.semantic_codex import SemanticCodex
from gaia_core.memory.codex_writer import CodexWriter
from typing import Generator, Dict, Any, List, Optional

from gaia_core.cognition.external_voice import ExternalVoice
from gaia_core.cognition.self_reflection import run_self_reflection, reflect_and_refine
from gaia_core.utils.prompt_builder import build_from_packet
from gaia_core.utils.output_router import route_output, _strip_think_tags_robust
# TODO: [GAIA-REFACTOR] chat_logger.py module not yet migrated.
# from app.utils.chat_logger import log_chat_entry, log_chat_entry_structured
log_chat_entry = lambda *args, **kwargs: None  # Placeholder
log_chat_entry_structured = lambda *args, **kwargs: None  # Placeholder
from gaia_core.utils.stream_observer import StreamObserver, Interrupt
from gaia_core.utils import mcp_client
from gaia_core.config import Config, get_config

# Get constants from config for backwards compatibility
_config = get_config()
constants = getattr(_config, 'constants', {})
from gaia_common.utils.thoughtstream import write as ts_write
# Persona switcher for dynamic persona/knowledge-base selection
from gaia_core.behavior.persona_switcher import get_persona_for_request
from gaia_core.cognition.cognitive_dispatcher import process_execution_results
from gaia_core.cognition.knowledge_enhancer import enhance_packet

# [GCP v0.3] Import new packet structure
from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Header, Persona, Routing, Model, Intent, Context, Content, Reasoning, Response, Governance, Safety, Metrics, TokenUsage, Status,
    PersonaRole, Origin, TargetEngine, SystemTask, PacketState,
    DataField, ReflectionLog, RelevantHistorySnippet, SessionHistoryRef, Cheatsheet, Constraints,
    # GCP Tool Routing System
    ToolRoutingState, ToolExecutionStatus, SelectedTool, ToolExecutionResult,
    # Output Routing (Spinal Column)
    OutputDestination, OutputRouting, DestinationTarget,
)
from gaia_core.cognition.nlu.intent_detection import detect_intent, Plan
from gaia_core.utils import gaia_rescue_helper as rescue_helper

# Loop Detection System
from gaia_core.cognition.loop_recovery import (
    LoopRecoveryManager,
    get_recovery_manager,
    build_loop_detection_config_from_constants,
    LoopInterrupt
)
from gaia_core.cognition.loop_detector import LoopDetectorConfig

logger = logging.getLogger("GAIA.AgentCore")
HISTORY_SUMMARY_THRESHOLD = 20
MAX_OUTPUT_LENGTH = 500  # Max length for command output

# Pattern to match <think>...</think> blocks (including multiline content)
# Uses DOTALL flag equivalent via (?s) to match newlines within the block
_THINK_TAG_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

# GCP section names that should never appear in user-facing output
_GCP_SECTIONS = ['HEADER', 'ROUTING', 'MODEL', 'CONTEXT', 'INTENT', 'CONTENT',
                 'REASONING', 'GOVERNANCE', 'METRICS', 'STATUS', 'RESPONSE']


# Use the migrated strip_think_tags from output_router
strip_think_tags = _strip_think_tags_robust


def _format_retrieved_session_context(results: dict) -> str:
    """Format RAG-retrieved session turns and topics into a readable context block."""
    parts = []
    turns = results.get("turns", [])
    topics = results.get("topics", [])
    if not turns and not topics:
        return ""
    if topics:
        parts.append("Earlier conversation topics:")
        for t in topics:
            parts.append(f"- {t.get('label', 'Topic')} (similarity: {t.get('similarity', '?')})")
    if turns:
        parts.append("Relevant earlier exchanges:")
        for t in turns:
            parts.append(f"[Turn {t.get('idx', '?')}, sim={t.get('similarity', '?')}]")
            parts.append(f"  User: {t.get('user', '')[:500]}")
            parts.append(f"  Assistant: {t.get('assistant', '')[:500]}")
    return "\n".join(parts)


# Known documents that can be recited, with keyword triggers and file paths
# Keywords are checked case-insensitively against the user's request
RECITABLE_DOCUMENTS = {
    "constitution": {
        "keywords": ["constitution", "gaia constitution", "the constitution"],
        "path": "knowledge/system_reference/core_documents/gaia_constitution.md",
        "title": "The GAIA Constitution",
    },
    "layered_identity": {
        "keywords": ["layered identity", "identity model", "tier i", "tier ii", "tier iii", "identity layers"],
        "path": "knowledge/system_reference/core_documents/layered_identity_model.md",
        "title": "The Layered Identity Model",
    },
    "declaration": {
        "keywords": ["declaration", "artisanal intelligence declaration", "declaration of artisanal"],
        "path": "knowledge/system_reference/core_documents/declaration_of_artisanal_intelligence.md",
        "title": "The Declaration of Artisanal Intelligence",
    },
    "cognition_protocol": {
        "keywords": ["cognition protocol", "cognitive protocol", "gaia cognition"],
        "path": "knowledge/system_reference/core_documents/gaia_cognitition_protocol.md",
        "title": "The GAIA Cognition Protocol",
    },
    "coalition_of_minds": {
        "keywords": ["coalition", "coalition of minds"],
        "path": "knowledge/system_reference/core_documents/coalition_of_minds.md",
        "title": "The Coalition of Minds",
    },
    "mindscape_manifest": {
        "keywords": ["mindscape", "manifest", "mindscape manifest"],
        "path": "knowledge/system_reference/core_documents/mindscape_manifest.md",
        "title": "The Mindscape Manifest",
    },
}


def find_recitable_document(user_input: str) -> Optional[Dict[str, str]]:
    """
    Check if the user's request matches a known recitable document.

    Args:
        user_input: The user's request text

    Returns:
        Dict with 'path', 'title', and 'content' if found, None otherwise
    """
    input_lower = user_input.lower()

    for doc_id, doc_info in RECITABLE_DOCUMENTS.items():
        for keyword in doc_info["keywords"]:
            if keyword in input_lower:
                # Found a match - try to load the document
                doc_path = Path(doc_info["path"])
                # Try multiple possible locations
                cwd = Path.cwd()
                # The doc_path starts with "knowledge/..." - also try just the part after "knowledge/"
                doc_subpath = doc_path.relative_to("knowledge") if str(doc_path).startswith("knowledge/") else doc_path
                possible_paths = [
                    cwd / doc_path,                                    # Relative to current working directory
                    doc_path,                                          # As-is (might be absolute)
                    Path("/gaia-assistant") / doc_path,                # Docker container path
                    Path("/knowledge") / doc_subpath,                  # Docker /knowledge mount
                    Path("/gaia/GAIA_Project/gaia-assistant") / doc_path,  # Development path
                ]

                logger.debug(f"Looking for recitable document '{doc_id}', cwd={cwd}")
                for try_path in possible_paths:
                    logger.debug(f"Trying path: {try_path} (exists={try_path.exists()})")
                    if try_path.exists():
                        try:
                            content = try_path.read_text(encoding="utf-8")
                            logger.info(f"Loaded recitable document: {doc_info['title']} from {try_path}")
                            return {
                                "path": str(try_path),
                                "title": doc_info["title"],
                                "content": content,
                                "doc_id": doc_id,
                            }
                        except Exception as e:
                            logger.warning(f"Failed to read document {try_path}: {e}")

                logger.warning(f"Document matched but file not found: {doc_info['path']} (tried {len(possible_paths)} paths from cwd={cwd})")
                return None

    return None


class AgentCore:
    """
    Encapsulates the core "Reason-Act-Reflect" loop for GAIA.
    This class is UI-agnostic and yields structured events back to the caller.
    It is session-aware and uses a prompt builder to manage context.
    """

    def __init__(self, ai_manager, ethical_sentinel=None):
        self.ai_manager = ai_manager
        self.model_pool = ai_manager.model_pool
        self.config = ai_manager.config
        self.ethical_sentinel = ethical_sentinel
        self.session_manager = ai_manager.session_manager
        # Ensure instance-level logger exists for use throughout the class
        # Some code paths reference self.logger; initialize it to the module logger
        try:
            self.logger = logging.getLogger("GAIA.AgentCore")
        except Exception:
            # Fallback to the module-level logger
            self.logger = logger
        
        # Initialize SemanticCodex and CodexWriter
        self.semantic_codex = SemanticCodex.instance(self.config)
        self.codex_writer = CodexWriter(self.config, self.semantic_codex)

    def _build_output_routing(self, source: str, destination: str, metadata: dict) -> OutputRouting:
        """Build OutputRouting from source/destination/metadata.

        Maps source strings (cli, discord_dm, discord_channel, web, api) to
        OutputDestination enums and constructs proper routing for the packet.
        """
        # Map source string to OutputDestination enum
        source_destination_map = {
            "cli": OutputDestination.CLI,
            "cli_chat": OutputDestination.CLI,
            "discord": OutputDestination.DISCORD,
            "discord_dm": OutputDestination.DISCORD,
            "discord_channel": OutputDestination.DISCORD,
            "web": OutputDestination.WEB,
            "api": OutputDestination.API,
            "webhook": OutputDestination.WEBHOOK,
        }

        # Determine the source destination (where input came from)
        source_dest = source_destination_map.get(source, OutputDestination.CLI)

        # Determine primary output destination
        dest_dest = source_destination_map.get(destination, source_dest)

        # Extract routing details from metadata
        is_dm = metadata.get("is_dm", False)
        channel_id = metadata.get("channel_id")
        user_id = metadata.get("user_id") or metadata.get("author_id")
        message_id = metadata.get("message_id")
        addressed_to_gaia = metadata.get("addressed_to_gaia", True)

        # Build the primary destination target
        primary = DestinationTarget(
            destination=dest_dest,
            channel_id=channel_id,
            user_id=user_id,
            reply_to_message_id=message_id,
            format_hint="markdown" if dest_dest == OutputDestination.DISCORD else None,
            metadata={
                "is_dm": is_dm,
                "source": source,
                "original_metadata": metadata,
            }
        )

        return OutputRouting(
            primary=primary,
            secondary=[],
            suppress_echo=False,
            addressed_to_gaia=addressed_to_gaia,
            source_destination=source_dest,
        )

    def _create_initial_packet(
        self,
        user_input: str,
        session_id: str,
        history: List[Dict[str, Any]],
        selected_model_name: str,
        source: str = "cli",
        destination: str = "cli_chat",
        metadata: dict = None
    ) -> CognitionPacket:
        """Creates the initial v0.3 Cognition Packet for this turn.

        Args:
            user_input: The user's message
            session_id: Session identifier
            history: Conversation history
            selected_model_name: Model to use for this turn
            source: Input source (cli, discord_dm, discord_channel, web, api)
            destination: Output destination (cli_chat, discord, web, etc.)
            metadata: Additional context (is_dm, user_id, channel_id, etc.)
        """
        now = datetime.utcnow()
        active_persona = self.ai_manager.active_persona        
        model_config = self.config # Use the main config object

        # Populate persona with an immutable Tier-1 identity pulled from config
        persona_identity = getattr(self.config, 'identity', 'GAIA - General Artisanal Intelligence Architecture')
        persona_safety_ref = str(getattr(self.config, 'identity_file_path', '') or '')

        # Build persona traits safely from active_persona (may be a simple object)
        persona_traits = {}
        try:
            persona_traits = getattr(active_persona, 'traits', {}) or {}
        except Exception:
            persona_traits = {}

        header = Header(
            datetime=now.isoformat(),
            session_id=session_id,
            packet_id=str(uuid.uuid4()),
            sub_id="001",
            parent_packet_id=None,
            lineage=[],
            persona=Persona(
                identity_id=persona_identity,
                persona_id=active_persona.name,
                role=PersonaRole.DEFAULT, # Placeholder, should be derived from persona
                tone_hint=persona_traits.get("tone", ""), # Get tone from traits dictionary
                safety_profile_id=persona_safety_ref or None,
                traits=persona_traits,
            ),
            origin=Origin.USER,
            routing=Routing(target_engine=TargetEngine.PRIME), # Placeholder
            model=Model(
                name=selected_model_name,
                provider=getattr(model_config, 'provider', 'unknown'),
                context_window_tokens=self.config.max_tokens_lite if selected_model_name == "lite" else getattr(model_config, 'context_length', 8192),
                max_output_tokens=self.config.max_tokens
            )
        )

        # Build output routing from source/destination/metadata
        _metadata = metadata or {}
        output_routing = self._build_output_routing(source, destination, _metadata)
        header.output_routing = output_routing

        intent = Intent(
            user_intent="", # To be filled by NLU
            system_task=SystemTask.INTENT_DETECTION,
            confidence=0.0,
            tags=[]
        )

        # Sliding window: only the most recent turn-pairs go into the packet directly.
        # Older turns are retrieved via semantic RAG (see session_history_indexer).
        SLIDING_WINDOW_SIZE = 6  # Last 3 turn-pairs
        all_history = history or []
        window = all_history[-SLIDING_WINDOW_SIZE:]

        relevant_history_snippet = []
        for i, msg in enumerate(window):
            msg_id = msg.get('id', f"auto_{i}")
            msg_role = msg.get('role', 'unknown')
            msg_content = msg.get('content', '')
            relevant_history_snippet.append(
                RelevantHistorySnippet(id=msg_id, role=msg_role, summary=strip_think_tags(msg_content[:2000]))
            )
        if any(m.get('id') is None for m in window):
            self.logger.warning("AgentCore: Missing IDs found in history; temporary auto_ IDs assigned.")

        # Discover available MCP tools
        available_tools = []
        try:
            tool_info = mcp_client.discover()
            if tool_info and tool_info.get("ok"):
                available_tools = tool_info.get("tools", [])
        except Exception as e:
            self.logger.error(f"Failed to discover MCP tools: {e}")

        context = Context(
            session_history_ref=SessionHistoryRef(type="hash", value=""), # Placeholder
            cheatsheets=[Cheatsheet(id='default', title='GAIA Protocol Cheatsheet', version='1.0', pointer=str(self.config.cheat_sheet_path))] if self.config.cheat_sheet else [],
            constraints=Constraints(
                max_tokens=self.config.max_tokens_lite if selected_model_name == "lite" else self.config.max_tokens,
                time_budget_ms=5000, # Placeholder
                safety_mode="normal",
                policies=[]
            ),
            relevant_history_snippet=relevant_history_snippet,
            available_mcp_tools=available_tools
        )

        content = Content(original_prompt=user_input)
        # Include the immutable Tier-1 identity and a short intro in the packet content
        try:
            content.data_fields.append(DataField(key='immutable_identity', value=getattr(self.config, 'identity', '')))
            intro = getattr(self.config, 'identity_intro', '')
            if intro:
                # keep the intro concise in the packet
                content.data_fields.append(DataField(key='immutable_identity_intro', value=intro[:200]))
            summary = getattr(self.config, 'identity_summary', '')
            if summary:
                content.data_fields.append(DataField(key='identity_summary', value=summary[:400]))
            
            # --- NEW: Add the world state as a DataField ---
            from gaia_core.utils.world_state import format_world_state_snapshot
            world_state_text = ""
            try:
                # Include output context so GAIA knows where she's communicating
                output_context = {
                    "source": source,
                    "destination": destination,
                    **(_metadata or {}),
                }
                world_state_text = format_world_state_snapshot(output_context=output_context)
            except Exception as e:
                self.logger.exception("AgentCore: Failed to format world state snapshot; world state will be missing from packet.")
                world_state_text = ""
            if world_state_text:
                content.data_fields.append(DataField(key='world_state_snapshot', value=world_state_text))
            # MCP capabilities are provided via dynamic world state; avoid duplicating in packet.
        except Exception:
            # best-effort; don't fail packet creation
            self.logger.exception("AgentCore: Failed to add identity and world state to packet.")
            pass

        # RAG retrieval for older session history (beyond the sliding window)
        if len(all_history) > SLIDING_WINDOW_SIZE:
            try:
                from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
                indexer = SessionHistoryIndexer.instance(session_id)
                results = indexer.retrieve(
                    query=user_input,
                    top_k_turns=3,
                    top_k_topics=2,
                    exclude_recent_n=SLIDING_WINDOW_SIZE
                )
                retrieved_context = _format_retrieved_session_context(results)
                if retrieved_context:
                    content.data_fields.append(DataField(
                        key='retrieved_session_context',
                        value=retrieved_context,
                        type='session_rag'
                    ))
                    self.logger.info(f"Session RAG: injected {len(results.get('turns', []))} turns + {len(results.get('topics', []))} topics")
            except Exception as e:
                self.logger.warning(f"Session RAG retrieval failed (non-fatal): {e}")

        # Initialize empty containers
        reasoning = Reasoning()
        response = Response(candidate="", confidence=0.0, stream_proposal=True)
        governance = Governance(safety=Safety(execution_allowed=False, dry_run=True))
        
        # --- NEW: Add system resource metrics ---
        # TODO: [GAIA-REFACTOR] telemetric_senses.py module not yet migrated.
        # from app.cognition.telemetric_senses import get_system_resources
        try:
            from gaia_core.cognition.telemetric_senses import get_system_resources
            system_resources = get_system_resources()
        except ImportError:
            system_resources = None  # Placeholder until migration
        
        metrics = Metrics(
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), 
            latency_ms=0,
            resources=system_resources
        )
        status = Status(finalized=False, state=PacketState.INITIALIZED, next_steps=["Detect intent"])

        return CognitionPacket(
            version="v0.3",
            schema_id="https://gaia.local/schemas/cognitive_packet/v0.3.json",
            header=header,
            intent=intent,
            context=context,
            content=content,
            reasoning=reasoning,
            response=response,
            governance=governance,
            metrics=metrics,
            status=status
        )

    def _run_pre_generation_safety_check(self, packet: CognitionPacket, assembled_prompt: str) -> (bool, str):
        """
        Run the EthicalSentinel (preferred) or CoreIdentityGuardian (fallback) to determine
        whether generation should proceed.

        Returns (allowed: bool, reason: str). reason is one of:
          - 'ok'
          - 'identity_violation'
          - 'system_condition'
          - 'guardian_exception'
        """
        try:
            # Ensure packet contains a usable identity. Prefer header.persona.identity_id,
            # fall back to content.data_fields['immutable_identity'] or configured identity.
            try:
                persona_identity = getattr(packet.header.persona, 'identity_id', None)
            except Exception:
                persona_identity = None
            if not persona_identity:
                # Attempt to pull from content.data_fields if present
                try:
                    for df in getattr(packet, 'content', None).data_fields or []:
                        if getattr(df, 'key', '') == 'immutable_identity' and df.value:
                            persona_identity = df.value
                            break
                except Exception:
                    persona_identity = None
            if not persona_identity:
                # As a last resort, inject the configured identity into the packet so
                # downstream guardians and observers can validate against it.
                try:
                    cfg_identity = getattr(self.config, 'identity', None) or getattr(self.config, 'IDENTITY', None)
                    if cfg_identity:
                        try:
                            packet.header.persona.identity_id = cfg_identity
                        except Exception:
                            # header.persona may be frozen or not support assignment; record in content
                            try:
                                packet.content.data_fields.append(DataField(key='immutable_identity', value=cfg_identity))
                            except Exception:
                                pass
                        logger.warning(f"AgentCore: injected missing persona identity into packet: {cfg_identity}")
                except Exception:
                    logger.debug("AgentCore: failed to inject config identity into packet", exc_info=True)
            # Prefer the ethical_sentinel if present (it will call the identity guardian internally if configured)
            if self.ethical_sentinel:
                try:
                    allowed = self.ethical_sentinel.run_full_safety_check(
                        persona_traits=getattr(packet.header.persona, 'traits', None) or {},
                        instructions=[r.summary for r in packet.reasoning.reflection_log],
                        prompt=assembled_prompt
                    )
                except Exception:
                    # Fail closed on sentinel exceptions
                    logger.exception("AgentCore: ethical_sentinel raised exception during safety check")
                    return False, 'guardian_exception'

                if not allowed:
                    # Disambiguate whether this was an identity violation or broader system condition
                    id_guardian = getattr(self.ethical_sentinel, 'identity_guardian', None)
                    id_ok = True
                    if id_guardian:
                        try:
                            id_ok = id_guardian.validate_prompt_stack(
                                getattr(packet.header.persona, 'traits', {}) or {},
                                [r.summary for r in packet.reasoning.reflection_log] or [],
                                assembled_prompt
                            )
                        except Exception:
                            id_ok = False
                    if not id_ok:
                        return False, 'identity_violation'
                    return False, 'system_condition'

                return True, 'ok'

            # Fallback: look for an identity_guardian on the agent or ai_manager.
            # When gathering persona traits, attempt to use active_persona traits if the
            # packet header doesn't include them (older v0.3 Persona lacks 'traits').
            identity_guardian = getattr(self, 'identity_guardian', None) or getattr(self.ai_manager, 'identity_guardian', None)
            if identity_guardian:
                # Build persona_traits robustly
                persona_traits = {}
                try:
                    persona_traits = getattr(packet.header.persona, 'traits', {}) or {}
                except Exception:
                    persona_traits = {}
                if not persona_traits:
                    try:
                        ap = getattr(self.ai_manager, 'active_persona', None)
                        if ap and hasattr(ap, 'traits'):
                            persona_traits = getattr(ap, 'traits') or {}
                    except Exception:
                        persona_traits = {}

                try:
                    allowed = identity_guardian.validate_prompt_stack(
                        persona_traits,
                        [r.summary for r in packet.reasoning.reflection_log] or [],
                        assembled_prompt
                    )
                except Exception:
                    logger.exception("AgentCore: identity_guardian raised exception during safety check")
                    return False, 'guardian_exception'

                if not allowed:
                    return False, 'identity_violation'
                return True, 'ok'

            # No guardian/sentinel configured -> permissive
            return True, 'ok'
        except Exception:
            logger.exception("AgentCore: unexpected error during pre-generation safety check")
            return False, 'guardian_exception'

    def run_turn(
        self,
        user_input: str,
        session_id: str,
        destination: str = "cli_chat",
        source: str = "cli",
        metadata: dict = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        The primary cognitive loop. Creates a packet and processes it through
        planning, reflection, and execution stages.

        Args:
            user_input: The user's message
            session_id: Session identifier (e.g., "discord_dm_12345" for DMs)
            destination: Output destination (cli_chat, discord, web, etc.)
            source: Input source (cli, discord_channel, discord_dm, web, api)
            metadata: Additional context (is_dm, user_id, channel_id, etc.)
        """
        self.logger.info("--- AGENT CORE: RUN TURN ---")
        # Normalize metadata
        _metadata = metadata or {}
        import json
        import os
        import sys
        from gaia_core.utils.prompt_builder import build_from_packet # Assumes this is updated for v0.3

        import time as _time
        t0 = _time.perf_counter()
        try:
            logger.info("AgentCore: run_turn start")
            logger.debug("[DEBUG] AgentCore.run_turn start session_id=%s input_len=%d destination=%s", session_id, len(user_input or ""), destination)
        except Exception:
            logger.debug("[DEBUG] AgentCore.run_turn start (metrics unavailable)")

        # --- Loop Detection: Initialize and check for active recovery ---
        try:
            loop_config = build_loop_detection_config_from_constants(constants)
            loop_manager = get_recovery_manager()
            loop_manager.config = loop_config
            loop_manager.enabled = constants.get("LOOP_DETECTION_ENABLED", True)
        except Exception:
            logger.debug("Loop detection initialization failed, continuing without", exc_info=True)
            loop_manager = None

        # Check for pending loop recovery context from previous reset
        loop_recovery_context = None
        if loop_manager:
            try:
                loop_recovery_context = loop_manager.get_recovery_context()
                if loop_recovery_context:
                    logger.info("Loop recovery context pending, will inject into prompt")
            except Exception:
                logger.debug("Failed to get loop recovery context", exc_info=True)

        # Determine persona and knowledge base for this turn
        persona_name, knowledge_base_name = get_persona_for_request(user_input)

        # Load the selected persona
        self.ai_manager.initialize(persona_name)
        
        # Role mapping (keep internal keys stable):
        # - "lite"  => Operator (CPU orchestrator: intent, tools, summaries, short answers)
        # - "prime"/"gpu_prime"/"cpu_prime" => Thinker (GPU/CPU polished or heavy answers)
        # - "oracle" => External/cloud escalation
        selected_model_name = None

        # 1. Model Selection (Simplified)
        # NEW: User's suggestion - if a knowledge base is triggered, prefer the GPU model.
        if knowledge_base_name:
            logger.info(f"[MODEL_SELECT] Knowledge base '{knowledge_base_name}' triggered, preferring gpu_prime.")
            for cand in ["gpu_prime", "prime"]:
                if cand in self.config.MODEL_CONFIGS:
                    selected_model_name = cand
                    logger.info(f"[MODEL_SELECT] Overriding model selection to '{selected_model_name}' due to RAG-enabled persona.")
                    break

        if not selected_model_name:
            # Respect runtime override via environment var GAIA_BACKEND or config.llm_backend
            backend_env = os.getenv("GAIA_BACKEND") or getattr(self.config, "llm_backend", None)
            logger.warning(f"[MODEL_SELECT DEBUG] backend_env={backend_env} pool_keys={list(self.model_pool.models.keys())}")
            if backend_env:
                # Use the backend if it's been loaded (or will be loaded); otherwise fallback
                if backend_env in self.model_pool.models:
                    selected_model_name = backend_env
                else:
                    logger.info(f"Requested GAIA_BACKEND='{backend_env}' not present in model pool; falling back to default selection")
                    logger.warning(f"[MODEL_SELECT DEBUG] pool keys at backend check: {list(self.model_pool.models.keys())}")

        if not selected_model_name:
            text_lower = user_input.lower()
            force_thinker = os.getenv("GAIA_FORCE_THINKER", "").lower() in ("1", "true", "yes")
            # Explicit callouts to Thinker (GPU) if available
            wants_thinker = force_thinker or any(tag in text_lower for tag in ["thinker:", "[thinker]", "::thinker"])
            # Default path: Operator (lite) handles most turns; escalate to Thinker if explicitly requested.
            if "oracle" in text_lower and self.config.use_oracle:
                selected_model_name = "oracle"
            elif wants_thinker:
                # Prefer GPU prime, then prime, then cpu_prime
                for cand in ["gpu_prime", "prime", "cpu_prime"]:
                    if cand in self.model_pool.models:
                        selected_model_name = cand
                        break
            # If still unset, prefer Operator (lite); fall back to prime if lite is missing.
            if not selected_model_name:
                if "lite" in self.model_pool.models:
                    selected_model_name = "lite"
                elif "prime" in self.model_pool.models:
                    selected_model_name = "prime"
                else:
                    selected_model_name = "gpu_prime" if "gpu_prime" in self.model_pool.models else "cpu_prime"
        
        # If the chosen model isn't present yet, try sensible fallbacks
        # Try lazy loading for the selected model first
        if selected_model_name not in self.model_pool.models:
            logger.info(f"Selected model '{selected_model_name}' not present in pool; attempting lazy load")
            try:
                self.model_pool.ensure_model_loaded(selected_model_name)
            except Exception as e:
                logger.warning(f"Lazy load failed for '{selected_model_name}': {e}")

        if selected_model_name not in self.model_pool.models:
            logger.info(f"Model '{selected_model_name}' still not available after lazy load; attempting fallback selection")
            logger.warning(f"[MODEL_SELECT DEBUG] pool keys before fallback: {list(self.model_pool.models.keys())}")
            # Try preferred backend first
            backend_env = os.getenv("GAIA_BACKEND") or getattr(self.config, "llm_backend", None)
            candidates = []
            if backend_env:
                candidates.append(backend_env)
            # Prefer Operator (lite) first, then Thinker tiers, then oracle/azrael
            candidates.extend(["lite", "gpu_prime", "prime", "cpu_prime", "oracle", "azrael"])
            found = None
            for cand in candidates:
                logger.warning(f"[MODEL_SELECT DEBUG] checking candidate: {cand}")
                # Try lazy loading for each candidate
                if cand not in self.model_pool.models:
                    logger.info(f"[MODEL_SELECT DEBUG] attempting lazy load for: {cand}")
                    try:
                        self.model_pool.ensure_model_loaded(cand)
                    except Exception as e:
                        logger.warning(f"Lazy load failed for '{cand}': {e}")
                if cand in self.model_pool.models:
                    found = cand
                    logger.warning(f"[MODEL_SELECT DEBUG] candidate found: {found}")
                    break
            if found:
                logger.info(f"Fallback selected model: {found}")
                selected_model_name = found
            else:
                yield {"type": "token", "value": "I am currently unable to process your request as my primary model is unavailable."}
                return

        # Heuristic escalation: if Operator (lite) was picked but the prompt looks complex,
        # auto-route to a Thinker when available (unless GAIA_FORCE_OPERATOR=1).
        try:
            if selected_model_name == "lite":
                force_operator = os.getenv("GAIA_FORCE_OPERATOR", "").lower() in ("1", "true", "yes")
                if not force_operator and self._should_escalate_to_thinker(user_input):
                    for cand in ["gpu_prime", "prime", "cpu_prime"]:
                        # Try lazy loading for escalation candidates
                        if cand not in self.model_pool.models:
                            try:
                                self.model_pool.ensure_model_loaded(cand)
                            except Exception:
                                pass
                        if cand in self.model_pool.models:
                            logger.info(f"[MODEL_SELECT DEBUG] escalating from lite-> {cand} based on heuristics")
                            selected_model_name = cand
                            break
        except Exception:
            logger.debug("Heuristic escalation check failed; continuing with selected model", exc_info=True)

        # Acquire the selected model. `selected_model_name` may already be a
        # concrete model key (e.g. 'gpu_prime' or 'prime') or it may be a role
        # name (e.g. 'responder'). Prefer using the acquire_model path even when
        # the concrete key exists so we enforce status updates and runtime
        # wrapping (ensuring callers receive a SafeModelProxy rather than a
        # raw backend object).
        if selected_model_name in self.model_pool.models:
            # Use acquire_model (not get) to ensure wrapping and busy status.
            selected_model = self.model_pool.acquire_model(selected_model_name)
        else:
            selected_model = self.model_pool.acquire_model_for_role(selected_model_name)
        logger.warning(f"[MODEL_SELECT DEBUG] acquired selected_model_name={selected_model_name} -> {selected_model}")
        logger.info(f"[RESPONDER] Selected responder model: '{selected_model_name}'")
        logger.info("AgentCore: selected model %s", selected_model_name)
        logger.debug("[DEBUG] AgentCore selected_model=%s", selected_model_name)
        # Extra runtime guard: if a raw llama_cpp.Llama escaped into the pool
        # (some scripts/tests assign directly to model_pool.models), wrap it
        # now so downstream calls always operate on SafeModelProxy.
        try:
            llama_mod = __import__("llama_cpp")
            Llama = getattr(llama_mod, "Llama", None)
            if Llama is not None and isinstance(selected_model, Llama):
                from gaia_core.models.model_pool import SafeModelProxy
                wrapped = SafeModelProxy(selected_model, pool=self.model_pool, role=selected_model_name)
                # update pool and local reference
                try:
                    self.model_pool.models[selected_model_name] = wrapped
                except Exception:
                    logger.debug("AgentCore: failed to replace pool entry with proxy for %s", selected_model_name, exc_info=True)
                selected_model = wrapped
                logger.warning("AgentCore: wrapped raw llama_cpp.Llama for '%s' into SafeModelProxy", selected_model_name)
        except Exception:
            # best-effort guard; continue even if wrapping fails
            logger.debug("AgentCore: runtime wrapping guard failed for %s", selected_model_name, exc_info=True)
        if not selected_model:
            yield {"type": "token", "value": "I am currently unable to process your request as my primary model is busy or unavailable."}
            return

        # 2. Create the initial v0.3 Cognition Packet
        history = self.session_manager.get_history(session_id).copy()
        packet = self._create_initial_packet(
            user_input, session_id, history, selected_model_name,
            source=source, destination=destination, metadata=_metadata
        )
        self.session_manager.add_message(session_id, "user", user_input)

        if knowledge_base_name:
            packet.content.data_fields.append(DataField(key='knowledge_base_name', value=knowledge_base_name, type='string'))
        
        # Enhance the packet with knowledge from the knowledge base
        packet = enhance_packet(packet)

        # RAG: Retrieve documents from vector store if a knowledge base is specified
        try:
            from gaia_core.utils import mcp_client
            
            # Extract knowledge_base_name from the packet's data_fields
            knowledge_base_name = None
            for field in packet.content.data_fields:
                if field.key == 'knowledge_base_name':
                    knowledge_base_name = field.value
                    break
            
            if knowledge_base_name:
                self.logger.info(f"Performing RAG query on knowledge base: {knowledge_base_name}")
                retrieved_docs = mcp_client.embedding_query(
                    packet.content.original_prompt, 
                    top_k=3, 
                    knowledge_base_name=knowledge_base_name
                )
                if retrieved_docs and retrieved_docs.get("ok"):
                    docs = retrieved_docs.get("results", [])  # Fixed: was "response", should be "results"
                    if docs:
                        packet.content.data_fields.append(DataField(key='retrieved_documents', value=docs, type='list'))
                        self.logger.info(f"Retrieved {len(docs)} documents from '{knowledge_base_name}'.")
                    else:
                        # No documents found - signal epistemic uncertainty
                        packet.content.data_fields.append(DataField(
                            key='rag_no_results',
                            value=True,
                            type='bool'
                        ))
                        self.logger.warning(f"RAG query returned no documents from '{knowledge_base_name}' - epistemic uncertainty flagged.")
                else:
                    # RAG query failed
                    error_msg = retrieved_docs.get("error", "Unknown error") if retrieved_docs else "No response"
                    self.logger.warning(f"RAG query failed for '{knowledge_base_name}': {error_msg}")
                    packet.content.data_fields.append(DataField(
                        key='rag_no_results',
                        value=True,
                        type='bool'
                    ))
            else:
                self.logger.info("No knowledge_base_name specified in the packet; skipping RAG query.")
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents from vector store: {e}")

        # If RAG query returns no results, attempt to find and embed the knowledge
        if any(df.key == 'rag_no_results' and df.value for df in packet.content.data_fields):
            packet = self._knowledge_acquisition_workflow(packet)



        # Compatibility shims removed: callers should use the v0.3 packet
        # contract (header.persona, context.cheatsheets, content.original_prompt,
        # and reasoning.reflection_log). The StreamObserver and other helpers
        # already check for legacy attributes via hasattr when needed.

        # Normalization is handled centrally by build_from_packet() in the prompt builder.
        # AgentCore no longer performs message role normalization here.

        # 3. Intent Detection
        # Prefer Lite explicitly; avoid Prime/vLLM here to reduce errors.
        lite_llm = None
        try:
            lite_llm = self.model_pool.models.get("lite")
            if lite_llm:
                self.model_pool.set_status("lite", "busy")
        except Exception:
            lite_llm = None
        prime_llm = None  # do not use prime for intent detection
        fallback_llm = lite_llm or selected_model
        plan = None
        try:
            plan = detect_intent(
                user_input,
                self.config,
                lite_llm=lite_llm,
                full_llm=prime_llm,
                fallback_llm=fallback_llm,
            )
        finally:
            # Use role-aware release via ModelPool API; let exceptions be logged but
            # don't rely on hasattr guards anymore since ModelPool now provides
            # `release_model_for_role`.
            try:
                if lite_llm:
                    self.model_pool.release_model_for_role("lite")
                self.model_pool.release_model_for_role("prime")
            except Exception:
                self.logger.debug("AgentCore: release_model_for_role failed during intent-detect cleanup", exc_info=True)
        
        # 3a. Fast local knowledge check: if a fact exists for this question,
        # bypass the model and respond immediately.
        try:
            local_fact = rescue_helper.recall_fact(key=user_input, limit=1)
        except Exception:
            local_fact = ""
        if local_fact and not local_fact.strip().startswith("🧠 Memory store is empty"):
            yield {"type": "token", "value": local_fact}
            # Also record in history as assistant reply
            self.session_manager.add_message(session_id, "assistant", local_fact)
            return

        packet.intent.user_intent = plan.intent
        packet.intent.confidence = 0.9 # Placeholder
        packet.status.state = PacketState.PROCESSING
        packet.content.data_fields.append(DataField(key='read_only_intent', value=plan.read_only, type='boolean'))
        ts_write({"type": "intent_detect", "intent": plan.intent, "read_only": plan.read_only}, session_id, source=source, destination_context=_metadata)

        # 3b. GCP Tool Routing System: Check if request needs MCP tools
        # This runs before the slim prompt path to properly route tool-related requests
        if self._should_use_tool_routing(plan, user_input):
            logger.info(f"Tool routing triggered for intent: {plan.intent}")
            packet = self._run_tool_routing_loop(
                packet=packet,
                user_input=user_input,
                session_id=session_id,
                source=source,
                metadata=_metadata
            )

            # If tool was executed successfully, we may have results to include
            if packet.tool_routing and packet.tool_routing.execution_status == ToolExecutionStatus.EXECUTED:
                # Tool was executed - the result is now in the packet
                # Continue to normal processing which will include the tool result in context
                logger.info("Tool executed successfully, continuing with enhanced context")
                ts_write({
                    "type": "tool_routing",
                    "stage": "continuing_with_result",
                    "tool": packet.tool_routing.selected_tool.tool_name if packet.tool_routing.selected_tool else None
                }, session_id, source=source, destination_context=_metadata)

        # Fast-path slim prompt: low-complexity intents (incl. list_tools) avoid the heavy
        # planning/reflector stack. Uses minimal identity + MCP summary + user input.
        if self._should_use_slim_prompt(plan, user_input):
            text = self._run_slim_prompt(selected_model_name, user_input, history, plan.intent, session_id=session_id, source=source, metadata=_metadata, packet=packet)
            yield {"type": "token", "value": text}
            self.session_manager.add_message(session_id, "assistant", text)
            return

        # 4. Initial Planning & Reflection
        if plan.intent:
            codex = SemanticCodex.instance(self.config)
            codex_symbol = f"§INTENT/{plan.intent.upper()}"
            entry = codex.get(codex_symbol)
            if entry:
                packet.content.data_fields.append(DataField(key='proactive_codex', value=entry.body, type='text'))
                logger.info(f"Proactively loaded codex entry '{codex_symbol}' for intent '{plan.intent}'.")

        # Dump the packet identity and data_fields so we can verify identity propagation
        try:
            pid = getattr(packet.header.persona, 'identity_id', None)
            df_repr = [(getattr(df, 'key', None), getattr(df, 'value', None)) for df in getattr(packet.content, 'data_fields', [])]
            logger.warning("AgentCore: packet persona.identity_id=%s, content.data_fields=%s", pid, df_repr)
        except Exception:
            logger.exception("AgentCore: failed to log packet identity/data_fields")

        plan_messages = build_from_packet(packet, task_instruction_key="initial_planning")
        # Emit the assembled plan messages to stderr for debugging so we can
        # confirm exactly what the model receives (ensures system message is first).
        try:
            _s = json.dumps(plan_messages, default=str, ensure_ascii=False, indent=2)
            print("===ASSEMBLED_PLAN_MESSAGES_START===", file=sys.stderr)
            # truncate to avoid excessive output
            print(_s[:20000], file=sys.stderr)
            print("===ASSEMBLED_PLAN_MESSAGES_END===", file=sys.stderr)
        except Exception:
            logger.exception("AgentCore: failed to dump plan_messages to stderr")
        # build_from_packet() now normalizes roles/alternation for chat formatters.
        # create_chat_completion may return a dict (non-stream) or a generator (stream).
        import types
        # Use ModelPool.forward_to_model to centralize acquisition, wrapping,
        # and release semantics (this avoids directly invoking backend objects
        # that may bypass our SafeModelProxy protections).
        logger.warning("AgentCore: entering planning call with model=%s", selected_model_name)
        try:
            # Release any earlier acquisition so forward_to_model manages lifecycle.
            try:
                self.model_pool.release_model(selected_model_name)
            except Exception:
                pass
            plan_res = self.model_pool.forward_to_model(
                selected_model_name,
                messages=plan_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        except Exception as exc:
            logger.exception("AgentCore: forward_to_model failed for %s", selected_model_name)
            yield {"type": "token", "value": f"[plan-error] {type(exc).__name__}: {exc}"}
            return
        logger.warning("AgentCore: planning call returned type=%s", type(plan_res))
        ts_write({"type": "planning_raw_response", "response": str(plan_res)}, session_id, source=source, destination_context=_metadata)
        if isinstance(plan_res, types.GeneratorType):
            pieces = []
            for item in plan_res:
                # item may be streaming delta dicts: {"choices":[{"delta": {"content": "..."}}]}
                try:
                    delta = item.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        pieces.append(text)
                        continue
                except Exception:
                    pass
                # fallback to non-stream shape
                try:
                    text = item.get("choices", [{}])[0].get("message", {}).get("content")
                    if text:
                        pieces.append(text)
                except Exception:
                    pass
            initial_plan_text = "".join(pieces).strip()
        else:
            try:
                initial_plan_text = plan_res["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.exception("AgentCore: failed to parse plan response (%s)", type(plan_res))
                yield {"type": "token", "value": f"[plan-parse-error] {type(exc).__name__}: {exc}"} 
                return
        packet.reasoning.reflection_log.append(ReflectionLog(step="initial_plan", summary=initial_plan_text, confidence=0.8)) # Placeholder confidence

        # pick a reflection model that's idle and not the selected model; prefer role resolution
        # Pick a reflection model that's idle and not the selected model.
        # Rely on `get_idle_model` being present on ModelPool; fall back to a
        # simple pool-key scan only if the call fails at runtime.
        reflection_model_name = None
        reflection_model = None
        preferred_reflector = os.getenv("GAIA_REFLECTION_MODEL", "").strip().lower()
        if preferred_reflector in ("", "prime", "gpu_prime", "selected", selected_model_name.lower()):
            reflection_model = selected_model
        elif preferred_reflector:
            try:
                reflection_model = self.model_pool.acquire_model_for_role(preferred_reflector)
                reflection_model_name = preferred_reflector
            except Exception:
                reflection_model = None
                reflection_model_name = None
        if reflection_model is None:
            try:
                reflection_model_name = self.model_pool.get_idle_model(exclude=[selected_model_name])
            except Exception:
                reflection_model_name = None
                try:
                    pool_keys = list(getattr(self.model_pool, "models", {}).keys())
                    candidates = [k for k in pool_keys if k != selected_model_name and k != "embed"]
                    reflection_model_name = candidates[0] if candidates else None
                except Exception:
                    reflection_model_name = None
            if reflection_model_name:
                # get() now supports lazy loading - no need to pre-check pool
                reflection_model = self.model_pool.get(reflection_model_name)
                if reflection_model is None:
                    reflection_model = self.model_pool.acquire_model_for_role(reflection_model_name)
            if reflection_model is None:
                reflection_model = selected_model
        try:
            refined_plan_text = reflect_and_refine(packet=packet, output=initial_plan_text, config=self.config, llm=reflection_model, ethical_sentinel=self.ethical_sentinel) # Instructions are now in the packet
            # reflect_and_refine now appends iteration logs directly; add a summary entry for the refined plan
            try:
                packet.reasoning.reflection_log.append(ReflectionLog(step="refined_plan", summary=refined_plan_text, confidence=0.88))
            except Exception:
                logger.debug("AgentCore: failed to append refined_plan reflection log")
            logger.debug(f"Reflection output: {refined_plan_text}")
            # Mark packet as ready to stream after successful reflection
            try:
                packet.status.state = PacketState.READY_TO_STREAM
                packet.status.next_steps.append("Ready for final generation")
            except Exception:
                logger.debug("AgentCore: failed to set packet to READY_TO_STREAM")
        finally:
            if reflection_model_name:
                try:
                    self.model_pool.release_model_for_role(reflection_model_name)
                except Exception:
                    logger.debug("release_model_for_role failed", exc_info=True)
        ts_write({"type": "reflection-pre", "packet_id": packet.header.packet_id, "out": refined_plan_text}, session_id, source=source, destination_context=_metadata)

        # 5. Final Response Generation
        final_messages = build_from_packet(packet)
        # Also print the final assembled messages immediately before generation.
        try:
            with open("final_prompt_for_review.json", "w") as f:
                _s = json.dumps(final_messages, default=str, ensure_ascii=False, indent=2)
                f.write(_s)
        except Exception:
            logger.exception("AgentCore: failed to write final_messages to file")
        # build_from_packet() normalizes messages; no runtime normalization required here.

        # Assemble a simple prompt string for guardian/sentinel checks (concatenate message contents)
        try:
            assembled_prompt = "\n".join([m.get('content', '') for m in final_messages if isinstance(m, dict)])
        except Exception:
            assembled_prompt = packet.content.original_prompt or ""

        # Pre-generation safety check: prefer EthicalSentinel (which may call the CoreIdentityGuardian)
        ok, reason = self._run_pre_generation_safety_check(packet, assembled_prompt)
        if not ok:
            # Fail closed: block generation and return a clear, distinct message based on reason
            packet.status.finalized = True
            packet.status.state = PacketState.ABORTED
            packet.status.next_steps.append(f"Blocked by safety check: {reason}")
            packet.metrics.errors.append(f"blocked:{reason}")
            if reason == 'identity_violation':
                user_msg = "I can't comply with that request because it would violate my core identity constraints."
            elif reason == 'system_condition':
                user_msg = "I can't process that right now due to system safety conditions (resource/loop/error)."
            else:
                user_msg = "I can't process that request due to an internal safety check failure."

            logger.error(f"[CoreIdentityGuardian] Blocking generation: {reason}")
            yield {"type": "token", "value": user_msg}
            return

        # pick an observer model that's idle if possible; fall back to config-based selection
        logger.info(f"[OBSERVER] Selecting observer model. Responder is '{selected_model_name}'")
        logger.info(f"[OBSERVER] Available models in pool: {list(getattr(self.model_pool, 'models', {}).keys())}")
        logger.info(f"[OBSERVER] Model statuses: {getattr(self.model_pool, 'model_status', {})}")

        # Check observer config for explicit model preference
        observer_config = self.config.constants.get("MODEL_CONFIGS", {}).get("observer", {})
        observer_enabled = observer_config.get("enabled", False)
        use_gpu_prime_for_observer = observer_config.get("use_gpu_prime", False)
        logger.info(f"[OBSERVER] Observer config: enabled={observer_enabled}, use_gpu_prime={use_gpu_prime_for_observer}")

        observer_model_name = None
        try:
            observer_model_name = self.model_pool.get_idle_model(exclude=[selected_model_name])
            logger.info(f"[OBSERVER] get_idle_model returned: '{observer_model_name}'")
        except Exception as e:
            logger.warning(f"[OBSERVER] get_idle_model failed: {e}")

        # If no idle model found but observer is configured to use gpu_prime, try to load it
        if observer_model_name is None and observer_enabled and use_gpu_prime_for_observer:
            logger.info("[OBSERVER] No idle model found, attempting to load gpu_prime for observer role")
            try:
                # Try to load gpu_prime on-demand via acquire_model_for_role
                if self.model_pool.acquire_model_for_role("gpu_prime", lazy_load=True):
                    observer_model_name = "gpu_prime"
                    logger.info("[OBSERVER] Successfully loaded gpu_prime for observer role")
                else:
                    logger.warning("[OBSERVER] Failed to load gpu_prime for observer role")
            except Exception as e:
                logger.warning(f"[OBSERVER] Exception loading gpu_prime: {e}")

        # Final fallback: scan pool for any available model
        if observer_model_name is None:
            try:
                pool_keys = list(getattr(self.model_pool, "models", {}).keys())
                candidates = [k for k in pool_keys if k != selected_model_name and k != "embed"]
                observer_model_name = candidates[0] if candidates else None
                logger.info(f"[OBSERVER] Fallback selection from candidates {candidates}: '{observer_model_name}'")
            except Exception as e2:
                logger.error(f"[OBSERVER] Fallback selection also failed: {e2}")
                observer_model_name = None

        def _env_bool(name: str) -> Optional[bool]:
            val = os.getenv(name)
            if val is None:
                return None
            return str(val).strip().lower() in ("1", "true", "yes", "on")

        observer_model = None
        if observer_model_name:
            # get() now supports lazy loading - no need to pre-check pool
            observer_model = self.model_pool.get(observer_model_name)
            if observer_model is None:
                observer_model = self.model_pool.acquire_model_for_role(observer_model_name)
            logger.info(f"[OBSERVER] Acquired observer model '{observer_model_name}': {observer_model is not None}")
        else:
            logger.warning("[OBSERVER] No observer model available")

        disable_observer_env = _env_bool("GAIA_DISABLE_OBSERVER")
        disable_observer_const = self.config.constants.get("OBSERVER_DISABLE_DEFAULT", False)
        if isinstance(disable_observer_const, str):
            disable_observer_const = disable_observer_const.strip().lower() in ("1", "true", "yes", "on")
        disable_observer = disable_observer_env if disable_observer_env is not None else bool(disable_observer_const)

        observer_use_llm = self.config.constants.get("OBSERVER_USE_LLM", False)
        logger.info(f"[OBSERVER] Config: disable_observer={disable_observer}, OBSERVER_USE_LLM={observer_use_llm}")

        post_stream_override = _env_bool("GAIA_OBSERVER_POST_STREAM")
        post_stream_const = self.config.constants.get("OBSERVER_POST_STREAM_DEFAULT", False)
        if isinstance(post_stream_const, str):
            post_stream_const = post_stream_const.strip().lower() in ("1", "true", "yes", "on")
        post_stream_observer = post_stream_override if post_stream_override is not None else bool(post_stream_const)

        observer_instance = None
        if not disable_observer and observer_model is not None:
            observer_instance = StreamObserver(config=self.config, llm=observer_model, name="AgentCore-Observer")
            observer_instance.post_stream_only = post_stream_observer
            logger.info(f"[OBSERVER] StreamObserver created: model='{observer_model_name}', post_stream_only={post_stream_observer}")
        else:
            reason = "disabled" if disable_observer else "no model available"
            logger.warning(f"[OBSERVER] StreamObserver NOT created: {reason}")

        active_stream_observer = None if post_stream_observer else observer_instance
        post_run_observer = observer_instance if post_stream_observer else None

        logger.info(f"[OBSERVER] Final setup: active_stream={active_stream_observer is not None}, post_run={post_run_observer is not None}")

        import json
        import sys
        print("---PROMPT TO BE SENT TO THE MODEL---", file=sys.stderr)
        print(json.dumps(final_messages, indent=2), file=sys.stderr)
        print("------------------------------------", file=sys.stderr)

        voice = ExternalVoice(
            model=selected_model,
            model_pool=self.model_pool,
            config=self.config,
            messages=final_messages,
            source="agent_core",
            observer=active_stream_observer,
            context={"packet": packet, "ethical_sentinel": self.ethical_sentinel},
            session_id=session_id,
        )
        lite_fallback_used = False
        lite_fallback_model_name = None
        lite_fallback_acquired = False
        try:
            try:
                stream_generator = voice.stream_response()
                pieces: List[str] = []
                # consume the generator and handle interruption events
                for item in stream_generator:
                    # item can be a token string or an event dict
                    if isinstance(item, dict):
                        ev = item.get("event")
                        if ev == "interruption":
                            reason_data = item.get("data", "observer interruption")
                            # data may be a dict with reason and suggestion (explain mode)
                            if isinstance(reason_data, dict):
                                reason = reason_data.get("reason") or reason_data.get("reason", "observer interruption")
                                suggestion = reason_data.get("suggestion")
                            else:
                                reason = reason_data
                                suggestion = None

                            logger.warning(f"AgentCore: Observer interruption during stream: {reason}")
                            # record in packet status if available
                            try:
                                packet.status.observer_trace.append(f"STREAM_INTERRUPT: {reason}")
                                packet.status.next_steps.append(f"Observer interruption: {reason}")
                            except Exception:
                                logger.debug("AgentCore: failed to update packet status with observer interruption")
                            # surface a user-facing notification; include suggestion when present
                            user_msg = f"--- Stream interrupted by observer: {reason} ---"
                            if suggestion:
                                user_msg += f"\nSuggestion: {suggestion}"
                            yield {"type": "token", "value": user_msg}
                            # abort consuming further stream tokens
                            break
                        else:
                            logger.debug(f"AgentCore: stream event ignored: {ev}")
                            continue
                    else:
                        pieces.append(str(item))

                full_response = "".join(pieces)
                logger.warning(
                    "AgentCore: collected %s stream chunks (len=%s chars)",
                    len(pieces),
                    len(full_response),
                )
            except Exception:
                # If the GPU/backed engine dies or the stream raises, attempt a
                # graceful fallback to the 'lite' CPU model so the session stays
                # usable. We surface a brief user-facing message and then try to
                # re-run generation on the lite model.
                logger.exception("AgentCore: Primary model stream failed; attempting fallback to 'lite' model")
                yield {"type": "token", "value": "Primary model failed during generation; falling back to CPU model (lite)."}
                try:
                    # Acquire lite for fallback
                    lite_model = self.model_pool.acquire_model_for_role("lite")
                    lite_fallback_acquired = True
                    lite_fallback_used = True
                    lite_fallback_model_name = "lite"
                    fallback_voice = ExternalVoice(model=lite_model, model_pool=self.model_pool, config=self.config, messages=final_messages, source="agent_core_fallback", observer=active_stream_observer, context={"packet": packet}, session_id=session_id)
                    pieces = []
                    stream_generator = fallback_voice.stream_response()
                    for item in stream_generator:
                        if isinstance(item, dict):
                            ev = item.get("event")
                            if ev == "interruption":
                                reason_data = item.get("data", "observer interruption")
                                if isinstance(reason_data, dict):
                                    reason = reason_data.get("reason") or reason_data.get("reason", "observer interruption")
                                    suggestion = reason_data.get("suggestion")
                                else:
                                    reason = reason_data
                                    suggestion = None
                                logger.warning(f"AgentCore: Observer interruption during fallback stream: {reason}")
                                try:
                                    packet.status.observer_trace.append(f"STREAM_INTERRUPT: {reason}")
                                    packet.status.next_steps.append(f"Observer interruption: {reason}")
                                except Exception:
                                    logger.debug("AgentCore: failed to update packet status with observer interruption")
                                user_msg = f"--- Stream interrupted by observer: {reason} ---"
                                if suggestion:
                                    user_msg += f"\nSuggestion: {suggestion}"
                                yield {"type": "token", "value": user_msg}
                                break
                            else:
                                logger.debug(f"AgentCore: fallback stream event ignored: {ev}")
                                continue
                        else:
                            pieces.append(str(item))
                    full_response = "".join(pieces)
                except Exception:
                    logger.exception("AgentCore: Fallback to 'lite' also failed")
                    yield {"type": "token", "value": "Fallback to CPU model also failed; I'm unable to complete your request right now."}
                    # Ensure any fallback acquisition is released in finally block
                    return
            full_response = self._suppress_repetition(full_response)
            logger.info(f"Full LLM response before routing: {full_response}")

            if post_run_observer and not disable_observer:
                try:
                    review = post_run_observer.observe(packet, full_response)
                    note = f"[Observer] {review.level.upper()}: {review.reason}"
                    if review.suggestion:
                        note += f" | Suggestion: {review.suggestion}"
                    logger.info("Post-stream observer review: %s", note)
                    yield {"type": "token", "value": f"\n\n{note}\n"}
                except Exception:
                    logger.warning("Post-stream observer review failed; continuing without interruption.", exc_info=True)

            # This function now needs to handle the v0.3 packet
            routed_output = route_output(full_response, packet, self.ai_manager, session_id, destination)
            user_facing_response = routed_output["response_to_user"]
            execution_results = routed_output["execution_results"]

            packet.reasoning.reflection_log.append(ReflectionLog(step="execution_results", summary=str(execution_results)))

            logger.debug(f"User-facing response after routing: {user_facing_response}")
            yield {"type": "token", "value": user_facing_response}
            logger.debug(f"Yielded to user: {user_facing_response}")

            # If the response came from the Oracle path, persist it as a learned fact
            # so future turns can answer locally without another external call.
            try:
                backend_env = (os.getenv("GAIA_BACKEND") or "").lower()
                if "oracle" in (selected_model_name or "").lower() or backend_env.startswith("oracle"):
                    rescue_helper.remember_fact(key=user_input, value=user_facing_response, note="sourced_from=oracle")
            except Exception:
                logger.debug("AgentCore: failed to log oracle-sourced fact", exc_info=True)
            
            # Use the routed user-facing response for all persistence.
            # full_response may be empty if the LLM stream yielded nothing,
            # but user_facing_response always has content (route_output provides fallbacks).
            packet.response.candidate = user_facing_response
            packet.response.confidence = 0.93 # Placeholder
            self.ai_manager.status["last_response"] = user_facing_response
            self.session_manager.add_message(session_id, "assistant", strip_think_tags(user_facing_response))

            if execution_results:
                process_execution_results(execution_results, self.session_manager, session_id, packet)

                messages = build_from_packet(packet, task_instruction_key="execution_feedback")
                voice = ExternalVoice(model=selected_model, model_pool=self.model_pool, config=self.config, messages=messages, source="agent_core", observer=active_stream_observer, context={"packet": packet}, session_id=session_id)
                stream_generator = voice.stream_response()
                concluding_response = "".join([str(token) for token in stream_generator if isinstance(token, str)])
                yield {"type": "token", "value": concluding_response}
                self.session_manager.add_message(session_id, "assistant", concluding_response)

            log_chat_entry(user_input, user_facing_response, source=source, session_id=session_id, metadata=_metadata)
            log_chat_entry_structured(user_input, user_facing_response, source=source, session_id=session_id, metadata=_metadata)
            ts_write({"type":"turn_end","user":user_input,"assistant":user_facing_response}, session_id, source=source, destination_context=_metadata)
            self.session_manager.record_last_activity()

            # --- Loop Detection: Record output and check for loops ---
            if loop_manager and loop_manager.enabled:
                try:
                    # Record the output for similarity detection
                    loop_manager.record_output(full_response)

                    # Record state snapshot
                    loop_manager.record_state(
                        goal=plan.intent if plan else "",
                        state_snapshot={
                            "packet_id": packet.header.packet_id,
                            "session_id": session_id,
                            "tool_routing": packet.tool_routing.execution_status.value if packet.tool_routing else None
                        }
                    )

                    # Check for loop patterns
                    loop_result = loop_manager.check_and_handle(
                        session_id=session_id,
                        packet_id=packet.header.packet_id,
                        goal=plan.intent if plan else "",
                        last_output=full_response
                    )

                    if loop_result and loop_result.is_loop:
                        # Update packet with loop state
                        from gaia_core.cognition.cognition_packet import LoopState, LoopAttempt
                        packet.loop_state = LoopState(
                            detected_at=loop_result.evidence.get("detected_at", ""),
                            loop_type=loop_result.primary_category.value,
                            pattern=loop_result.pattern,
                            reset_count=loop_result.reset_count,
                            confidence=loop_result.confidence,
                            triggered_by=loop_result.triggered_by,
                            in_recovery=not loop_result.should_warn,
                            warned=loop_result.should_warn
                        )

                        # Notify user
                        notification = loop_manager.get_notification(loop_result)
                        if loop_result.should_warn:
                            # Warning only - add note to output
                            warn_msg = f"\n\n---\n⚠️ **Loop Warning**: {notification.get('toast', {}).get('body', 'Repetitive pattern detected.')}\n---"
                            yield {"type": "token", "value": warn_msg}
                            logger.warning(f"Loop warning issued: {notification.get('status_line', '')}")
                        else:
                            # Reset triggered - will inject recovery context next turn
                            reset_msg = f"\n\n---\n🔄 **Loop Detected**: {notification.get('toast', {}).get('body', 'Resetting to try a different approach.')}\n---"
                            yield {"type": "token", "value": reset_msg}
                            logger.warning(f"Loop reset triggered: {notification.get('status_line', '')}")

                        ts_write({
                            "type": "loop_detection",
                            "pattern": loop_result.pattern,
                            "confidence": loop_result.confidence,
                            "action": "warn" if loop_result.should_warn else "reset",
                            "reset_count": loop_result.reset_count
                        }, session_id, source=source, destination_context=_metadata)

                    # Clear recovery context if we successfully completed without looping
                    if not loop_result or not loop_result.is_loop:
                        loop_manager.clear_recovery_context()

                except Exception:
                    logger.debug("Loop detection: post-turn check failed", exc_info=True)

            packet.status.finalized = True
            packet.status.state = PacketState.COMPLETED
            packet.compute_hashes() # Compute final integrity hashes
            logger.info(f"🧠 Final CognitionPacket: {packet.to_json()}")

            # Write a serializable version to thoughtstream to avoid Enum serialization issues
            try:
                ts_write({"type": "cognition_packet", "packet": packet.to_serializable_dict()}, session_id, source=source, destination_context=_metadata)
            except Exception:
                logger.exception("Failed to write cognition_packet to thoughtstream using serializable dict")

            # Analyze dev_matrix for task completion
            # TODO: [GAIA-REFACTOR] dev_matrix_analyzer.py module not yet migrated.
            try:
                from gaia_core.utils.dev_matrix_analyzer import DevMatrixAnalyzer
                dev_matrix_analyzer = DevMatrixAnalyzer(self.config)
                newly_resolved = dev_matrix_analyzer.analyze_and_update()
            except ImportError:
                newly_resolved = []  # Placeholder until migration

            # If we're in Discord and the Discord task was just resolved, log it prominently
            if newly_resolved and "discord" in source.lower():
                for task in newly_resolved:
                    task_name = task.get('task', '')
                    if 'discord' in task_name.lower():
                        logger.info(f"🎉 GAIA autonomously marked '{task_name}' as resolved while using Discord!")
                        ts_write({
                            "type": "autonomous_task_completion",
                            "task": task_name,
                            "context": "Resolved while actively using Discord integration",
                            "source": source,
                        }, session_id)
            
            packet.metrics.latency_ms = int((_time.perf_counter() - t0) * 1000)
            logger.info(f"AgentCore: run_turn total took {{packet.metrics.latency_ms / 1000:.2f}}s")
        finally:
            # Release observer/selected models using the ModelPool API directly.
            try:
                if observer_model_name:
                    self.model_pool.release_model_for_role(observer_model_name)
                self.model_pool.release_model_for_role(selected_model_name)
                # If we acquired a lite fallback, release it as well
                try:
                    if 'lite_fallback_acquired' in locals() and lite_fallback_acquired:
                        self.model_pool.release_model_for_role('lite')
                except Exception:
                    logger.debug("AgentCore: failed to release lite fallback model", exc_info=True)
            except Exception:
                self.logger.debug("AgentCore: release_model_for_role failed during final cleanup", exc_info=True)
            logger.info(f"AgentCore: Released model {selected_model_name}")

    def _knowledge_acquisition_workflow(self, packet: CognitionPacket) -> CognitionPacket:
        """
        Attempt to find, embed, and query relevant knowledge when the initial RAG query fails.
        """
        self.logger.info("Knowledge acquisition workflow ENTERED.")
        try:
            from gaia_core.utils import mcp_client
            knowledge_base_name = None
            for field in packet.content.data_fields:
                if field.key == 'knowledge_base_name':
                    knowledge_base_name = field.value
                    break
            
            if not knowledge_base_name:
                self.logger.warning("Knowledge acquisition workflow cannot proceed without a knowledge base name.")
                return packet

            # Step 1: Find relevant documents
            self.logger.info(f"Searching for relevant documents for query: {packet.content.original_prompt}")
            found_docs = mcp_client.call_jsonrpc(
                "find_relevant_documents",
                {"query": packet.content.original_prompt, "knowledge_base_name": knowledge_base_name}
            )
            self.logger.info(f"Found documents: {found_docs}")

            response_body = found_docs.get("response", {})
            # The tool output is nested inside the 'result' key of the JSON-RPC response
            result = response_body.get("result", {})
            
            if found_docs.get("ok") and result.get("files"):
                # Select the first matching file from the list
                doc_path = result["files"] 
                self.logger.info(f"Found relevant document: {doc_path}")

                # Step 2: Embed the document
                self.logger.info(f"Embedding document: {doc_path}")
                embed_result = mcp_client.call_jsonrpc(
                    "embed_documents",
                    {"knowledge_base_name": knowledge_base_name, "file_path": doc_path}
                )

                if embed_result and embed_result.get("ok"):
                    self.logger.info("Document embedded successfully. Re-running RAG query.")
                    
                    # Step 3: Re-run RAG query
                    retrieved_docs = mcp_client.embedding_query(
                        packet.content.original_prompt,
                        top_k=3,
                        knowledge_base_name=knowledge_base_name
                    )

                    if retrieved_docs and retrieved_docs.get("ok"):
                        docs = retrieved_docs.get("results", [])
                        if docs:
                            # Remove the old rag_no_results flag
                            packet.content.data_fields = [df for df in packet.content.data_fields if df.key != 'rag_no_results']
                            # Add the new documents
                            packet.content.data_fields.append(DataField(key='retrieved_documents', value=docs, type='list'))
                            self.logger.info(f"Successfully retrieved {len(docs)} documents after autonomous embedding.")
                else:
                    self.logger.error("Failed to embed document.")
            else:
                self.logger.info("No relevant documents found to embed.")

        except Exception as e:
            self.logger.error(f"Error during knowledge acquisition workflow: {e}")

        return packet
        
    def _suppress_repetition(self, text: str, max_repeat: int = 2) -> str:
        """
        Collapse runaway repetition by limiting how many times a sentence-level fragment
        can appear in the final response. Keeps the earliest occurrences and drops
        subsequent duplicates beyond `max_repeat`.
        """
        if not text:
            return text
        try:
            import regex as _re
            sentences = _re.split(r'(?<=[.!?])\s+', text)
        except Exception:
            sentences = text.splitlines()
        seen: Dict[str, int] = {}
        result: List[str] = []
        for sentence in sentences:
            key = sentence.strip().lower()
            if not key:
                continue
            count = seen.get(key, 0) + 1
            seen[key] = count
            if count <= max_repeat:
                result.append(sentence)
        if not result:
            return text
        return " ".join(result)

    def _should_escalate_to_thinker(self, text: str) -> bool:
        """
        Lightweight heuristic to decide if a request should bypass Operator (lite)
        and use a Thinker model. Signals: long/complex asks, code/architecture/debug,
        or explicit multi-step planning words.
        """
        if not text:
            return False
        t = text.lower()
        # Recitation/quote-style asks stay on Operator for stability.
        recite_markers = ["recite", "poem", "verse", "lyrics", "quote", "memorize", "memorised", "cite", "quote"]
        if any(m in t for m in recite_markers):
            return False
        # very long => escalate
        if len(t.split()) > 120:
            return True
        markers = [
            "code", "script", "function", "class", "api", "stack trace", "traceback",
            "debug", "optimiz", "profile", "benchmark", "architecture", "design",
            "plan", "steps", "multistep", "algorithm", "performance", "latency",
            "concurrency", "thread", "process", "gpu", "cuda", "memory leak",
        ]
        return any(m in t for m in markers)

    def _should_use_slim_prompt(self, plan: Plan, user_input: str) -> bool:
        """
        Decide whether to bypass the full plan/reflect pipeline and use a light prompt.
        Heuristics:
        - Respect GAIA_FORCE_FULL_PROMPT=1 to disable.
        - Respect GAIA_FORCE_SLIM_PROMPT=1 to always enable.
        - Always slim for list_tools and list_tree intents.
        - For simple 'other' intents, allow slim when the request is short and not
          explicitly asking for multi-step plans or code/tool execution.
        """
        force_full = os.getenv("GAIA_FORCE_FULL_PROMPT", "").lower() in ("1", "true", "yes")
        if force_full:
            return False
        force_slim = os.getenv("GAIA_FORCE_SLIM_PROMPT", "").lower() in ("1", "true", "yes")
        if force_slim:
            return True
        if plan.intent == "recitation":
            return True
        if plan.intent in ("list_tools", "list_tree", "find_file", "list_files", "read_file"):
            return True
        if plan.intent != "other":
            return False
        text = (user_input or "").lower()
        if len(text.split()) > 120:
            return False
        complex_markers = ["step", "plan", "analy", "investigat", "code", "script", "run ", "execute", "file", "shell"]
        if any(marker in text for marker in complex_markers):
            return False
        return True

    def _run_slim_prompt(self, selected_model_name: str, user_input: str, history: List[Dict[str, Any]], intent: str = "", session_id: str = "", source: str = "cli", metadata: dict = None, packet: CognitionPacket = None) -> str:
        """Execute a minimal prompt for low-complexity intents, or invoke MCP directly for tool-like intents.

        Args:
            packet: Optional pre-built CognitionPacket with RAG context. If provided, this packet
                   will be used instead of creating a new one, preserving retrieved_documents etc.
        """
        self.logger.info(f"--- RUNNING SLIM PROMPT for intent: {intent} ---")
        _meta = metadata or {}
        # Direct MCP path for toolish intents
        if intent == "list_tools":
            try:
                from gaia_core.utils import mcp_client
                ts_write({"type": "sketch", "intent": "list_tools", "plan": [
                    "Plan: enumerate available MCP tools",
                    "1) Call MCP list_tools.",
                    "2) Return the list to the operator."
                ]}, session_id, source=source, destination_context=_meta)
                resp = mcp_client.call_jsonrpc("list_tools", {})
                if resp.get("ok") and isinstance(resp.get("response"), dict):
                    tools = resp["response"].get("result") or []
                    return "Available MCP tools:\n- " + "\n- ".join(str(t) for t in tools)
                return f"Unable to list tools: {resp.get('error') or resp}"
            except Exception as exc:
                self.logger.exception("list_tools MCP call failed")
                return f"I encountered an error while listing tools: {exc}"

        if intent == "list_tree":
            try:
                # Annotate intent for logging/trace
                ts_write({"type": "sketch", "intent": "list_tree", "plan": [
                    "Plan: list a bounded directory tree",
                    "1) Call MCP list_tree with safe depth/entry limits.",
                    "2) If output is long, save full tree to a file and show a preview."
                ]}, session_id, source=source, destination_context=_meta)
                ts_write({"type": "mcp_call", "intent": "list_tree"}, session_id, source=source, destination_context=_meta)
                return self._run_mcp_list_tree()
            except Exception as exc:
                self.logger.exception("list_tree MCP call failed")
                return f"I couldn't retrieve the directory tree: {exc}"

        if intent == "list_files":
            try:
                # Annotate intent for logging/trace
                ts_write({"type": "sketch", "intent": "list_files", "plan": [
                    "Plan: list files in a directory",
                    "1) Call MCP list_files with a default path.",
                    "2) Return the list to the operator."
                ]}, session_id, source=source, destination_context=_meta)
                ts_write({"type": "mcp_call", "intent": "list_files"}, session_id, source=source, destination_context=_meta)
                return self._run_mcp_list_files()
            except Exception as exc:
                self.logger.exception("list_files MCP call failed")
                return f"I couldn't retrieve the file list: {exc}"

        if intent == "find_file":
            try:
                from gaia_core.utils import mcp_client
                ts_write({"type": "sketch", "intent": "find_file", "plan": [
                    "Plan: locate target file and summarize it",
                    "1) If an explicit path is provided, read it directly (if allowed).",
                    "2) Otherwise search under /gaia-assistant (skip hidden/cache dirs).",
                    "3) If exactly one match is found, read it and return a short preview.",
                    "4) If multiple matches are found, show a shortlist and await choice to read.",
                    "5) On failure/no match, prompt for a narrower path or filename."
                ]}, session_id, source=source, destination_context=_meta)
                ts_write({"type": "mcp_call", "intent": "find_file"}, session_id, source=source, destination_context=_meta)
                raw = user_input or ""
                allow_roots = [Path("/gaia-assistant").resolve(), Path("/models").resolve()]

                # Direct path extraction (if the user pasted a path)
                path_hits = re.findall(r"/[A-Za-z0-9_./-]+", raw)
                for p_raw in path_hits:
                    p = Path(p_raw).resolve()
                    if any(str(p).startswith(str(a)) for a in allow_roots) and p.is_file():
                        read = mcp_client.call_jsonrpc("read_file", {"path": str(p)})
                        if read.get("ok") and isinstance(read.get("response"), dict):
                            rres = read["response"].get("result") or read["response"]
                            content = ""
                            if isinstance(rres, dict):
                                content = rres.get("content") or ""
                            preview = (content[:1200] + "...") if len(content) > 1200 else content
                            summary_lines = preview.strip().splitlines()[:40]
                            summary_text = "\n".join(summary_lines)
                            return f"Found path: {p}\n\nPreview:\n{summary_text}"
                        return f"Found path: {p} (unable to read content automatically)."

                # Derive a filename-like query token from the text
                tokens = re.findall(r"[A-Za-z0-9_.-]+", raw)
                search_term = None
                for tok in tokens:
                    if "dev" in tok.lower() and "matrix" in tok.lower():
                        search_term = tok
                        break
                if not search_term and tokens:
                    # fallback to longest token
                    search_term = max(tokens, key=len)
                query_term = search_term or raw

                # Use the derived query; bounded search on the sidecar
                resp = mcp_client.call_jsonrpc("find_files", {"query": query_term, "max_depth": 5, "max_results": 50})
                if resp.get("ok") and isinstance(resp.get("response"), dict):
                    result = resp["response"].get("result") or resp["response"].get("result", resp["response"])
                    if isinstance(result, dict) and result.get("ok"):
                        matches = result.get("results") or []
                        if not matches:
                            return "I searched for that filename but found no matches under /gaia-assistant."
                        # If exactly one match, attempt to read and summarize it.
                        if len(matches) == 1:
                            path = matches[0]
                            read = mcp_client.call_jsonrpc("read_file", {"path": path})
                            if read.get("ok") and isinstance(read.get("response"), dict):
                                rres = read["response"].get("result") or read["response"]
                                content = ""
                                if isinstance(rres, dict):
                                    content = rres.get("content") or ""
                                preview = (content[:1200] + "...") if len(content) > 1200 else content
                                summary_lines = preview.strip().splitlines()[:40]
                                summary_text = "\n".join(summary_lines)
                                return f"Found one match: {path}\n\nPreview:\n{summary_text}"
                            return f"Found one match: {path} (unable to read content automatically)."
                        # Multiple matches: rank and present a shortlist
                        matches_sorted = sorted(matches, key=lambda p: (len(p), p.lower()))
                        lines = "\n".join(matches_sorted[:8])
                        suffix = ""
                        if len(matches_sorted) > 8:
                            suffix = f"\n... and {len(matches_sorted)-8} more (refine the query to narrow results)."
                        best_guess = matches_sorted[0] if matches_sorted else ""
                        return f"Found multiple matches. Best guess: {best_guess}\nCandidates:\n{lines}{suffix}\nAsk me to read one of these for a summary."
                return f"find_files failed: {resp.get('error') or resp}"
            except Exception as exc:
                self.logger.exception("find_file MCP call failed")
                return f"I encountered an error while searching for that file: {exc}"

        if intent == "read_file":
            try:
                from gaia_core.utils import mcp_client
                ts_write({"type": "sketch", "intent": "read_file", "plan": [
                    "Plan: read a file safely and summarize it",
                    "1) If a direct path is present, read it.",
                    "2) Otherwise search for likely matches.",
                    "3) If exactly one match is found, read and summarize it.",
                    "4) Store a preview to sketchpad for recall."
                ]}, session_id, source=source, destination_context=_meta)
                ts_write({"type": "mcp_call", "intent": "read_file"}, session_id, source=source, destination_context=_meta)
                raw = user_input or ""
                allow_roots = [Path("/gaia-assistant").resolve(), Path("/models").resolve()]

                # Direct path extraction (if the user pasted a path)
                path_hits = re.findall(r"/[A-Za-z0-9_./-]+", raw)
                for p_raw in path_hits:
                    p = Path(p_raw).resolve()
                    if any(str(p).startswith(str(a)) for a in allow_roots) and p.is_file():
                        read = mcp_client.call_jsonrpc("read_file", {"path": str(p)})
                        if read.get("ok") and isinstance(read.get("response"), dict):
                            rres = read["response"].get("result") or read["response"]
                            content = ""
                            if isinstance(rres, dict):
                                content = rres.get("content") or ""
                            preview = (content[:1200] + "...") if len(content) > 1200 else content
                            summary_lines = preview.strip().splitlines()[:40]
                            summary_text = "\n".join(summary_lines)
                            try:
                                rescue_helper.sketchpad_write(
                                    "LastFileRead",
                                    f"Path: {p}\n\nPreview:\n{summary_text}"
                                )
                            except Exception:
                                self.logger.debug("AgentCore: failed to write LastFileRead sketchpad entry", exc_info=True)
                            return f"Read: {p}\n\nPreview:\n{summary_text}\n(sketchpad: LastFileRead)"
                        return f"Found path: {p} (unable to read content automatically)."

                # Derive a filename-like query token from the text
                tokens = re.findall(r"[A-Za-z0-9_.-]+", raw)
                search_term = None
                for tok in tokens:
                    if tok.lower().endswith((".md", ".txt", ".json", ".py", ".yml", ".yaml", ".sh")):
                        search_term = tok
                        break
                if not search_term and tokens:
                    search_term = max(tokens, key=len)
                query_term = search_term or raw

                resp = mcp_client.call_jsonrpc("find_files", {"query": query_term, "max_depth": 5, "max_results": 50})
                if resp.get("ok") and isinstance(resp.get("response"), dict):
                    result = resp["response"].get("result") or resp["response"].get("result", resp["response"])
                    if isinstance(result, dict) and result.get("ok"):
                        matches = result.get("results") or []
                        if not matches:
                            return "I couldn't find that file. Share a path or a more specific filename."
                        if len(matches) == 1:
                            path = matches[0]
                            read = mcp_client.call_jsonrpc("read_file", {"path": path})
                            if read.get("ok") and isinstance(read.get("response"), dict):
                                rres = read["response"].get("result") or read["response"]
                                content = ""
                                if isinstance(rres, dict):
                                    content = rres.get("content") or ""
                                preview = (content[:1200] + "...") if len(content) > 1200 else content
                                summary_lines = preview.strip().splitlines()[:40]
                                summary_text = "\n".join(summary_lines)
                                try:
                                    rescue_helper.sketchpad_write(
                                        "LastFileRead",
                                        f"Path: {path}\n\nPreview:\n{summary_text}"
                                    )
                                except Exception:
                                    self.logger.debug("AgentCore: failed to write LastFileRead sketchpad entry", exc_info=True)
                                return f"Read: {path}\n\nPreview:\n{summary_text}\n(sketchpad: LastFileRead)"
                            return f"Found one match: {path} (unable to read content automatically)."
                        matches_sorted = sorted(matches, key=lambda p: (len(p), p.lower()))
                        lines = "\n".join(matches_sorted[:8])
                        suffix = ""
                        if len(matches_sorted) > 8:
                            suffix = f"\n... and {len(matches_sorted)-8} more (refine the filename)."
                        best_guess = matches_sorted[0] if matches_sorted else ""
                        return f"Found multiple matches. Best guess: {best_guess}\nCandidates:\n{lines}{suffix}\nTell me which path to read."
                return f"read_file lookup failed: {resp.get('error') or resp}"
            except Exception as exc:
                self.logger.exception("read_file MCP call failed")
                return f"I encountered an error while reading that file: {exc}"

        # Custom command to trigger documentation
        document_command_match = re.match(
            r'GAIA, DOCUMENT "(?P<title>[^"]+)" AS "(?P<symbol>[^"]+)" ABOUT "(?P<content>.+)"',
            user_input,
            re.IGNORECASE
        )
        if document_command_match:
            try:
                title = document_command_match.group("title")
                symbol = document_command_match.group("symbol").upper().replace(" ", "_") # Normalize symbol
                content_to_document = document_command_match.group("content")
                tags = ["user-generated", "documentation", symbol.lower()] # Default tags

                self.logger.info(f"Document command detected: title='{title}', symbol='{symbol}'")
                
                # Call the CodexWriter to document the information
                documented_path = self.codex_writer.document_information(
                    packet=packet,
                    info_to_document=content_to_document,
                    symbol=symbol,
                    title=title,
                    tags=tags,
                    llm_model=None  # Model acquired separately if CodexWriter needs LLM refinement
                )

                if documented_path:
                    response_text = f"Acknowledged. I have documented '{title}' (Symbol: {symbol}) to {documented_path}."
                else:
                    response_text = f"I encountered an issue while trying to document '{title}' (Symbol: {symbol})."

                return response_text
            except Exception as e:
                self.logger.exception(f"Error processing document command: {e}")
                return f"I encountered an error while trying to document that information: {e}"

        # Check if this is a recitation/long-form request that should use fragmentation
        fragmentation_enabled = os.getenv("GAIA_FRAGMENTATION", "true").lower() in ("1", "true", "yes")

        # Early bypass: if the request matches a known recitable document, force
        # recitation intent regardless of what the NLU classified.  This ensures
        # GAIA's own core documents are always routed through the document
        # recitation path even if the intent detector misclassifies the request.
        if intent != "recitation" and fragmentation_enabled:
            early_doc = find_recitable_document(user_input)
            if early_doc:
                self.logger.info(
                    "Early bypass: request matches recitable document '%s' "
                    "but intent was '%s'; overriding to 'recitation'",
                    early_doc["title"], intent,
                )
                print(
                    f"DEBUG: Early bypass override: intent '{intent}' -> 'recitation' "
                    f"for document '{early_doc['title']}'",
                    file=sys.stderr,
                )
                intent = "recitation"

        if intent == "recitation" and fragmentation_enabled:
            print("DEBUG: Entered recitation intent block", file=sys.stderr)
            print(f"DEBUG: Calling assess_task_confidence, fragmentation_enabled={fragmentation_enabled}", file=sys.stderr)
            self.logger.info("Recitation intent detected, checking for known documents first...")

            # First, check if this is a known recitable document we can load directly
            recitable_doc = find_recitable_document(user_input)

            if recitable_doc:
                # We have the actual document - no need for confidence checks
                self.logger.info(f"Found recitable document: {recitable_doc['title']}")
                print(f"DEBUG: Found recitable document: {recitable_doc['title']}", file=sys.stderr)

                try:
                    return self._run_with_document_recitation(
                        user_input=user_input,
                        document=recitable_doc,
                        selected_model_name=selected_model_name,
                        history=history or [],
                        session_id=session_id,
                        output_as_file=True,  # Long-form document recitations go to file
                    )
                except Exception as exc:
                    self.logger.exception("Document recitation failed, falling back to standard generation")
                    # Fall through to confidence-based approach

            # No known document found - use confidence-based approach
            self.logger.info("No known document matched, assessing task confidence...")

            # Pre-task epistemic check: Does GAIA actually know this content?
            # Uses full GCP pipeline so GAIA has identity, world state, and tool awareness
            confidence_check = self.assess_task_confidence(
                intent=intent,
                user_input=user_input,
                model_name="lite",
                session_id=session_id
            )

            try:
                self.logger.info(f"Task confidence assessment: score={confidence_check.get('confidence_score')}, "
                               f"can_attempt={confidence_check.get('can_attempt')}")
                self.logger.info(f"Confidence reasoning: {confidence_check.get('reasoning', '')[:200]}")
            except Exception as e:
                self.logger.exception(f"Error logging task confidence assessment: {e}")

            # Gate on confidence - if too low, offer alternative instead of producing garbage
            # Threshold is configurable; default 0.5 means "more uncertain than confident"
            confidence_threshold = float(os.getenv("GAIA_CONFIDENCE_THRESHOLD", "0.5"))
            confidence_score = confidence_check.get("confidence_score", 0.5)
            can_attempt = confidence_check.get("can_attempt", True)

            if not can_attempt or confidence_score < confidence_threshold:
                self.logger.warning(f"Low confidence ({confidence_score}) - declining to attempt recitation")
                alternative = confidence_check.get("alternative_offer", "")
                if alternative:
                    return f"I need to be honest with you: {confidence_check.get('reasoning', 'I am not confident I can accurately complete this task.')}\n\n{alternative}"
                return f"I need to be honest: {confidence_check.get('reasoning', 'I do not have this content accurately memorized and would likely produce errors if I attempted it.')}"

            # Proceed with fragmented generation if confidence is adequate
            self.logger.info("Confidence adequate, proceeding with fragmented generation")
            try:
                return self._run_with_fragmentation(
                    user_input=user_input,
                    selected_model_name=selected_model_name,
                    history=history or [],
                    session_id=session_id,
                    max_fragments=5,
                    output_as_file=True,  # Long-form recitations go to file
                )
            except Exception as exc:
                self.logger.exception("Fragmented generation failed, falling back to standard")
                # Fall through to standard generation

        # Otherwise, use the canonical GCP prompt builder (world state + identity).
        try:
            # Use provided packet if available (preserves RAG context), otherwise create new
            if packet is None:
                packet = self._create_initial_packet(
                    user_input=user_input,
                    session_id=session_id or "system",
                    history=history or [],
                    selected_model_name=selected_model_name,
                )
            try:
                packet.intent.user_intent = intent or "other"
            except Exception:
                pass
            messages = build_from_packet(packet)
            self.logger.info("AgentCore: slim prompt routed through GCP builder")
            # Use configured MAX_ALLOWED_RESPONSE_TOKENS (default 1000) instead of hardcoded 256
            max_resp_tokens = (
                getattr(self.config, 'MAX_ALLOWED_RESPONSE_TOKENS', None) or
                self.config.constants.get("MAX_ALLOWED_RESPONSE_TOKENS", 1000)
            )
            res = self.model_pool.forward_to_model(
                selected_model_name,
                messages=messages,
                max_tokens=min(self.config.max_tokens, max_resp_tokens),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            content = res["choices"][0]["message"]["content"]
            # Optional polish via Thinker: if Operator (lite) handled this turn and a Thinker is available,
            # send a short polish request and return the Thinker output as final.
            try:
                polish_flag = os.getenv("GAIA_THINKER_POLISH", "").lower() in ("1", "true", "yes")
                if polish_flag and selected_model_name == "lite":
                    thinker_name = None
                    for cand in ["gpu_prime", "prime", "cpu_prime"]:
                        if cand in self.model_pool.models:
                            thinker_name = cand
                            break
                    if thinker_name:
                        polish_messages = [
                            {"role": "system", "content": "Polish the assistant reply for clarity and GAIA voice. Do not change facts or add refusals."},
                            {"role": "user", "content": content},
                        ]
                        polished = self.model_pool.forward_to_model(
                            thinker_name,
                            messages=polish_messages,
                            max_tokens=min(self.config.max_tokens, 200),
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                        content = polished["choices"][0]["message"]["content"]
            except Exception:
                self.logger.debug("Thinker polish step failed; returning Operator output", exc_info=True)
            return content
        except Exception as exc:
            self.logger.exception("slim prompt call failed")
            return f"I encountered an error while answering: {exc}"

    def _run_with_document_recitation(
        self,
        user_input: str,
        document: Dict[str, str],
        selected_model_name: str,
        history: List[Dict[str, Any]],
        session_id: str = "",
        output_as_file: bool = False,
    ) -> str:
        """
        Handle recitation of a known document by providing the actual content to the model.

        Instead of asking the model to recite from memory, we load the document
        and ask the model to present it appropriately based on the user's request.

        Args:
            user_input: The user's original request
            document: Dict with 'title', 'content', 'path' from find_recitable_document()
            selected_model_name: Model to use for generation
            history: Conversation history
            session_id: Session ID for logging
            output_as_file: If True, write result to file instead of returning as response

        Returns:
            The model's presentation of the document, or file path message if output_as_file=True
        """
        doc_title = document.get("title", "Document")
        doc_content = document.get("content", "")

        self.logger.info(f"Running document recitation for: {doc_title} ({len(doc_content)} chars)")

        # Build a prompt that gives the model the actual document content
        # and asks it to present/recite it appropriately
        recitation_prompt = f"""The user has asked you to recite or share {doc_title}.

Here is the complete, authoritative content of {doc_title}:

--- BEGIN {doc_title.upper()} ---
{doc_content}
--- END {doc_title.upper()} ---

USER'S REQUEST: {user_input}

INSTRUCTIONS:
1. Present this document to the user as requested
2. You may add a brief introduction acknowledging their request
3. Present the content faithfully - this is YOUR constitution/document, speak it with conviction
4. You may format it nicely for readability (preserve markdown structure)
5. If the document is very long, you may summarize sections if appropriate, but prefer completeness
6. Do NOT fabricate or add content that isn't in the document above

Present {doc_title}:"""

        # Build messages with identity context
        packet = self._create_initial_packet(
            user_input=recitation_prompt,
            session_id=session_id or "document_recitation",
            history=history,
            selected_model_name=selected_model_name,
        )

        try:
            packet.intent.user_intent = "recitation"
        except Exception:
            pass

        messages = build_from_packet(packet)

        # Use a generous token limit since documents can be long
        max_resp_tokens = (
            getattr(self.config, 'MAX_ALLOWED_RESPONSE_TOKENS', None) or
            self.config.constants.get("MAX_ALLOWED_RESPONSE_TOKENS", 1000)
        )
        # Allow 3x normal limit for document recitation
        recitation_max_tokens = min(self.config.max_tokens * 3, max_resp_tokens * 3, 4096)

        try:
            res = self.model_pool.forward_to_model(
                selected_model_name,
                messages=messages,
                max_tokens=recitation_max_tokens,
                temperature=self.config.temperature * 0.7,  # Lower temp for faithful recitation
                top_p=self.config.top_p,
            )
            content = res["choices"][0]["message"]["content"]

            # Strip any think tags from the response
            content = strip_think_tags(content)

            self.logger.info(f"Document recitation complete: {len(content)} chars")

            # If file output mode, write to file
            if output_as_file and content:
                import uuid
                request_id = str(uuid.uuid4())[:8]
                return self._write_assembled_to_file(
                    content=content,
                    original_request=user_input,
                    request_id=request_id,
                    filename=None,  # Auto-generate from doc_title
                )

            return content

        except Exception as e:
            self.logger.exception("Document recitation generation failed")
            # Fallback: just return the raw document with a header
            fallback_content = f"Here is {doc_title}:\n\n{doc_content}"
            if output_as_file:
                import uuid
                request_id = str(uuid.uuid4())[:8]
                return self._write_assembled_to_file(
                    content=fallback_content,
                    original_request=user_input,
                    request_id=request_id,
                )
            return fallback_content

    def _run_with_fragmentation(self, user_input: str, selected_model_name: str,
                                 history: List[Dict[str, Any]], session_id: str = "",
                                 max_fragments: int = 5,
                                 output_as_file: bool = False,
                                 output_filename: Optional[str] = None) -> str:
        """
        Run generation with sketchpad-based fragmentation for long-form content.

        Uses a "fragment, store, reflect, assemble" workflow:
        1. Generate content fragments, storing each to the sketchpad
        2. After all fragments collected, run a separate assembly turn
        3. The model reads its own fragments and assembles them intelligently
        4. Optionally write the final content to a file instead of returning as response

        This delegates assembly intelligence to the model rather than relying
        on naive Python string concatenation.

        Args:
            user_input: The user's request
            selected_model_name: Model to use for generation
            history: Conversation history
            session_id: Session ID for logging
            max_fragments: Maximum number of continuation attempts
            output_as_file: If True, write assembled content to file and return file path message
            output_filename: Optional filename for file output (default: auto-generated based on request)

        Returns:
            Complete assembled response from the model's assembly turn,
            or a message with file path if output_as_file=True
        """
        import uuid
        from gaia_core.utils import mcp_client

        request_id = str(uuid.uuid4())[:8]
        fragment_keys: List[str] = []
        fragment_contents: Dict[str, str] = {}  # In-memory fallback storage
        fragment_sequence = 0

        # Get max tokens config
        max_resp_tokens = (
            getattr(self.config, 'MAX_ALLOWED_RESPONSE_TOKENS', None) or
            self.config.constants.get("MAX_ALLOWED_RESPONSE_TOKENS", 1000)
        )

        current_prompt = user_input
        is_continuation = False

        self.logger.info(f"Starting sketchpad-based fragmented generation for request {request_id}...")

        # Phase 1: Generate and store fragments to sketchpad
        while fragment_sequence < max_fragments:
            self.logger.info(f"Generating fragment {fragment_sequence + 1}/{max_fragments}")

            # Build messages for this fragment
            if is_continuation:
                # For continuations, provide context about what's in sketchpad
                fragment_list = ", ".join(fragment_keys)
                messages = [
                    {"role": "system", "content": (
                        "You are GAIA, continuing a long-form generation task. "
                        f"You have already written fragments to your sketchpad: [{fragment_list}]. "
                        "Continue generating the next portion. Do NOT repeat content you've already written. "
                        "Focus only on the new content that comes next."
                    )},
                    {"role": "user", "content": current_prompt}
                ]
            else:
                # First fragment uses full packet/prompt building
                packet = self._create_initial_packet(
                    user_input=current_prompt,
                    session_id=session_id or "system",
                    history=history or [],
                    selected_model_name=selected_model_name,
                )
                from gaia_core.utils.prompt_builder import build_from_packet
                messages = build_from_packet(packet)

            # Generate this fragment
            try:
                res = self.model_pool.forward_to_model(
                    selected_model_name,
                    messages=messages,
                    max_tokens=min(self.config.max_tokens, max_resp_tokens),
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                content = res["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.exception(f"Fragment {fragment_sequence} generation failed")
                break

            # Check for truncation
            truncation_info = self.detect_truncation(content, max_tokens=max_resp_tokens)

            # Store fragment to sketchpad with unique key
            fragment_key = f"recitation_fragment_{request_id}_{fragment_sequence}"
            try:
                rescue_helper.sketchpad_write(fragment_key, content)
                fragment_keys.append(fragment_key)
                self.logger.info(f"Stored fragment to sketchpad: {fragment_key}")
            except Exception as e:
                self.logger.warning(f"Failed to store fragment to sketchpad: {e}")
                # Fallback: keep in memory with same key for consistency
                fragment_contents[fragment_key] = content
                fragment_keys.append(fragment_key)
                self.logger.info(f"Stored fragment to memory fallback: {fragment_key}")

            # Also store via MCP for auditing (non-critical)
            try:
                mcp_client.call_jsonrpc("fragment_write", {
                    "parent_request_id": request_id,
                    "sequence": fragment_sequence,
                    "content": content,
                    "continuation_hint": truncation_info.get("continuation_hint", ""),
                    "is_complete": not truncation_info["truncated"],
                    "token_count": truncation_info.get("approx_tokens", 0)
                })
            except Exception:
                pass  # MCP storage is optional

            fragment_sequence += 1

            # If not truncated, we're done collecting fragments
            if not truncation_info["truncated"]:
                self.logger.info(f"Fragment collection complete after {fragment_sequence} fragment(s)")
                break

            # Use model reflection to generate continuation prompt
            self.logger.info(f"Fragment {fragment_sequence} truncated (reason: {truncation_info['reason']}), reflecting...")

            # Read fragments from sketchpad for reflection context
            combined_so_far = self._read_fragments_from_sketchpad(fragment_keys, fragment_contents)

            reflection = self.reflect_on_truncation(
                original_request=user_input,
                truncated_output=combined_so_far,
                model_name="lite"  # Use lite model for fast reflection
            )

            self.logger.info(f"Reflection: progress={reflection.get('estimated_progress')}, "
                           f"repetition={reflection.get('repetition_detected')}")
            self.logger.debug(f"Reflection reasoning: {reflection.get('reasoning')}")

            # If repetition detected, note it but continue
            if reflection.get("repetition_detected"):
                self.logger.warning("Repetition detected in output, assembly turn will handle cleanup...")

            # Use the model-generated continuation prompt
            current_prompt = reflection.get("continuation_prompt", "")
            self.logger.info(f"Generated continuation prompt: {current_prompt}")
            if not current_prompt:
                self.logger.warning("No continuation prompt generated, stopping fragmentation")
                break

            is_continuation = True

        # Phase 2: Model-driven assembly turn
        self.logger.info(f"Starting assembly turn with {len(fragment_keys)} fragments...")
        final_response = self._run_assembly_turn(
            original_request=user_input,
            fragment_keys=fragment_keys,
            selected_model_name=selected_model_name,
            session_id=session_id,
            memory_fallback=fragment_contents,
        )

        # Cleanup: clear fragments from MCP storage
        try:
            mcp_client.call_jsonrpc("fragment_clear", {"parent_request_id": request_id})
        except Exception:
            pass

        self.logger.info(f"Sketchpad-based generation complete: {fragment_sequence} fragments, "
                        f"{len(final_response)} chars final output")

        # Phase 3: File output mode - write to file instead of returning as response
        if output_as_file and final_response:
            return self._write_assembled_to_file(
                content=final_response,
                original_request=user_input,
                request_id=request_id,
                filename=output_filename,
            )

        return final_response

    def _read_fragments_from_sketchpad(self, fragment_keys: List[str],
                                        memory_fallback: Optional[Dict[str, str]] = None) -> str:
        """
        Read and concatenate fragments from sketchpad for reflection context.

        Args:
            fragment_keys: List of sketchpad keys for fragments
            memory_fallback: Optional dict of key->content for fragments stored in memory

        Returns:
            Combined fragment content
        """
        memory_fallback = memory_fallback or {}
        parts = []
        for key in fragment_keys:
            # First try memory fallback (faster and works when sketchpad fails)
            if key in memory_fallback:
                parts.append(memory_fallback[key])
                continue

            # Then try sketchpad
            try:
                content = rescue_helper.sketchpad_read(key)
                # sketchpad_read returns formatted output, extract just the content
                if content and not content.startswith("[No sketch"):
                    # Format is "[timestamp] title\ncontent" - extract content after first newline
                    lines = content.split("\n", 1)
                    if len(lines) > 1:
                        parts.append(lines[1])
                    else:
                        parts.append(content)
            except Exception as e:
                self.logger.debug(f"Failed to read fragment {key} from sketchpad: {e}")
        return "\n".join(parts)

    def _run_assembly_turn(self, original_request: str, fragment_keys: List[str],
                          selected_model_name: str, session_id: str = "",
                          memory_fallback: Optional[Dict[str, str]] = None) -> str:
        """
        Run the assembly turn where the model reads its fragments and assembles them.

        This is the key innovation: instead of Python concatenation, the model
        itself reviews and assembles the fragments into a coherent whole.

        Args:
            original_request: The original user request
            fragment_keys: List of sketchpad keys containing fragments
            selected_model_name: Model to use for assembly
            session_id: Session ID for logging
            memory_fallback: Optional dict of key->content for fragments stored in memory

        Returns:
            The model's assembled response
        """
        memory_fallback = memory_fallback or {}

        # If only one fragment or none, no assembly needed
        if len(fragment_keys) <= 1:
            if fragment_keys:
                content = self._read_fragments_from_sketchpad(fragment_keys, memory_fallback)
                return strip_think_tags(content)
            return ""

        # Read all fragments for the assembly prompt
        fragment_contents = []
        for i, key in enumerate(fragment_keys):
            # First try memory fallback
            if key in memory_fallback:
                # Strip <think> tags from fragment content before assembly
                clean_content = strip_think_tags(memory_fallback[key])
                fragment_contents.append(f"--- Fragment {i + 1} ({key}) ---\n{clean_content}")
                continue

            # Then try sketchpad
            try:
                content = rescue_helper.sketchpad_read(key)
                if content and not content.startswith("[No sketch"):
                    lines = content.split("\n", 1)
                    fragment_text = lines[1] if len(lines) > 1 else content
                    # Strip <think> tags from fragment content before assembly
                    clean_fragment = strip_think_tags(fragment_text)
                    fragment_contents.append(f"--- Fragment {i + 1} ({key}) ---\n{clean_fragment}")
            except Exception as e:
                self.logger.debug(f"Failed to read fragment {key} for assembly: {e}")

        if not fragment_contents:
            self.logger.warning("No fragments available for assembly")
            return ""

        fragments_text = "\n\n".join(fragment_contents)

        # Build the assembly prompt
        assembly_prompt = f"""You have generated several fragments of a long-form response to this request:

ORIGINAL REQUEST: {original_request}

Your fragments are stored in your sketchpad. Here they are:

{fragments_text}

--- END OF FRAGMENTS ---

ASSEMBLY INSTRUCTIONS:
1. Review all fragments above carefully
2. Identify any overlapping or repeated content between fragments
3. Assemble them into a single, clean, complete response
4. Remove any duplications or awkward transitions
5. Ensure the final output is coherent and flows naturally
6. Present ONLY the final assembled text - no commentary or explanation

Assembled response:"""

        messages = [
            {"role": "system", "content": (
                "You are GAIA, performing an assembly task. You have previously generated "
                "fragments of a long response which are now in your sketchpad. Your job is to "
                "read these fragments and assemble them into a single, coherent, complete response. "
                "Output ONLY the assembled content, nothing else."
            )},
            {"role": "user", "content": assembly_prompt}
        ]

        # Use a higher token limit for assembly since we're producing the final output
        max_resp_tokens = (
            getattr(self.config, 'MAX_ALLOWED_RESPONSE_TOKENS', None) or
            self.config.constants.get("MAX_ALLOWED_RESPONSE_TOKENS", 1000)
        )
        # Allow more tokens for assembly (up to 2x normal limit)
        assembly_max_tokens = min(self.config.max_tokens * 2, max_resp_tokens * 2)

        try:
            res = self.model_pool.forward_to_model(
                selected_model_name,
                messages=messages,
                max_tokens=assembly_max_tokens,
                temperature=self.config.temperature * 0.8,  # Slightly lower temp for assembly
                top_p=self.config.top_p,
            )
            assembled = res["choices"][0]["message"]["content"]
            # Strip any <think> reasoning blocks from the assembled output
            assembled = strip_think_tags(assembled)
            self.logger.info(f"Assembly turn complete: {len(assembled)} chars")
            return assembled
        except Exception as e:
            self.logger.exception("Assembly turn failed, falling back to simple concatenation")
            # Fallback to simple concatenation if assembly fails
            return self._read_fragments_from_sketchpad(fragment_keys, memory_fallback)

    def _write_assembled_to_file(self, content: str, original_request: str,
                                  request_id: str, filename: Optional[str] = None) -> str:
        """
        Write assembled content to a file and return a message with the file path.

        This is the final step in file output mode for fragmentation. Instead of
        returning the content directly as a chat response, we write it to a file
        in the sandbox and return a message pointing to the file.

        Args:
            content: The assembled content to write
            original_request: The original user request (used to derive filename if not specified)
            request_id: Unique request ID for this generation
            filename: Optional explicit filename, otherwise auto-generated

        Returns:
            A message indicating the file was written and its path
        """
        import re
        from datetime import datetime

        # Generate filename if not provided
        if not filename:
            # Try to extract a meaningful name from the request
            # Look for quoted titles or common patterns
            title_match = re.search(r'["\']([^"\']+)["\']', original_request)
            if title_match:
                base_name = title_match.group(1)
            else:
                # Look for "The X" pattern common in titles
                the_match = re.search(r'\b(The\s+\w+(?:\s+\w+)?)\b', original_request, re.IGNORECASE)
                if the_match:
                    base_name = the_match.group(1)
                else:
                    # Fall back to generic name with request ID
                    base_name = f"assembled_content_{request_id}"

            # Sanitize filename: lowercase, replace spaces with underscores, remove special chars
            safe_name = re.sub(r'[^\w\s-]', '', base_name.lower())
            safe_name = re.sub(r'[\s-]+', '_', safe_name).strip('_')
            filename = f"{safe_name}.txt"

        # Ensure we write to the sandbox directory
        output_path = f"/sandbox/{filename}"

        # Write the file using MCP client
        result = mcp_client.ai_write(output_path, content)

        if result.get("ok"):
            file_size = result.get("bytes", len(content))
            self.logger.info(f"Wrote assembled content to file: {output_path} ({file_size} bytes)")

            # Return both the file notification AND the actual content
            # The content will be sent to Discord, and the file provides a backup
            return (
                f"*[Saved to `{output_path}` ({file_size:,} bytes)]*\n\n"
                f"{content}"
            )
        else:
            # File write failed - log error and fall back to returning content directly
            error = result.get("error", "Unknown error")
            self.logger.error(f"Failed to write assembled content to file: {error}")
            return (
                f"I assembled the content but couldn't save it to a file ({error}).\n\n"
                f"Here is the content:\n\n{content}"
            )

    def _assemble_fragments(self, fragments: List[str]) -> str:
        """
        Assemble fragments into a single response, handling potential overlaps.

        Args:
            fragments: List of fragment strings in order

        Returns:
            Assembled content
        """
        if not fragments:
            return ""

        if len(fragments) == 1:
            return fragments[0]

        assembled_parts = []
        for i, content in enumerate(fragments):
            if i == 0:
                assembled_parts.append(content)
                continue

            # Check for overlap with previous fragment
            prev_content = assembled_parts[-1]
            overlap_window = 150  # chars to check for overlap

            prev_tail = prev_content[-overlap_window:] if len(prev_content) > overlap_window else prev_content
            curr_head = content[:overlap_window] if len(content) > overlap_window else content

            # Find overlap by checking if prev_tail ends with something curr_head starts with
            overlap_found = 0
            for overlap_len in range(min(len(prev_tail), len(curr_head)), 10, -1):
                if prev_tail.endswith(curr_head[:overlap_len]):
                    overlap_found = overlap_len
                    break

            if overlap_found > 0:
                self.logger.debug(f"Fragment {i}: removed {overlap_found} char overlap")
                content = content[overlap_found:]

            assembled_parts.append(content)

        return "".join(assembled_parts)

    def _run_mcp_list_tree(self) -> str:
        """Call MCP sidecar to list a bounded directory tree with approval."""
        from gaia_core.utils import mcp_client
        params = {"path": "/gaia-assistant", "max_depth": 2, "max_entries": 120, "_allow_pending": True}
        bypass = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")

        def _handle_tree_result(result_dict: dict) -> str:
            tree = result_dict.get("tree") or ""
            truncated_flag = result_dict.get("truncated")
            if not tree:
                return "No entries found."
            # If the tree is long, persist it and return a short preview
            if len(tree) > 4000:
                preview = "\n".join(tree.splitlines()[:20])
                target_path = "/gaia-assistant/knowledge/system_reference/tree_latest.txt"
                # Write via MCP so it is auditable/approved
                write_req = mcp_client.request_approval_via_mcp("ai_write", {"path": target_path, "content": tree, "_allow_pending": True})
                if write_req.get("ok") and write_req.get("action_id") and write_req.get("challenge"):
                    approval = write_req["challenge"][::-1]
                    write_appr = mcp_client.approve_action_via_mcp(write_req["action_id"], approval)
                    if not write_appr.get("ok"):
                        return f"(Saved tree skipped due to approval error: {write_appr.get('error')})\nPreview:\n{preview}"
                    try:
                        rescue_helper.sketchpad_write(
                            "DirectoryTreeLatest",
                            f"Saved full tree to: {target_path}\n\nPreview:\n{preview}"
                        )
                    except Exception:
                        self.logger.debug("AgentCore: failed to write tree preview to sketchpad", exc_info=True)
                elif bypass:
                    # In bypass mode, attempt direct write
                    mcp_client.call_jsonrpc("ai_write", {"path": target_path, "content": tree})
                    try:
                        rescue_helper.sketchpad_write(
                            "DirectoryTreeLatest",
                            f"Saved full tree to: {target_path}\n\nPreview:\n{preview}"
                        )
                    except Exception:
                        self.logger.debug("AgentCore: failed to write tree preview to sketchpad (bypass)", exc_info=True)
                else:
                    return f"(Could not request approval to save tree: {write_req.get('error')})\nPreview:\n{preview}"
                suffix = "\n\n(full tree saved to knowledge/system_reference/tree_latest.txt)"
                if truncated_flag:
                    suffix += " (truncated at source)"
                return f"{preview}\n... [preview ends]{suffix}\n(sketchpad: DirectoryTreeLatest)"
            # Short enough; return inline
            return tree + ("\n\n(truncated)" if truncated_flag else "")

        if bypass:
            resp = mcp_client.call_jsonrpc("list_tree", params)
            if resp.get("ok") and isinstance(resp.get("response"), dict):
                result = resp["response"].get("result") or {}
                if result.get("ok"):
                    return _handle_tree_result(result)
                return f"list_tree error: {result}"
            return f"list_tree failed: {resp.get('error') or resp}"

        # Request approval then auto-approve using reversed challenge
        req = mcp_client.request_approval_via_mcp("list_tree", params)
        if not req.get("ok"):
            return f"Could not request approval for list_tree: {req.get('error')}"
        action_id = req.get("action_id")
        challenge = req.get("challenge")
        if not action_id or not challenge:
            return "Approval request did not return action_id/challenge."
        approval = challenge[::-1]
        appr = mcp_client.approve_action_via_mcp(action_id, approval)
        if not appr.get("ok"):
            return f"Approval failed: {appr.get('error')}"
        result = appr.get("result") or {}
        if isinstance(result, dict) and result.get("ok") and "tree" in result:
            return _handle_tree_result(result)
        # If server wraps result under 'result' for JSON-RPC payload
        if isinstance(result, dict) and "result" in result:
            inner = result["result"]
            if isinstance(inner, dict) and inner.get("ok") and "tree" in inner:
                return _handle_tree_result(inner)
        return f"list_tree execution returned unexpected payload: {result}"

    def _run_mcp_list_files(self) -> str:
        """Call MCP sidecar to list files in a directory with approval."""
        from gaia_core.utils import mcp_client
        params = {"path": "/gaia-assistant", "_allow_pending": True}
        bypass = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")

        def _handle_list_result(result_dict: dict) -> str:
            files = result_dict.get("files") or []
            if not files:
                return "No files found."
            return "\n".join(files)

        if bypass:
            resp = mcp_client.call_jsonrpc("list_files", params)
            if resp.get("ok") and isinstance(resp.get("response"), dict):
                result = resp["response"].get("result") or {}
                if result.get("ok"):
                    return _handle_list_result(result)
                return f"list_files error: {result}"
            return f"list_files failed: {resp.get('error') or resp}"

        # Request approval then auto-approve using reversed challenge
        req = mcp_client.request_approval_via_mcp("list_files", params)
        if not req.get("ok"):
            return f"Could not request approval for list_files: {req.get('error')}"
        action_id = req.get("action_id")
        challenge = req.get("challenge")
        if not action_id or not challenge:
            return "Approval request did not return action_id/challenge."
        approval = challenge[::-1]
        appr = mcp_client.approve_action_via_mcp(action_id, approval)
        if not appr.get("ok"):
            return f"Approval failed: {appr.get('error')}"
        result = appr.get("result") or {}
        if isinstance(result, dict) and result.get("ok") and "files" in result:
            return _handle_list_result(result)
        # If server wraps result under 'result' for JSON-RPC payload
        if isinstance(result, dict) and "result" in result:
            inner = result["result"]
            if isinstance(inner, dict) and inner.get("ok") and "files" in inner:
                return _handle_list_result(inner)
        return f"list_files execution returned unexpected payload: {result}"

    # --- Response Fragmentation Helpers ---

    def detect_truncation(self, response: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Detect if a response appears to be truncated.

        Checks for:
        - Response ending mid-sentence (no terminal punctuation)
        - Response ending mid-word
        - Response length approaching token limit
        - Repetition patterns indicating model confusion

        Returns:
            Dict with:
                - truncated: bool
                - reason: str (why we think it's truncated)
                - last_complete_sentence: str (for continuation context)
                - continuation_hint: str (suggested continuation point)
        """
        if not response or not response.strip():
            return {"truncated": False, "reason": "empty_response"}

        response = response.strip()

        # Approximate token count (rough: ~4 chars per token for English)
        approx_tokens = len(response) // 4

        # Check for mid-sentence truncation
        terminal_punct = {'.', '!', '?', '"', "'", ')', ']', '}'}
        ends_with_terminal = response[-1] in terminal_punct if response else False

        # Check for mid-word truncation (ends with letter/number, no space before)
        ends_mid_word = response[-1].isalnum() and len(response) > 1 and response[-2].isalnum()

        # Check for repetition (same phrase appearing multiple times near end)
        repetition_detected = False
        if len(response) > 200:
            last_200 = response[-200:]
            # Look for repeated phrases of 20+ chars
            for phrase_len in range(30, 15, -1):
                phrase = last_200[-phrase_len:]
                if last_200.count(phrase) > 1:
                    repetition_detected = True
                    break

        # Determine if truncated
        truncated = False
        reason = ""

        if repetition_detected:
            truncated = True
            reason = "repetition_detected"
        elif approx_tokens >= max_tokens * 0.9:
            truncated = True
            reason = "approaching_token_limit"
        elif ends_mid_word and not ends_with_terminal:
            truncated = True
            reason = "mid_word_truncation"
        elif not ends_with_terminal and approx_tokens > 100:
            # Only flag mid-sentence if response is substantial
            truncated = True
            reason = "mid_sentence_truncation"

        # Find last complete sentence for continuation context
        last_complete = ""
        if truncated:
            # Find the last sentence-ending punctuation
            for i in range(len(response) - 1, -1, -1):
                if response[i] in {'.', '!', '?'}:
                    # Check it's not part of an abbreviation or number
                    if i < len(response) - 1 and response[i + 1] in {' ', '\n', '"', "'"}:
                        last_complete = response[:i + 1].strip()
                        break
                    elif i == len(response) - 1:
                        last_complete = response
                        break

        # Generate continuation hint
        continuation_hint = ""
        if truncated and last_complete:
            # Take last ~50 chars of complete content as context
            hint_start = max(0, len(last_complete) - 100)
            continuation_hint = last_complete[hint_start:]

        return {
            "truncated": truncated,
            "reason": reason,
            "approx_tokens": approx_tokens,
            "last_complete_sentence": last_complete,
            "continuation_hint": continuation_hint
        }

    def build_continuation_prompt(self, original_request: str, previous_content: str,
                                   continuation_hint: str = "") -> str:
        """
        Build a prompt for continuing a truncated response.

        Args:
            original_request: The user's original request
            previous_content: Content generated so far
            continuation_hint: Optional hint about where to continue from

        Returns:
            A prompt string for the model to continue generation
        """
        # Get last ~150 chars of previous content for context
        context_len = min(200, len(previous_content))
        context_tail = previous_content[-context_len:] if previous_content else ""

        prompt = f"""Continue the response to this request. Pick up EXACTLY where the previous output ended - do not repeat content already generated.

Original request: {original_request}

Previous output ended with:
...{context_tail}

Continue from that point. Do not include any preamble like "Continuing from..." - just continue the content naturally."""

        if continuation_hint:
            prompt += f"\n\nHint: {continuation_hint}"

        return prompt

    def assess_task_confidence(self, intent: str, user_input: str,
                                model_name: str = "lite", session_id: str = "") -> Dict[str, Any]:
        print("DEBUG: Entered assess_task_confidence function", file=sys.stderr)
        print(f"DEBUG: Manually printing from AgentCore logger (INFO level): {self.logger.isEnabledFor(logging.INFO)}", file=sys.stderr)

        """
        Pre-task epistemic assessment using the full GCP pipeline.

        This ensures GAIA has her identity, world state, and tool awareness when
        evaluating whether she can complete a task accurately. The GCP provides:
        - Identity context (who GAIA is)
        - World state (available tools, knowledge base structure)
        - MCP capabilities (file read, etc.)

        This is the "embarrassment signal" that humans have - knowing when you
        don't know something well enough to attempt it confidently.

        Args:
            intent: The detected intent (e.g., "recitation")
            user_input: The user's request
            model_name: Which model to use for assessment
            session_id: Session ID for packet creation

        Returns:
            Dict with:
                - confidence_score: 0.0-1.0 self-assessed confidence
                - can_attempt: bool - should we even try?
                - reasoning: Why this confidence level
                - alternative_offer: What to offer if confidence is low
        """
        # Build the assessment request - this will be wrapped in a full GCP packet
        assessment_request = f"""CONFIDENCE ASSESSMENT TASK

Before attempting the following task, perform an honest self-assessment.

TASK TYPE: {intent}
USER REQUEST: {user_input}

Consider your capabilities:
1. Do you have this content ACCURATELY memorized word-for-word?
2. If not memorized, does this content EXIST in your knowledge base where you could READ it?
   (Check: knowledge/system_reference/core_documents/, knowledge/personas/, etc.)
3. Do you have MCP tools (like ai_read, list_tree) that could help you access the content?

IMPORTANT:
- If you don't have something memorized but CAN access it via file read tools, that's a VALID approach!
- It's better to read from source than to hallucinate from memory.
- Your knowledge base contains your own documents (constitution, blueprints, etc.)

Respond in EXACTLY this format:
CONFIDENCE: [0.0-1.0 - can be high if you can read from file]
CAN_ATTEMPT: [yes/no]
REASONING: [explain your plan - memorized? read from file? which file?]
ALTERNATIVE: [if low confidence: what could you do instead?]"""

        try:
            # Create a proper CognitionPacket with full GCP context
            # This gives the model identity, world state, and tool awareness
            packet = self._create_initial_packet(
                user_input=assessment_request,
                session_id=session_id or "confidence_assessment",
                history=[],
                selected_model_name=model_name,
            )

            # Mark this as a confidence assessment task
            try:
                packet.intent.system_task = SystemTask.INTENT_DETECTION
                packet.intent.user_intent = "confidence_assessment"
            except Exception:
                pass

            # Build messages using the GCP pipeline
            # This includes identity, world state, MCP tools, knowledge base awareness
            messages = build_from_packet(packet, task_instruction_key="confidence_assessment")

            response = self.model_pool.forward_to_model(
                model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.3,
            )

            assessment_text = response["choices"][0]["message"]["content"]
            self.logger.debug(f"Raw confidence assessment (GCP): {assessment_text}")

            # Parse the response
            import re

            confidence_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", assessment_text, re.IGNORECASE)
            confidence_score = float(confidence_match.group(1)) if confidence_match else 0.5
            # Clamp to valid range
            confidence_score = max(0.0, min(1.0, confidence_score))

            can_attempt_match = re.search(r"CAN_ATTEMPT:\s*(yes|no)", assessment_text, re.IGNORECASE)
            can_attempt = can_attempt_match.group(1).lower() == "yes" if can_attempt_match else True

            reasoning_match = re.search(r"REASONING:\s*(.+?)(?=ALTERNATIVE:|$)", assessment_text, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            alternative_match = re.search(r"ALTERNATIVE:\s*(.+)", assessment_text, re.IGNORECASE | re.DOTALL)
            alternative_offer = alternative_match.group(1).strip() if alternative_match else ""

            # VERIFICATION STEP: Check if any claimed file paths actually exist
            # Models often hallucinate about having files they don't have
            claimed_paths = re.findall(r'knowledge/[^\s,\'"]+|/knowledge/[^\s,\'"]+', reasoning + assessment_text)
            verified_paths = []
            missing_paths = []

            for claimed_path in claimed_paths:
                # Normalize path
                if not claimed_path.startswith('/'):
                    claimed_path = '/' + claimed_path
                # Check if file exists
                if os.path.exists(claimed_path):
                    verified_paths.append(claimed_path)
                else:
                    missing_paths.append(claimed_path)

            # If the model claimed high confidence based on files that don't exist, downgrade
            if missing_paths and confidence_score > 0.5:
                self.logger.warning(
                    f"Confidence downgrade: model claimed paths that don't exist: {missing_paths}"
                )
                # Downgrade confidence significantly - the model is hallucinating
                confidence_score = 0.2
                can_attempt = False
                reasoning = (
                    f"VERIFICATION FAILED: Model claimed to have files at {missing_paths} "
                    f"but these paths do not exist. Original reasoning: {reasoning}"
                )
                alternative_offer = (
                    "I don't have this content in my knowledge base. "
                    "I could attempt to recall it from my training data, but accuracy is not guaranteed."
                )

            return {
                "confidence_score": confidence_score,
                "can_attempt": can_attempt,
                "reasoning": reasoning,
                "alternative_offer": alternative_offer,
                "raw_assessment": assessment_text,
                "verified_paths": verified_paths,
                "missing_paths": missing_paths,
            }

        except Exception as e:
            self.logger.exception("Confidence assessment failed")
            return {
                "confidence_score": 0.5,
                "can_attempt": True,
                "reasoning": f"Assessment failed: {e}",
                "alternative_offer": ""
            }

    def reflect_on_truncation(self, original_request: str, truncated_output: str,
                               model_name: str = "lite") -> Dict[str, Any]:
        """
        Use a model to reflect on truncated output and generate a natural continuation prompt.

        This is the intelligent, GAIA-aligned approach: rather than using regex or heuristics,
        we ask the model to understand what was generated and craft an appropriate continuation.

        Args:
            original_request: The user's original request
            truncated_output: The output that was truncated
            model_name: Which model to use for reflection (default: lite for speed)

        Returns:
            Dict with:
                - continuation_prompt: Natural language prompt for continuing
                - analysis: Model's analysis of what was generated
                - estimated_progress: Rough estimate of completion (e.g., "~50%", "stanza 9/18")
                - repetition_detected: Whether the model noticed repetition
                - reasoning: The model's reasoning process
        """
        # Trim output for reflection (last ~1500 chars for context, first ~500 for beginning)
        output_beginning = truncated_output[:500] if len(truncated_output) > 500 else truncated_output
        output_end = truncated_output[-1500:] if len(truncated_output) > 1500 else truncated_output

        reflection_prompt = f"""You are helping GAIA (an AI assistant) continue a response that was truncated due to length limits.

ORIGINAL REQUEST FROM USER:
{original_request}

BEGINNING OF GAIA'S RESPONSE:
{output_beginning}
{"..." if len(truncated_output) > 500 else ""}

END OF GAIA'S RESPONSE (where it was cut off):
{"..." if len(truncated_output) > 1500 else ""}{output_end}

Please analyze this truncated response and help GAIA continue it properly.

Respond in this exact format:
ANALYSIS: [Brief analysis of what was generated and where it stopped]
REPETITION: [yes/no - did you notice any repeated content near the end?]
PROGRESS: [Estimate how much of the requested content was completed, e.g., "stanza 9 of 18" or "~60%"]
CONTINUATION_PROMPT: [Write the exact prompt GAIA should use to continue. Be specific about where to pick up, e.g., "Continue reciting The Raven from stanza 10, which begins 'Then, methought, the air grew denser...' Do not repeat any previous stanzas."]
REASONING: [Brief explanation of your analysis]"""

        try:
            # Use the specified model for reflection (get() now supports lazy loading)
            model = self.model_pool.get(model_name)
            if not model:
                # Fallback to any available model - get() has lazy loading built in
                for fallback in ["lite", "gpu_prime", "prime"]:
                    model = self.model_pool.get(fallback)
                    if model:
                        break

            if not model:
                self.logger.warning("No model available for truncation reflection")
                return {
                    "continuation_prompt": self.build_continuation_prompt(original_request, truncated_output),
                    "analysis": "No model available for reflection",
                    "estimated_progress": "unknown",
                    "repetition_detected": False,
                    "reasoning": "Fallback to basic continuation"
                }

            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes AI outputs and helps with continuation."},
                {"role": "user", "content": reflection_prompt}
            ]

            response = self.model_pool.forward_to_model(
                model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            reflection_text = response["choices"][0]["message"]["content"]

            # Parse the structured response
            result = {
                "continuation_prompt": "",
                "analysis": "",
                "estimated_progress": "unknown",
                "repetition_detected": False,
                "reasoning": "",
                "raw_reflection": reflection_text
            }

            # Extract each field from the response
            lines = reflection_text.split("\n")
            current_field = None
            current_value = []

            for line in lines:
                line_upper = line.upper().strip()
                if line_upper.startswith("ANALYSIS:"):
                    if current_field and current_value:
                        result[current_field] = " ".join(current_value).strip()
                    current_field = "analysis"
                    current_value = [line.split(":", 1)[1].strip() if ":" in line else ""]
                elif line_upper.startswith("REPETITION:"):
                    if current_field and current_value:
                        result[current_field] = " ".join(current_value).strip()
                    current_field = "repetition_detected"
                    val = line.split(":", 1)[1].strip().lower() if ":" in line else ""
                    result["repetition_detected"] = "yes" in val
                    current_field = None
                    current_value = []
                elif line_upper.startswith("PROGRESS:"):
                    if current_field and current_value:
                        result[current_field] = " ".join(current_value).strip()
                    current_field = "estimated_progress"
                    current_value = [line.split(":", 1)[1].strip() if ":" in line else ""]
                elif line_upper.startswith("CONTINUATION_PROMPT:"):
                    if current_field and current_value:
                        result[current_field] = " ".join(current_value).strip()
                    current_field = "continuation_prompt"
                    current_value = [line.split(":", 1)[1].strip() if ":" in line else ""]
                elif line_upper.startswith("REASONING:"):
                    if current_field and current_value:
                        result[current_field] = " ".join(current_value).strip()
                    current_field = "reasoning"
                    current_value = [line.split(":", 1)[1].strip() if ":" in line else ""]
                elif current_field:
                    current_value.append(line.strip())

            # Capture the last field
            if current_field and current_value:
                result[current_field] = " ".join(current_value).strip()

            # If we couldn't parse a continuation prompt, fall back to basic method
            if not result["continuation_prompt"]:
                result["continuation_prompt"] = self.build_continuation_prompt(original_request, truncated_output)
                result["reasoning"] = (result.get("reasoning", "") + " [Fallback to basic continuation prompt]").strip()

            self.logger.info(f"Truncation reflection complete: progress={result['estimated_progress']}, repetition={result['repetition_detected']}")

            # Topic-alignment check: ensure continuation prompt relates to original request
            if result["continuation_prompt"]:
                is_aligned, alignment_reason = self._check_topic_alignment(
                    original_request, result["continuation_prompt"]
                )
                if not is_aligned:
                    self.logger.warning(f"Continuation prompt misaligned with original request: {alignment_reason}")
                    # Fall back to a simple, grounded continuation prompt
                    result["continuation_prompt"] = self._build_grounded_continuation(original_request, truncated_output)
                    result["reasoning"] = (result.get("reasoning", "") + f" [Topic misalignment detected: {alignment_reason}. Using grounded continuation.]").strip()
                    result["topic_aligned"] = False
                else:
                    result["topic_aligned"] = True

            return result

        except Exception as e:
            self.logger.exception("Error during truncation reflection")
            return {
                "continuation_prompt": self.build_continuation_prompt(original_request, truncated_output),
                "analysis": f"Reflection failed: {e}",
                "estimated_progress": "unknown",
                "repetition_detected": False,
                "reasoning": "Error during reflection, using fallback"
            }

    def _check_topic_alignment(self, original_request: str, continuation_prompt: str) -> tuple:
        """
        Check if a continuation prompt is topically aligned with the original request.

        Uses keyword extraction and semantic similarity to detect when the model
        has drifted off-topic in its continuation prompt.

        Args:
            original_request: The user's original request
            continuation_prompt: The generated continuation prompt to validate

        Returns:
            Tuple of (is_aligned: bool, reason: str)
        """
        import re

        # Extract significant words from both texts (lowercase, 4+ chars, no stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                     'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were',
                     'they', 'this', 'that', 'with', 'from', 'what', 'which', 'their',
                     'will', 'would', 'could', 'should', 'there', 'where', 'when', 'who',
                     'how', 'why', 'about', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further', 'then',
                     'once', 'here', 'some', 'more', 'most', 'other', 'only', 'same',
                     'than', 'very', 'just', 'also', 'now', 'continue', 'continuing',
                     'recite', 'reciting', 'explain', 'explaining', 'describe', 'next',
                     'portion', 'content', 'please', 'focus', 'specifically', 'gaia'}

        def extract_keywords(text: str) -> set:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            return {w for w in words if w not in stopwords}

        original_keywords = extract_keywords(original_request)
        continuation_keywords = extract_keywords(continuation_prompt)

        if not original_keywords:
            # Can't validate if we can't extract keywords from original
            return (True, "No keywords to validate against")

        # Check for keyword overlap
        overlap = original_keywords & continuation_keywords
        overlap_ratio = len(overlap) / len(original_keywords) if original_keywords else 0

        # Also check if continuation introduces many new topic-specific words
        # that weren't in the original (sign of topic drift)
        new_keywords = continuation_keywords - original_keywords

        # Known topic-drift indicators: specific nouns that indicate a different subject
        # If continuation has many proper nouns/specific terms not in original, it's drifting
        drift_indicators = {
            'sumerian', 'babylonian', 'mesopotamian', 'enlil', 'inanna', 'enki', 'anu',
            'cuneiform', 'tablet', 'ziggurat', 'temple', 'deity', 'deities', 'pantheon',
            'archaeological', 'dynasty', 'inscription', 'ritual', 'worship', 'mythology'
        }

        drift_terms_found = new_keywords & drift_indicators

        # Decision logic
        if drift_terms_found and overlap_ratio < 0.3:
            return (False, f"Topic drift detected: introduced '{', '.join(list(drift_terms_found)[:3])}' with low overlap ({overlap_ratio:.0%})")

        if overlap_ratio < 0.15 and len(new_keywords) > len(original_keywords):
            return (False, f"Low topic overlap ({overlap_ratio:.0%}) with many new terms")

        # Check for specific mismatches (e.g., poem request getting historical response)
        poem_indicators = {'poem', 'poetry', 'recite', 'stanza', 'verse', 'rhyme', 'jabberwocky',
                          'carroll', 'lewis', 'looking', 'glass', 'brillig', 'slithy', 'toves'}
        history_indicators = {'history', 'historical', 'ancient', 'civilization', 'empire',
                             'dynasty', 'archaeological', 'evidence', 'cuneiform'}

        original_is_poem = bool(original_keywords & poem_indicators)
        continuation_is_history = bool(continuation_keywords & history_indicators)

        if original_is_poem and continuation_is_history and not (continuation_keywords & poem_indicators):
            return (False, "Original request was about poetry but continuation drifted to history")

        return (True, f"Topic aligned (overlap: {overlap_ratio:.0%})")

    def _build_grounded_continuation(self, original_request: str, truncated_output: str) -> str:
        """
        Build a simple, grounded continuation prompt that stays on topic.

        This is used when the model-generated continuation prompt has drifted off-topic.
        Instead of trusting the model's reflection, we build a minimal prompt that
        directly references the original request.

        Args:
            original_request: The user's original request
            truncated_output: The output that was truncated

        Returns:
            A grounded continuation prompt
        """
        # Extract the last meaningful line from the truncated output
        lines = [l.strip() for l in truncated_output.strip().split('\n') if l.strip()]
        last_line = lines[-1] if lines else ""

        # Truncate last_line if too long
        if len(last_line) > 100:
            last_line = last_line[:100] + "..."

        # Build a simple, direct continuation prompt
        prompt = f'Continue responding to this request: "{original_request[:200]}"'
        if last_line:
            prompt += f'\n\nYou left off at: "{last_line}"'
        prompt += "\n\nContinue from where you stopped. Stay on topic and do not introduce new subjects."

        return prompt

    # -------------------------------------------------------------------------
    # Self-Improvement Orchestration
    # -------------------------------------------------------------------------
    # These methods enable GAIA to review her own code, propose fixes, and
    # apply them safely with automatic rollback on failure.

    def _find_relevant_files(self, topic: str, max_files: int = 10) -> List[Dict[str, Any]]:
        """
        Find code files relevant to a given topic by searching filenames and content.

        Args:
            topic: The topic to search for (e.g., "Discord integration", "intent detection")
            max_files: Maximum number of files to return

        Returns:
            List of dicts with {path, relevance_score, match_type}
        """
        import subprocess

        results = []
        app_dir = Path("/gaia-assistant/app") if Path("/gaia-assistant/app").exists() else Path.cwd() / "app"

        # Keywords derived from topic
        keywords = [w.lower() for w in topic.split() if len(w) > 2]

        try:
            # Search filenames first
            for py_file in app_dir.rglob("*.py"):
                if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                    continue

                filename_lower = py_file.name.lower()
                stem_lower = py_file.stem.lower()

                # Check filename match
                filename_score = sum(1 for kw in keywords if kw in filename_lower or kw in stem_lower)
                if filename_score > 0:
                    results.append({
                        "path": str(py_file),
                        "relevance_score": filename_score * 2,  # Filename matches weighted higher
                        "match_type": "filename",
                    })

            # Search content with grep
            for kw in keywords:
                try:
                    grep_result = subprocess.run(
                        ["grep", "-l", "-r", "-i", kw, str(app_dir)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if grep_result.returncode == 0:
                        for line in grep_result.stdout.strip().split("\n"):
                            if line and line.endswith(".py") and "__pycache__" not in line:
                                # Check if already in results
                                existing = next((r for r in results if r["path"] == line), None)
                                if existing:
                                    existing["relevance_score"] += 1
                                    existing["match_type"] = "filename+content"
                                else:
                                    results.append({
                                        "path": line,
                                        "relevance_score": 1,
                                        "match_type": "content",
                                    })
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Grep timed out searching for '{kw}'")
                except Exception as e:
                    self.logger.warning(f"Grep failed for '{kw}': {e}")

            # Sort by relevance and limit
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:max_files]

        except Exception as e:
            self.logger.error(f"Error finding relevant files for '{topic}': {e}")
            return []

    def _analyze_code_for_topic(
        self,
        topic: str,
        file_paths: List[str],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Use the LLM to analyze code files related to a topic.

        Args:
            topic: The topic/task being analyzed
            file_paths: List of file paths to analyze
            task_context: Optional context from dev_matrix task

        Returns:
            Dict with {summary, issues, suggestions, files_analyzed}
        """
        # Read file contents (limited to prevent context overflow)
        file_contents = []
        total_chars = 0
        max_chars = 50000  # Limit total content

        for fp in file_paths:
            if total_chars >= max_chars:
                break
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                # Truncate very large files
                if len(content) > 10000:
                    content = content[:10000] + "\n... [truncated]"
                file_contents.append(f"=== FILE: {fp} ===\n{content}")
                total_chars += len(content)
            except Exception as e:
                self.logger.warning(f"Failed to read {fp}: {e}")

        if not file_contents:
            return {
                "summary": "No files could be read for analysis.",
                "issues": [],
                "suggestions": [],
                "files_analyzed": [],
            }

        # Build analysis prompt
        task_desc = ""
        if task_context:
            task_desc = f"\nTask from dev_matrix: {task_context.get('task', topic)}\nPurpose: {task_context.get('purpose', 'N/A')}\nStatus: {task_context.get('status', 'open')}\n"

        analysis_prompt = f"""You are GAIA, analyzing your own codebase to understand the implementation of: {topic}
{task_desc}
Review the following code files and provide:
1. SUMMARY: A brief summary of how this topic is currently implemented
2. ISSUES: Any bugs, incomplete implementations, or problems you notice (list each as a bullet)
3. SUGGESTIONS: Specific improvements that could be made (list each as a bullet)

Be specific and reference file paths and line numbers where relevant.

CODE FILES:
{chr(10).join(file_contents)}

Respond in this format:
SUMMARY:
<your summary>

ISSUES:
- <issue 1>
- <issue 2>
...

SUGGESTIONS:
- <suggestion 1>
- <suggestion 2>
..."""

        try:
            llm = self.model_pool.get_model_for_role("prime")
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=2000,
            )

            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            response_text = strip_think_tags(response_text)

            # Parse the response
            parsed = {"summary": "", "issues": [], "suggestions": [], "files_analyzed": file_paths}

            current_section = None
            for line in response_text.split("\n"):
                line_stripped = line.strip()
                line_upper = line_stripped.upper()

                if line_upper.startswith("SUMMARY:"):
                    current_section = "summary"
                    rest = line_stripped[8:].strip()
                    if rest:
                        parsed["summary"] = rest
                elif line_upper.startswith("ISSUES:"):
                    current_section = "issues"
                elif line_upper.startswith("SUGGESTIONS:"):
                    current_section = "suggestions"
                elif current_section == "summary" and line_stripped:
                    parsed["summary"] += " " + line_stripped
                elif current_section == "issues" and line_stripped.startswith("-"):
                    parsed["issues"].append(line_stripped[1:].strip())
                elif current_section == "suggestions" and line_stripped.startswith("-"):
                    parsed["suggestions"].append(line_stripped[1:].strip())

            parsed["summary"] = parsed["summary"].strip()
            self.logger.info(f"Code analysis complete: {len(parsed['issues'])} issues, {len(parsed['suggestions'])} suggestions")
            return parsed

        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return {
                "summary": f"Analysis failed: {e}",
                "issues": [],
                "suggestions": [],
                "files_analyzed": file_paths,
            }

    def _propose_code_fix(
        self,
        file_path: str,
        issue_description: str,
        suggestion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Have the LLM propose a specific code fix for a file.

        Args:
            file_path: Path to the file to fix
            issue_description: Description of the issue to fix
            suggestion: Optional suggestion for how to fix it

        Returns:
            Dict with {ok, original_content, proposed_content, explanation, error}
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()
        except Exception as e:
            return {"ok": False, "error": f"Failed to read file: {e}"}

        # Truncate for very large files
        if len(original_content) > 30000:
            return {
                "ok": False,
                "error": "File too large for safe automated editing. Consider manual review.",
            }

        fix_prompt = f"""You are GAIA, editing your own source code.

FILE: {file_path}

ISSUE: {issue_description}
{f"SUGGESTED APPROACH: {suggestion}" if suggestion else ""}

CURRENT CODE:
```python
{original_content}
```

Provide the COMPLETE fixed file content. Make minimal changes to fix the issue.
Do NOT add unnecessary comments, docstrings, or refactoring beyond the fix.
If the issue cannot be fixed or is unclear, respond with "CANNOT_FIX: <reason>".

Respond with ONLY the complete file content (no markdown code blocks, no explanation).
Start your response with the first line of the file."""

        try:
            llm = self.model_pool.get_model_for_role("prime")
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,  # Low temp for precise code generation
                max_tokens=min(len(original_content) * 2, 8000),
            )

            proposed_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            proposed_content = strip_think_tags(proposed_content)

            # Check for refusal
            if proposed_content.strip().startswith("CANNOT_FIX:"):
                reason = proposed_content.strip()[11:].strip()
                return {"ok": False, "error": f"LLM declined to fix: {reason}"}

            # Basic validation - should start like Python code
            if not proposed_content.strip():
                return {"ok": False, "error": "LLM returned empty response"}

            # Strip markdown code blocks if present
            if proposed_content.strip().startswith("```"):
                lines = proposed_content.strip().split("\n")
                # Remove first and last lines if they're code fences
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                proposed_content = "\n".join(lines)

            return {
                "ok": True,
                "original_content": original_content,
                "proposed_content": proposed_content,
                "file_path": file_path,
                "issue": issue_description,
                "explanation": f"Fix for: {issue_description}",
            }

        except Exception as e:
            self.logger.error(f"Failed to propose fix: {e}")
            return {"ok": False, "error": str(e)}

    def _apply_code_fix(
        self,
        file_path: str,
        new_content: str,
        reason: str,
        run_syntax_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply a code fix using SnapshotManager with automatic rollback on failure.

        Args:
            file_path: Path to the file to edit
            new_content: The new content to write
            reason: Reason for the edit (for backup metadata)
            run_syntax_check: Whether to validate Python syntax after writing

        Returns:
            Dict with {ok, backup_path, validated, rolled_back, error}
        """
        # TODO: [GAIA-REFACTOR] snapshot_manager.py module not yet migrated.
        try:
            from gaia_core.utils.code_analyzer.snapshot_manager import SnapshotManager, validate_python_syntax
        except ImportError:
            return {"ok": False, "error": "snapshot_manager module not yet migrated"}

        try:
            sm = SnapshotManager(self.config)

            validator = validate_python_syntax if run_syntax_check else None

            result = sm.safe_edit(
                file_path=file_path,
                new_content=new_content,
                reason=reason,
                validator=validator,
            )

            if result.get("ok"):
                self.logger.info(f"Code fix applied successfully to {file_path}")
            else:
                self.logger.warning(f"Code fix failed for {file_path}: {result.get('error')}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to apply code fix: {e}")
            return {"ok": False, "error": str(e)}

    def _update_dev_matrix_task(
        self,
        task_context: Dict[str, Any],
        topic: str,
        fixes_applied: int,
        files_modified: List[str],
        analysis_summary: str,
    ) -> Dict[str, Any]:
        """
        Update a dev_matrix task after successful self-improvement fixes.

        Args:
            task_context: The original task dict from dev_matrix
            topic: The topic that was improved
            fixes_applied: Number of fixes successfully applied
            files_modified: List of file paths that were modified
            analysis_summary: Summary from the code analysis

        Returns:
            Dict with {ok, task, status_changed, audit_added, error}
        """
        from datetime import datetime, timezone
        import json

        try:
            from gaia_core.memory.dev_matrix import GAIADevMatrix
            dm = GAIADevMatrix(self.config)

            task_label = task_context.get("task", topic)

            # Find the task in the matrix
            task_found = None
            for task in dm.tasks:
                if task.get("task") == task_label:
                    task_found = task
                    break

            if not task_found:
                return {"ok": False, "error": f"Task '{task_label}' not found in dev_matrix"}

            # Create audit entry
            audit_entry = {
                "by": "gaia_self_improvement",
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": "auto_fix_applied",
                "fixes_applied": fixes_applied,
                "files_modified": files_modified,
                "summary": analysis_summary[:500] if analysis_summary else "",
            }

            # Add audit trail
            if "audit" not in task_found:
                task_found["audit"] = []
            task_found["audit"].append(audit_entry)

            # Determine if we should mark as resolved
            # Only mark resolved if the task was previously "open" and we applied fixes
            status_changed = False
            previous_status = task_found.get("status", "open")

            # For now, we add progress but don't auto-resolve
            # The user or a subsequent review should confirm resolution
            if "progress" not in task_found:
                task_found["progress"] = []
            task_found["progress"].append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "fixes": fixes_applied,
                "files": files_modified,
            })

            # Save the updated matrix
            dm._save()

            self.logger.info(f"Updated dev_matrix task '{task_label}': +{fixes_applied} fixes, audit added")

            return {
                "ok": True,
                "task": task_label,
                "status_changed": status_changed,
                "previous_status": previous_status,
                "audit_added": True,
                "fixes_recorded": fixes_applied,
            }

        except Exception as e:
            self.logger.error(f"Failed to update dev_matrix task: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def run_self_improvement(
        self,
        topic: str,
        auto_apply: bool = False,
        max_files: int = 5,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Orchestrate a self-improvement cycle for a given topic.

        This method:
        1. Finds relevant code files for the topic
        2. Analyzes the code for issues and suggestions
        3. Proposes fixes for identified issues
        4. Optionally applies fixes with automatic rollback on failure

        Args:
            topic: The topic to improve (e.g., "Discord integration")
            auto_apply: If True, automatically apply proposed fixes. If False, only propose.
            max_files: Maximum number of files to analyze

        Yields:
            Progress events as dicts with {stage, status, data}
        """
        yield {"stage": "start", "status": "Starting self-improvement cycle", "data": {"topic": topic}}

        # Stage 1: Find relevant files
        yield {"stage": "find_files", "status": "Searching for relevant files...", "data": None}
        relevant_files = self._find_relevant_files(topic, max_files=max_files)

        if not relevant_files:
            yield {
                "stage": "find_files",
                "status": "complete",
                "data": {"error": f"No relevant files found for topic: {topic}"},
            }
            return

        yield {
            "stage": "find_files",
            "status": "complete",
            "data": {"files": relevant_files, "count": len(relevant_files)},
        }

        # Stage 2: Check dev_matrix for related tasks
        task_context = None
        try:
            from gaia_core.memory.dev_matrix import GAIADevMatrix
            dm = GAIADevMatrix(self.config)
            open_tasks = dm.get_open_tasks()
            # Find matching task
            topic_lower = topic.lower()
            for task in open_tasks:
                if topic_lower in task.get("task", "").lower() or topic_lower in task.get("purpose", "").lower():
                    task_context = task
                    yield {
                        "stage": "dev_matrix",
                        "status": "Found related task in dev_matrix",
                        "data": task,
                    }
                    break
        except Exception as e:
            self.logger.warning(f"Could not check dev_matrix: {e}")

        # Stage 3: Analyze code
        yield {"stage": "analyze", "status": "Analyzing code...", "data": None}
        file_paths = [f["path"] for f in relevant_files]
        analysis = self._analyze_code_for_topic(topic, file_paths, task_context)

        yield {
            "stage": "analyze",
            "status": "complete",
            "data": analysis,
        }

        # Stage 4: Propose fixes for issues
        if not analysis.get("issues"):
            yield {
                "stage": "propose",
                "status": "complete",
                "data": {"message": "No issues found requiring fixes."},
            }
            return

        proposed_fixes = []
        for i, issue in enumerate(analysis["issues"][:3]):  # Limit to 3 fixes per run
            yield {
                "stage": "propose",
                "status": f"Proposing fix for issue {i+1}...",
                "data": {"issue": issue},
            }

            # Find the most relevant file for this issue
            # Simple heuristic: look for file references in the issue text
            target_file = file_paths[0]  # Default to most relevant file
            for fp in file_paths:
                if Path(fp).name in issue or Path(fp).stem in issue:
                    target_file = fp
                    break

            suggestion = analysis["suggestions"][i] if i < len(analysis["suggestions"]) else None
            fix_proposal = self._propose_code_fix(target_file, issue, suggestion)

            if fix_proposal.get("ok"):
                proposed_fixes.append(fix_proposal)
                yield {
                    "stage": "propose",
                    "status": f"Fix proposed for {Path(target_file).name}",
                    "data": {
                        "file": target_file,
                        "issue": issue,
                        "has_fix": True,
                    },
                }
            else:
                yield {
                    "stage": "propose",
                    "status": f"Could not propose fix",
                    "data": {
                        "file": target_file,
                        "issue": issue,
                        "error": fix_proposal.get("error"),
                    },
                }

        # Stage 5: Apply fixes (if auto_apply is True)
        successful_fixes = 0
        if auto_apply and proposed_fixes:
            yield {"stage": "apply", "status": "Applying fixes...", "data": None}

            applied_results = []
            for fix in proposed_fixes:
                yield {
                    "stage": "apply",
                    "status": f"Applying fix to {Path(fix['file_path']).name}...",
                    "data": {"file": fix["file_path"]},
                }

                apply_result = self._apply_code_fix(
                    file_path=fix["file_path"],
                    new_content=fix["proposed_content"],
                    reason=f"Self-improvement: {fix['issue'][:100]}",
                    run_syntax_check=True,
                )

                applied_results.append({
                    "file": fix["file_path"],
                    "result": apply_result,
                })

                if apply_result.get("ok"):
                    successful_fixes += 1
                    yield {
                        "stage": "apply",
                        "status": f"Fix applied to {Path(fix['file_path']).name}",
                        "data": apply_result,
                    }
                else:
                    yield {
                        "stage": "apply",
                        "status": f"Fix failed for {Path(fix['file_path']).name}",
                        "data": apply_result,
                    }

            yield {
                "stage": "apply",
                "status": "complete",
                "data": {"results": applied_results, "successful_count": successful_fixes},
            }

            # Stage 5b: Update dev_matrix if we have a task context and fixes succeeded
            if task_context and successful_fixes > 0:
                yield {"stage": "dev_matrix_update", "status": "Updating task status...", "data": None}
                try:
                    update_result = self._update_dev_matrix_task(
                        task_context=task_context,
                        topic=topic,
                        fixes_applied=successful_fixes,
                        files_modified=[r["file"] for r in applied_results if r["result"].get("ok")],
                        analysis_summary=analysis.get("summary", ""),
                    )
                    yield {
                        "stage": "dev_matrix_update",
                        "status": "complete" if update_result.get("ok") else "failed",
                        "data": update_result,
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to update dev_matrix: {e}")
                    yield {
                        "stage": "dev_matrix_update",
                        "status": "failed",
                        "data": {"error": str(e)},
                    }

        elif proposed_fixes:
            yield {
                "stage": "apply",
                "status": "skipped",
                "data": {
                    "message": "Fixes proposed but not applied (auto_apply=False)",
                    "proposed_count": len(proposed_fixes),
                },
            }

        # Stage 6: Summary
        yield {
            "stage": "complete",
            "status": "Self-improvement cycle complete",
            "data": {
                "topic": topic,
                "files_analyzed": len(file_paths),
                "issues_found": len(analysis.get("issues", [])),
                "fixes_proposed": len(proposed_fixes),
                "fixes_applied": successful_fixes,
                "task_updated": task_context is not None and successful_fixes > 0,
            },
        }

    # =========================================================================
    # GCP Tool Routing System Methods
    # =========================================================================

    def _run_tool_routing_loop(
        self,
        packet: CognitionPacket,
        user_input: str,
        session_id: str = "",
        source: str = "cli",
        metadata: dict = None
    ) -> CognitionPacket:
        """
        Execute the tool routing loop for packets that need MCP tools.

        This method:
        1. Selects appropriate tool with low-temp generation
        2. Reviews selection for confidence
        3. Executes if confidence is high
        4. Injects result back into packet

        Returns the modified packet.
        """
        from gaia_core.cognition.tool_selector import (
            needs_tool_routing, select_tool, review_selection,
            initialize_tool_routing, inject_tool_result_into_packet,
            AVAILABLE_TOOLS
        )

        _meta = metadata or {}

        # Initialize tool routing state if not present
        packet = initialize_tool_routing(packet)

        # Safety check: prevent infinite loops
        if packet.tool_routing.reinjection_count >= packet.tool_routing.max_reinjections:
            logger.warning(f"Max reinjections reached for packet {packet.header.packet_id}")
            packet.tool_routing.execution_status = ToolExecutionStatus.SKIPPED
            return packet

        packet.tool_routing.reinjection_count += 1
        packet.tool_routing.routing_requested = True

        # Get tool routing config
        tool_config = self.config.constants.get("TOOL_ROUTING", {})
        selection_temp = tool_config.get("SELECTION_TEMPERATURE", 0.15)
        review_temp = tool_config.get("REVIEW_TEMPERATURE", 0.3)
        confidence_threshold = tool_config.get("CONFIDENCE_THRESHOLD", 0.7)

        # Step 1: Tool Selection (low temperature)
        logger.info(f"Tool routing: selecting tool for packet {packet.header.packet_id}")
        ts_write({
            "type": "tool_routing",
            "stage": "selection_start",
            "packet_id": packet.header.packet_id,
            "reinjection_count": packet.tool_routing.reinjection_count
        }, session_id, source=source, destination_context=_meta)

        # Use Lite model for selection if available, otherwise use first available model
        # Note: acquire_model() now supports lazy loading - no need to pre-check pool
        selection_model = None
        selection_model_name = None
        for cand in ["lite", "prime", "gpu_prime", "cpu_prime"]:
            try:
                selection_model = self.model_pool.acquire_model(cand)
                if selection_model is not None:
                    selection_model_name = cand
                    break
            except Exception:
                continue

        if not selection_model:
            logger.error("No model available for tool selection")
            packet.tool_routing.execution_status = ToolExecutionStatus.FAILED
            return packet

        try:
            primary_tool, alternatives = select_tool(
                packet=packet,
                user_input=user_input,
                model=selection_model,
                temperature=selection_temp
            )
        finally:
            if selection_model_name:
                try:
                    self.model_pool.release_model(selection_model_name)
                except Exception:
                    pass

        if primary_tool is None:
            logger.info("Tool selector determined no tool needed")
            packet.tool_routing.needs_tool = False
            packet.tool_routing.execution_status = ToolExecutionStatus.SKIPPED
            ts_write({
                "type": "tool_routing",
                "stage": "no_tool_needed",
                "packet_id": packet.header.packet_id
            }, session_id, source=source, destination_context=_meta)
            return packet

        packet.tool_routing.needs_tool = True
        packet.tool_routing.selected_tool = primary_tool
        packet.tool_routing.alternative_tools = alternatives

        ts_write({
            "type": "tool_routing",
            "stage": "tool_selected",
            "tool": primary_tool.tool_name,
            "params": primary_tool.params,
            "confidence": primary_tool.selection_confidence
        }, session_id, source=source, destination_context=_meta)

        # Step 2: Confidence Review (Prime model)
        logger.info(f"Tool routing: reviewing selection {primary_tool.tool_name}")

        # Use Prime model for review if available
        # Note: acquire_model() now supports lazy loading - no need to pre-check pool
        review_model = None
        review_model_name = None
        for cand in ["prime", "gpu_prime", "lite", "cpu_prime"]:
            try:
                review_model = self.model_pool.acquire_model(cand)
                if review_model is not None:
                    review_model_name = cand
                    break
            except Exception:
                continue

        if review_model:
            try:
                confidence, reasoning = review_selection(
                    packet=packet,
                    selected_tool=primary_tool,
                    model=review_model,
                    temperature=review_temp
                )
            finally:
                if review_model_name:
                    try:
                        self.model_pool.release_model(review_model_name)
                    except Exception:
                        pass
        else:
            # No review model available, use selection confidence directly
            confidence = primary_tool.selection_confidence
            reasoning = "No review model available; using selection confidence"

        packet.tool_routing.review_confidence = confidence
        packet.tool_routing.review_reasoning = reasoning

        ts_write({
            "type": "tool_routing",
            "stage": "review_complete",
            "confidence": confidence,
            "reasoning": reasoning
        }, session_id, source=source, destination_context=_meta)

        # Step 3: Confidence Gate
        if confidence < confidence_threshold:
            logger.warning(f"Tool selection confidence too low: {confidence} < {confidence_threshold}")
            packet.tool_routing.execution_status = ToolExecutionStatus.SKIPPED
            # Add to reasoning log
            packet.reasoning.reflection_log.append(ReflectionLog(
                step="tool_routing",
                summary=f"Tool {primary_tool.tool_name} skipped due to low confidence ({confidence:.2f})",
                confidence=confidence
            ))
            ts_write({
                "type": "tool_routing",
                "stage": "confidence_gate_failed",
                "confidence": confidence,
                "threshold": confidence_threshold
            }, session_id, source=source, destination_context=_meta)
            return packet

        # Step 4: Execute Tool
        logger.info(f"Tool routing: executing {primary_tool.tool_name}")
        packet.tool_routing.execution_status = ToolExecutionStatus.APPROVED

        ts_write({
            "type": "tool_routing",
            "stage": "executing",
            "tool": primary_tool.tool_name
        }, session_id, source=source, destination_context=_meta)

        try:
            result = self._execute_mcp_tool(primary_tool)
            packet.tool_routing.execution_result = result
            packet.tool_routing.execution_status = (
                ToolExecutionStatus.EXECUTED if result.success
                else ToolExecutionStatus.FAILED
            )

            # Add result to reasoning log
            packet.reasoning.reflection_log.append(ReflectionLog(
                step="tool_execution",
                summary=f"Executed {primary_tool.tool_name}: {'success' if result.success else 'failed'}",
                confidence=confidence
            ))

            ts_write({
                "type": "tool_routing",
                "stage": "execution_complete",
                "success": result.success,
                "execution_time_ms": result.execution_time_ms
            }, session_id, source=source, destination_context=_meta)

            # --- Loop Detection: Record tool call ---
            try:
                loop_mgr = get_recovery_manager()
                if loop_mgr and loop_mgr.enabled:
                    result_str = str(result.output)[:500] if result.output else ""
                    loop_mgr.record_tool_call(
                        tool=primary_tool.tool_name,
                        args=primary_tool.params,
                        result=result_str
                    )
                    # Record error if failed
                    if not result.success and result.error:
                        loop_mgr.record_error(
                            error_type=f"tool_{primary_tool.tool_name}",
                            error_message=result.error,
                            was_success=False
                        )
            except Exception:
                logger.debug("Loop detection: failed to record tool call", exc_info=True)

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            packet.tool_routing.execution_status = ToolExecutionStatus.FAILED
            packet.tool_routing.execution_result = ToolExecutionResult(
                success=False,
                error=str(e)
            )

            # --- Loop Detection: Record error ---
            try:
                loop_mgr = get_recovery_manager()
                if loop_mgr and loop_mgr.enabled:
                    loop_mgr.record_error(
                        error_type=f"tool_{primary_tool.tool_name}",
                        error_message=str(e),
                        was_success=False
                    )
            except Exception:
                logger.debug("Loop detection: failed to record tool error", exc_info=True)

        # Step 5: Inject result into packet context
        packet = inject_tool_result_into_packet(packet)

        return packet

    def _execute_mcp_tool(self, tool: SelectedTool) -> ToolExecutionResult:
        """
        Execute an MCP tool and return the result.
        """
        import time

        start_time = time.time()

        # Get tool config for safety settings
        tool_config = self.config.constants.get("TOOL_ROUTING", {})
        allow_write = tool_config.get("ALLOW_WRITE_TOOLS", False)
        allow_execute = tool_config.get("ALLOW_EXECUTE_TOOLS", False)

        try:
            if tool.tool_name == "ai.read":
                result = mcp_client.ai_read(tool.params.get("path", ""))

            elif tool.tool_name == "ai.write":
                if not allow_write:
                    return ToolExecutionResult(
                        success=False,
                        error="Write operations are disabled. Set TOOL_ROUTING.ALLOW_WRITE_TOOLS=true to enable."
                    )
                result = mcp_client.ai_write(
                    tool.params.get("path", ""),
                    tool.params.get("content", "")
                )

            elif tool.tool_name == "ai.execute":
                if not allow_execute:
                    return ToolExecutionResult(
                        success=False,
                        error="Execute operations are disabled. Set TOOL_ROUTING.ALLOW_EXECUTE_TOOLS=true to enable."
                    )
                result = mcp_client.ai_execute(
                    tool.params.get("command", ""),
                    dry_run=not allow_execute
                )

            elif tool.tool_name == "embedding.query":
                result = mcp_client.embedding_query(
                    tool.params.get("query", ""),
                    top_k=tool.params.get("top_k", 5)
                )

            else:
                # Dispatch via MCP JSON-RPC for tools not handled locally
                logger.info(f"Dispatching unknown tool '{tool.tool_name}' via MCP JSON-RPC")
                rpc_result = mcp_client.call_jsonrpc(
                    method=tool.tool_name,
                    params=tool.params or {}
                )
                elapsed_ms = int((time.time() - start_time) * 1000)
                if rpc_result.get("ok"):
                    rpc_response = rpc_result.get("response", {})
                    actual_result = rpc_response.get("result", rpc_response)
                    return ToolExecutionResult(
                        success=actual_result.get("ok", True) if isinstance(actual_result, dict) else True,
                        output=actual_result,
                        error=actual_result.get("error") if isinstance(actual_result, dict) else None,
                        execution_time_ms=elapsed_ms
                    )
                else:
                    return ToolExecutionResult(
                        success=False,
                        error=rpc_result.get("error", "MCP call failed"),
                        execution_time_ms=elapsed_ms
                    )

            elapsed_ms = int((time.time() - start_time) * 1000)

            return ToolExecutionResult(
                success=result.get("ok", False),
                output=result,
                error=result.get("error"),
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ToolExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms
            )

    def _should_use_tool_routing(self, plan: Plan, user_input: str) -> bool:
        """
        Decide if the request should go through the tool routing system.

        Returns True if tool routing should be attempted.
        """
        # Check if tool routing is enabled
        tool_config = self.config.constants.get("TOOL_ROUTING", {})
        if not tool_config.get("ENABLED", True):
            return False

        # Check for tool_routing intent from intent detection
        if plan.intent == "tool_routing":
            return True

        # Quick heuristic check for tool-related requests
        from gaia_core.cognition.tool_selector import needs_tool_routing

        # Create a minimal packet for the check
        # We don't have a full packet yet, so we check against the raw input
        lowered = (user_input or "").lower()

        # File operation indicators
        file_indicators = [
            "read ", "open ", "show ", "view ", "cat ",
            "write ", "save ", "create file",
            ".md", ".txt", ".json", ".py", ".yaml", ".yml",
            "/gaia", "/knowledge", "/app", "/docs"
        ]

        # Command execution indicators
        exec_indicators = [
            "run ", "execute ", "shell ", "command ",
            "ls ", "pwd", "git ", "docker "
        ]

        # Search/query indicators
        search_indicators = [
            "search ", "find ", "look for ", "where is ",
            "semantic search", "query "
        ]

        for indicator in file_indicators + exec_indicators + search_indicators:
            if indicator in lowered:
                logger.debug(f"Tool routing triggered by indicator: '{indicator}'")
                return True

        return False
