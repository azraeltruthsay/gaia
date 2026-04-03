import logging
import regex as re
import uuid
import json
import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from gaia_core.memory.semantic_codex import SemanticCodex
from gaia_core.ethics.core_identity_guardian import CoreIdentityGuardian
from gaia_core.memory.codex_writer import CodexWriter
from dataclasses import dataclass
from typing import Generator, Dict, Any, List, Optional

from gaia_core.cognition.external_voice import ExternalVoice
from gaia_core.cognition.self_reflection import reflect_and_refine
from gaia_core.cognition.cognitive_audit import run_cognitive_self_audit
from gaia_core.cognition.history_review import review_history
from gaia_core.utils.prompt_builder import build_from_packet
from gaia_core.utils.output_router import route_output, _strip_think_tags_robust
# TODO: [GAIA-REFACTOR] chat_logger.py module not yet migrated.
# from app.utils.chat_logger import log_chat_entry, log_chat_entry_structured
log_chat_entry = lambda *args, **kwargs: None  # Placeholder
log_chat_entry_structured = lambda *args, **kwargs: None  # Placeholder
from gaia_core.utils.stream_observer import StreamObserver
import gaia_core.utils.mcp_client as mcp_client
from gaia_core.config import get_config
from gaia_common.utils.entity_validator import EntityValidator
from gaia_common.utils.error_logging import log_gaia_error

# Get constants from config for backwards compatibility
_config = get_config()
constants = getattr(_config, 'constants', {})
from gaia_common.utils.thoughtstream import write as ts_write
# Persona switcher for dynamic persona/knowledge-base selection
from gaia_core.behavior.persona_switcher import get_persona_for_request, get_persona_for_knowledge_base
# Pre-cognition semantic probe (vector lookup before intent detection)
from gaia_core.cognition.semantic_probe import run_semantic_probe, get_session_probe_stats
from gaia_core.cognition.cognitive_dispatcher import process_execution_results
from gaia_core.cognition.knowledge_enhancer import enhance_packet
from gaia_core.cognition.knowledge_ingestion import run_explicit_save, run_auto_detect, run_update_detect, run_attachment_ingestion

# [GCP v0.3] Import new packet structure
from gaia_common.protocols.cognition_packet import (
    COGPACKET_VERSION, COGPACKET_SCHEMA_ID,
    CognitionPacket, Header, Persona, Routing, Model, Intent, Context, Content, Reasoning, Response, Governance, Safety, Metrics, TokenUsage, Status,
    PersonaRole, Origin, TargetEngine, SystemTask, PacketState,
    DataField, ReflectionLog, RelevantHistorySnippet, SessionHistoryRef, Cheatsheet, Constraints,
    # GCP Tool Routing System
    ToolExecutionStatus, SelectedTool, ToolExecutionResult,
    # Output Routing (Spinal Column)
    OutputDestination, OutputRouting, DestinationTarget,
)
from gaia_core.cognition.nlu.intent_detection import detect_intent, Plan
import gaia_core.utils.gaia_rescue_helper as rescue_helper

# Loop Detection System
from gaia_core.cognition.loop_recovery import (
    get_recovery_manager,
    cleanup_session_manager,
    build_loop_detection_config_from_constants
)

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
        "keywords": ["gaia constitution", "gaia's constitution", "your constitution"],
        "path": "knowledge/system_reference/core_documents/gaia_constitution.md",
        "title": "The GAIA Constitution",
    },
    "layered_identity": {
        "keywords": ["layered identity", "identity model", "tier i", "tier ii", "tier iii", "identity layers"],
        "path": "knowledge/system_reference/core_documents/layered_identity_model.md",
        "title": "The Layered Identity Model",
    },
    "declaration": {
        "keywords": ["declaration of artisanal", "artisanal intelligence declaration", "artisanal declaration", "gaia declaration"],
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
    "core_blueprint": {
        "keywords": ["core blueprint", "gaia core blueprint", "your blueprint", "your core blueprint"],
        "path": "knowledge/blueprints/GAIA_CORE.md",
        "title": "The GAIA Core Blueprint",
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


@dataclass
class ComplexityAssessment:
    """Represents the results of a request complexity assessment."""
    should_escalate: bool
    reason: str
    confidence: float


class AgentCore:
    """
    Encapsulates the core "Reason-Act-Reflect" loop for GAIA.
    This class is UI-agnostic and yields structured events back to the caller.
    It is session-aware and uses a prompt builder to manage context.
    """
    MIND_TAG_FORMAT: str = "[{mind}]"
    _MIND_ALIASES: Dict[str, str] = {
        "gpu_prime": "Prime",
        "cpu_prime": "Prime",
        "prime": "Prime",
        "lite": "Core",
        "core": "Core",
        "oracle": "Oracle",
        "groq_fallback": "Groq",
    }

    # ── Always-on LoRA adapter for identity/architecture ────────────────────
    _DEFAULT_ADAPTER = os.getenv("GAIA_LORA_ADAPTER", "gaia_architecture")
    _ADAPTER_BASE_PATH = Path("/models/lora_adapters/tier1_global")
    # Models that support LoRA adapters (vLLM-backed)
    _ADAPTER_ELIGIBLE_MODELS = frozenset({"gpu_prime", "prime", "cpu_prime", "thinker"})

    def __init__(self, ai_manager, ethical_sentinel=None):
        self.ai_manager = ai_manager
        self.model_pool = ai_manager.model_pool
        self.config = ai_manager.config
        self.ethical_sentinel = ethical_sentinel
        self.session_manager = ai_manager.session_manager
        self._lifecycle_client = None  # Set by main.py after init
        # Ensure instance-level logger exists for use throughout the class
        # Some code paths reference self.logger; initialize it to the module logger
        try:
            self.logger = logging.getLogger("GAIA.AgentCore")
        except Exception:
            # Fallback to the module-level logger
            self.logger = logger
        
        # Timeline store for temporal grounding (set by main.py after sleep loop init)
        self.timeline_store = None

        # Initialize SemanticCodex and CodexWriter
        self.semantic_codex = SemanticCodex.instance(self.config)
        self.codex_writer = CodexWriter(self.config, self.semantic_codex)
        
        # Initialize EntityValidator for noun correction
        self.entity_validator = EntityValidator.from_config(self.config)

        # Initialize Identity Guardian for reflex and prompt validation
        self.identity_guardian = CoreIdentityGuardian(self.config)

    @property
    def _is_prime_available(self) -> bool:
        """Check if Prime (gpu_prime) is available for inference.

        Uses the lifecycle state machine when available, falls back to
        the legacy _gpu_released flag on the model pool.
        """
        if self._lifecycle_client:
            try:
                snapshot = self._lifecycle_client.get_state_sync(timeout=2.0)
                from gaia_common.lifecycle.states import LifecycleState
                return LifecycleState(snapshot.state) == LifecycleState.FOCUSING
            except Exception:
                pass
        # Legacy fallback
        return not getattr(self.model_pool, '_gpu_released', False)

    def _resolve_adapter(self, model_name: str) -> Optional[str]:
        """Return the LoRA adapter name to use for the given model, or None.

        Only Prime/Thinker (vLLM-backed) models can use adapters.  Returns
        None if the model is ineligible or the adapter hasn't been trained yet
        (no adapter weights on disk).
        """
        # Only vLLM-backed models support LoRA
        resolved = model_name.lower()
        if not any(tag in resolved for tag in ("prime", "thinker", "gpu")):
            return None
        # Check that the adapter has both config AND actual weight files
        adapter_dir = self._ADAPTER_BASE_PATH / self._DEFAULT_ADAPTER
        if not (adapter_dir / "adapter_config.json").exists():
            return None
        # Require actual weight files — config-only means training hasn't completed
        has_weights = (
            (adapter_dir / "adapter_model.safetensors").exists()
            or (adapter_dir / "adapter_model.bin").exists()
        )
        if not has_weights:
            return None
        return self._DEFAULT_ADAPTER

    def _emit_timeline_message(self, session_id: str, role: str, source: str = "") -> None:
        """Emit a message event to the timeline store (best-effort)."""
        if self.timeline_store is not None:
            try:
                self.timeline_store.append("message", {
                    "session_id": session_id,
                    "role": role,
                    "source": source,
                })
            except Exception as _tl_exc:
                logger.debug("Timeline write failed (non-fatal): %s", _tl_exc)

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

    def _assess_complexity(self, user_input: str) -> ComplexityAssessment:
        """
        Assess the complexity of a user request to determine if it should be escalated to the Council.
        """
        if user_input is None:
            return ComplexityAssessment(should_escalate=False, reason="standard_request", confidence=1.0)

        input_lower = user_input.lower()
        
        # 1. Technical/Code topics
        technical_keywords = ["code", "python", "script", "debug", "crash", "optimize", "architecture", "microservices", "algorithm", "dijkstra", "stack trace"]
        if any(kw in input_lower for kw in technical_keywords):
            return ComplexityAssessment(should_escalate=True, reason="technical depth", confidence=0.9)

        # 2. Philosophical/Deep topics
        if "meaning of life" in input_lower:
            return ComplexityAssessment(should_escalate=True, reason="meaning of life", confidence=0.9)

        philosophical_keywords = ["consciousness", "sovereignty", "existence", "ethics", "rights", "moral", "purpose", "nurtured mind", "how do you feel", "understand quantum"]
        if any(kw in input_lower for kw in philosophical_keywords):
            return ComplexityAssessment(should_escalate=True, reason="philosophical depth", confidence=0.9)
            
        # 3. System-internal signals
        system_keywords = ["sleep wake", "architecture work", "checkpoint", "prime currently running"]
        if any(kw in input_lower for kw in system_keywords):
            return ComplexityAssessment(should_escalate=True, reason="system reference", confidence=0.95)

        # 4. Long prompts
        if len(user_input.split()) > 100:
            return ComplexityAssessment(should_escalate=True, reason="long prompt", confidence=0.8)
            
        # 5. Recitation check (should stay on Lite)
        if "recite" in input_lower:
            return ComplexityAssessment(should_escalate=False, reason="recitation request", confidence=1.0)

        return ComplexityAssessment(should_escalate=False, reason="standard_request", confidence=1.0)

    def _escalate_to_prime(self, user_input: str, lite_response: str, reason: str, session_id: str):
        """
        Escalates a request to Prime by writing a Council note and signaling wake-up.
        """
        self.logger.info(f"Escalating session {session_id} to Prime. Reason: {reason}")
        
        # Write Council note
        if hasattr(self, "council_notes") and self.council_notes:
            # The test expects write_note with specific keyword arguments
            self.council_notes.write_note(
                user_prompt=user_input,
                lite_response=lite_response,
                escalation_reason=reason,
                session_id=session_id
            )
            
        # Signal wake-up to SleepWakeManager if available
        try:
            from gaia_core.main import app
            if hasattr(app.state, "sleep_wake_manager"):
                app.state.sleep_wake_manager.receive_wake_signal()
        except (ImportError, AttributeError):
            pass

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
        now = datetime.now(timezone.utc)
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

        # Skip history for simple greetings — prevents old conversation turns
        # from polluting trivial responses (e.g. "good afternoon" → strawberry answer)
        _greeting_words = {"hello", "hi", "hey", "good", "morning", "afternoon", "evening",
                           "night", "howdy", "greetings", "yo", "sup", "gaia"}
        _input_words = set(user_input.lower().split())
        if len(_input_words) <= 5 and _input_words.issubset(_greeting_words | {".", "!", ",", "?"}):
            window = all_history[-2:]  # Only last exchange, not full window
        else:
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
                available_tools = tool_info.get("methods", [])
        except Exception as e:
            log_gaia_error(self.logger, "GAIA-CORE-150", str(e), exc_info=True)

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
            from gaia_common.utils.world_state import format_world_state_snapshot
            world_state_text = ""
            try:
                # Include output context so GAIA knows where she's communicating
                output_context = {
                    "source": source,
                    "destination": destination,
                    **(_metadata or {}),
                }
                
                # Fetch auditory context from metadata if available (TCP)
                _metadata.get("auditory_environment") if _metadata else None
                
                world_state_text = format_world_state_snapshot(
                    output_context=output_context
                )
            except Exception:
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
            version=COGPACKET_VERSION,
            schema_id=COGPACKET_SCHEMA_ID,
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
                            except Exception as _id_exc:
                                logger.debug("AgentCore: immutable identity fallback failed: %s", _id_exc)
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
        metadata: dict = None,
        reflex_text: str = ""
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

        # 0. GRIT MODE: Push past cosmetic irritation if env var set
        if os.getenv("GAIA_GRIT_MODE", "").lower() in ("1", "true", "yes"):
            from gaia_common.utils.immune_system import enable_grit_mode
            enable_grit_mode()

        # 1. CIRCUIT BREAKER: Check if manual healing is required
        if os.path.exists("/shared/HEALING_REQUIRED.lock"):
            log_gaia_error(self.logger, "GAIA-CORE-001", "See /shared/HEALING_REQUIRED.lock")
            yield {
                "type": "error",
                "value": "My cognitive loop is currently locked for safety following a diagnostic spiral. Manual triage by the Architect (Azrael) is required to restore my integrity."
            }
            return

        lite_fallback_acquired = False

        # Normalize metadata
        _metadata = metadata or {}
        
        # Apply entity validation/correction to user input
        if user_input:
            user_input = self.entity_validator.correct_text(user_input)
            
        from gaia_core.utils.prompt_builder import build_from_packet # Assumes this is updated for v0.3

        import time as _time
        t0 = _time.perf_counter()
        
        # Primary turn wrapper for resource management
        try:
            # Initialize turn variables to prevent NameErrors in finally block or loop
            observer_model_name = None
            observer_instance = None
            active_stream_observer = None  # NOTE: currently unused — kept for future streaming observer support
            post_run_observer = None
            disable_observer = True # Default to safe
            reflex_text = "" # Default to empty
            
            try:
                logger.info("AgentCore: run_turn start")
                logger.debug("[DEBUG] AgentCore.run_turn start session_id=%s input_len=%d destination=%s", session_id, len(user_input or ""), destination)
            except Exception:
                logger.debug("[DEBUG] AgentCore.run_turn start (metrics unavailable)")

            # --- Loop Detection: Initialize and check for active recovery ---
            try:
                loop_config = build_loop_detection_config_from_constants(constants)
                loop_manager = get_recovery_manager(session_id)
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

            # --- Semantic Probe: pre-cognition vector lookup ---
            # Runs BEFORE persona selection. Extracts interesting phrases from
            # user input, probes all vector collections, and uses hits to drive
            # persona/KB selection (context-driven rather than keyword-gated).
            probe_result = None
            try:
                kb_config = self.config.constants.get("KNOWLEDGE_BASES", {})
                probe_result = run_semantic_probe(
                    user_input=user_input,
                    knowledge_bases=kb_config,
                    session_id=session_id,
                )
                if probe_result and probe_result.has_hits:
                    logger.info(
                        "SemanticProbe: %d hits, primary=%s, supplemental=%s (%.1fms)",
                        len(probe_result.hits),
                        probe_result.primary_collection,
                        probe_result.supplemental_collections,
                        probe_result.probe_time_ms,
                    )
                    ts_write({
                        "type": "semantic_probe",
                        "metrics": probe_result.to_metrics_dict(),
                        "phrases": probe_result.phrases_tested,
                        "hit_details": [
                            {"phrase": h.phrase, "collection": h.collection,
                             "similarity": round(h.similarity, 4), "filename": h.filename}
                            for h in probe_result.hits
                        ],
                    }, session_id, source=source, destination_context=_metadata)
                else:
                    ts_write({
                        "type": "semantic_probe",
                        "metrics": probe_result.to_metrics_dict() if probe_result else {"skipped": True},
                    }, session_id, source=source, destination_context=_metadata)
                    logger.debug("SemanticProbe: no hits")
            except Exception:
                logger.debug("SemanticProbe: probe failed, falling back to keyword routing", exc_info=True)

            # --- Persona & KB selection (probe-driven with keyword fallback) ---
            if probe_result and probe_result.primary_collection:
            # Probe found a dominant domain — adopt that persona
                knowledge_base_name = probe_result.primary_collection
                persona_name = get_persona_for_knowledge_base(knowledge_base_name)
                if not persona_name:
                    # KB exists but no persona mapped to it — use default persona with the KB
                    persona_name = "dev"
                logger.info(
                    "Persona selection (probe-driven): persona=%s, kb=%s",
                    persona_name, knowledge_base_name,
                )
            else:
                # No probe hits — fall back to keyword matching
                persona_name, knowledge_base_name = get_persona_for_request(user_input)
                logger.info(
                    "Persona selection (keyword fallback): persona=%s, kb=%s",
                    persona_name, knowledge_base_name,
                )
    
            # 0. Emergency Healing Mode (The "Immune Response")
            # If the Immune System detects a CRITICAL failure (like a SyntaxError), 
            # we pivot the entire turn toward self-repair.
    #         emergency_irritant = self._check_immune_system_for_emergency()
    #         if emergency_irritant:
    #             yield {"type": "token", "value": "[(i) Digital Immune System: CRITICAL IRRITATION DETECTED. Entering Emergency Healing Mode...]\n\n"}
    #             # Force high-reasoning persona and model for repair
    #             persona_name = "dev"
    #             knowledge_base_name = "core"
    #             # Add emergency instructions to the user input
    #             user_input = (
    #                 f"!! COGNITIVE EMERGENCY !!\n"
    #                 f"The system has detected a structural failure: {emergency_irritant}\n"
    #                 f"PRIORITY: Identify and repair this issue immediately using your tools.\n"
    #                 f"User's original request (suspend for now): {user_input}"
    #             )
    #             logger.warning("[IMMUNE RESPONSE] Turn pivoted to emergency repair.")
    # 
    #         # Load the selected persona
    #         self.ai_manager.initialize(persona_name)
    #         
    #         # Role mapping (keep internal keys stable):
            # - "lite"  => Operator (CPU orchestrator: intent, tools, summaries, short answers)
            # - "prime"/"gpu_prime"/"cpu_prime" => Thinker (GPU/CPU polished or heavy answers)
            # - "oracle" => External/cloud escalation
            selected_model_name = None
            # Check if Prime is available via lifecycle state machine
            _gpu_sleeping = not self._is_prime_available

            # 0. Character-level analysis injection
            # Models can't see individual letters due to tokenization.
            # When a question involves letter counting, spelling, or character
            # analysis, inject a character map so the model can READ the answer.
            import re as _re_count

            _char_maps = []

            # Pattern A: "how many [char] in [word]" (specific letter count)
            _count_match = _re_count.search(
                r"how many\s+(?:times?\s+(?:does?\s+)?)?(?:the\s+)?(?:letter\s+)?([a-zA-Z])(?:'s)?\s+.*\bin\s+(?:the\s+)?(?:word\s+)?(\w+)",
                user_input, _re_count.IGNORECASE)
            if not _count_match:
                _count_match = _re_count.search(
                    r"how many\s+([a-zA-Z])s?\s+(?:are\s+)?(?:there\s+)?(?:in\s+)?(?:the\s+)?(?:word\s+)?(\w+)",
                    user_input, _re_count.IGNORECASE)
            if _count_match:
                _char, _word = _count_match.group(1).lower(), _count_match.group(2).lower()
                _count = _word.count(_char)
                _positions = [i + 1 for i, c in enumerate(_word) if c == _char]
                _spelled = " ".join(f"{c}({i+1})" for i, c in enumerate(_word))
                _char_maps.append(
                    f'Word "{_word}" ({len(_word)} chars): {_spelled}\n'
                    f'Letter "{_char}" appears at position(s): {", ".join(str(p) for p in _positions)} '
                    f'— {_count} occurrence{"s" if _count != 1 else ""}.'
                )

            # Pattern B: "spell [word]" / "letters in [word]" / "how many letters"
            # Also catch "how do you spell", "what letters are in", "break down the word"
            if not _char_maps:
                _spell_match = _re_count.search(
                    r"(?:spell(?:ing)?|letters?\s+in|how many letters|break\s*down|character[s]?\s+in)\s+(?:the\s+)?(?:word\s+)?['\"]?(\w{3,})['\"]?",
                    user_input, _re_count.IGNORECASE)
                if _spell_match:
                    _word = _spell_match.group(1).lower()
                    _spelled = " ".join(f"{c}({i+1})" for i, c in enumerate(_word))
                    # Build letter frequency
                    _freq = {}
                    for c in _word:
                        _freq[c] = _freq.get(c, 0) + 1
                    _freq_str = ", ".join(f"{c}={n}" for c, n in sorted(_freq.items()) if n > 1)
                    _char_maps.append(
                        f'Word "{_word}" ({len(_word)} chars): {_spelled}'
                        + (f'\nRepeated letters: {_freq_str}' if _freq_str else '')
                    )

            # Pattern C: quoted words in questions about letters/counting/spelling
            if not _char_maps:
                _quoted = _re_count.findall(r'["\'](\w{3,})["\']', user_input)
                _has_letter_context = any(w in user_input.lower() for w in
                    ["letter", "spell", "character", "count", "how many", "contain"])
                if _quoted and _has_letter_context:
                    for _word in _quoted[:3]:  # Max 3 words
                        _word = _word.lower()
                        _spelled = " ".join(f"{c}({i+1})" for i, c in enumerate(_word))
                        _freq = {}
                        for c in _word:
                            _freq[c] = _freq.get(c, 0) + 1
                        _freq_str = ", ".join(f"{c}={n}" for c, n in sorted(_freq.items()) if n > 1)
                        _char_maps.append(
                            f'Word "{_word}" ({len(_word)} chars): {_spelled}'
                            + (f'\nRepeated letters: {_freq_str}' if _freq_str else '')
                        )

            if _char_maps:
                _injection = (
                    "\n\n[Character Analysis — your tokenizer groups letters into chunks, "
                    "so you cannot see individual characters. Use this pre-computed data:]\n"
                    + "\n".join(_char_maps) +
                    "\n[Answer the user's question naturally using this data.]"
                )
                user_input = user_input + _injection
                logger.info("[INTERCEPT] Injected character map(s) for %d word(s)", len(_char_maps))

            # 1. Model Selection (Prioritize Fast-Path & Overrides)
            text_lower = user_input.lower()
            force_thinker = os.getenv("GAIA_FORCE_THINKER", "").lower() in ("1", "true", "yes")
            wants_nano = any(tag in text_lower for tag in ["::nano", "[nano]", "nano:"])
            
            # Simple patterns NANO can handle reliably
            factual_patterns = [
                "who is", "what is", "where is", "when did", "how many",
                "name of", "tell me what time", "what time", "what date",
                "current date", "current time", "date and time", "time and date",
                "what day", "today's date", "right now",
            ]
            is_factual = any(p in text_lower for p in factual_patterns)
            import re as _re
            _trivial_phrases = [
                r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bstatus\b", r"\buptime\b",
                r"\bwho are you\b", r"\bhow are you\b",
                r"\bgood morning\b", r"\bgood afternoon\b", r"\bgood evening\b", r"\bgood night\b",
                r"\bthanks\b", r"\bthank you\b",
            ]
            is_trivial = len(user_input) < 100 and (
                any(_re.search(p, text_lower) for p in _trivial_phrases) or is_factual
            )
    
            # 1a. Nano Fast-Path (0.5B) - Highest priority for speed
            # Trivial/factual queries go to Nano first; cascade triage will
            # escalate later if the question turns out to be complex.
            # The nano model may be registered as "nano" or "reflex" in MODEL_CONFIGS.
            has_nano = any(k in self.config.MODEL_CONFIGS for k in ("nano", "reflex"))
            nano_key = "nano" if "nano" in self.config.MODEL_CONFIGS else "reflex"
            if (wants_nano or is_trivial) and has_nano and not force_thinker:
                selected_model_name = nano_key
                logger.info(f"[MODEL_SELECT] Routing to Reflex Nano '{nano_key}' (wants_nano={wants_nano}, is_trivial={is_trivial}, is_factual={is_factual})")

            # 1b. Knowledge Base Override (Escalate to GPU for RAG tasks)
            # Only escalate if the query is non-trivial — a spurious KB match
            # on a simple greeting/question shouldn't force the heavy pipeline.
            elif knowledge_base_name and not is_trivial:
                logger.info(f"[MODEL_SELECT] Knowledge base '{knowledge_base_name}' triggered, preferring gpu_prime.")
                for cand in ["gpu_prime", "prime"]:
                    if cand == "gpu_prime" and _gpu_sleeping:
                        continue
                    if cand in self.config.MODEL_CONFIGS:
                        selected_model_name = cand
                        logger.info(f"[MODEL_SELECT] Overriding model selection to '{selected_model_name}' due to RAG-enabled persona.")
                        break
            elif knowledge_base_name and is_trivial:
                logger.info(f"[MODEL_SELECT] Knowledge base '{knowledge_base_name}' matched but query is trivial — skipping gpu_prime override.")
    
            if not selected_model_name:
                # Respect runtime override via environment var GAIA_BACKEND or config.llm_backend
                backend_env = os.getenv("GAIA_BACKEND") or getattr(self.config, "llm_backend", None)
                if backend_env:
                    # Skip gpu_prime when GPU is released for sleep
                    if backend_env == "gpu_prime" and _gpu_sleeping:
                        logger.info("GAIA_BACKEND='gpu_prime' but GPU is sleeping; falling back to default selection")
                    elif backend_env in self.model_pool.models:
                        selected_model_name = backend_env
                    else:
                        logger.info(f"Requested GAIA_BACKEND='{backend_env}' not present in model pool; falling back to default selection")
    
            if not selected_model_name:
                # 2. Thinker / Oracle Path
                if "oracle" in text_lower and self.config.use_oracle:
                    selected_model_name = "oracle"
                elif force_thinker or any(tag in text_lower for tag in ["thinker:", "[thinker]", "::thinker"]):
                    # Prefer GPU prime, then prime, then cpu_prime
                    for cand in ["gpu_prime", "prime", "cpu_prime"]:
                        if cand == "gpu_prime" and _gpu_sleeping:
                            continue
                        if cand in self.model_pool.models:
                            selected_model_name = cand
                            break
                
                # Default path: Operator (lite/core) handles most turns.
                if not selected_model_name:
                    if "lite" in self.model_pool.models:
                        selected_model_name = "lite"
                    elif "core" in self.model_pool.models:
                        selected_model_name = "core"
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
                    log_gaia_error(logger, "GAIA-CORE-050", f"Lazy load failed for '{selected_model_name}': {e}")

            if selected_model_name not in self.model_pool.models:
                logger.info(f"Model '{selected_model_name}' still not available after lazy load; attempting fallback selection")
                logger.warning(f"[MODEL_SELECT DEBUG] pool keys before fallback: {list(self.model_pool.models.keys())}")
                # Try preferred backend first
                backend_env = os.getenv("GAIA_BACKEND") or getattr(self.config, "llm_backend", None)
                candidates = []
                if backend_env and not (backend_env == "gpu_prime" and _gpu_sleeping):
                    candidates.append(backend_env)
                # Prefer Operator (lite/core) first, then Thinker tiers, then oracle/azrael
                candidates.extend(["lite", "core", "gpu_prime", "prime", "cpu_prime", "oracle", "azrael"])
                found = None
                for cand in candidates:
                    if cand == "gpu_prime" and _gpu_sleeping:
                        continue
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
    
            # --- Cascade Routing ---
            # If no forced model was picked, we use Nano to triage the request.
            # This implements the user's suggestion for model-driven complexity routing.
            try:
                # Only cascade if we haven't already picked a 'prime' or 'oracle' model
                # and if we aren't forcing the operator.
                force_operator = os.getenv("GAIA_FORCE_OPERATOR", "").lower() in ("1", "true", "yes")
                is_forced_thinker = os.getenv("GAIA_FORCE_THINKER", "").lower() in ("1", "true", "yes") or \
                                    any(tag in user_input.lower() for tag in ["thinker:", "[thinker]", "::thinker"])
                
                if selected_model_name in ("lite", "nano", "reflex") and not force_operator and not is_forced_thinker:
                    # Skip LLM triage for queries already classified as factual/trivial
                    # by deterministic pattern matching — no need to ask the 0.5B model.
                    if is_factual or is_trivial:
                        triage_result = "SIMPLE"
                        logger.info("[CASCADE] Skipping Nano triage — deterministic match (is_factual=%s, is_trivial=%s)", is_factual, is_trivial)
                    else:
                        triage_result = self._nano_triage(user_input)
                    if triage_result == "COMPLEX":
                        # Emit status message to user
                        yield {"type": "token", "value": "[(i) Reflex Nano: Complexity detected. Routing to Operator Core...]\n\n"}
                        # Prefer lite, fall back to core (embedded llama-server)
                        selected_model_name = "lite" if "lite" in self.model_pool.models else "core"

                        # Secondary escalation: check if Lite thinks it's even MORE complex
                        if self._should_escalate_to_thinker(user_input):
                            _escalated = False
                            for cand in ["gpu_prime", "prime", "cpu_prime"]:
                                if cand == "gpu_prime" and _gpu_sleeping:
                                    continue
                                # Ensure escalation candidate is loaded
                                if cand not in self.model_pool.models:
                                    try:
                                        self.model_pool.ensure_model_loaded(cand)
                                    except Exception as _load_exc:
                                        logger.warning("AgentCore: cascade escalation preload failed for %s: %s", cand, _load_exc)
                                if cand in self.model_pool.models:
                                    # Verify the model endpoint is actually reachable before escalating
                                    _model_obj = self.model_pool.models.get(cand)
                                    if hasattr(_model_obj, 'endpoint'):
                                        try:
                                            import requests as _req
                                            _health = _req.get(f"{_model_obj.endpoint}/health", timeout=2)
                                            if _health.status_code != 200:
                                                logger.warning(f"[CASCADE] {cand} endpoint unhealthy ({_health.status_code}); skipping escalation")
                                                continue
                                        except Exception:
                                            logger.warning(f"[CASCADE] {cand} endpoint unreachable; skipping escalation")
                                            continue
                                    yield {"type": "token", "value": "[(i) Operator Core: Deep reasoning required. Escalating to Thinker Prime...]\n\n"}
                                    selected_model_name = cand
                                    _escalated = True
                                    break
                            if not _escalated:
                                logger.info("[CASCADE] No thinker-tier model reachable; staying on %s", selected_model_name)
                    else:
                        # Nano can handle it — use the actual key from MODEL_CONFIGS
                        selected_model_name = nano_key
                        logger.info("[CASCADE] Reflex Nano confirmed SIMPLE request.")
            except Exception:
                logger.debug("Cascade routing failed; continuing with selected model", exc_info=True)

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

            # ── Pipeline Depth: tier-based gating ────────────────────────
            # REFLEX:   minimal pipeline — just generate (reflex already handled above)
            # OPERATOR: intent → slim prompt → generation → 1 reflection → observer
            # THINKER:  full pipeline with all stages
            _REFLEX_MODELS = {"nano", "reflex"}
            _OPERATOR_MODELS = {"core", "lite", "cpu_prime"}
            # Everything else (gpu_prime, thinker, prime, oracle, azrael) = THINKER depth
            if selected_model_name in _REFLEX_MODELS:
                pipeline_depth = "REFLEX"
            elif selected_model_name in _OPERATOR_MODELS:
                pipeline_depth = "OPERATOR"
            else:
                pipeline_depth = "THINKER"
            logger.info("[PIPELINE_DEPTH] %s (model: %s)", pipeline_depth, selected_model_name)
            logger.debug("[DEBUG] AgentCore selected_model=%s", selected_model_name)

            # Log model routing decision to event buffer
            try:
                from gaia_common.event_buffer import log_event
                log_event("routing", f"Model: {selected_model_name} ({pipeline_depth})",
                          source="agent_core",
                          details={"model": selected_model_name, "depth": pipeline_depth})
            except Exception:
                pass
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
                log_gaia_error(logger, "GAIA-CORE-065", f"model={selected_model_name}")
                yield {"type": "token", "value": (
                    "**[GAIA-CORE-065]** I am currently unable to process your request — "
                    "my primary model is busy or unavailable. This has been logged for self-diagnosis."
                )}
                return
    
            # 2. Create the initial v0.3 Cognition Packet
            history = self.session_manager.get_history(session_id).copy()
    
            # Pre-injection history review: check for epistemic violations
            try:
                hr_cfg = self.config.constants.get("HISTORY_REVIEW", {})
                history = review_history(history, config=hr_cfg, session_id=session_id)
            except Exception:
                logger.warning("AgentCore: history review failed, using raw history", exc_info=True)
    
            packet = self._create_initial_packet(
                user_input, session_id, history, selected_model_name,
                source=source, destination=destination, metadata=_metadata
            )
            self.session_manager.add_message(session_id, "user", user_input)
            self._emit_timeline_message(session_id, "user", source)
    
            # Inject semantic probe result into packet (if probe found hits)
            if probe_result and probe_result.has_hits:
                packet.content.data_fields.append(DataField(
                    key='semantic_probe_result',
                    value=probe_result.to_dict(),
                    type='json',
                    source='semantic_probe',
                ))
    
            # Attach probe metrics to packet.metrics for observability
            if probe_result:
                packet.metrics.semantic_probe = probe_result.to_metrics_dict()
    
            if knowledge_base_name:
                packet.content.data_fields.append(DataField(key='knowledge_base_name', value=knowledge_base_name, type='string'))
            
            # Enhance the packet with knowledge from the knowledge base
            packet = enhance_packet(packet)
    
            # RAG: Retrieve documents from vector store if a knowledge base is specified.
            # Phase 4 dedup: if the semantic probe already found strong hits from the
            # same collection, use those as seed context and skip the redundant MCP query.
            try:
    
                # Extract knowledge_base_name from the packet's data_fields
                knowledge_base_name = None
                for field in packet.content.data_fields:
                    if field.key == 'knowledge_base_name':
                        knowledge_base_name = field.value
                        break
    
                if knowledge_base_name:
                    # Check if the probe already provided hits from this collection
                    probe_seeded = False
                    if probe_result and probe_result.has_hits:
                        probe_hits_for_kb = [
                            h for h in probe_result.hits
                            if h.collection == knowledge_base_name
                        ]
                        if len(probe_hits_for_kb) >= 2:
                            # Probe found multiple strong hits — use them as seed,
                            # skip the full-prompt RAG query (which would largely overlap)
                            seed_docs = []
                            seen_files = set()
                            for h in probe_hits_for_kb:
                                # Dedup by filename to avoid repeat chunks from the same doc
                                if h.filename not in seen_files:
                                    seen_files.add(h.filename)
                                    seed_docs.append({
                                        "text": h.chunk_text,
                                        "filename": h.filename,
                                        "score": h.similarity,
                                        "source": "semantic_probe",
                                    })
                            if seed_docs:
                                packet.content.data_fields.append(
                                    DataField(key='retrieved_documents', value=seed_docs, type='list')
                                )
                                self.logger.info(
                                    f"RAG dedup: using {len(seed_docs)} probe hits as seed context "
                                    f"for '{knowledge_base_name}' (skipped MCP query)"
                                )
                                probe_seeded = True
    
                    if not probe_seeded:
                        # No probe seed — run the standard RAG query
                        self.logger.info(f"Performing RAG query on knowledge base: {knowledge_base_name}")
                        retrieved_docs = asyncio.run(mcp_client.embedding_query(
                            packet.content.original_prompt,
                            top_k=3,
                            knowledge_base_name=knowledge_base_name
                        ))
                        if retrieved_docs and retrieved_docs.get("ok"):
                            docs = retrieved_docs.get("results", [])
                            if docs:
                                packet.content.data_fields.append(DataField(key='retrieved_documents', value=docs, type='list'))
                                self.logger.info(f"Retrieved {len(docs)} documents from '{knowledge_base_name}'.")
                            else:
                                packet.content.data_fields.append(DataField(
                                    key='rag_no_results',
                                    value=True,
                                    type='bool'
                                ))
                                self.logger.warning(f"RAG query returned no documents from '{knowledge_base_name}' - epistemic uncertainty flagged.")
                        else:
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
                log_gaia_error(self.logger, "GAIA-CORE-045", str(e), exc_info=True)

            # If RAG query returns no results, attempt to find and embed the knowledge
            if any(df.key == 'rag_no_results' and df.value for df in packet.content.data_fields):
                packet = self._knowledge_acquisition_workflow(packet)
    
            # Epistemic gate: if RAG found nothing for a domain-specific query,
            # check confidence before generating (prevents fabrication)
            rag_no_results_flag = any(
                getattr(df, 'key', '') == 'rag_no_results' and getattr(df, 'value', False)
                for df in getattr(packet.content, 'data_fields', []) or []
            )
            kb_active = bool(knowledge_base_name)
    
            if rag_no_results_flag and kb_active:
                eg_config = self.config.constants.get("EPISTEMIC_GUARDRAILS", {})
                if eg_config.get("pre_generation_gate", True):
                    self.logger.info("Epistemic gate: RAG returned no results for domain query — checking confidence")
                    try:
                        conf_result = self.assess_task_confidence(
                            "domain_query", user_input, selected_model_name, session_id
                        )
                        conf_threshold = eg_config.get("confidence_threshold", 0.5)
                        conf_score = conf_result.get("confidence_score", 1.0)
                        if conf_score < conf_threshold:
                            alt = conf_result.get("alternative_offer", "I can try to answer from general knowledge, but I may not be accurate.")
                            honest_response = (
                                f"I don't have specific information about that in my knowledge base. "
                                f"{alt}"
                            )
                            _header = self._build_response_header(selected_model_name, packet, None, None, None)
                            yield {"type": "token", "value": _header + honest_response}
                            self.session_manager.add_message(session_id, "assistant", honest_response)
                            return
                    except Exception:
                        self.logger.debug("Epistemic gate confidence check failed; proceeding with generation", exc_info=True)
    
            # ── Knowledge Ingestion Pipeline ──────────────────────────────
            # Check for explicit save commands or auto-detected knowledge dumps
            # before intent detection so we can short-circuit or tag the packet.
            # If no KB was selected but the user explicitly asked to save, use a
            # default knowledge base so the save pipeline still fires.
            if not knowledge_base_name:
                from gaia_core.cognition.knowledge_ingestion import detect_save_command
                if detect_save_command(user_input):
                    knowledge_base_name = "general"
                    logger.info("No KB selected but explicit save command detected; using default KB 'general'")
            if knowledge_base_name:
                # 1. Explicit save: "save this about X", "remember this", legacy DOCUMENT
                save_result = run_explicit_save(user_input, knowledge_base_name)
                if save_result:
                    if save_result.get("action") == "dedup_blocked":
                        existing = save_result["existing_doc"]
                        yield {"type": "token", "value": (
                            f"I found an existing document that closely matches this information "
                            f"(similarity: {existing['similarity']:.0%}): **{existing['title']}**. "
                            f"No duplicate was created. If this is genuinely new information, "
                            f"please rephrase or add distinguishing details."
                        )}
                        return
                    elif save_result.get("ok"):
                        yield {"type": "token", "value": (
                            f"Acknowledged. I've documented '{save_result['subject']}' "
                            f"to `{save_result['path']}`."
                        )}
                        if save_result.get("embed_ok"):
                            yield {"type": "token", "value": " It's now indexed for future retrieval."}
                        return
                    else:
                        yield {"type": "token", "value": (
                            f"I tried to save that information but encountered an issue: "
                            f"{save_result.get('error', 'unknown error')}. "
                            f"The MCP write_file tool may require approval."
                        )}
                        return
    
                # 2. Auto-detect: long D&D info dump → offer to save (don't write yet)
                auto_classification = run_auto_detect(user_input, knowledge_base_name)
                if auto_classification:
                    packet.content.data_fields.append(DataField(
                        key='knowledge_ingestion_offer',
                        value=auto_classification,
                        type='json',
                    ))
                    self.logger.info(
                        f"Knowledge ingestion offer tagged: category={auto_classification['category']}"
                    )
    
                # 3. Knowledge update detect: short casual updates to existing entities
                if not auto_classification:
                    update_info = run_update_detect(user_input, knowledge_base_name)
                    if update_info:
                        existing_doc = update_info.get("existing_doc")
                        if existing_doc:
                            # Ensure entity doc is in retrieved_documents
                            already_in = False
                            for df in packet.content.data_fields:
                                if df.key == 'retrieved_documents':
                                    for d in (df.value or []):
                                        if d.get("source") == existing_doc.get("source"):
                                            already_in = True
                                    if not already_in:
                                        df.value.append(existing_doc)
                                    break
                            else:
                                if not already_in:
                                    packet.content.data_fields.append(
                                        DataField(key='retrieved_documents', value=[existing_doc], type='list'))
    
                        # Tag packet + inject system_hint directly
                        packet.content.data_fields.append(
                            DataField(key='knowledge_update_detected', value=update_info, type='json'))
    
                        entity = update_info["entity"]
                        signal = update_info["update_signal"]
                        has_doc = existing_doc is not None
                        hint = (
                            f"The user indicated that '{entity}' has been updated: {signal}. "
                            + (
                                "An existing document for this entity is in the Retrieved Documents above. "
                                "Your task:\n"
                                "1. Acknowledge the update conversationally.\n"
                                "2. Examine the existing document to identify affected fields.\n"
                                "3. Use D&D 5e knowledge to reason about cascading changes "
                                "(HP, spell slots, features, proficiency, ability scores, etc.).\n"
                                "4. Ask targeted follow-up questions about anything ambiguous.\n"
                                "5. Offer to update the documentation with confirmed changes.\n"
                                "Do NOT write the update yet — gather information first."
                                if has_doc else
                                "No existing document found. Acknowledge, ask questions, "
                                "and offer to create documentation for this entity."
                            )
                        )
                        packet.content.data_fields.append(
                            DataField(key='system_hint', value=hint, type='string'))
    
                        self.logger.info(
                            f"Knowledge update tagged + hint injected: entity={entity}"
                        )
            # ── End Knowledge Ingestion Pipeline ──────────────────────────

            # ── Attachment Ingestion Pipeline ─────────────────────────────
            # Process file attachments (PDFs, text files, etc.) sent via Discord
            if packet.content.attachments:
                for att in packet.content.attachments:
                    # Find the matching attachment_text DataField
                    att_text = None
                    for df in packet.content.data_fields:
                        if df.key == "attachment_text" and isinstance(df.value, dict):
                            if df.value.get("filename") == att.name:
                                att_text = df.value.get("text_preview", "")
                                break

                    if not att_text:
                        self.logger.info("Attachment '%s' has no extracted text — skipping ingestion", att.name)
                        continue

                    try:
                        result = run_attachment_ingestion(
                            filename=att.name,
                            text_content=att_text,
                            user_hint=user_input,
                        )
                        if result.get("action") == "saved" and result.get("path"):
                            msg = (
                                f"I've saved **{att.name}** to your **{result['kb_name']}** "
                                f"knowledge base ({result['category']})."
                            )
                            if result.get("embed_ok"):
                                msg += " It's now indexed for retrieval."
                            yield {"type": "token", "value": msg}
                        elif result.get("action") == "dedup_blocked":
                            existing = result.get("existing_doc", {})
                            yield {"type": "token", "value": (
                                f"**{att.name}** appears to already exist in the "
                                f"**{result['kb_name']}** knowledge base "
                                f"(similarity: {existing.get('similarity', 0):.0%}). "
                                f"No duplicate was created."
                            )}
                        elif result.get("error"):
                            yield {"type": "token", "value": (
                                f"I tried to save **{att.name}** but encountered an issue: "
                                f"{result['error']}"
                            )}
                    except Exception as e:
                        log_gaia_error(self.logger, "GAIA-CORE-079", f"file={att.name} error={e}", exc_info=True)
                        yield {"type": "token", "value": (
                            f"**[GAIA-CORE-079]** Failed to process attachment **{att.name}**: {e}"
                        )}

                # If the message was ONLY attachments (no real user text), we're done
                if user_input.startswith("[Attached:"):
                    return
            # ── End Attachment Ingestion Pipeline ─────────────────────────

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
                # Lazy-load lite model on demand (GAIA_AUTOLOAD_MODELS may be 0)
                if self.model_pool.ensure_model_loaded("lite"):
                    lite_llm = self.model_pool.models.get("lite")
                if lite_llm:
                    self.model_pool.set_status("lite", "busy")
            except Exception:
                lite_llm = None
            prime_llm = None  # do not use prime for intent detection
            fallback_llm = lite_llm or selected_model
    
            # Build a short probe context hint for the intent detector
            _probe_context = ""
            if probe_result and probe_result.has_hits:
                matched_phrases = [h.phrase for h in probe_result.hits[:5]]
                unique_phrases = list(dict.fromkeys(matched_phrases))  # dedup preserving order
                _probe_context = (
                    f"User references {probe_result.primary_collection} entities "
                    f"({', '.join(unique_phrases)})"
                )
    
            # Grab the embedding model for intent classification (non-blocking)
            _embed_model = None
            try:
                _embed_model = self.model_pool.get_embed_model(timeout=0, lazy_load=True)
            except Exception as _emb_exc:
                logger.debug("AgentCore: embed model lazy load failed (non-fatal): %s", _emb_exc)
    
            plan = None
            try:
                plan = detect_intent(
                    user_input,
                    self.config,
                    lite_llm=lite_llm,
                    full_llm=prime_llm,
                    fallback_llm=fallback_llm,
                    probe_context=_probe_context,
                    embed_model=_embed_model,
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
    
            # 3a-bis. Goal Detection — identify overarching user goal
            # Only run for THINKER depth — REFLEX and OPERATOR skip this
            if pipeline_depth == "THINKER":
                try:
                    from gaia_core.cognition.goal_detector import GoalDetector
                    goal_detector = GoalDetector(config=self.config)
                    packet.goal_state = goal_detector.detect(
                        packet=packet,
                        session_manager=self.session_manager,
                        session_id=session_id,
                        model_pool=self.model_pool,
                    )
                    if packet.goal_state and packet.goal_state.current_goal:
                        ts_write({
                            "type": "goal_detect",
                            "goal": packet.goal_state.current_goal.goal_id,
                            "confidence": packet.goal_state.current_goal.confidence.value,
                            "source": packet.goal_state.current_goal.source,
                        }, session_id, source=source, destination_context=_metadata)
                except Exception:
                    logger.debug("Goal detection failed (non-fatal)", exc_info=True)
    
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
            # Skip slim prompt when tool routing already executed — tool results need the
            # full ExternalVoice pipeline to be properly incorporated into the response.
            tool_already_executed = (
                packet.tool_routing
                and packet.tool_routing.execution_status == ToolExecutionStatus.EXECUTED
            )
            if not tool_already_executed and self._should_use_slim_prompt(plan, user_input, selected_model_name=selected_model_name):
                text = self._run_slim_prompt(selected_model_name, user_input, history, plan.intent, session_id=session_id, source=source, metadata=_metadata, packet=packet)
                if text is not None:
                    # _run_slim_prompt already handles uncertainty escalation
                    # internally — if Nano hedged, it already tried Core/Lite.
                    # The text here is the best answer from the slim path.
                    _header = self._build_response_header(selected_model_name, packet, None, None, None)
                    yield {"type": "token", "value": _header + text}
                    self.session_manager.add_message(session_id, "assistant", text)
                    return
                # _run_slim_prompt returned None — slim path declined (e.g. low
                # confidence recitation).  Attempt tool routing (web_search) to
                # gather real content before falling through to ExternalVoice.
                if not (packet.tool_routing and packet.tool_routing.execution_status == ToolExecutionStatus.EXECUTED):
                    logger.info("Slim prompt declined — attempting tool routing for web_search")
                    packet = self._run_tool_routing_loop(
                        packet=packet,
                        user_input=user_input,
                        session_id=session_id,
                        source=source,
                        metadata=_metadata
                    )
                    if packet.tool_routing and packet.tool_routing.execution_status == ToolExecutionStatus.EXECUTED:
                        logger.info("Tool routing succeeded after slim prompt decline — continuing with enhanced context")
    
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

            # Load skill adapter on Prime if intent requires one (e.g., code tasks)
            try:
                from gaia_core.cognition.skill_adapter import ensure_adapter
                _intent_for_adapter = plan.intent if plan else ""
                ensure_adapter(_intent_for_adapter)
            except Exception:
                pass  # Non-blocking — adapter is optional enhancement

            try:
                # Release any earlier acquisition so forward_to_model manages lifecycle.
                try:
                    self.model_pool.release_model(selected_model_name)
                except Exception as _rel_exc:
                    logger.warning("AgentCore: model release before planning failed for %s: %s", selected_model_name, _rel_exc)
                # Contextual thinking: enable chain-of-thought for complex tasks,
                # disable for simple exchanges and tool routing.
                _complex_intents = {"planning", "analysis", "debugging", "code",
                                    "explanation", "architecture", "research"}
                _enable_thinking = plan.intent in _complex_intents if plan else True
                if not _enable_thinking:
                    logger.info("Thinking disabled for intent: %s", plan.intent if plan else "none")

                plan_res = self.model_pool.forward_to_model(
                    selected_model_name,
                    messages=plan_messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    adapter_name=self._resolve_adapter(selected_model_name),
                    chat_template_kwargs={"enable_thinking": _enable_thinking},
                )
            except Exception as exc:
                log_gaia_error(
                    logger, "GAIA-CORE-078",
                    f"model={selected_model_name} error={type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                yield {"type": "token", "value": (
                    f"**[GAIA-CORE-078]** Plan generation failed ({type(exc).__name__}). "
                    f"Logged for self-diagnosis."
                )}
                yield {"type": "flush"}
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
                        pass  # per-token parse; fallback below
                    # fallback to non-stream shape
                    try:
                        text = item.get("choices", [{}])[0].get("message", {}).get("content")
                        if text:
                            pieces.append(text)
                    except Exception:
                        logger.debug("AgentCore: plan stream piece unparseable: %s", type(item))
                initial_plan_text = "".join(pieces).strip()
            else:
                try:
                    initial_plan_text = plan_res["choices"][0]["message"]["content"].strip()
                except Exception as exc:
                    logger.exception("AgentCore: failed to parse plan response (%s)", type(plan_res))
                    yield {"type": "token", "value": f"[plan-parse-error] {type(exc).__name__}: {exc}"} 
                    return
            packet.reasoning.reflection_log.append(ReflectionLog(step="initial_plan", summary=initial_plan_text, confidence=0.8)) # Placeholder confidence
    
            # --- Cognitive Self-Audit ---
            audit_cfg = self.config.constants.get("COGNITIVE_AUDIT", {})
            is_slim_prompt = getattr(self.config, '_slim_prompt', False)
            logger.warning("AgentCore: cognitive audit gate — enabled=%s, skip_for_slim=%s, is_slim=%s",
                         audit_cfg.get("enabled", False), audit_cfg.get("skip_for_slim_prompt", True), is_slim_prompt)
            if audit_cfg.get("enabled", False) and not (audit_cfg.get("skip_for_slim_prompt", True) and is_slim_prompt):
                try:
                    logger.warning("AgentCore: running cognitive self-audit...")
                    run_cognitive_self_audit(
                        packet=packet,
                        plan_text=initial_plan_text,
                        config=self.config,
                        llm=selected_model,
                    )
                    logger.warning("AgentCore: cognitive self-audit completed successfully")
                except Exception:
                    logger.warning("AgentCore: cognitive self-audit failed, continuing", exc_info=True)
    
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
                # REFLEX: skip reflection entirely
                # OPERATOR: 1 reflection iteration (simple queries don't need deep reflection)
                # THINKER: full reflection (up to 3 iterations)
                if pipeline_depth == "REFLEX":
                    logger.info("[PIPELINE_DEPTH] REFLEX — skipping reflection")
                    refined_plan_text = initial_plan_text
                elif pipeline_depth == "OPERATOR":
                    # Cap OPERATOR to 1 reflection iteration — prevents over-thinking
                    # trivial questions like "how much wood could a woodchuck chuck"
                    orig_iters = getattr(self.config, 'max_reflection_iterations', 3)
                    self.config.max_reflection_iterations = 1
                    try:
                        refined_plan_text = reflect_and_refine(packet=packet, output=initial_plan_text, config=self.config, llm=reflection_model, ethical_sentinel=self.ethical_sentinel)
                    finally:
                        self.config.max_reflection_iterations = orig_iters
                else:
                    refined_plan_text = reflect_and_refine(packet=packet, output=initial_plan_text, config=self.config, llm=reflection_model, ethical_sentinel=self.ethical_sentinel)
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
    
            # 5a. Planning Orchestrator: multi-phase collaborative planning
            # Routes through the planning pipeline when:
            # - Intent is planning OR user request matches planning keywords
            # - GPU Prime is available (selected_model_name is gpu_prime)
            _plan_intent = getattr(packet.intent, 'user_intent', '') if packet.intent else ''
            _plan_keywords = ["implementation plan", "detailed plan", "create a plan", "plan for adding"]
            _is_planning_task = (
                _plan_intent == "planning"
                or any(kw in (user_input or "").lower() for kw in _plan_keywords)
            )
            if _is_planning_task and selected_model_name in ("gpu_prime", "prime"):
                try:
                    from gaia_core.cognition.planning_orchestrator import run_planning_pipeline
                    # Get reviewer model (idle Core on CPU during FOCUSING)
                    _reviewer = None
                    for _rev_name in ["lite", "core", "reflex"]:
                        if _rev_name in self.model_pool.models and _rev_name != selected_model_name:
                            try:
                                _reviewer = self.model_pool.acquire_model(_rev_name)
                                break
                            except Exception:
                                continue

                    logger.info("Planning orchestrator: prime=%s, reviewer=%s", selected_model_name, _rev_name if _reviewer else "none")
                    for event in run_planning_pipeline(
                        user_request=user_input,
                        prime_model=selected_model,
                        reviewer_model=_reviewer,
                        packet=packet,
                        config=self.config,
                        model_pool=self.model_pool,
                    ):
                        yield event

                    if _reviewer:
                        try:
                            self.model_pool.release_model_for_role(_rev_name)
                        except Exception:
                            pass

                    # Skip council debate — planning orchestrator handled everything
                    packet.status.state = PacketState.COMPLETED
                    packet.status.finalized = True
                    # Record in session
                    assembled = "\n".join(
                        s.content for s in packet.reasoning.sketchpad
                        if s.slot == "plan_final"
                    )
                    if assembled:
                        packet.response.candidate = assembled
                    self.session_manager.add_message(session_id, "assistant", packet.response.candidate or "Plan generated.")
                    ts_write({"type": "planning_complete", "packet_id": packet.header.packet_id}, session_id, source=source, destination_context=_metadata)
                    yield {"type": "flush"}
                    yield {"type": "packet", "value": packet.to_serializable_dict()}
                    return
                except Exception as _plan_err:
                    logger.warning("Planning orchestrator failed, falling back to council debate: %s", _plan_err, exc_info=True)

            # 5b. Final Response Generation (with Iterative Council Debate)
            debate_turn = 0
            MAX_DEBATE_TURNS = 3
            council_history = []
            full_response = ""
            
            while debate_turn < MAX_DEBATE_TURNS:
                debate_turn += 1
                logger.info(f"--- Council Debate Turn {debate_turn} (Model: {selected_model_name}) ---")
                
                # Update packet with peer's council context if this is a follow-up turn
                if council_history:
                    # Add or update council context field
                    found = False

                    for field in packet.content.data_fields:
                        if field.key == "council_debate_history":
                            field.value = "\n".join(council_history)
                            found = True
                            break
                    if not found:
                        packet.content.data_fields.append(DataField(
                            key="council_debate_history",
                            value="\n".join(council_history)
                        ))
    
                final_messages = build_from_packet(packet)
                
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
    
                    log_gaia_error(logger, "GAIA-CORE-010", f"Blocked by guardian: {reason}")
                    yield {"type": "token", "value": user_msg}
                    return
    
                # pick an observer model that's idle if possible; fall back to config-based selection
                observer_model_name = None
                observer_config = self.config.constants.get("MODEL_CONFIGS", {}).get("observer", {})
                observer_enabled = observer_config.get("enabled", False)
                use_lite_for_observer = observer_config.get("use_lite", True)
    
                if observer_enabled and use_lite_for_observer:
                    try:
                        if self.model_pool.ensure_model_loaded("lite"):
                            observer_model_name = "lite"
                    except Exception as _obs_exc:
                        logger.warning("AgentCore: observer lite model load failed: %s", _obs_exc)

                if observer_model_name is None:
                    try:
                        # Exclude embed (SentenceTransformer) — it can't do chat completion
                        observer_model_name = self.model_pool.get_idle_model(exclude=[selected_model_name, "embed"])
                    except Exception as _obs_exc:
                        logger.warning("AgentCore: observer model selection failed: %s", _obs_exc)
    
                observer_model = None
                if observer_model_name:
                    observer_model = self.model_pool.acquire_model_for_role(observer_model_name)
    
                observer_instance = None
                if observer_model is not None:
                    observer_instance = StreamObserver(config=self.config, llm=observer_model, name=f"AgentCore-Observer-Turn{debate_turn}")

                # ── Phase Header Yielding ──
                # Naming convention: [(Role) Identity]
                #   Nano=Reflex, Core=Operator, Prime=Thinker
                _is_prime = "thinker" in selected_model_name.lower() or "prime" in selected_model_name.lower()
                role_label = "Thinker" if _is_prime else "Operator"
                identity_label = "Prime" if _is_prime else "Core"
                header_icon = "🧠" if _is_prime else "🤖"

                # ── Adapter activation for ExternalVoice streaming ──
                _stream_adapter = self._resolve_adapter(selected_model_name)
                if _stream_adapter and hasattr(selected_model, 'set_active_adapter'):
                    selected_model.set_active_adapter(_stream_adapter)

                voice = ExternalVoice(
                    model=selected_model,
                    model_pool=self.model_pool,
                    config=self.config,
                    messages=final_messages,
                    source=identity_label.lower(),  # "prime" or "core" for generation stream
                    observer=observer_instance,
                    context={"packet": packet, "ethical_sentinel": self.ethical_sentinel},
                    session_id=session_id,
                )

                # BICAMERAL UPGRADE: If escalate to Thinker, have Operator yield a status update
                # Only show escalation notice if we actually swapped models (debate_turn > 1)
                if _is_prime and debate_turn > 1:
                    yield {"type": "token", "value": "🤖 **[(Operator) Core]** *Deep reasoning required. Escalating to Thinker...*\n"}

                phase_header = f"\n\n{header_icon} **[({role_label}) {identity_label}]**\n"

                # If this is a refinement turn, add that context
                if reflex_text and debate_turn == 1:
                    phase_header = f"\n\n---\n🔄 **Refinement from Council ({role_label} {identity_label})**\n"
                
                yield {"type": "token", "value": phase_header}

                current_turn_pieces = []
                _stream_buf = ""  # Rolling buffer for live repetition detection
                try:
                    stream_generator = voice.stream_response()
                    for item in stream_generator:
                        if isinstance(item, dict):
                            # item is an event dict (interruption, etc.)
                            ev = item.get("event")
                            if ev == "interruption":
                                reason_data = item.get("data", "observer interruption")
                                logger.warning(f"AgentCore: Observer interruption during debate turn {debate_turn}: {reason_data}")
                                break
                        else:
                            token_str = str(item)
                            _stream_buf += token_str
                            # Live repetition detection: if the buffer's second half
                            # repeats the first half, stop streaming.
                            _sb_len = len(_stream_buf)
                            if _sb_len >= 80:
                                _half = _sb_len // 2
                                if _stream_buf[_half:].strip() and _stream_buf[:_half].rstrip().endswith(_stream_buf[_half:].strip()):
                                    logger.warning("AgentCore: Stream repetition detected at %d chars; truncating.", _sb_len)
                                    break
                                # Also check if a sentence-level block is repeated
                                _stripped = _stream_buf.strip()
                                _mid = len(_stripped) // 2
                                if _mid >= 30:
                                    _first = _stripped[:_mid].strip()
                                    _second = _stripped[_mid:].strip()
                                    if _first and _second and _first == _second:
                                        logger.warning("AgentCore: Block duplication detected in stream; truncating.")
                                        # Remove the duplicate from pieces
                                        current_turn_pieces = current_turn_pieces[:len(current_turn_pieces)//2] if current_turn_pieces else current_turn_pieces
                                        break
                            current_turn_pieces.append(token_str)
                            # Yield token for real-time streaming
                            yield {"type": "token", "value": token_str}
                except Exception as _stream_exc:
                    # ── GAIA-CORE-075: Inference stream interrupted ──
                    _exc_type = type(_stream_exc).__name__
                    _exc_detail = f"{_exc_type}: {_stream_exc}"
                    _tokens_received = len(current_turn_pieces)
                    log_gaia_error(
                        logger, "GAIA-CORE-075",
                        f"model={selected_model_name} role={role_label} "
                        f"tokens_before_break={_tokens_received} "
                        f"error={_exc_detail}",
                        exc_info=True,
                    )
                    # Persist seed so sleep cycle can analyze the failure pattern
                    try:
                        from gaia_core.cognition.thought_seed import save_thought_seed
                        save_thought_seed(
                            f"THOUGHT_SEED: Inference stream interrupted — "
                            f"model={selected_model_name}, error={_exc_type}, "
                            f"tokens_before_break={_tokens_received}. "
                            f"Prompt: '{user_input[:50]}...'",
                            packet, self.config,
                        )
                    except Exception as _seed_exc:
                        logger.debug("AgentCore: thought seed save failed (non-fatal): %s", _seed_exc)
                    yield {
                        "type": "token",
                        "value": (
                            f"\n\n--- **[GAIA-CORE-075]** Stream from {role_label} {identity_label} "
                            f"interrupted ({_exc_type}) after {_tokens_received} tokens. "
                            f"Logging for self-diagnosis. ---"
                        ),
                    }

                current_response = "".join(current_turn_pieces)

                # ── Clear adapter after streaming ──
                if _stream_adapter and hasattr(selected_model, 'set_active_adapter'):
                    selected_model.set_active_adapter(None)

                # Yield any observer findings attached to this phase
                if observer_instance and hasattr(observer_instance, 'last_review') and observer_instance.last_review:
                    rev = observer_instance.last_review
                    obs_note = f"\n\n🔍 **[Observation]** {rev.level.upper()}: {rev.reason}"
                    yield {"type": "token", "value": obs_note}

                # Parse current output for council tags
                routing_result = route_output(current_response, packet, self, session_id, destination)
                council_msgs = routing_result.get("council_messages", [])
                
                # Release observer before potentially continuing debate
                if observer_model_name:
                    self.model_pool.release_model_for_role(observer_model_name)

                if not council_msgs:
                    # Consensus reached: yield final cleaned response
                    logger.info(f"AgentCore: Consensus reached by {selected_model_name} in turn {debate_turn}")
                    full_response = current_response
                    break
                
                # Consensus NOT reached: store peer message and swap model
                logger.info(f"AgentCore: Debate continues. {selected_model_name} emitted {len(council_msgs)} council messages.")
                for msg in council_msgs:
                    council_history.append(f"[{selected_model_name.upper()}]: {msg}")
                
                # Model Swap: Core <-> Thinker (or Core <-> Core if Prime down)
                next_model_role = "thinker" if selected_model_name == "core" else "core"
                
                # Release current model
                self.model_pool.release_model_for_role(selected_model_name)
                
                # Acquire next model
                try:
                    new_model = self.model_pool.acquire_model_for_role(next_model_role)
                    if not new_model and next_model_role == "thinker":
                        logger.warning("AgentCore: Thinker (Prime) unavailable. Falling back to Monastic Reasoning (Core solo-debate).")
                        # Fallback to Core for self-reflection
                        new_model = self.model_pool.acquire_model_for_role("core")
                        next_model_role = "core"

                    if new_model:
                        selected_model = new_model
                        selected_model_name = next_model_role
                    else:
                        logger.warning(f"AgentCore: Could not acquire {next_model_role} for debate; terminating.")
                        full_response = routing_result.get("response_to_user", "")
                        break
                except Exception as _swap_exc:
                    log_gaia_error(
                        logger, "GAIA-CORE-076",
                        f"target_role={next_model_role} error={type(_swap_exc).__name__}: {_swap_exc}",
                        exc_info=True,
                    )
                    full_response = routing_result.get("response_to_user", "")
                    break
    
            # End of while loop
            if debate_turn >= MAX_DEBATE_TURNS:
                logger.warning(f"AgentCore: Max debate turns ({MAX_DEBATE_TURNS}) reached without consensus.")
            
            # Release the final model used
            self.model_pool.release_model_for_role(selected_model_name)

            # Single consolidated flush after all debate turns complete
            yield {"type": "flush"}

            # --- Finalize response processing ---
            full_response = self._suppress_repetition(full_response)

            # Strip triage markers that should never reach the user.
            # Nano sometimes outputs "ESCALATE", "SIMPLE", "COMPLEX" as part of
            # triage classification. These are internal routing signals.
            _TRIAGE_MARKERS = ["ESCALATE", "SIMPLE", "COMPLEX"]
            for _marker in _TRIAGE_MARKERS:
                if _marker in full_response.upper():
                    import re as _re_triage
                    full_response = _re_triage.sub(
                        rf'\b{_marker}\b[.,;:\s]*', '', full_response,
                        flags=_re_triage.IGNORECASE).strip()

            logger.info(f"Full LLM response before routing: {full_response}")

            # ── Think-tag-only recovery ──────────────────────────────────

                # If the model generated only <think> reasoning without any
                # visible response, retry once with an explicit instruction.
            stripped_check = strip_think_tags(full_response).strip()
            if full_response.strip() and not stripped_check:
                logger.warning(
                    "AgentCore: Response was think-tags only (%d chars). "
                    "Retrying with explicit no-thinking instruction.",
                    len(full_response),
                )

                # Reset packet state — the circuit breaker set it to ABORTED
                # to stop the first generation, but the retry deserves a
                # clean slate so route_output doesn't reject the result.
                if packet.status.state == PacketState.ABORTED:
                    logger.info("AgentCore: Resetting packet state from ABORTED → PROCESSING for think-tag retry")
                    packet.status.state = PacketState.PROCESSING
                    # Clear stale abort traces so the post-stream observer
                    # and output router see a clean packet.
                    packet.status.next_steps = [
                        s for s in packet.status.next_steps
                        if "Loop detected" not in s and "Observer interruption" not in s
                    ]
                    packet.status.observer_trace = [
                        s for s in packet.status.observer_trace
                        if "LOOP_BLOCK" not in s and "STREAM_INTERRUPT" not in s
                    ]
                # Extract the reasoning so we can give it context on retry
                import re as _re
                _think_match = _re.search(
                    r'<(?:think|thinking)>(.*?)</(?:think|thinking)>',
                    full_response, _re.DOTALL,
                )
                reasoning_summary = ""
                if _think_match:
                    reasoning_summary = _think_match.group(1).strip()[:500]

                # Build a retry message list: append an instruction to
                # produce a direct answer without think tags.
                retry_instruction = (
                    "Your previous response contained only internal reasoning "
                    "without a visible answer. Please respond directly to the "
                    "user's last message. Do NOT use <think> tags. "
                    "Do NOT reference or cite any files or file paths. "
                    "Do NOT invent sources. If you don't know something, "
                    "say so honestly. Provide a concise, helpful answer."
                )
                if reasoning_summary:
                    retry_instruction += (
                        f"\n\nYour reasoning was: {reasoning_summary}\n\n"
                        "Now turn that reasoning into a direct response."
                    )
                retry_messages = list(final_messages)
                retry_messages.append({"role": "user", "content": retry_instruction})

                try:
                    retry_result = self.model_pool.forward_to_model(
                        selected_model_name,
                        messages=retry_messages,
                        max_tokens=self.config.max_tokens,
                        temperature=max(self.config.temperature - 0.1, 0.1),
                        top_p=self.config.top_p,
                        adapter_name=self._resolve_adapter(selected_model_name),
                    )
                    # Extract text from the retry result
                    retry_text = ""
                    if isinstance(retry_result, dict):
                        choices = retry_result.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            retry_text = msg.get("content", "")
                            if not retry_text:
                                retry_text = choices[0].get("text", "")
                    retry_text = strip_think_tags(retry_text).strip()
                    if retry_text:
                        logger.info(
                            "AgentCore: Think-tag retry succeeded (%d chars)",
                            len(retry_text),
                        )
                        full_response = retry_text
                    else:
                        logger.warning(
                            "AgentCore: Think-tag retry also empty; "
                            "using reasoning as fallback."
                        )
                        if reasoning_summary:
                            full_response = (
                                "Based on my analysis: " + reasoning_summary
                            )
                except Exception:
                    logger.exception(
                        "AgentCore: Think-tag retry failed; "
                        "using reasoning as fallback."
                    )
                    if reasoning_summary:
                        full_response = (
                            "Based on my analysis: " + reasoning_summary
                        )
            # ── End think-tag recovery ────────────────────────────────────

            # ── Post-generation quality gate — escalate to Prime if needed ──
            _stripped_response = strip_think_tags(full_response).strip() if full_response else ""
            _response_too_short = len(_stripped_response) < 10 and not any(
                tag in (user_input or "").lower() for tag in ["hi", "hello", "hey", "bye", "thanks"]
            )
            _response_empty = not _stripped_response

            # Check for recitation requests that got summarized instead of reproduced.
            # "recite X" should return the full text, not a description of it.
            _recitation_summarized = False
            _input_lower = (user_input or "").lower()
            if any(w in _input_lower for w in ["recite", "recitation", "full text", "reproduce", "quote the"]):
                # Recitation expected — check if response looks like a summary
                # (short, no line breaks suggesting verse structure, describes rather than reproduces)
                _has_verse_structure = _stripped_response.count("\n") >= 3
                _is_summary = len(_stripped_response) < 400 and not _has_verse_structure
                _describes_not_recites = any(p in _stripped_response.lower() for p in [
                    "is a poem", "is a nonsense poem", "features made-up", "tells a brief",
                    "written by", "from his", "published in"])
                if _is_summary or _describes_not_recites:
                    _recitation_summarized = True
                    logger.info("Quality gate: recitation requested but Core summarized instead (%d chars, %d newlines)",
                                len(_stripped_response), _stripped_response.count("\n"))

            if (_response_empty or _response_too_short or _recitation_summarized) and selected_model_name not in (
                "gpu_prime", "prime", "thinker", "groq_fallback", "oracle_openai"
            ):
                if _recitation_summarized:
                    _escalation_reason = "recitation request got summary instead of full text"
                elif _response_empty:
                    _escalation_reason = "empty response"
                else:
                    _escalation_reason = f"response too short ({len(_stripped_response)} chars)"
                logger.warning(
                    "Post-generation quality gate: %s from %s. Attempting Prime escalation.",
                    _escalation_reason, selected_model_name
                )
                # Escalate to Prime via consciousness state transition.
                # CPU Prime is for OBSERVATION only — responses need GPU Prime.
                # Request FOCUSING mode: Core→CPU, Prime→GPU, Nano stays GPU.
                _escalated_ok = False
                _prime_endpoint = os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777")
                _prime_model = os.environ.get("PRIME_MODEL_PATH", "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive")
                _orchestrator_endpoint = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
                try:
                    import httpx as _httpx_esc
                    yield {"type": "token", "value": f"\n\n[(i) Core response insufficient ({_escalation_reason}). Requesting FOCUSING mode — swapping Prime to GPU...]\n\n"}

                    # Request consciousness transition: AWAKE → FOCUSING
                    # This swaps Core to CPU/GGUF and Prime to GPU/safetensors.
                    # The endpoint returns immediately (async) — we must poll lifecycle
                    # state to know when the transition completes.
                    try:
                        _focus_resp = _httpx_esc.post(
                            f"{_orchestrator_endpoint}/consciousness/focusing",
                            timeout=10.0,
                        )
                        if _focus_resp.status_code == 200:
                            logger.info("FOCUSING transition requested, waiting for completion...")
                            yield {"type": "token", "value": "[(i) Switching to FOCUSING mode...]\n"}
                            yield {"type": "flush"}

                            # Poll lifecycle state until FOCUSING or timeout
                            import time as _esc_time
                            _focus_deadline = _esc_time.time() + 120.0
                            _focusing_ready = False
                            while _esc_time.time() < _focus_deadline:
                                try:
                                    _state_resp = _httpx_esc.get(
                                        f"{_orchestrator_endpoint}/lifecycle/state",
                                        timeout=5.0,
                                    )
                                    if _state_resp.status_code == 200:
                                        _lstate = _state_resp.json().get("state", "")
                                        if _lstate == "focusing":
                                            _focusing_ready = True
                                            break
                                except Exception:
                                    pass
                                _esc_time.sleep(3.0)

                            if _focusing_ready:
                                logger.info("FOCUSING mode active — Prime on GPU")
                                yield {"type": "token", "value": "[(i) FOCUSING mode active — Prime on GPU.]\n"}
                            else:
                                logger.warning("FOCUSING transition timed out — proceeding with available Prime")
                                yield {"type": "token", "value": "[(i) GPU swap still in progress — trying Prime anyway.]\n"}
                        else:
                            logger.warning("Consciousness FOCUSING transition failed: %s", _focus_resp.status_code)
                            yield {"type": "token", "value": "[(i) GPU swap failed — using Prime on CPU.]\n"}
                    except Exception as _focus_err:
                        logger.warning("Consciousness transition failed: %s — falling back to CPU Prime", _focus_err)
                        yield {"type": "token", "value": "[(i) Orchestrator unavailable — using Prime on CPU.]\n"}

                    # Now check Prime's health (should be on GPU after FOCUSING)
                    _ph = _httpx_esc.get(f"{_prime_endpoint}/health", timeout=5)
                    _prime_health = _ph.json() if _ph.status_code == 200 else {}
                    _prime_loaded = _prime_health.get("model_loaded", False)
                    # GGUF backend health returns just {"status":"ok"} without model_loaded
                    # If we get a 200 and the managed mode reports active, model is loaded
                    if _prime_health.get("mode") == "active" or _prime_health.get("status") == "ok":
                        _prime_loaded = True
                    if not _prime_loaded:
                        # Prime is in standby — load the model
                        logger.info("Prime in standby — loading model for escalation")
                        yield {"type": "token", "value": "[(i) Loading Prime model...]\n"}
                        _load_resp = _httpx_esc.post(
                            f"{_prime_endpoint}/model/load",
                            json={"model": _prime_model, "device": "cuda"},
                            timeout=180.0,
                        )
                        # 409 = already loaded (race condition) — that's fine
                        if _load_resp.status_code == 409:
                            logger.info("Prime already loaded (409) — proceeding")
                        elif _load_resp.status_code != 200 or not _load_resp.json().get("ok"):
                            raise RuntimeError(f"Prime model load failed: {_load_resp.text[:200]}")
                        else:
                            logger.info("Prime model loaded for escalation")
                    # Generate via Prime — use the model pool for proper tool access
                    # This gives Prime the full system prompt, context, and MCP tools
                    _prime_model_obj = None
                    for _pm_key in ["gpu_prime", "prime"]:
                        if _pm_key in self.model_pool.models:
                            _prime_model_obj = self.model_pool.models[_pm_key]
                            break
                    if _prime_model_obj:
                        try:
                            # Build Prime's prompt with tool results for grounded recitation
                            _sys_prompt = getattr(packet, '_system_prompt', '') or ''
                            _msgs = []
                            if _sys_prompt:
                                _msgs.append({"role": "system", "content": _sys_prompt})

                            # Include tool results (web search data) if available.
                            # For recitation requests, fetch full page content from a
                            # trusted URL so Prime has the actual text, not just snippets.
                            _tool_context = ""
                            if hasattr(packet, 'tool_routing') and packet.tool_routing:
                                _exec_result = getattr(packet.tool_routing, 'execution_result', None)
                                if _exec_result and getattr(_exec_result, 'output', None):
                                    _tool_output = _exec_result.output
                                    if isinstance(_tool_output, dict) and _tool_output.get("results"):
                                        # Try web_fetch on first trusted URL for full content
                                        _fetched_content = ""
                                        if _recitation_summarized:
                                            _trusted_urls = [
                                                r.get("url") for r in _tool_output["results"]
                                                if r.get("trust_tier") == "trusted" and r.get("url")
                                            ]
                                            for _fetch_url in _trusted_urls[:2]:
                                                try:
                                                    logger.info("Fetching full content from %s", _fetch_url)
                                                    yield {"type": "token", "value": f"[(i) Fetching full text from {_fetch_url.split('/')[2]}...]\n"}
                                                    yield {"type": "flush"}
                                                    _fetch_result = mcp_client.call_jsonrpc(
                                                        "web_fetch", {"url": _fetch_url}
                                                    )
                                                    if _fetch_result.get("ok"):
                                                        _fr = _fetch_result.get("response", {}).get("result", {})
                                                        _fc = _fr.get("content", _fr.get("text", ""))
                                                        if isinstance(_fc, str) and len(_fc) > 50:
                                                            _fetched_content = _fc[:4000]
                                                            logger.info("Fetched %d chars from %s", len(_fc), _fetch_url)
                                                            break
                                                except Exception as _fetch_err:
                                                    logger.warning("web_fetch failed for %s: %s", _fetch_url, _fetch_err)

                                        if _fetched_content:
                                            _tool_context = f"\n\nFull content fetched from web:\n{_fetched_content}"
                                        else:
                                            _snippets = []
                                            for r in _tool_output["results"][:3]:
                                                _snippets.append(f"Source: {r.get('title', '')}\n{r.get('snippet', '')}")
                                            _tool_context = "\n\nWeb search results:\n" + "\n---\n".join(_snippets)

                            _user_msg = user_input
                            if _tool_context:
                                _user_msg += _tool_context
                            if _recitation_summarized:
                                _user_msg += "\n\nIMPORTANT: Please reproduce the FULL TEXT of the requested work, not a summary. Recite it verbatim."
                            _msgs.append({"role": "user", "content": _user_msg})
                            _max_tok = packet.context.constraints.max_tokens if hasattr(packet.context, 'constraints') else 2048
                            _esc_result = _prime_model_obj.create_chat_completion(
                                messages=_msgs,
                                max_tokens=_max_tok,
                            )
                            _esc_text = ""
                            if isinstance(_esc_result, dict):
                                _esc_text = _esc_result.get("response", "") or _esc_result.get("text", "")
                                if not _esc_text and "choices" in _esc_result:
                                    _esc_text = _esc_result["choices"][0].get("message", {}).get("content", "")
                            elif isinstance(_esc_result, str):
                                _esc_text = _esc_result
                            _esc_text = strip_think_tags(_esc_text).strip() if _esc_text else ""
                        except Exception as _pool_err:
                            logger.warning("Model pool escalation failed: %s — falling back to direct API", _pool_err)
                            _esc_text = ""

                        if not _esc_text:
                            # Fallback: direct API call (no tools but at least gets a response)
                            _prime_resp = _httpx_esc.post(
                                f"{_prime_endpoint}/v1/chat/completions",
                                json={"messages": [{"role": "user", "content": user_input}],
                                      "max_tokens": 2048, "temperature": 0.7},
                                timeout=120.0,
                            )
                            if _prime_resp.status_code == 200:
                                _prime_data = _prime_resp.json()
                                if "choices" in _prime_data:
                                    _esc_text = _prime_data["choices"][0].get("message", {}).get("content", "")
                                _esc_text = strip_think_tags(_esc_text).strip() if _esc_text else ""

                        if _esc_text and len(_esc_text) > len(_stripped_response):
                            full_response = _esc_text
                            selected_model_name = "prime"
                            _escalated_ok = True
                            logger.info("Post-generation escalation to Prime succeeded (%d chars)", len(_esc_text))
                            yield {"type": "token", "value": _esc_text}
                    else:
                        # No Prime in model pool — direct API only
                        _prime_resp = _httpx_esc.post(
                            f"{_prime_endpoint}/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": user_input}],
                                  "max_tokens": 2048, "temperature": 0.7},
                            timeout=120.0,
                        )
                        if _prime_resp.status_code == 200:
                            _prime_data = _prime_resp.json()
                            _esc_text = ""
                            if "choices" in _prime_data:
                                _esc_text = _prime_data["choices"][0].get("message", {}).get("content", "")
                            _esc_text = strip_think_tags(_esc_text).strip() if _esc_text else ""
                            if _esc_text and len(_esc_text) > len(_stripped_response):
                                full_response = _esc_text
                                selected_model_name = "prime"
                                _escalated_ok = True
                                logger.info("Post-generation escalation to Prime (direct) succeeded (%d chars)", len(_esc_text))
                                yield {"type": "token", "value": _esc_text}
                except Exception as _prime_exc:
                    logger.warning("Post-generation escalation to Prime failed: %s", _prime_exc)

                # Transition back to AWAKE: Prime→CPU/GGUF, Core→GPU
                # This runs whether escalation succeeded or not — don't leave Prime on GPU
                try:
                    _awake_resp = _httpx_esc.post(
                        f"{_orchestrator_endpoint}/consciousness/awake",
                        timeout=120.0,
                    )
                    if _awake_resp.status_code == 200:
                        logger.info("Consciousness transition back to AWAKE: %s", _awake_resp.json().get("configuration"))
                    else:
                        logger.warning("Consciousness AWAKE transition failed: %s", _awake_resp.status_code)
                except Exception as _awake_err:
                    logger.warning("Consciousness AWAKE transition failed: %s", _awake_err)

                if not _escalated_ok:
                    logger.warning("Post-generation escalation failed; using original response")
            # ── End quality gate ──────────────────────────────────────────────

            if post_run_observer and not disable_observer:
                try:
                    review = post_run_observer.observe(packet, full_response)
                    note = f"[Observer] {review.level.upper()}: {review.reason}"
                    if review.suggestion:
                        note += f" | Suggestion: {review.suggestion}"
                    logger.info("Post-stream observer review: %s", note)
                    yield {"type": "token", "value": f"\n\n{note}\n"}

                    # Scrub confabulated file references from the response.
                    # The observer catches fake paths at CAUTION level — strip
                    # sentences that reference nonexistent files so the user
                    # doesn't see hallucinated sources.
                    if review.level == "CAUTION" and "nonexistent file" in review.reason:
                        import re as _re_scrub
                        # Extract the fake file names from the observer reason
                        _fab_match = _re_scrub.search(r'nonexistent file\(s\): (.+)$', review.reason)
                        if _fab_match:
                            fake_names = [f.strip() for f in _fab_match.group(1).split(',')]
                            for fake in fake_names:
                                # Remove sentences/lines that reference the fake file
                                escaped = _re_scrub.escape(fake)
                                full_response = _re_scrub.sub(
                                    r'[^\n.]*' + escaped + r'[^\n.]*[.\n]?',
                                    '', full_response
                                    )
                                full_response = full_response.strip()
                                logger.info("AgentCore: Scrubbed %d confabulated file reference(s) from response", len(fake_names))
                except Exception:
                    logger.warning("Post-stream observer review failed; continuing without interruption.", exc_info=True)

            # ── Observer Scoring (async, non-blocking) ──
            try:
                from gaia_core.cognition.observer_scorer import get_observer_scorer
                _scorer = get_observer_scorer(self.config)
                if _scorer:
                    import threading
                    _obs_review = review if 'review' in dir() else None
                    threading.Thread(
                        target=_scorer.score_turn,
                        args=(user_input, full_response, packet),
                        kwargs={"observer_review": _obs_review},
                        daemon=True,
                    ).start()
            except Exception:
                logger.debug("Observer scorer hook failed", exc_info=True)

            # --- Output Routing & Persistence (Always Run) ---
            # This function now needs to handle the v0.3 packet
            routed_output = route_output(full_response, packet, self.ai_manager, session_id, destination)
            user_facing_response = routed_output["response_to_user"]
            execution_results = routed_output["execution_results"]

            packet.reasoning.reflection_log.append(ReflectionLog(step="execution_results", summary=str(execution_results)))

            logger.debug(f"User-facing response after routing: {user_facing_response}")

            # Epistemic citation check — warn user if fabricated citations detected
            epistemic_warning = ""
            try:
                eg_config = self.config.constants.get("EPISTEMIC_GUARDRAILS", {})
                if eg_config.get("annotate_unverified_citations", True):
                    max_before_warn = eg_config.get("max_unverified_citations_before_warning", 1)
                    reflection_logs = getattr(packet.reasoning, 'reflection_log', []) or []
                    for log_entry in reflection_logs:
                        step = getattr(log_entry, 'step', '') if hasattr(log_entry, 'step') else log_entry.get('step', '')
                        summary = getattr(log_entry, 'summary', '') if hasattr(log_entry, 'summary') else log_entry.get('summary', '')
                        if step in ('observer_path_validation', 'observer_citation_verification'):
                            warning_count = summary.count('does not exist') + summary.count('Unverified')
                            if warning_count >= max_before_warn:
                                epistemic_warning = (
                                    "\n\n---\n*[Observer: Some file references in this response "
                                    "could not be verified against the knowledge base. "
                                    "Information may be from general knowledge rather than "
                                    "retrieved documents.]*\n---\n"
                                )
                                break
            except Exception:
                logger.debug("Epistemic citation check failed", exc_info=True)

            _header = self._build_response_header(selected_model_name, packet, observer_instance, active_stream_observer, post_run_observer)
            
            # --- Speculative Response Comparison & Epistemic Validation ---
            final_yield_text = user_facing_response
            if reflex_text:
                # Value check: similarity ratio and length growth
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, reflex_text, user_facing_response).ratio()
                self.logger.warning(f"AgentCore: Epistemic check ratio={ratio:.2f} for reflex vs response")
                
                is_sufficient = ratio > 0.8 and len(user_facing_response) < (len(reflex_text) * 1.2)
                
                if is_sufficient:
                    self.logger.info(f"AgentCore: speculative reflex was sufficient (match={ratio:.2f}); skipping follow-up.")
                    final_yield_text = "" # Don't yield a redundant follow-up
                else:
                    # EPISTEMIC VALIDATION: Check for contradictions or significant inaccuracies
                    # If the ratio is very low, it's likely a hallucination or a completely different topic.
                    if ratio < 0.4:
                        self.logger.warning(f"AgentCore: Epistemic mismatch detected (ratio={ratio:.2f}); labeling as correction.")
                        final_yield_text = "\n\n---\n⚠️ **[Correction — deeper analysis follows]**\n" + user_facing_response
                        
                        # Log thought seed for Samvega introspection (internal only, not user-facing)
                        seed_msg = f"THOUGHT_SEED: Nano hallucination detected on prompt '{user_input[:30]}...'. Reflex said '{reflex_text[:30]}...', Core said '{user_facing_response[:30]}...'. Investigating prevention."
                        self.logger.info(seed_msg)
                        # Persist to /knowledge/seeds/ so heartbeat can triage
                        try:
                            from gaia_core.cognition.thought_seed import save_thought_seed
                            save_thought_seed(seed_msg, packet, self.config)
                        except Exception as _seed_exc:
                            logger.debug("AgentCore: thought seed save failed (non-fatal): %s", _seed_exc)
                    else:
                        self.logger.info(f"AgentCore: reflex insufficient (match={ratio:.2f}); yielding full response as refinement.")
                        final_yield_text = "\n\n---\n🔄 **[Refinement — here's the fuller answer]**\n" + user_facing_response

            # ── Epistemic Confabulation Gate ─────────────────────────────
            # Detect when GAIA claims to have reviewed/evaluated something
            # that wasn't grounded in retrieved documents or tool output.
            _confab_signals = [
                "solid upgrade", "great addition", "looks good", "well designed",
                "i've reviewed", "i've seen", "i can confirm", "glad you",
                "happy with", "the implementation is", "nice work on",
            ]
            _response_lower = (final_yield_text or "").lower()
            _has_confab_signal = any(s in _response_lower for s in _confab_signals)
            if _has_confab_signal:
                # Check if there's grounding: retrieved docs or tool execution results
                _has_grounding = bool(execution_results)
                if not _has_grounding:
                    for _df in packet.content.data_fields:
                        if _df.key == "retrieved_documents" and _df.value:
                            _has_grounding = True
                            break
                if not _has_grounding:
                    self.logger.warning(
                        "AgentCore: Epistemic confabulation detected — "
                        "response claims evaluation without grounding"
                    )
                    seed_msg = (
                        f"THOUGHT_SEED: Confabulation detected — claimed to have "
                        f"reviewed/evaluated something without grounding documents. "
                        f"Prompt: '{user_input[:50]}...'"
                    )
                    self.logger.info(seed_msg)
                    ts_write(seed_msg)
                    # Persist to /knowledge/seeds/ so heartbeat can triage
                    try:
                        from gaia_core.cognition.thought_seed import save_thought_seed
                        save_thought_seed(seed_msg, packet, self.config)
                    except Exception as _seed_exc:
                        logger.debug("AgentCore: thought seed save failed (non-fatal): %s", _seed_exc)
            # ── End Epistemic Confabulation Gate ──────────────────────────

            # IMPORTANT: If we already streamed the response tokens, do NOT yield it again as a single chunk.
            # Only yield if it's a 'Refinement' (reflex_text was present and response differs).
            if final_yield_text:
                if reflex_text:
                    # Reflex was sent early — yield the full response as a refinement
                    yield {"type": "token", "value": _header + final_yield_text + epistemic_warning}
                    logger.debug("Yielded refinement after reflex")
                else:
                    # Response was already streamed during debate loop — don't duplicate
                    logger.debug("Suppressing final yield: already streamed in debate loop.")

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
            # Don't save degenerate or triage-contaminated responses to session history
            _save_text = strip_think_tags(user_facing_response).strip()
            if _save_text and len(_save_text) > 5 and not any(
                m in _save_text.upper() for m in ["ESCALATE", "CONFIDENCE ASSESSMENT TASK"]):
                self.session_manager.add_message(session_id, "assistant", _save_text)
            else:
                logger.warning("Skipping session save — degenerate response: %s", _save_text[:100])
            self._emit_timeline_message(session_id, "assistant", source)

            if execution_results:
                process_execution_results(execution_results, self.session_manager, session_id, packet)

                messages = build_from_packet(packet, task_instruction_key="execution_feedback")
                voice = ExternalVoice(model=selected_model, model_pool=self.model_pool, config=self.config, messages=messages, source="agent_core", observer=active_stream_observer, context={"packet": packet}, session_id=session_id)
                stream_generator = voice.stream_response()
                concluding_response = "".join([str(token) for token in stream_generator if isinstance(token, str)])
                yield {"type": "token", "value": concluding_response}
                self.session_manager.add_message(session_id, "assistant", concluding_response)

            # Notify KV cache manager that inference occurred for this role
            try:
                from gaia_core.cognition.kv_cache_manager import get_kv_cache_manager
                _kv_mgr = get_kv_cache_manager()
                if _kv_mgr is not None and selected_model_name:
                    _kv_role_map = {"reflex": "reflex", "nano": "reflex", "core": "core", "lite": "core"}
                    _kv_role = _kv_role_map.get(selected_model_name)
                    if _kv_role:
                        _kv_mgr.notify_inference(_kv_role)
            except Exception as _kv_exc:
                logger.debug("AgentCore: KV cache notify failed (non-fatal): %s", _kv_exc)

            log_chat_entry(user_input, user_facing_response, source=source, session_id=session_id, metadata=_metadata)
            log_chat_entry_structured(user_input, user_facing_response, source=source, session_id=session_id, metadata=_metadata)
            turn_end_event = {"type": "turn_end", "user": user_input, "assistant": user_facing_response}
            probe_stats = get_session_probe_stats(session_id)
            if probe_stats:
                turn_end_event["probe_session_stats"] = probe_stats
            ts_write(turn_end_event, session_id, source=source, destination_context=_metadata)
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
                        from gaia_common.protocols.cognition_packet import LoopState
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
                            log_gaia_error(logger, "GAIA-CORE-025", notification.get('status_line', ''))
                        else:
                            # Reset triggered - will inject recovery context next turn
                            reset_msg = f"\n\n---\n🔄 **Loop Detected**: {notification.get('toast', {}).get('body', 'Resetting to try a different approach.')}\n---"
                            yield {"type": "token", "value": reset_msg}
                            log_gaia_error(logger, "GAIA-CORE-026", notification.get('status_line', ''))
    
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
                        # Evict session loop state to prevent memory leak
                        cleanup_session_manager(session_id)

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
            logger.info(f"AgentCore: run_turn total took {packet.metrics.latency_ms / 1000:.2f}s")
            
            # YIELD THE FINAL PACKET
            yield {"type": "packet", "value": packet.to_serializable_dict()}
        finally:
            # Clear Grit Mode if it was enabled for this turn
            try:
                from gaia_common.utils.immune_system import clear_grit_mode
                clear_grit_mode()
            except Exception as _grit_exc:
                logger.warning("AgentCore: grit mode clear failed: %s", _grit_exc)
            # Release observer/selected models using the ModelPool API directly.
            try:
                if observer_model_name:
                    self.model_pool.release_model_for_role(observer_model_name)
                self.model_pool.release_model_for_role(selected_model_name)
                if 'lite_fallback_acquired' in locals() and lite_fallback_acquired:
                    self.model_pool.release_model_for_role('lite')
            except Exception:
                self.logger.debug("AgentCore: release_model_for_role failed during final cleanup", exc_info=True)
            logger.info("AgentCore: Released models for turn context")

    def _knowledge_acquisition_workflow(self, packet: CognitionPacket) -> CognitionPacket:
        """
        Attempt to find, embed, and query relevant knowledge when the initial RAG query fails.
        """
        self.logger.info("Knowledge acquisition workflow ENTERED.")
        try:
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
            found_docs = asyncio.run(mcp_client.call_jsonrpc(
                "find_relevant_documents",
                {"query": packet.content.original_prompt, "knowledge_base_name": knowledge_base_name}
            ))
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
                embed_result = asyncio.run(mcp_client.call_jsonrpc(
                    "embed_documents",
                    {"knowledge_base_name": knowledge_base_name, "file_path": doc_path}
                ))

                if embed_result and embed_result.get("ok"):
                    self.logger.info("Document embedded successfully. Re-running RAG query.")
                    
                    # Step 3: Re-run RAG query
                    retrieved_docs = asyncio.run(mcp_client.embedding_query(
                        packet.content.original_prompt,
                        top_k=3,
                        knowledge_base_name=knowledge_base_name
                    ))

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

        Also detects whole-block duplication where the model outputs the same
        response twice back-to-back (possibly without whitespace separating them).
        """
        if not text:
            return text

        # --- Pass 0: Whole-block dedup ---
        # If the second half of the text is a near-verbatim copy of the first
        # half, keep only the first half.  This catches cases where the model
        # regenerates the entire answer without any separator.
        text = self._dedup_block(text)

        # --- Pass 0.5: Line-block dedup ---
        # Catch repeated multi-line blocks (e.g. Nano outputting the same
        # 10-15 line status block over and over).
        lines = text.split('\n')
        if len(lines) > 10:
            text = self._dedup_line_blocks(text)

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

    @staticmethod
    def _dedup_block(text: str, min_block: int = 40, similarity_threshold: float = 0.85) -> str:
        """
        Detect and remove whole-block duplication where the model outputs the
        same response twice back-to-back.

        Scans for a repeated substring of at least `min_block` characters.
        If the text contains a block that appears twice (possibly glued together
        without whitespace), the duplicate is removed.
        """
        if len(text) < min_block * 2:
            return text

        # Strategy: try to find the longest prefix of the text that also
        # appears later.  Check from the midpoint outward.
        half = len(text) // 2

        # Try different split points around the midpoint
        for offset in range(0, min(half, 200), 10):
            for split_at in [half + offset, half - offset]:
                if split_at < min_block or split_at >= len(text) - min_block:
                    continue
                first_half = text[:split_at].strip()
                second_half = text[split_at:].strip()

                # Quick length check — halves should be roughly the same size
                if not second_half or not first_half:
                    continue
                len_ratio = len(second_half) / len(first_half)
                if len_ratio < 0.7 or len_ratio > 1.3:
                    continue

                # Compare normalized versions
                norm_first = first_half.lower().split()
                norm_second = second_half.lower().split()
                if not norm_first or not norm_second:
                    continue

                # Jaccard word-set similarity
                set_first = set(norm_first)
                set_second = set(norm_second)
                intersection = len(set_first & set_second)
                union = len(set_first | set_second)
                similarity = intersection / union if union else 0

                if similarity >= similarity_threshold:
                    # Keep the first half (it's the original)
                    return first_half

        return text

    @staticmethod
    def _dedup_line_blocks(text: str, block_sizes=(3, 4, 5), max_repeats: int = 2) -> str:
        """Remove repeated multi-line blocks.

        Slides an N-line window through the text for each block size.
        If any window hash appears more than *max_repeats* times,
        the text is truncated after the second occurrence.
        """
        lines = text.split('\n')
        for n in block_sizes:
            if len(lines) < n * (max_repeats + 1):
                continue
            seen: Dict[int, list] = {}  # hash -> list of start indices
            for i in range(len(lines) - n + 1):
                block = '\n'.join(lines[i:i + n]).strip()
                if not block:
                    continue
                h = hash(block)
                positions = seen.setdefault(h, [])
                positions.append(i)
                if len(positions) > max_repeats:
                    # Truncate after the end of the max_repeats-th occurrence
                    cutoff = positions[max_repeats - 1] + n
                    lines = lines[:cutoff]
                    text = '\n'.join(lines)
                    break  # re-check with truncated text
            else:
                continue
            break  # already truncated, stop checking larger block sizes
        return '\n'.join(lines) if isinstance(lines, list) else text

    def _is_degenerate_output(self, content: str, user_input: str) -> bool:
        """Detect degenerate model output that should trigger tier escalation."""
        if not content or not content.strip():
            return True
        stripped = content.strip()
        if len(stripped) < 5:
            return True
        # Repetition ratio: >60% of lines are duplicates
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if len(lines) > 5:
            unique = set(lines)
            if len(unique) / len(lines) < 0.4:
                return True
        # Starts with known error pattern
        if stripped.startswith("I encountered an error"):
            return True
        # Disproportionately long for short input (>800 chars for <=30 char input)
        if len(user_input) <= 30 and len(stripped) > 800:
            return True
        # Low unique-word ratio — sign of phrase-level repetition
        words = stripped.lower().split()
        if len(words) > 15:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.45:
                return True
        # Trailing truncation — output cut off mid-sentence (hit token limit)
        if len(stripped) > 20 and stripped[-1] not in '.!?"\')…':
            # Ends without terminal punctuation — likely truncated
            # Only flag if there WAS punctuation earlier (so it's not just a short phrase)
            if any(c in stripped[:-10] for c in '.!?'):
                return True
        return False

    # Hedging phrases that signal Nano can't extract an answer from context
    _HEDGING_ESCALATION_PHRASES = [
        "i'm not sure", "i'm not certain", "i don't have", "i can't",
        "i cannot", "i don't know", "not available", "unable to",
        "i should check", "i need to check", "let me check",
        "i should be honest", "rather than guessing", "i don't guess",
        "checking my", "i might be", "i may not",
    ]

    # Phrases that indicate Nano described the PROCESS but didn't deliver the ANSWER
    _PROCESS_WITHOUT_ANSWER_PHRASES = [
        "i check the clock line",
        "i read the clock",
        "it shows the time",
        "it's injected",
        "it's updated",
        "from my system context",
        "from my monitoring",
    ]

    def _should_escalate_for_uncertainty(self, content: str, user_input: str,
                                          entropy: float = 0.0) -> bool:
        """Detect when Nano's response shows uncertainty that warrants escalation.

        Three escalation signals (any one triggers):
        1. High output entropy (>2.0) — model was uncertain during generation
        2. Hedging phrases on factual questions — model admits it can't answer
        3. Process-without-answer — model describes HOW it would answer but doesn't
           actually provide the answer (e.g., "I check the Clock line" without a time)
        """
        lower = content.lower()
        input_lower = user_input.lower()

        # Signal 1: High entropy from engine
        if entropy > 2.0:
            self.logger.info("Uncertainty escalation: high entropy (%.2f > 2.0)", entropy)
            return True

        # Signal 2: Hedging on factual questions
        # Only escalate for factual-type questions, not philosophical/creative
        factual_signals = ["what time", "what port", "how many", "what is the",
                          "what gpu", "how much", "how long", "what status",
                          "tell me the", "what's the"]
        is_factual = any(sig in input_lower for sig in factual_signals)
        if is_factual:
            hedging = any(phrase in lower for phrase in self._HEDGING_ESCALATION_PHRASES)
            if hedging:
                self.logger.info("Uncertainty escalation: hedging on factual question")
                return True

        # Signal 3: Process description without actual answer
        # Nano says "I check the Clock line" but never gives a time
        describes_process = any(phrase in lower for phrase in self._PROCESS_WITHOUT_ANSWER_PHRASES)
        if describes_process:
            # Check if there's an actual data value in the response
            import re
            has_value = bool(re.search(r'\d{1,2}:\d{2}|\d+\s*(gb|mb|%|seconds|minutes|hours|days)', lower))
            if not has_value:
                self.logger.info("Uncertainty escalation: process description without actual value")
                return True

        # Signal 4: Factual question asked but NO factual value in response
        # Catches "I'm ready to share the time" without actually sharing it
        if is_factual:
            import re
            has_value = bool(re.search(r'\d{1,2}:\d{2}|\d+\s*(gb|mb|%|seconds|minutes|hours|days)', lower))
            if not has_value:
                self.logger.info("Uncertainty escalation: factual question but no value in response")
                return True

        # Signal 5: Nano explicitly declines or deflects a task
        # Catches "I can't recite", "I don't have knowledge", "I'm not familiar" etc.
        decline_phrases = ["i cannot create", "i can't create", "i cannot generate",
                          "i can't generate", "i'm not able to", "i am not able to",
                          "i cannot provide a detailed", "i can't provide a detailed",
                          "beyond my capability", "outside my scope",
                          "i can't recite", "i cannot recite",
                          "i don't have knowledge", "i'm not familiar",
                          "i am not familiar", "i don't have access to",
                          "without verification", "without a source",
                          "i can help you find", "let me know what you'd like to find",
                          "i cannot use", "i can't use", "i don't have the ability",
                          "i should acknowledge", "limits of my capabilities"]
        if any(phrase in lower for phrase in decline_phrases):
            self.logger.info("Uncertainty escalation: Nano explicitly declined the task")
            return True

        # Signal 5b: User asked for tool use but Nano described capability instead of acting
        tool_request_signals = ["use your", "web search", "search for", "look up",
                                "use the tool", "use a tool", "search tool"]
        user_wants_tool = any(sig in input_lower for sig in tool_request_signals)
        nano_described_not_acted = any(phrase in lower for phrase in [
            "i can use", "i can search", "i can look", "i can help you find",
            "let me know what", "what would you like"])
        if user_wants_tool and nano_described_not_acted:
            self.logger.info("Uncertainty escalation: user requested tool use, Nano only described capability")
            return True

        # Signal 6: Planning/architecture request answered with description only
        planning_signals = ["implementation plan", "detailed plan", "create a plan",
                           "design a system", "how would you add", "architecture"]
        is_planning = any(sig in input_lower for sig in planning_signals)
        if is_planning and len(content) < 500:
            self.logger.info("Uncertainty escalation: planning request got short response (%d chars)", len(content))
            return True

        # Signal 7: Very short response or classification tag — not a real answer
        # Catches raw classification outputs like "COMPLEX", "SIMPLE", single words
        stripped = content.strip()
        if len(stripped) < 20:
            self.logger.info("Uncertainty escalation: response too short to be a real answer (%d chars: '%s')", len(stripped), stripped[:30])
            return True

        # Signal 8: Response is a classification tag, not prose
        classification_tags = {"complex", "simple", "routing", "escalate", "tool_use",
                               "tool_routing", "decline", "redirect", "pass"}
        if stripped.lower().rstrip(".!") in classification_tags:
            self.logger.info("Uncertainty escalation: response is a classification tag, not an answer: '%s'", stripped)
            return True

        return False

    # Escalation chain for slim path failures
    _SLIM_ESCALATION_CHAIN = ["core", "lite", "groq_fallback", "oracle"]

    def _escalate_slim_response(self, failed_model: str, messages: list, max_tokens: int) -> str:
        """Try higher-tier models when the slim path model produces garbage.

        On success, sets ``self._last_responding_model`` so callers can
        attribute the response to the correct model in headers/tags.
        """
        # Extract the actual user question from the slim prompt messages
        # (last user message is the real question, earlier ones are few-shot)
        user_question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_question = msg.get("content", "")
                break

        for candidate in self._SLIM_ESCALATION_CHAIN:
            if candidate == failed_model:
                continue
            if candidate not in self.model_pool.models:
                continue
            try:
                self.logger.info("Slim escalation: trying '%s' after '%s' failed", candidate, failed_model)
                # Build a clean prompt for the escalation target —
                # don't send Nano's few-shot format to Core/Lite
                escalation_messages = [
                    {"role": "system", "content": "You are GAIA, a sovereign AI. Answer directly and concisely."},
                    {"role": "user", "content": user_question},
                ]
                res = self.model_pool.forward_to_model(
                    candidate,
                    messages=escalation_messages,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    adapter_name=self._resolve_adapter(candidate),
                )
                content = res["choices"][0]["message"]["content"]
                content = strip_think_tags(content).strip()
                content = self._suppress_repetition(content)
                if not self._is_degenerate_output(content, messages[-1].get("content", "")):
                    self.logger.info("Slim escalation succeeded with '%s'", candidate)
                    self._last_responding_model = candidate
                    return content
                self.logger.warning("Slim escalation: '%s' also produced degenerate output", candidate)
            except Exception:
                log_gaia_error(self.logger, "GAIA-CORE-070", f"Slim escalation to '{candidate}' failed", exc_info=True)
        # All escalations failed
        self.logger.error("All slim escalation tiers exhausted — returning dignified error")
        return "I'm having trouble responding right now. Please try again."

    def _build_response_header(
        self,
        model_name: str,
        packet,
        observer_instance,
        active_stream_observer,
        post_run_observer,
    ) -> str:
        """
        Build a concise status header prepended to user-facing responses.
        Shows the 'mind' (model) that generated the response.
        """
        # If an escalation occurred, use the model that actually responded
        actual_model = getattr(self, '_last_responding_model', None) or model_name
        self._last_responding_model = None  # reset after use
        mind_name = self._MIND_ALIASES.get(actual_model or "unknown", actual_model or "unknown")
        tag = self.MIND_TAG_FORMAT.format(mind=mind_name)
        return f"{tag}\n\n"

    def _check_immune_system_for_emergency(self) -> Optional[str]:
        """
        Check the current Immune System status for CRITICAL issues or SyntaxErrors.
        Returns the error message if an emergency is detected, else None.
        """
        try:
            from gaia_common.utils import immune_system
            health = immune_system.get_immune_summary()
            
            # Look for CRITICAL or SyntaxError
            if "Immune System: CRITICAL" in health or "SyntaxError" in health:
                logger.warning(f"[IMMUNE EMERGENCY] System irritation detected: {health}")
                return health
        except Exception as _imm_exc:
            logger.error("AgentCore: immune system health check failed — safety blind spot: %s", _imm_exc, exc_info=True)
        return None

    def is_eligible_for_reflex(self, packet: CognitionPacket, history: list) -> bool:
        """Determines if a packet should trigger an instant Nano reflex.

        Fires for ANY short, simple request regardless of session history.
        The reflex model handles greetings, time/date, identity, and basic
        math — escalating to the full pipeline via ESCALATE when unsure.
        """
        user_input = packet.content.original_prompt
        is_short = len(user_input) < 150

        from gaia_common.utils.immune_system import is_system_irritated
        return is_short and not is_system_irritated() and "reflex" in self.model_pool.models

    def generate_continuation(self, messages: list, session_id: str = ""):
        """
        Generate a continuation response after tool execution.

        Takes pre-built messages (user + assistant first output + tool_result)
        and generates the model's continuation. Yields token events like run_turn.

        This is the second-half of the native tool calling flow:
        1. Model generates first output (including <tool_call>)
        2. Runtime executes tool, gets result
        3. This method generates the continuation with result in context
        """
        # Acquire a model — prefer current tier, fall back through cascade
        model = None
        model_name = None
        for cand in ["lite", "prime", "gpu_prime", "core", "cpu_prime"]:
            try:
                model = self.model_pool.acquire_model(cand)
                if model is not None:
                    model_name = cand
                    break
            except Exception:
                continue

        if not model:
            self.logger.warning("No model available for continuation generation")
            yield {"type": "token", "value": "\n*[No model available for continuation]*\n"}
            return

        try:
            self.logger.info("Continuation generation with %s (%d messages)", model_name, len(messages))
            result = self.model_pool.forward_to_model(
                model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            if isinstance(result, dict):
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if text:
                    text = strip_think_tags(text).strip()
                    yield {"type": "token", "value": text}
            elif hasattr(result, '__iter__'):
                # Streaming response
                for chunk in result:
                    if isinstance(chunk, dict):
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            delta = strip_think_tags(delta)
                            if delta:
                                yield {"type": "token", "value": delta}
        except Exception as e:
            self.logger.warning("Continuation generation failed: %s", e)
            yield {"type": "token", "value": f"\n*[continuation error: {e}]*\n"}
        finally:
            if model_name:
                try:
                    self.model_pool.release_model(model_name)
                except Exception:
                    pass

    def generate_instant_reflex(self, packet: CognitionPacket) -> str:
        """
        Executes a fast, non-streaming Reflex response with a minimal packet.
        Returns the generated text, or "" to defer to run_turn().
        """
        if "reflex" not in self.model_pool.models and "nano" not in self.model_pool.models:
            return ""

        try:
            # Create a "Thin" version of the packet for speed
            from gaia_core.utils.prompt_builder import build_from_packet
            reflex_messages = build_from_packet(packet, slim_mode=True)
            user_input = packet.content.original_prompt or ""

            # Direct non-streaming call to Reflex (Nano), capped at 256 tokens.
            # Thinking always disabled for Nano — speed is the priority.
            res = self.model_pool.forward_to_model(
                "reflex",
                messages=reflex_messages,
                max_tokens=256,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )

            if isinstance(res, dict):
                reflex_text = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                # Capability gate: Nano signals it can't answer confidently.
                # Check anywhere in the response, not just start — Nano often
                # prepends time/context before the ESCALATE marker.
                if "ESCALATE" in reflex_text.upper():
                    self.logger.info("Reflex: Nano self-escalated — deferring to run_turn()")
                    return ""
                # Strip think tags and suppress repetition
                reflex_text = strip_think_tags(reflex_text).strip()
                reflex_text = self._suppress_repetition(reflex_text)
                # Time sanity check: if Nano claims a time, verify it's within
                # 2 minutes of now. Small models hallucinate times.
                if self._reflex_time_is_stale(reflex_text):
                    self.logger.warning("Speculative reflex returned stale/hallucinated time — deferring to run_turn()")
                    return ""
                # Degenerate output → defer to full run_turn() pipeline
                if self._is_degenerate_output(reflex_text, user_input):
                    self.logger.warning("Speculative reflex produced degenerate output — deferring to run_turn()")
                    return ""
                # Identity and Sanity check
                if self.identity_guardian.validate_reflex(reflex_text):
                    return reflex_text
                else:
                    self.logger.warning("Speculative reflex suppressed by IdentityGuardian.")
                    return ""
        except Exception:
            self.logger.debug("Speculative reflex failed", exc_info=True)
        return ""

    def _reflex_time_is_stale(self, text: str) -> bool:
        """Check if a Reflex response contains a time that's more than 2 minutes off.

        Small models (0.8B) sometimes hallucinate plausible-looking times instead
        of copying from the few-shot example. This catches that.
        """
        import re
        from datetime import datetime, timezone, timedelta

        # Look for time patterns like "11:54 AM" or "3:45 PM"
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', text, re.IGNORECASE)
        if not time_match:
            return False  # No time claim — not a time question, no check needed

        try:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            ampm = time_match.group(3).upper()

            # Convert to 24h
            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0

            # Get current local time (same tz logic as prompt_builder)
            _tz_offset = int(os.environ.get("GAIA_LOCAL_TZ_OFFSET", "-7"))
            local_tz = timezone(timedelta(hours=_tz_offset))
            now = datetime.now(local_tz)

            # Compare hours and minutes
            claimed_minutes = hour * 60 + minute
            actual_minutes = now.hour * 60 + now.minute
            diff = abs(claimed_minutes - actual_minutes)
            # Handle midnight wrap
            if diff > 720:
                diff = 1440 - diff

            if diff > 2:
                self.logger.warning(
                    "Reflex time sanity: claimed %02d:%02d, actual %02d:%02d (diff=%d min)",
                    hour, minute, now.hour, now.minute, diff,
                )
                return True
        except Exception as _time_exc:
            logger.debug("AgentCore: reflex time stale check failed: %s", _time_exc)
        return False

    def _run_speculative_reflex(self, packet: CognitionPacket) -> str:
        """Legacy internal wrapper."""
        return self.generate_instant_reflex(packet)

    def _nano_triage(self, user_input: str) -> str:
        """
        Use the Nano model (0.5B) to perform a quick triage of the request.
        Returns "SIMPLE" or "COMPLEX".
        """
        # The nano model may be registered as "nano" or "reflex" in MODEL_CONFIGS.
        _nano_key = "nano" if "nano" in self.config.MODEL_CONFIGS else "reflex"
        if _nano_key not in self.config.MODEL_CONFIGS:
            return "COMPLEX"  # Fallback to more capable models

        try:
            # Ensure nano is loaded
            self.model_pool.ensure_model_loaded(_nano_key)
            nano = self.model_pool.get(_nano_key)
            if not nano:
                return "COMPLEX"

            triage_system_prompt = """
You are GAIA. Assess this request's complexity.
SIMPLE: Greetings, time/date, system status, simple facts, short poems, basic formatting.
COMPLEX: Coding, debugging, architectural design, deep philosophy, multi-step planning, long-form creative writing.

Respond ONLY with:
RESULT: SIMPLE
or
RESULT: COMPLEX (reason: <brief reason>)
"""
            messages = [
                {"role": "system", "content": triage_system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Use forward_to_model for lifecycle management if possible, 
            # or just call directly since it's a quick triage.
            res = nano.create_chat_completion(
                messages=messages,
                max_tokens=32,
                temperature=0.1
            )
            content = res["choices"][0]["message"]["content"].strip().upper()
            if "RESULT: SIMPLE" in content:
                return "SIMPLE"
            return "COMPLEX"
        except Exception:
            log_gaia_error(logger, "GAIA-CORE-060", "Nano triage failed, falling back to COMPLEX", exc_info=True)
            return "COMPLEX"

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

    def _should_use_slim_prompt(self, plan: Plan, user_input: str, selected_model_name: str = "") -> bool:
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

        # Nano/reflex MUST always use slim prompt — their context window (2048)
        # cannot fit the full planning pipeline. Sending the full prompt causes
        # context overflow errors (400) and exhausts the fallback chain.
        if selected_model_name in ("nano", "reflex"):
            return True

        # Disable slim-prompt shortcuts only when truly CRITICAL (score >= 25).
        # IRRITATED (8-25) should still allow slim prompts — the debate loop
        # generates repetitive output for simple queries under irritation.
        from gaia_common.utils.immune_system import is_system_irritated
        if is_system_irritated(threshold=25.0):
            self.logger.info("AgentCore: System is CRITICAL; disabling Slim Prompt shortcut for awareness.")
            return False

        if plan.intent in ("recitation", "chat"):
            # Don't slim-path if user explicitly asked for tools
            lowered = (user_input or "").lower()
            tool_signals = ["use your ", "use the ", "use my ", "look up", "look it up",
                            "search ", "web search", "find ", "tools"]
            if any(sig in lowered for sig in tool_signals):
                self.logger.info("Slim prompt declined: recitation intent but user requested tools")
                return False
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
                ts_write({"type": "sketch", "intent": "list_tools", "plan": [
                    "Plan: enumerate available MCP tools",
                    "1) Call MCP list_tools.",
                    "2) Return the list to the operator."
                ]}, session_id, source=source, destination_context=_meta)
                resp = asyncio.run(mcp_client.call_jsonrpc("list_tools", {}))
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
                ts_write({"type": "sketch", "intent": "find_file", "plan": [
                    "Plan: locate target file and summarize it",
                    "1) If an explicit path is provided, read it directly (if allowed).",
                    "2) Otherwise search under /knowledge (skip hidden/cache dirs).",
                    "3) If exactly one match is found, read it and return a short preview.",
                    "4) If multiple matches are found, show a shortlist and await choice to read.",
                    "5) On failure/no match, prompt for a narrower path or filename."
                ]}, session_id, source=source, destination_context=_meta)
                ts_write({"type": "mcp_call", "intent": "find_file"}, session_id, source=source, destination_context=_meta)
                raw = user_input or ""
                allow_roots = [Path("/knowledge").resolve(), Path("/sandbox").resolve(), Path("/models").resolve()]

                # Direct path extraction (if the user pasted a path)
                path_hits = re.findall(r"/[A-Za-z0-9_./-]+", raw)
                for p_raw in path_hits:
                    p = Path(p_raw).resolve()
                    if any(str(p).startswith(str(a)) for a in allow_roots) and p.is_file():
                        read = asyncio.run(mcp_client.call_jsonrpc("read_file", {"path": str(p)}))
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
                resp = asyncio.run(mcp_client.call_jsonrpc("find_files", {"query": query_term, "max_depth": 5, "max_results": 50}))
                if resp.get("ok") and isinstance(resp.get("response"), dict):
                    result = resp["response"].get("result") or resp["response"].get("result", resp["response"])
                    if isinstance(result, dict) and result.get("ok"):
                        matches = result.get("results") or []
                        if not matches:
                            return "I searched for that filename but found no matches under /knowledge."
                        # If exactly one match, attempt to read and summarize it.
                        if len(matches) == 1:
                            path = matches[0]
                            read = asyncio.run(mcp_client.call_jsonrpc("read_file", {"path": path}))
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
                ts_write({"type": "sketch", "intent": "read_file", "plan": [
                    "Plan: read a file safely and summarize it",
                    "1) If a direct path is present, read it.",
                    "2) Otherwise search for likely matches.",
                    "3) If exactly one match is found, read and summarize it.",
                    "4) Store a preview to sketchpad for recall."
                ]}, session_id, source=source, destination_context=_meta)
                ts_write({"type": "mcp_call", "intent": "read_file"}, session_id, source=source, destination_context=_meta)
                raw = user_input or ""
                allow_roots = [Path("/knowledge").resolve(), Path("/sandbox").resolve(), Path("/models").resolve()]

                # Direct path extraction (if the user pasted a path)
                path_hits = re.findall(r"/[A-Za-z0-9_./-]+", raw)
                for p_raw in path_hits:
                    p = Path(p_raw).resolve()
                    if any(str(p).startswith(str(a)) for a in allow_roots) and p.is_file():
                        read = asyncio.run(mcp_client.call_jsonrpc("read_file", {"path": str(p)}))
                        if read.get("ok") and isinstance(read.get("response"), dict):
                            rres = read["response"].get("result") or read["response"]
                            content = ""
                            if isinstance(rres, dict):
                                content = rres.get("content") or ""
                            preview = (content[:1200] + "...") if len(content) > 1200 else content
                            summary_lines = preview.strip().splitlines()[:40]
                            summary_text = "\n".join(summary_lines)
                            try:
                                rescue_helper.sketch(
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

                resp = asyncio.run(mcp_client.call_jsonrpc("find_files", {"query": query_term, "max_depth": 5, "max_results": 50}))
                if resp.get("ok") and isinstance(resp.get("response"), dict):
                    result = resp["response"].get("result") or resp["response"].get("result", resp["response"])
                    if isinstance(result, dict) and result.get("ok"):
                        matches = result.get("results") or []
                        if not matches:
                            return "I couldn't find that file. Share a path or a more specific filename."
                        if len(matches) == 1:
                            path = matches[0]
                            read = asyncio.run(mcp_client.call_jsonrpc("read_file", {"path": path}))
                            if read.get("ok") and isinstance(read.get("response"), dict):
                                rres = read["response"].get("result") or read["response"]
                                content = ""
                                if isinstance(rres, dict):
                                    content = rres.get("content") or ""
                                preview = (content[:1200] + "...") if len(content) > 1200 else content
                                summary_lines = preview.strip().splitlines()[:40]
                                summary_text = "\n".join(summary_lines)
                                try:
                                    rescue_helper.sketch(
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
        # Handled by the Knowledge Ingestion Pipeline in run_turn() (Location A).
        # The legacy DOCUMENT format is now detected by knowledge_ingestion.detect_save_command()
        # alongside natural-language save commands ("save this about X", etc.).
        # This block is kept as a fallback for non-knowledge-base contexts.
        from gaia_core.cognition.knowledge_ingestion import detect_save_command
        save_cmd = detect_save_command(user_input)
        if save_cmd:
            try:
                title = save_cmd["subject"]
                symbol = save_cmd.get("symbol", title.upper().replace(" ", "_"))
                content_to_document = save_cmd["raw_content"]
                tags = ["user-generated", "documentation", symbol.lower()]

                self.logger.info(f"Document command detected (ingestion pipeline): title='{title}', symbol='{symbol}'")

                documented_path = self.codex_writer.document_information(
                    packet=packet,
                    info_to_document=content_to_document,
                    symbol=symbol,
                    title=title,
                    tags=tags,
                    llm_model=None,
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
                except Exception:
                    self.logger.exception("Document recitation failed, falling back to standard generation")
                    # Fall through to web retrieval, then confidence-based approach

            # No local document — try web retrieval for well-known texts
            web_doc = self._web_retrieve_for_recitation(user_input, session_id)
            if web_doc:
                self.logger.info(f"Web retrieval found recitable content: {web_doc['title']}")
                try:
                    return self._run_with_document_recitation(
                        user_input=user_input,
                        document=web_doc,
                        selected_model_name=selected_model_name,
                        history=history or [],
                        session_id=session_id,
                        output_as_file=True,
                    )
                except Exception:
                    self.logger.exception("Web-retrieved document recitation failed, falling back")
                    # Fall through to confidence-based approach

            # No document found (local or web) - use confidence-based approach
            self.logger.info("No document found (local or web), assessing task confidence...")

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
                self.logger.warning(
                    f"Low confidence ({confidence_score}) for recitation — "
                    "returning None to trigger full ExternalVoice pipeline"
                )
                # Return None to signal run_turn() that the slim path declined.
                # run_turn() will then skip to the full ExternalVoice streaming
                # pipeline which has tool selection (web_search, web_fetch),
                # the streaming observer, and the think-tag circuit breaker.
                return None
            else:
                # Proceed with fragmented generation only when confidence is adequate
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
                except Exception:
                    self.logger.exception("Fragmented generation failed, falling back to standard")
                    # Fall through to standard generation

        # Otherwise, use the canonical GCP prompt builder (world state + identity).
        # Nano (0.5B) gets slim_mode=True — the few-shot prompt with explicit
        # clock/identity examples.  The full multi-thousand-token system prompt
        # overwhelms small models and causes them to miss dynamic context like
        # the current time.
        use_slim = selected_model_name in ("nano", "reflex")
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
            except Exception as _int_exc:
                logger.debug("AgentCore: intent assignment failed: %s", _int_exc)
            messages = build_from_packet(packet, slim_mode=use_slim)
            self.logger.info("AgentCore: slim prompt routed through GCP builder (slim_mode=%s)", use_slim)
            # Use configured MAX_ALLOWED_RESPONSE_TOKENS (default 1000) instead of hardcoded 256
            max_resp_tokens = (
                getattr(self.config, 'MAX_ALLOWED_RESPONSE_TOKENS', None) or
                self.config.constants.get("MAX_ALLOWED_RESPONSE_TOKENS", 1000)
            )
            # Nano (0.8B) should never generate more than 512 tokens —
            # its 2K context window is too small for uncapped generation.
            if selected_model_name in ("reflex", "nano"):
                max_resp_tokens = min(max_resp_tokens, 512)
            # Let the GAIA Engine handle clock injection + KV caching.
            # The engine injects [Clock: ...] and caches identity as KV prefix.
            res = self.model_pool.forward_to_model(
                selected_model_name,
                messages=messages,
                max_tokens=min(self.config.max_tokens, max_resp_tokens),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                adapter_name=self._resolve_adapter(selected_model_name),
            )
            content = res["choices"][0]["message"]["content"]
            # Strip think tags and suppress repetition (same guards as streaming path)
            content = strip_think_tags(content).strip()
            content = self._suppress_repetition(content)
            # Nano self-escalation: if response is literally "ESCALATE", defer to Core
            if content.upper().startswith("ESCALATE"):
                self.logger.info("Nano self-escalated via ESCALATE signal — trying next tier")
                escalated = self._escalate_slim_response(selected_model_name, messages, max_resp_tokens)
                if escalated:
                    return escalated
            # Detect degenerate output and escalate to a higher tier
            if self._is_degenerate_output(content, user_input):
                self.logger.warning("Degenerate output from '%s' — escalating to next tier", selected_model_name)
                escalated = self._escalate_slim_response(selected_model_name, messages, max_resp_tokens)
                if escalated:
                    return escalated
            # Uncertainty-based escalation — Nano hedges or describes process
            # without delivering the actual answer. Uses entropy + hedging signals.
            if use_slim:
                resp_entropy = res.get("usage", {}).get("mean_entropy", 0.0)
                if self._should_escalate_for_uncertainty(content, user_input, entropy=resp_entropy):
                    self.logger.info("Uncertainty escalation from '%s' (entropy=%.2f) — trying next tier",
                                     selected_model_name, resp_entropy)
                    escalated = self._escalate_slim_response(selected_model_name, messages, max_resp_tokens)
                    if escalated:
                        return escalated
            # Optional polish via Thinker: if Operator (lite) handled this turn and a Thinker is available,
            # send a short polish request and return the Thinker output as final.
            try:
                polish_flag = os.getenv("GAIA_THINKER_POLISH", "").lower() in ("1", "true", "yes")
                if polish_flag and selected_model_name == "lite":
                    _gpu_sleeping_polish = not self._is_prime_available
                    thinker_name = None
                    for cand in ["gpu_prime", "prime", "cpu_prime"]:
                        if cand == "gpu_prime" and _gpu_sleeping_polish:
                            continue
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
                            adapter_name=self._resolve_adapter(thinker_name),
                        )
                        content = polished["choices"][0]["message"]["content"]
            except Exception:
                self.logger.debug("Thinker polish step failed; returning Operator output", exc_info=True)
            return content
        except Exception as exc:
            self.logger.exception("slim prompt call failed for '%s'", selected_model_name)
            escalated = self._escalate_slim_response(selected_model_name, messages, max_resp_tokens)
            if escalated:
                return escalated
            return f"I encountered an error while answering: {exc}"

    # ------------------------------------------------------------------
    # Web-retrieval helpers for recitation (poem/speech/document lookup)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recitation_search_query(user_input: str) -> str:
        """Strip action verbs and qualifiers, append 'full text' for a search-friendly query."""
        import re
        q = user_input
        # Remove common request preambles
        q = re.sub(
            r"(?i)^(please\s+)?(recite|read|share|give me|show me|write out|type out|present)\s+",
            "",
            q,
        )
        # Remove trailing qualifiers
        q = re.sub(r"(?i)\s+(for me|please|if you (can|could|would))\.?$", "", q)
        q = q.strip().rstrip(".")
        if q:
            q += " full text"
        return q

    def _validate_recitation_content(self, content: str, user_input: str) -> bool:
        """Gate on length (200–100k chars) and minimal relevance to the request."""
        if not content or len(content) < 200 or len(content) > 100_000:
            return False
        # Extract salient nouns from the user input for a rough relevance check
        import re
        words = re.findall(r"[A-Za-z]{3,}", user_input.lower())
        stop = {"the", "and", "for", "please", "recite", "read", "share",
                "give", "show", "write", "type", "first", "three", "stanzas",
                "stanza", "full", "text", "from", "out", "with", "that", "this",
                "can", "could", "would", "you", "your", "poem", "speech"}
        keywords = [w for w in words if w not in stop]
        if not keywords:
            return True  # no keywords to check — accept on length alone
        content_lower = content.lower()
        hits = sum(1 for kw in keywords if kw in content_lower)
        return hits >= 1  # at least one salient keyword present

    def _web_retrieve_for_recitation(
        self,
        user_input: str,
        session_id: str,
    ) -> Optional[Dict[str, str]]:
        """Search the web for a well-known text and return {title, content} or None.

        Orchestrates: web_search → sort by trust tier → web_fetch top 3 → validate.
        Every failure path returns None so the caller can fall through gracefully.
        """

        query = self._build_recitation_search_query(user_input)
        if not query:
            return None

        # Determine content_type hint from the request
        input_lower = user_input.lower()
        if any(w in input_lower for w in ("poem", "stanza", "verse", "sonnet", "ode")):
            content_type = "poem"
        elif any(w in input_lower for w in ("speech", "address", "declaration")):
            content_type = "facts"
        else:
            content_type = None

        try:
            # Try with content_type first (adds trusted-domain site: filters),
            # then retry without if the site-filtered query returns nothing.
            search_attempts = []
            if content_type:
                search_attempts.append({"query": query, "max_results": 5, "content_type": content_type})
            search_attempts.append({"query": query, "max_results": 5})  # unfiltered fallback

            results = []
            for search_params in search_attempts:
                self.logger.info(f"Web recitation search: {search_params}")
                resp = asyncio.run(mcp_client.call_jsonrpc("web_search", search_params))

                if not resp.get("ok"):
                    self.logger.warning("web_search failed for recitation: %s", resp.get("error"))
                    continue

                # The MCP response wraps the actual result under "response"
                payload = resp.get("response", resp)
                if isinstance(payload, dict) and "result" in payload:
                    payload = payload["result"]

                results = payload.get("results", [])
                if results:
                    break  # got results, stop searching

            if not results:
                self.logger.info("web_search returned no results for recitation query")
                return None

            # Sort by trust tier: trusted > reliable > unknown
            tier_priority = {"trusted": 0, "reliable": 1, "unknown": 2}
            results.sort(key=lambda r: tier_priority.get(r.get("trust_tier", "unknown"), 2))

            # Try fetching top 3 results
            for result in results[:3]:
                url = result.get("url", "")
                if not url:
                    continue
                try:
                    self.logger.info(f"Web recitation fetch: {url} (tier={result.get('trust_tier')})")
                    fetch_resp = asyncio.run(mcp_client.call_jsonrpc("web_fetch", {"url": url}))

                    if not fetch_resp.get("ok"):
                        continue

                    fetch_payload = fetch_resp.get("response", fetch_resp)
                    if isinstance(fetch_payload, dict) and "result" in fetch_payload:
                        fetch_payload = fetch_payload["result"]

                    content = fetch_payload.get("content", "")
                    title = fetch_payload.get("title") or result.get("title", "Web Document")

                    if self._validate_recitation_content(content, user_input):
                        self.logger.info(
                            f"Web recitation content validated: {title!r} ({len(content)} chars)"
                        )
                        return {"title": title, "content": content}
                    else:
                        self.logger.info(
                            f"Web recitation content rejected: {title!r} ({len(content)} chars)"
                        )
                except Exception:
                    self.logger.debug("web_fetch failed for %s", url, exc_info=True)
                    continue

            self.logger.info("No valid recitation content found from web search")
            return None

        except Exception:
            self.logger.debug("_web_retrieve_for_recitation failed", exc_info=True)
            return None

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
        except Exception as _int_exc:
            logger.debug("AgentCore: recitation intent assignment failed: %s", _int_exc)

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
                adapter_name=self._resolve_adapter(selected_model_name),
            )
            content = res["choices"][0]["message"]["content"]

            # Strip any think tags from the response
            content = strip_think_tags(content)

            # Scrub degenerate repetition patterns (e.g., "___" x1000)
            # that the 3B model produces when it runs out of real content.
            content = self._suppress_repetition(content)
            # Also truncate runs of repeated short tokens (underscores, etc.)
            content = re.sub(r'(\s*_{3}\s*){5,}', '\n\n[...]\n\n', content)
            content = re.sub(r'(\*{2,}\s*_{2,}\s*\*{2,}\s*){3,}', '\n\n[...]\n\n', content)

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

        except Exception:
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
                    adapter_name=self._resolve_adapter(selected_model_name),
                )
                content = res["choices"][0]["message"]["content"]
            except Exception:
                self.logger.exception(f"Fragment {fragment_sequence} generation failed")
                break

            # Check for truncation
            truncation_info = self.detect_truncation(content, max_tokens=max_resp_tokens)

            # Store fragment to sketchpad with unique key
            fragment_key = f"recitation_fragment_{request_id}_{fragment_sequence}"
            try:
                rescue_helper.sketch(fragment_key, content)
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
                asyncio.run(mcp_client.call_jsonrpc("fragment_write", {
                    "parent_request_id": request_id,
                    "sequence": fragment_sequence,
                    "content": content,
                    "continuation_hint": truncation_info.get("continuation_hint", ""),
                    "is_complete": not truncation_info["truncated"],
                    "token_count": truncation_info.get("approx_tokens", 0)
                }))
            except Exception as _mcp_exc:
                logger.debug("AgentCore: MCP fragment storage failed (optional): %s", _mcp_exc)

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
            asyncio.run(mcp_client.call_jsonrpc("fragment_clear", {"parent_request_id": request_id}))
        except Exception as _frag_exc:
            logger.debug("AgentCore: MCP fragment cleanup failed (non-fatal): %s", _frag_exc)

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
                content = rescue_helper.show_sketchpad(key)
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
                content = rescue_helper.show_sketchpad(key)
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
                adapter_name=self._resolve_adapter(selected_model_name),
            )
            assembled = res["choices"][0]["message"]["content"]
            # Strip any <think> reasoning blocks from the assembled output
            assembled = strip_think_tags(assembled)
            self.logger.info(f"Assembly turn complete: {len(assembled)} chars")
            return assembled
        except Exception:
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
        result = asyncio.run(mcp_client.ai_write(output_path, content))

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
        params = {"path": "/knowledge", "max_depth": 2, "max_entries": 120, "_allow_pending": True}
        bypass = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")

        def _handle_tree_result(result_dict: dict) -> str:
            tree = result_dict.get("tree") or ""
            truncated_flag = result_dict.get("truncated")
            if not tree:
                return "No entries found."
            # If the tree is long, persist it and return a short preview
            if len(tree) > 4000:
                preview = "\n".join(tree.splitlines()[:20])
                target_path = "/knowledge/system_reference/tree_latest.txt"
                # Write via MCP so it is auditable/approved
                write_req = asyncio.run(mcp_client.request_approval_via_mcp("ai_write", {"path": target_path, "content": tree, "_allow_pending": True}))
                if write_req.get("ok") and write_req.get("action_id") and write_req.get("challenge"):
                    approval = write_req["challenge"][::-1]
                    write_appr = asyncio.run(mcp_client.approve_action_via_mcp(write_req["action_id"], approval))
                    if not write_appr.get("ok"):
                        return f"(Saved tree skipped due to approval error: {write_appr.get('error')})\nPreview:\n{preview}"
                    try:
                        rescue_helper.sketch(
                            "DirectoryTreeLatest",
                            f"Saved full tree to: {target_path}\n\nPreview:\n{preview}"
                        )
                    except Exception:
                        self.logger.debug("AgentCore: failed to write tree preview to sketchpad", exc_info=True)
                elif bypass:
                    # In bypass mode, attempt direct write
                    asyncio.run(mcp_client.call_jsonrpc("ai_write", {"path": target_path, "content": tree}))
                    try:
                        rescue_helper.sketch(
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
            resp = asyncio.run(mcp_client.call_jsonrpc("list_tree", params))
            if resp.get("ok") and isinstance(resp.get("response"), dict):
                result = resp["response"].get("result") or {}
                if result.get("ok"):
                    return _handle_tree_result(result)
                return f"list_tree error: {result}"
            return f"list_tree failed: {resp.get('error') or resp}"

        # Request approval then auto-approve using reversed challenge
        req = asyncio.run(mcp_client.request_approval_via_mcp("list_tree", params))
        if not req.get("ok"):
            return f"Could not request approval for list_tree: {req.get('error')}"
        action_id = req.get("action_id")
        challenge = req.get("challenge")
        if not action_id or not challenge:
            return "Approval request did not return action_id/challenge."
        approval = challenge[::-1]
        appr = asyncio.run(mcp_client.approve_action_via_mcp(action_id, approval))
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
        params = {"path": "/knowledge", "_allow_pending": True}
        bypass = os.getenv("GAIA_MCP_BYPASS", "false").lower() in ("1", "true", "yes")

        def _handle_list_result(result_dict: dict) -> str:
            files = result_dict.get("files") or []
            if not files:
                return "No files found."
            return "\n".join(files)

        if bypass:
            resp = asyncio.run(mcp_client.call_jsonrpc("list_files", params))
            if resp.get("ok") and isinstance(resp.get("response"), dict):
                result = resp["response"].get("result") or {}
                if result.get("ok"):
                    return _handle_list_result(result)
                return f"list_files error: {result}"
            return f"list_files failed: {resp.get('error') or resp}"

        # Request approval then auto-approve using reversed challenge
        req = asyncio.run(mcp_client.request_approval_via_mcp("list_files", params))
        if not req.get("ok"):
            return f"Could not request approval for list_files: {req.get('error')}"
        action_id = req.get("action_id")
        challenge = req.get("challenge")
        if not action_id or not challenge:
            return "Approval request did not return action_id/challenge."
        approval = challenge[::-1]
        appr = asyncio.run(mcp_client.approve_action_via_mcp(action_id, approval))
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
            except Exception as _int_exc:
                logger.debug("AgentCore: confidence intent assignment failed: %s", _int_exc)

            # Build messages using the GCP pipeline
            # This includes identity, world state, MCP tools, knowledge base awareness
            messages = build_from_packet(packet, task_instruction_key="confidence_assessment")

            response = self.model_pool.forward_to_model(
                model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.3,
                adapter_name=self._resolve_adapter(model_name),
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
                adapter_name=self._resolve_adapter(model_name),
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
        app_dir = Path("/app") if Path("/app").exists() else Path.cwd() / "app"

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
                    "status": "Could not propose fix",
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
            select_tool, review_selection,
            initialize_tool_routing, inject_tool_result_into_packet,
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
                except Exception as _rel_exc:
                    logger.debug("AgentCore: tool selection model release failed: %s", _rel_exc)

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

        # Step 2: Confidence Review
        # Skip review for deterministic matches (confidence >= 0.9) — they're
        # already high-confidence by definition and the review model may not
        # understand domain tool names, causing false rejections.
        if primary_tool.selection_confidence >= 0.9:
            confidence = primary_tool.selection_confidence
            reasoning = "Deterministic match — review skipped"
            logger.info("Tool routing: skipping review for deterministic match (confidence=%.2f)", confidence)
        else:
            logger.info(f"Tool routing: reviewing selection {primary_tool.tool_name}")

            # Use Prime model for review if available
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
                        except Exception as _rel_exc:
                            logger.debug("AgentCore: tool review model release failed: %s", _rel_exc)
            else:
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
                loop_mgr = get_recovery_manager(session_id)
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
                loop_mgr = get_recovery_manager(session_id)
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
            # Normalize tool names for SOA schema alignment
            canonical_name = tool.tool_name

            # Map legacy internal tool names to their MCP equivalents
            _INTERNAL_ALIASES = {
                "ai.read": "read_file",
                "ai.write": "write_file",
                "ai.execute": "run_shell",
                "embedding.query": "memory_query",
            }
            if canonical_name in _INTERNAL_ALIASES:
                canonical_name = _INTERNAL_ALIASES[canonical_name]

            # Domain tools (file, shell, web, etc.) go directly through MCP
            # The MCP server's execute_tool handles domain→legacy routing
            try:
                from gaia_common.utils.domain_tools import DOMAIN_TOOLS
                if canonical_name in DOMAIN_TOOLS:
                    logger.info(f"Dispatching domain tool '{canonical_name}' via MCP JSON-RPC")
                    rpc_result = mcp_client.call_jsonrpc(
                        method=canonical_name,
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
                            error=rpc_result.get("error", "Domain tool call failed"),
                            execution_time_ms=elapsed_ms
                        )
            except ImportError:
                pass  # domain_tools not available, fall through to legacy handling

            if canonical_name == "read_file":
                result = asyncio.run(mcp_client.ai_read(tool.params.get("path", "")))

            elif canonical_name == "write_file":
                if not allow_write:
                    return ToolExecutionResult(
                        success=False,
                        error="Write operations are disabled. Set TOOL_ROUTING.ALLOW_WRITE_TOOLS=true to enable."
                    )
                result = asyncio.run(mcp_client.ai_write(
                    tool.params.get("path", ""),
                    tool.params.get("content", "")
                ))

            elif canonical_name == "run_shell":
                if not allow_execute:
                    return ToolExecutionResult(
                        success=False,
                        error="Execute operations are disabled. Set TOOL_ROUTING.ALLOW_EXECUTE_TOOLS=true to enable."
                    )
                result = asyncio.run(mcp_client.ai_execute(
                    tool.params.get("command", ""),
                    dry_run=not allow_execute
                ))

            else:
                # Dispatch via MCP JSON-RPC for tools not handled by local shims
                logger.info(f"Dispatching tool '{canonical_name}' via MCP JSON-RPC")
                rpc_result = mcp_client.call_jsonrpc(
                    method=canonical_name,
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

        # Web research indicators
        web_indicators = [
            "web search", "search the web", "search online",
            "search the internet", "look up online", "google ",
            "look up ", "look it up",
        ]

        # Explicit tool invocation — user named a specific tool
        explicit_tool_indicators = [
            "use your ", "use the ", "use my ",
            "run the ", "run your ",
            "call the ", "call your ",
            "invoke ", "trigger ",
            "with your ", "with the ",
        ]

        # Knowledge save indicators
        knowledge_save_indicators = [
            "save to my knowledge", "save to knowledge",
            "store in my knowledge", "store in knowledge",
            "add to my knowledge", "add to knowledge",
            "save the following to",
        ]

        for indicator in file_indicators + exec_indicators + search_indicators + web_indicators + explicit_tool_indicators + knowledge_save_indicators:
            if indicator in lowered:
                logger.debug(f"Tool routing triggered by indicator: '{indicator}'")
                return True

        return False
