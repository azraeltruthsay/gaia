"""
Packet utility functions for GAIA services.

This module provides utility functions for working with CognitionPackets:
- Safety checking for execution gating
- Version upgrade/migration helpers
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from gaia_common.protocols import CognitionPacket

# Tools that can execute without human approval (read-only, memory, fragments)
SAFE_SIDECAR_TOOLS = {
    "read_file", "list_dir", "list_files", "list_tree", "find_files",
    "find_relevant_documents", "world_state", "memory_status", "memory_query",
    "query_knowledge", "embed_documents", "add_document",
    "fragment_write", "fragment_read", "fragment_assemble",
    "fragment_list_pending", "fragment_clear",
}


def is_execution_safe(packet: "CognitionPacket") -> bool:
    """
    Check if a packet is approved for action execution.

    Uses a tiered approach:
    - If governance explicitly allows execution (whitelist set), all actions pass.
    - Otherwise, only actions in SAFE_SIDECAR_TOOLS are allowed through.
    - Sensitive tools (write_file, run_shell, etc.) are routed to MCP approval.

    Args:
        packet: The CognitionPacket to check

    Returns:
        True if execution is safe, False otherwise
    """
    if not packet.response.sidecar_actions:
        return True

    safety = packet.governance.safety
    # If governance explicitly allows, pass everything
    if safety.execution_allowed and safety.allowed_commands_whitelist_id is not None:
        return True

    # Otherwise, allow only if ALL actions are non-sensitive
    return all(
        action.action_type in SAFE_SIDECAR_TOOLS
        for action in packet.response.sidecar_actions
    )


def upgrade_v2_to_v3_packet(old_packet_data: Dict[str, Any]) -> "CognitionPacket":
    """
    Convert a legacy flat packet dictionary into the v0.3 CognitionPacket structure.

    This is a transitional shim for migrating from v2 flat packets to the new
    nested v0.3 structure. The conversion is intentionally lossy, focusing on
    preserving prompt, history, and identity/persona information required for
    prompt building and routing.

    Args:
        old_packet_data: Legacy v2 packet dictionary

    Returns:
        CognitionPacket v0.3 instance

    Raises:
        RuntimeError: If unable to import required classes
    """
    # Import here to avoid circular imports at module level
    from gaia_common.protocols import (
        CognitionPacket, Header, Persona, Routing, Model, Intent, Context,
        SessionHistoryRef, Cheatsheet, Constraints, Content, DataField,
        Reasoning, Response, Governance, Safety, Metrics, TokenUsage, Status,
        PacketState, PersonaRole, Origin, TargetEngine, SystemTask
    )

    # Helper to safely get fields from old dict
    def g(key: str, default: Any = None) -> Any:
        return old_packet_data.get(key, default)

    # Header / Persona
    datetime_str = g("time_date") or datetime.now(timezone.utc).isoformat()
    session_id = g("session_id") or g("session") or "legacy-session"
    packet_id = g("packet_id") or "legacy-pkt"
    sub_id = g("sub_packet_id") or ""

    persona_id = str(g("persona") or g("persona_id") or "legacy_persona")
    identity_id = str(g("identity") or "legacy_identity")
    persona_role = PersonaRole.OTHER

    persona = Persona(identity_id=identity_id, persona_id=persona_id, role=persona_role)
    routing = Routing(target_engine=TargetEngine.PRIME)

    model_name = g("model") or g("selected_model") or "prime"
    model = Model(
        name=str(model_name),
        provider=g("model_provider") or "local",
        context_window_tokens=int(g("context_window_tokens", 4096))
    )

    header = Header(
        datetime=datetime_str,
        session_id=session_id,
        packet_id=packet_id,
        sub_id=sub_id,
        persona=persona,
        origin=Origin.USER,
        routing=routing,
        model=model
    )

    # Intent
    user_intent = g("intent") or g("intent_name") or ""
    system_task = SystemTask.GENERATE_DRAFT
    intent_conf = float(g("intent_confidence", 0.0) or 0.0)
    intent = Intent(user_intent=user_intent, system_task=system_task, confidence=intent_conf)

    # Context: minimal mapping
    session_ref = SessionHistoryRef(type="session", value=session_id)
    cheatsheets: list = []
    constraints = Constraints(
        max_tokens=int(g("max_tokens", 4096)),
        time_budget_ms=int(g("time_budget_ms", 60000)),
        safety_mode=str(g("safety_mode", "standard"))
    )
    context = Context(session_history_ref=session_ref, cheatsheets=cheatsheets, constraints=constraints)

    # Content
    original_prompt = g("prompt") or g("original_prompt") or g("user_input") or ""
    data_fields = []
    df_map = g("data_fields") or g("data") or {}
    if isinstance(df_map, dict):
        for k, v in df_map.items():
            data_fields.append(DataField(key=str(k), value=v))

    content = Content(original_prompt=original_prompt, data_fields=data_fields)

    # Reasoning / Response / Governance / Metrics / Status - minimal defaults
    reasoning = Reasoning()
    response = Response(
        candidate=str(g("response", "")),
        confidence=float(g("response_confidence", 0.0) or 0.0),
        stream_proposal=False
    )
    safety = Safety(execution_allowed=False, allowed_commands_whitelist_id=None, dry_run=True)
    governance = Governance(safety=safety)
    token_usage = TokenUsage(
        prompt_tokens=int(g("prompt_tokens", 0)),
        completion_tokens=int(g("completion_tokens", 0)),
        total_tokens=int(g("total_tokens", 0))
    )
    metrics = Metrics(token_usage=token_usage, latency_ms=int(g("latency_ms", 0)))
    status = Status(finalized=False, state=PacketState.INITIALIZED)

    pkt = CognitionPacket(
        version=str(g("version", "v0.3-upgraded")),
        header=header,
        intent=intent,
        context=context,
        content=content,
        reasoning=reasoning,
        response=response,
        governance=governance,
        metrics=metrics,
        status=status,
        schema_id=g("schema_id") or None,
        council=None,
    )

    return pkt


__all__ = [
    "SAFE_SIDECAR_TOOLS",
    "is_execution_safe",
    "upgrade_v2_to_v3_packet",
]
