"""
Canonical CognitionPacket construction.

Every service that needs to build a CognitionPacket should use ``build_packet``
instead of hand-assembling the 30+ nested dataclasses.  Source-specific defaults
(destination, engine, constraints, etc.) are encoded in ``_SOURCE_DEFAULTS`` so
callers only provide the fields they actually know.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from gaia_common.protocols.cognition_packet import (
    Constraints,
    CognitionPacket,
    Content,
    Context,
    DataField,
    DestinationTarget,
    Governance,
    Header,
    Intent,
    Metrics,
    Model,
    OperationalStatus,
    Origin,
    OutputDestination,
    OutputRouting,
    PacketState,
    Persona,
    PersonaRole,
    Reasoning,
    Response,
    Routing,
    Safety,
    SessionHistoryRef,
    Status,
    SystemTask,
    TargetEngine,
    TokenUsage,
    ToolRoutingState,
)


# ---------------------------------------------------------------------------
# Public enum: callers pick one of these to get correct defaults
# ---------------------------------------------------------------------------

class PacketSource(Enum):
    WEB = "web"
    DISCORD = "discord"
    AUDIO_GATEWAY = "audio_gateway"
    VOICE_PRIME = "voice_prime"
    VOICE_LITE = "voice_lite"


# ---------------------------------------------------------------------------
# Source-specific defaults (mirrors the table in the plan)
# ---------------------------------------------------------------------------

_SOURCE_DEFAULTS: Dict[PacketSource, Dict[str, Any]] = {
    PacketSource.WEB: {
        "destination": OutputDestination.WEB,
        "target_engine": TargetEngine.PRIME,
        "priority": 5,
        "tone_hint": "neutral",
        "max_tokens": 2048,
        "time_budget_ms": 30000,
        "safety_mode": "strict",
        "dry_run": True,
        "model_name": "default_model",
        "model_provider": "default_provider",
        "context_window": 8192,
        "system_task": SystemTask.GENERATE_DRAFT,
        "initial_state": PacketState.INITIALIZED,
        "session_history_type": "web_session",
    },
    PacketSource.DISCORD: {
        "destination": OutputDestination.DISCORD,
        "target_engine": TargetEngine.PRIME,
        "priority": 5,
        "tone_hint": "conversational",
        "max_tokens": 2048,
        "time_budget_ms": 30000,
        "safety_mode": "strict",
        "dry_run": True,
        "model_name": "default_model",
        "model_provider": "default_provider",
        "context_window": 8192,
        "system_task": SystemTask.GENERATE_DRAFT,
        "initial_state": PacketState.INITIALIZED,
        "session_history_type": "discord_channel",
    },
    PacketSource.AUDIO_GATEWAY: {
        "destination": OutputDestination.AUDIO,
        "target_engine": TargetEngine.PRIME,
        "priority": 5,
        "tone_hint": None,
        "max_tokens": 2048,
        "time_budget_ms": 30000,
        "safety_mode": "standard",
        "dry_run": False,
        "model_name": "auto",
        "model_provider": "auto",
        "context_window": 8192,
        "system_task": SystemTask.GENERATE_DRAFT,
        "initial_state": PacketState.INITIALIZED,
        "session_history_type": "ref",
    },
    PacketSource.VOICE_PRIME: {
        "destination": OutputDestination.AUDIO,
        "target_engine": TargetEngine.PRIME,
        "priority": 5,
        "tone_hint": "conversational",
        "max_tokens": 512,
        "time_budget_ms": 15000,
        "safety_mode": "strict",
        "dry_run": True,
        "model_name": "default_model",
        "model_provider": "default_provider",
        "context_window": 8192,
        "system_task": SystemTask.GENERATE_DRAFT,
        "initial_state": PacketState.INITIALIZED,
        "session_history_type": "discord_voice",
    },
    PacketSource.VOICE_LITE: {
        "destination": OutputDestination.AUDIO,
        "target_engine": TargetEngine.LITE,
        "priority": 8,
        "tone_hint": "conversational",
        "max_tokens": 128,
        "time_budget_ms": 5000,
        "safety_mode": "strict",
        "dry_run": True,
        "model_name": "lite",
        "model_provider": "local",
        "context_window": 4096,
        "system_task": SystemTask.GENERATE_DRAFT,
        "initial_state": PacketState.INITIALIZED,
        "session_history_type": "discord_voice",
    },
}


# ---------------------------------------------------------------------------
# Sentinel for distinguishing "caller passed None" from "caller omitted arg"
# ---------------------------------------------------------------------------
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Session-ID helpers (source-specific patterns)
# ---------------------------------------------------------------------------

def _default_session_id(
    source: PacketSource,
    *,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    is_dm: bool = False,
) -> str:
    if source == PacketSource.WEB:
        return "web_ui_session"
    if source == PacketSource.DISCORD:
        if is_dm:
            return f"discord_dm_{user_id or 'unknown'}"
        return f"discord_channel_{channel_id or 'unknown'}"
    if source in (PacketSource.VOICE_PRIME, PacketSource.VOICE_LITE):
        return f"discord_voice_{user_id or 'voice_user'}"
    if source == PacketSource.AUDIO_GATEWAY:
        return "voice_session"
    return "session"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_packet(
    source: PacketSource,
    user_input: str,
    *,
    # Identity / routing
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    # Discord-specific
    reply_to_message_id: Optional[str] = None,
    is_dm: bool = False,
    author_name: Optional[str] = None,
    # Overrides (any source)
    target_engine: Optional[TargetEngine] = None,
    priority: Optional[int] = None,
    max_tokens: Optional[int] = None,
    time_budget_ms: Optional[int] = None,
    safety_mode: Optional[str] = None,
    tone_hint: Optional[str] = _SENTINEL,  # type: ignore[assignment]
    dry_run: Optional[bool] = None,
    extra_data_fields: Optional[List[DataField]] = None,
    system_hint: Optional[str] = None,
    compute_hashes: bool = True,
) -> CognitionPacket:
    """Build a fully-formed CognitionPacket for *source* with sane defaults.

    Explicit keyword arguments override the per-source defaults.  Fields that
    are never meaningful at construction time (e.g. ``response.candidate``,
    ``metrics.latency_ms``) are zero-valued.
    """
    d = _SOURCE_DEFAULTS[source]

    # Resolve overrides (explicit kwarg > source default)
    _target_engine = target_engine or d["target_engine"]
    _priority = priority if priority is not None else d["priority"]
    _max_tokens = max_tokens if max_tokens is not None else d["max_tokens"]
    _time_budget = time_budget_ms if time_budget_ms is not None else d["time_budget_ms"]
    _safety = safety_mode or d["safety_mode"]
    _tone = d["tone_hint"] if tone_hint is _SENTINEL else tone_hint
    _dry_run = dry_run if dry_run is not None else d["dry_run"]

    _session_id = session_id or _default_session_id(
        source, user_id=user_id, channel_id=channel_id, is_dm=is_dm,
    )

    now = datetime.now().isoformat()
    packet_id = str(uuid.uuid4())

    # -- Destination metadata --
    dest_meta: Dict[str, Any] = {}
    if source == PacketSource.DISCORD:
        if is_dm is not None:
            dest_meta["is_dm"] = is_dm
        if author_name:
            dest_meta["author_name"] = author_name
    elif source in (PacketSource.VOICE_PRIME, PacketSource.VOICE_LITE):
        dest_meta["source"] = "discord_voice"
        dest_meta["user"] = user_id or "voice_user"

    # -- Data fields --
    data_fields: List[DataField] = [
        DataField(key="user_message", value=user_input, type="text"),
    ]
    if system_hint:
        data_fields.append(
            DataField(key="system_hint", value=system_hint, type="text"),
        )
    if extra_data_fields:
        data_fields.extend(extra_data_fields)

    # -- Build packet --
    packet = CognitionPacket(
        version="0.3",
        header=Header(
            datetime=now,
            session_id=_session_id,
            packet_id=packet_id,
            sub_id="0",
            persona=Persona(
                identity_id="default_user",
                persona_id="default_persona" if source != PacketSource.WEB else "default_web_user",
                role=PersonaRole.DEFAULT,
                tone_hint=_tone,
            ),
            origin=Origin.USER,
            routing=Routing(
                target_engine=_target_engine,
                priority=_priority,
            ),
            model=Model(
                name=d["model_name"],
                provider=d["model_provider"],
                context_window_tokens=d["context_window"],
            ),
            output_routing=OutputRouting(
                primary=DestinationTarget(
                    destination=d["destination"],
                    channel_id=channel_id or _session_id,
                    user_id=user_id or ("web_user" if source == PacketSource.WEB else None),
                    reply_to_message_id=reply_to_message_id,
                    metadata=dest_meta,
                ),
                source_destination=d["destination"],
                addressed_to_gaia=True,
            ),
            operational_status=OperationalStatus(status="initialized"),
        ),
        intent=Intent(
            user_intent="chat",
            system_task=d["system_task"],
            confidence=0.0,
        ),
        context=Context(
            session_history_ref=SessionHistoryRef(
                type=d["session_history_type"],
                value=_session_id,
            ),
            cheatsheets=[],
            constraints=Constraints(
                max_tokens=_max_tokens,
                time_budget_ms=_time_budget,
                safety_mode=_safety,
            ),
        ),
        content=Content(
            original_prompt=user_input,
            data_fields=data_fields,
        ),
        reasoning=Reasoning(),
        response=Response(candidate="", confidence=0.0, stream_proposal=False),
        governance=Governance(
            safety=Safety(execution_allowed=False, dry_run=_dry_run),
        ),
        metrics=Metrics(
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            latency_ms=0,
        ),
        status=Status(
            finalized=False,
            state=d["initial_state"],
            next_steps=[],
        ),
        tool_routing=ToolRoutingState(),
    )

    if compute_hashes:
        packet.compute_hashes()

    return packet
