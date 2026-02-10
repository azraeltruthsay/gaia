"""Tests for packet utility functions."""

import pytest
from gaia_common.utils.packet_utils import is_execution_safe
from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    Header,
    Persona,
    PersonaRole,
    Origin,
    Routing,
    TargetEngine,
    Model,
    Intent,
    SystemTask,
    Context,
    SessionHistoryRef,
    Constraints,
    Content,
    Reasoning,
    Response,
    Governance,
    Safety,
    Metrics,
    TokenUsage,
    Status,
    PacketState,
    SidecarAction,
)


def _make_packet(execution_allowed=False, whitelist_id=None, sidecar_actions=None):
    """Helper to build a packet with specific safety settings."""
    return CognitionPacket(
        version="0.3",
        header=Header(
            datetime="2026-01-01T00:00:00",
            session_id="s1",
            packet_id="p1",
            sub_id="0",
            persona=Persona(identity_id="G", persona_id="d", role=PersonaRole.DEFAULT),
            origin=Origin.USER,
            routing=Routing(target_engine=TargetEngine.PRIME),
            model=Model(name="m", provider="p", context_window_tokens=4096),
        ),
        intent=Intent(user_intent="t", system_task=SystemTask.GENERATE_DRAFT, confidence=1.0),
        context=Context(
            session_history_ref=SessionHistoryRef(type="t", value="v"),
            cheatsheets=[],
            constraints=Constraints(max_tokens=512, time_budget_ms=5000, safety_mode="strict"),
        ),
        content=Content(original_prompt="test"),
        reasoning=Reasoning(),
        response=Response(
            candidate="ok",
            confidence=1.0,
            stream_proposal=False,
            sidecar_actions=sidecar_actions or [],
        ),
        governance=Governance(
            safety=Safety(
                execution_allowed=execution_allowed,
                allowed_commands_whitelist_id=whitelist_id,
            )
        ),
        metrics=Metrics(
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            latency_ms=0,
        ),
        status=Status(finalized=False, state=PacketState.INITIALIZED, next_steps=[]),
    )


class TestIsExecutionSafe:
    def test_no_sidecar_actions_always_safe(self):
        pkt = _make_packet(execution_allowed=False, whitelist_id=None, sidecar_actions=[])
        assert is_execution_safe(pkt) is True

    def test_allowed_with_whitelist_is_safe(self):
        pkt = _make_packet(
            execution_allowed=True,
            whitelist_id="safe_tools_v1",
            sidecar_actions=[SidecarAction(action_type="run")],
        )
        assert is_execution_safe(pkt) is True

    def test_allowed_without_whitelist_is_unsafe(self):
        pkt = _make_packet(
            execution_allowed=True,
            whitelist_id=None,
            sidecar_actions=[SidecarAction(action_type="run")],
        )
        assert is_execution_safe(pkt) is False

    def test_not_allowed_with_whitelist_is_unsafe(self):
        pkt = _make_packet(
            execution_allowed=False,
            whitelist_id="safe_tools_v1",
            sidecar_actions=[SidecarAction(action_type="run")],
        )
        assert is_execution_safe(pkt) is False
