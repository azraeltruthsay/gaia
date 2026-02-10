"""Tests for the CognitionPacket protocol and related dataclasses."""

import json
import pytest

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
    DataField,
    Reasoning,
    Response,
    Governance,
    Safety,
    Metrics,
    TokenUsage,
    Status,
    PacketState,
    OutputDestination,
    DestinationTarget,
    OutputRouting,
)


def _make_minimal_packet(**overrides) -> CognitionPacket:
    """Create a minimal valid CognitionPacket for testing."""
    defaults = dict(
        version="0.3",
        header=Header(
            datetime="2026-02-07T00:00:00",
            session_id="test-session",
            packet_id="pkt-001",
            sub_id="0",
            persona=Persona(
                identity_id="GAIA",
                persona_id="default",
                role=PersonaRole.DEFAULT,
            ),
            origin=Origin.USER,
            routing=Routing(target_engine=TargetEngine.PRIME),
            model=Model(name="test-model", provider="test", context_window_tokens=4096),
        ),
        intent=Intent(user_intent="test", system_task=SystemTask.GENERATE_DRAFT, confidence=0.9),
        context=Context(
            session_history_ref=SessionHistoryRef(type="test", value="sess-1"),
            cheatsheets=[],
            constraints=Constraints(max_tokens=512, time_budget_ms=5000, safety_mode="strict"),
        ),
        content=Content(original_prompt="Hello", data_fields=[]),
        reasoning=Reasoning(),
        response=Response(candidate="Hi there", confidence=0.8, stream_proposal=False),
        governance=Governance(safety=Safety(execution_allowed=False, dry_run=True)),
        metrics=Metrics(
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=100,
        ),
        status=Status(finalized=False, state=PacketState.INITIALIZED, next_steps=[]),
    )
    defaults.update(overrides)
    return CognitionPacket(**defaults)


class TestCognitionPacketCreation:
    """Test that packets can be created with valid data."""

    def test_minimal_packet_creates(self):
        pkt = _make_minimal_packet()
        assert pkt.version == "0.3"
        assert pkt.header.packet_id == "pkt-001"

    def test_packet_response_content(self):
        pkt = _make_minimal_packet()
        assert pkt.response.candidate == "Hi there"
        assert pkt.response.confidence == 0.8


class TestCognitionPacketSerialization:
    """Test round-trip serialization."""

    def test_to_json_produces_valid_json(self):
        pkt = _make_minimal_packet()
        json_str = pkt.to_json()
        parsed = json.loads(json_str)
        assert parsed["version"] == "0.3"

    def test_to_dict_and_from_dict_roundtrip(self):
        pkt = _make_minimal_packet()
        d = pkt.to_dict()
        restored = CognitionPacket.from_dict(d)
        assert restored.header.packet_id == pkt.header.packet_id
        assert restored.response.candidate == pkt.response.candidate

    def test_to_serializable_dict_converts_enums(self):
        pkt = _make_minimal_packet()
        sd = pkt.to_serializable_dict()
        # Enums should be string values, not Enum objects
        assert sd["header"]["origin"] == "user"
        assert sd["status"]["state"] == "initialized"


class TestCognitionPacketHashes:
    """Test integrity hash computation."""

    def test_compute_hashes_sets_values(self):
        pkt = _make_minimal_packet()
        pkt.compute_hashes()
        assert pkt.governance.signatures.header_hash is not None
        assert pkt.governance.signatures.content_hash is not None

    def test_different_content_different_hash(self):
        pkt_a = _make_minimal_packet(
            content=Content(original_prompt="Hello", data_fields=[])
        )
        pkt_b = _make_minimal_packet(
            content=Content(original_prompt="Goodbye", data_fields=[])
        )
        pkt_a.compute_hashes()
        pkt_b.compute_hashes()
        assert pkt_a.governance.signatures.content_hash != pkt_b.governance.signatures.content_hash

    def test_same_content_same_hash(self):
        pkt_a = _make_minimal_packet()
        pkt_b = _make_minimal_packet()
        pkt_a.compute_hashes()
        pkt_b.compute_hashes()
        assert pkt_a.governance.signatures.content_hash == pkt_b.governance.signatures.content_hash


class TestTokenBudget:
    """Test the token budget check."""

    def test_within_budget(self):
        pkt = _make_minimal_packet()
        pkt.header.model.max_output_tokens = 1000
        pkt.header.model.response_buffer_tokens = 100
        pkt.metrics.token_usage.projected_tokens = 500
        assert pkt.check_token_budget() is True

    def test_over_budget(self):
        pkt = _make_minimal_packet()
        pkt.header.model.max_output_tokens = 1000
        pkt.header.model.response_buffer_tokens = 100
        pkt.metrics.token_usage.projected_tokens = 950
        assert pkt.check_token_budget() is False

    def test_no_projection_is_ok(self):
        pkt = _make_minimal_packet()
        # No projected_tokens set → should pass
        assert pkt.check_token_budget() is True
