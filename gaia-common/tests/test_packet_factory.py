"""Tests for gaia_common.utils.packet_factory — build_packet() round-trips."""

import json
import pytest

from gaia_common.utils.packet_factory import build_packet, PacketSource
from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    DataField,
    OutputDestination,
    PacketState,
    SystemTask,
    TargetEngine,
)


# ---------------------------------------------------------------------------
# 1. Build every source variant and check defaults
# ---------------------------------------------------------------------------

class TestBuildPacketAllSources:
    """Each PacketSource produces a valid packet with the right defaults."""

    @pytest.mark.parametrize("source", list(PacketSource))
    def test_version_is_0_3(self, source: PacketSource):
        pkt = build_packet(source, "hello")
        assert pkt.version == "0.3"

    @pytest.mark.parametrize(
        "source, expected_dest",
        [
            (PacketSource.WEB, OutputDestination.WEB),
            (PacketSource.DISCORD, OutputDestination.DISCORD),
            (PacketSource.AUDIO_GATEWAY, OutputDestination.AUDIO),
            (PacketSource.VOICE_PRIME, OutputDestination.AUDIO),
            (PacketSource.VOICE_LITE, OutputDestination.AUDIO),
        ],
    )
    def test_destination(self, source, expected_dest):
        pkt = build_packet(source, "hello")
        assert pkt.header.output_routing.primary.destination == expected_dest

    @pytest.mark.parametrize(
        "source, expected_engine",
        [
            (PacketSource.WEB, TargetEngine.PRIME),
            (PacketSource.DISCORD, TargetEngine.PRIME),
            (PacketSource.AUDIO_GATEWAY, TargetEngine.PRIME),
            (PacketSource.VOICE_PRIME, TargetEngine.PRIME),
            (PacketSource.VOICE_LITE, TargetEngine.LITE),
        ],
    )
    def test_target_engine(self, source, expected_engine):
        pkt = build_packet(source, "hello")
        assert pkt.header.routing.target_engine == expected_engine

    @pytest.mark.parametrize(
        "source, expected_tokens, expected_budget",
        [
            (PacketSource.WEB, 2048, 30000),
            (PacketSource.DISCORD, 2048, 30000),
            (PacketSource.AUDIO_GATEWAY, 2048, 30000),
            (PacketSource.VOICE_PRIME, 512, 15000),
            (PacketSource.VOICE_LITE, 128, 5000),
        ],
    )
    def test_constraints(self, source, expected_tokens, expected_budget):
        pkt = build_packet(source, "hello")
        assert pkt.context.constraints.max_tokens == expected_tokens
        assert pkt.context.constraints.time_budget_ms == expected_budget

    @pytest.mark.parametrize("source", list(PacketSource))
    def test_hashes_computed(self, source):
        pkt = build_packet(source, "hello")
        assert pkt.governance.signatures.header_hash is not None
        assert pkt.governance.signatures.content_hash is not None

    @pytest.mark.parametrize("source", list(PacketSource))
    def test_system_task_is_generate_draft(self, source):
        pkt = build_packet(source, "hello")
        assert pkt.intent.system_task == SystemTask.GENERATE_DRAFT

    @pytest.mark.parametrize("source", list(PacketSource))
    def test_state_is_initialized(self, source):
        pkt = build_packet(source, "hello")
        assert pkt.status.state == PacketState.INITIALIZED


# ---------------------------------------------------------------------------
# 2. Round-trip serialization: build → dict → JSON → from_dict → compare
# ---------------------------------------------------------------------------

class TestRoundTripSerialization:
    @pytest.mark.parametrize("source", list(PacketSource))
    def test_round_trip(self, source: PacketSource):
        original = build_packet(source, "round trip test")
        d = original.to_serializable_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        restored = CognitionPacket.from_dict(loaded)

        assert restored.version == original.version
        assert restored.header.packet_id == original.header.packet_id
        assert restored.header.session_id == original.header.session_id
        assert restored.content.original_prompt == original.content.original_prompt
        assert restored.context.constraints.max_tokens == original.context.constraints.max_tokens


# ---------------------------------------------------------------------------
# 3. Overrides
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_max_tokens_override(self):
        pkt = build_packet(PacketSource.WEB, "hi", max_tokens=4096)
        assert pkt.context.constraints.max_tokens == 4096

    def test_priority_override(self):
        pkt = build_packet(PacketSource.WEB, "hi", priority=10)
        assert pkt.header.routing.priority == 10

    def test_time_budget_override(self):
        pkt = build_packet(PacketSource.WEB, "hi", time_budget_ms=60000)
        assert pkt.context.constraints.time_budget_ms == 60000

    def test_target_engine_override(self):
        pkt = build_packet(PacketSource.WEB, "hi", target_engine=TargetEngine.LITE)
        assert pkt.header.routing.target_engine == TargetEngine.LITE

    def test_safety_mode_override(self):
        pkt = build_packet(PacketSource.WEB, "hi", safety_mode="standard")
        assert pkt.context.constraints.safety_mode == "standard"

    def test_dry_run_override(self):
        pkt = build_packet(PacketSource.WEB, "hi", dry_run=False)
        assert pkt.governance.safety.dry_run is False

    def test_tone_hint_override_to_none(self):
        """Explicitly passing tone_hint=None should set it, not use the default."""
        pkt = build_packet(PacketSource.WEB, "hi", tone_hint=None)
        assert pkt.header.persona.tone_hint is None

    def test_non_overridden_defaults_unchanged(self):
        pkt = build_packet(PacketSource.WEB, "hi", max_tokens=4096)
        # priority should still be the WEB default
        assert pkt.header.routing.priority == 5
        assert pkt.context.constraints.safety_mode == "strict"


# ---------------------------------------------------------------------------
# 4. Discord-specific fields
# ---------------------------------------------------------------------------

class TestDiscordSpecificFields:
    def test_dm_session_id(self):
        pkt = build_packet(
            PacketSource.DISCORD, "hi",
            user_id="user123", is_dm=True,
        )
        assert pkt.header.session_id == "discord_dm_user123"

    def test_channel_session_id(self):
        pkt = build_packet(
            PacketSource.DISCORD, "hi",
            channel_id="chan456", is_dm=False,
        )
        assert pkt.header.session_id == "discord_channel_chan456"

    def test_metadata_contains_discord_fields(self):
        pkt = build_packet(
            PacketSource.DISCORD, "hi",
            user_id="u1", channel_id="c1",
            reply_to_message_id="msg99",
            is_dm=True, author_name="Alice",
        )
        meta = pkt.header.output_routing.primary.metadata
        assert meta["is_dm"] is True
        assert meta["author_name"] == "Alice"
        assert pkt.header.output_routing.primary.reply_to_message_id == "msg99"

    def test_explicit_session_id_wins(self):
        pkt = build_packet(
            PacketSource.DISCORD, "hi",
            session_id="custom_session",
            user_id="u1", is_dm=True,
        )
        assert pkt.header.session_id == "custom_session"


# ---------------------------------------------------------------------------
# 5. System hint adds extra data field
# ---------------------------------------------------------------------------

class TestSystemHint:
    def test_system_hint_creates_second_data_field(self):
        pkt = build_packet(
            PacketSource.VOICE_LITE, "hi",
            system_hint="Be warm and brief.",
        )
        fields = pkt.content.data_fields
        assert len(fields) == 2
        assert fields[0].key == "user_message"
        assert fields[1].key == "system_hint"
        assert fields[1].value == "Be warm and brief."

    def test_no_system_hint_means_one_field(self):
        pkt = build_packet(PacketSource.WEB, "hi")
        assert len(pkt.content.data_fields) == 1


# ---------------------------------------------------------------------------
# 6. Hashes disabled
# ---------------------------------------------------------------------------

class TestNoHashesWhenDisabled:
    def test_no_hashes(self):
        pkt = build_packet(PacketSource.WEB, "hi", compute_hashes=False)
        assert pkt.governance.signatures.header_hash is None
        assert pkt.governance.signatures.content_hash is None
