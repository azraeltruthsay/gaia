"""Unit tests for [Lite]/[Prime] response tagging (Council Protocol Phase 1)."""

from unittest.mock import MagicMock

import pytest

from gaia_core.cognition.agent_core import AgentCore


# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent_core():
    """Build an AgentCore with mocked dependencies."""
    ai_manager = MagicMock()
    ai_manager.config = MagicMock()
    ai_manager.config.constants = {}
    ai_manager.config.SHARED_DIR = "/tmp/test_shared"
    ai_manager.model_pool = MagicMock()
    ai_manager.session_manager = MagicMock()
    return AgentCore(ai_manager)


def _make_packet(state="processing"):
    """Build a minimal mock packet."""
    packet = MagicMock()
    packet.status.state.value = state
    return packet


# ── Mind Tag Format ──────────────────────────────────────────────────


class TestMindTags:
    def test_lite_tag(self):
        ac = _make_agent_core()
        header = ac._build_response_header("lite", _make_packet(), None, None, None)
        assert header == "[Lite]\n\n"

    def test_gpu_prime_tag(self):
        ac = _make_agent_core()
        header = ac._build_response_header("gpu_prime", _make_packet(), None, None, None)
        assert header == "[Prime]\n\n"

    def test_cpu_prime_tag(self):
        ac = _make_agent_core()
        header = ac._build_response_header("cpu_prime", _make_packet(), None, None, None)
        assert header == "[Prime]\n\n"

    def test_prime_tag(self):
        ac = _make_agent_core()
        header = ac._build_response_header("prime", _make_packet(), None, None, None)
        assert header == "[Prime]\n\n"

    def test_oracle_tag(self):
        ac = _make_agent_core()
        header = ac._build_response_header("oracle", _make_packet(), None, None, None)
        assert header == "[Oracle]\n\n"

    def test_unknown_model_uses_name(self):
        ac = _make_agent_core()
        header = ac._build_response_header("groq_fallback", _make_packet(), None, None, None)
        assert header == "[groq_fallback]\n\n"

    def test_none_model(self):
        ac = _make_agent_core()
        header = ac._build_response_header(None, _make_packet(), None, None, None)
        assert header == "[unknown]\n\n"


# ── Header Format Consistency ────────────────────────────────────────


class TestHeaderFormat:
    def test_header_is_parseable(self):
        """Headers should be extractable with a simple regex."""
        import re
        ac = _make_agent_core()
        for model in ("lite", "gpu_prime", "prime", "cpu_prime", "oracle"):
            header = ac._build_response_header(model, _make_packet(), None, None, None)
            match = re.match(r"^\[(\w+)\]\n\n$", header)
            assert match is not None, f"Header for {model} is not parseable: {header!r}"

    def test_header_uses_mind_tag_format(self):
        ac = _make_agent_core()
        header = ac._build_response_header("lite", _make_packet(), None, None, None)
        expected = ac.MIND_TAG_FORMAT.format(mind="Lite") + "\n\n"
        assert header == expected

    def test_no_verbose_debug_in_header(self):
        """Header should not contain Model:, State:, or Observer: labels."""
        ac = _make_agent_core()
        observer = MagicMock()
        header = ac._build_response_header(
            "gpu_prime", _make_packet(), observer, observer, observer,
        )
        assert "Model:" not in header
        assert "State:" not in header
        assert "Observer:" not in header
