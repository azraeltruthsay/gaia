"""Unit tests for complexity assessment and Council escalation (Phase 3)."""

from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.agent_core import AgentCore, ComplexityAssessment


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


# ── No-Escalation Cases ─────────────────────────────────────────────


class TestNoEscalation:
    def test_empty_input(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("")
        assert not result.should_escalate

    def test_none_input(self):
        ac = _make_agent_core()
        result = ac._assess_complexity(None)
        assert not result.should_escalate

    def test_simple_greeting(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Hey, how are you?")
        assert not result.should_escalate

    def test_short_casual_chat(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("What's the weather like?")
        assert not result.should_escalate

    def test_recitation_stays_on_lite(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Can you recite a poem for me?")
        assert not result.should_escalate
        assert "recit" in result.reason.lower()

    def test_lyrics_stays_on_lite(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("What are the lyrics to that song?")
        assert not result.should_escalate

    def test_quote_stays_on_lite(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Quote me something from Shakespeare")
        assert not result.should_escalate


# ── Escalation: Technical Depth ──────────────────────────────────────


class TestTechnicalEscalation:
    def test_code_request(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Can you write a Python script to sort numbers?")
        assert result.should_escalate
        assert "technical" in result.reason.lower() or "depth" in result.reason.lower()

    def test_debug_request(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("I need to debug this crash")
        assert result.should_escalate

    def test_architecture_question(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("What's the best architecture for a microservices app?")
        assert result.should_escalate

    def test_performance_question(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("How can I optimize the performance of my API?")
        assert result.should_escalate

    def test_algorithm_question(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Explain the algorithm for Dijkstra's shortest path")
        assert result.should_escalate

    def test_stack_trace(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Here's a stack trace from my app")
        assert result.should_escalate


# ── Escalation: Long Prompts ────────────────────────────────────────


class TestLongPromptEscalation:
    def test_long_prompt_escalates(self):
        ac = _make_agent_core()
        long_text = " ".join(["word"] * 130)
        result = ac._assess_complexity(long_text)
        assert result.should_escalate
        assert "long prompt" in result.reason

    def test_short_prompt_no_escalation(self):
        ac = _make_agent_core()
        short_text = " ".join(["word"] * 50)
        result = ac._assess_complexity(short_text)
        assert not result.should_escalate


# ── Escalation: Emotional / Philosophical ────────────────────────────


class TestEmotionalEscalation:
    def test_meaning_of_life(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("What is the meaning of life?")
        assert result.should_escalate
        assert "meaning of" in result.reason

    def test_how_do_you_feel(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("How do you feel about being an AI?")
        assert result.should_escalate

    def test_help_me_understand(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Help me understand quantum mechanics")
        assert result.should_escalate


# ── Escalation: System Internals ─────────────────────────────────────


class TestSystemEscalation:
    def test_gaia_self_reference(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Tell me about GAIA's sleep wake system")
        assert result.should_escalate

    def test_architecture_self_reference(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("How does your architecture work?")
        assert result.should_escalate

    def test_checkpoint_question(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("What's in your last checkpoint?")
        assert result.should_escalate

    def test_prime_reference(self):
        ac = _make_agent_core()
        result = ac._assess_complexity("Is prime currently running?")
        assert result.should_escalate


# ── ComplexityAssessment Structure ───────────────────────────────────


class TestComplexityAssessmentStructure:
    def test_dataclass_fields(self):
        ca = ComplexityAssessment(should_escalate=True, reason="test", confidence=0.8)
        assert ca.should_escalate is True
        assert ca.reason == "test"
        assert ca.confidence == 0.8

    def test_confidence_range(self):
        ac = _make_agent_core()
        for prompt in [
            "Hello!",
            "Write me a complex distributed system in Rust",
            "What is the meaning of existence?",
            "Tell me about GAIA's observer",
        ]:
            result = ac._assess_complexity(prompt)
            assert 0.0 <= result.confidence <= 1.0


# ── Escalate to Prime ───────────────────────────────────────────────


class TestEscalateToPrime:
    def test_escalation_writes_council_note(self):
        ac = _make_agent_core()
        ac.council_notes = MagicMock()

        with patch.dict("sys.modules", {"gaia_core.main": MagicMock()}):
            ac._escalate_to_prime(
                user_input="Complex question",
                lite_response="My quick take...",
                reason="technical depth (code)",
                session_id="sess-001",
            )

        ac.council_notes.write_note.assert_called_once_with(
            user_prompt="Complex question",
            lite_response="My quick take...",
            escalation_reason="technical depth (code)",
            session_id="sess-001",
        )

    def test_escalation_sends_wake_signal(self):
        ac = _make_agent_core()
        ac.council_notes = MagicMock()

        mock_swm = MagicMock()
        mock_app = MagicMock()
        mock_app.state.sleep_wake_manager = mock_swm

        with patch.dict("sys.modules", {"gaia_core.main": MagicMock(app=mock_app)}):
            ac._escalate_to_prime(
                user_input="Complex question",
                lite_response="Quick response",
                reason="technical depth",
                session_id="sess-001",
            )

        mock_swm.receive_wake_signal.assert_called_once()
