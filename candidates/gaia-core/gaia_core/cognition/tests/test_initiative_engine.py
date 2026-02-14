"""Unit tests for InitiativeEngine."""

from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.initiative_engine import InitiativeEngine, GIL_SESSION_ID


class FakeConfig:
    SLEEP_ENABLED = True


@pytest.fixture
def config():
    return FakeConfig()


@pytest.fixture
def mock_agent_core():
    core = MagicMock()
    # run_turn is a generator — simulate yielding token events
    core.run_turn.return_value = iter([
        {"type": "token", "value": "I"},
        {"type": "token", "value": " will"},
        {"type": "status", "value": "complete"},
    ])
    return core


# ------------------------------------------------------------------
# No topics
# ------------------------------------------------------------------


class TestNoTopics:
    @patch("gaia_core.cognition.topic_manager.prioritize_topics", return_value=[])
    def test_returns_none_when_no_topics(self, mock_pt, config, mock_agent_core):
        engine = InitiativeEngine(config=config, agent_core=mock_agent_core)
        result = engine.execute_turn()
        assert result is None
        mock_agent_core.run_turn.assert_not_called()


# ------------------------------------------------------------------
# No agent_core
# ------------------------------------------------------------------


class TestNoAgentCore:
    def test_returns_none_without_agent_core(self, config):
        engine = InitiativeEngine(config=config, agent_core=None)
        result = engine.execute_turn()
        assert result is None


# ------------------------------------------------------------------
# Successful turn
# ------------------------------------------------------------------


class TestSuccessfulTurn:
    @patch("gaia_core.cognition.topic_manager.prioritize_topics")
    def test_executes_turn_with_topic(self, mock_pt, config, mock_agent_core):
        mock_pt.return_value = [{"topic_id": "topic-1", "topic": "explore autonomy"}]

        engine = InitiativeEngine(config=config, agent_core=mock_agent_core)
        result = engine.execute_turn()

        assert result == {"topic_id": "topic-1", "status": "complete"}
        mock_agent_core.run_turn.assert_called_once()

    @patch("gaia_core.cognition.topic_manager.prioritize_topics")
    def test_uses_gil_session_id(self, mock_pt, config, mock_agent_core):
        mock_pt.return_value = [{"topic_id": "t-2", "topic": "test"}]

        engine = InitiativeEngine(config=config, agent_core=mock_agent_core)
        engine.execute_turn()

        call_kwargs = mock_agent_core.run_turn.call_args
        assert call_kwargs[1]["session_id"] == GIL_SESSION_ID


# ------------------------------------------------------------------
# Self-prompt construction
# ------------------------------------------------------------------


class TestSelfPrompt:
    def test_prompt_contains_topic_description(self):
        topic = {"topic_id": "t-3", "topic": "quantum consciousness"}
        prompt = InitiativeEngine._build_self_prompt(topic)
        assert "quantum consciousness" in prompt

    def test_prompt_contains_metadata(self):
        topic = {"topic_id": "t-4", "topic": "test", "urgency": 0.9}
        prompt = InitiativeEngine._build_self_prompt(topic)
        assert "urgency" in prompt
        assert "0.9" in prompt

    def test_prompt_contains_reflection_header(self):
        topic = {"topic_id": "t-5", "topic": "x"}
        prompt = InitiativeEngine._build_self_prompt(topic)
        assert "[Autonomous Reflection Cycle]" in prompt


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    @patch("gaia_core.cognition.topic_manager.prioritize_topics")
    def test_agent_core_error_returns_error_status(self, mock_pt, config):
        mock_pt.return_value = [{"topic_id": "t-err", "topic": "fail"}]
        core = MagicMock()
        core.run_turn.side_effect = RuntimeError("model crashed")

        engine = InitiativeEngine(config=config, agent_core=core)
        result = engine.execute_turn()

        assert result == {"topic_id": "t-err", "status": "error"}
