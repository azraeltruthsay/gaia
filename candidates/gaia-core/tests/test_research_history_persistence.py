"""Regression test for GAIA_Project-axd8: the research router early return
in run_turn must persist the user's message to session history before the
handler appends its assistant response — otherwise history ends up with
adjacent assistant turns, breaking prompt construction and vector indexing.
"""
import pytest
from unittest.mock import MagicMock, patch
from gaia_core.cognition.agent_core import AgentCore


@pytest.fixture
def mock_agent_core():
    config = MagicMock()
    config.get_persona_instructions.return_value = "Test Persona"
    config.max_reflection_tokens = 500
    config.max_history_len = 20
    config.max_tokens = 2048
    config.context_length = 8192
    config._slim_prompt = False
    config.cheat_sheet_path = "/tmp/cheat.json"
    config.identity_file_path = "/tmp/id.json"
    # Real string, not a MagicMock — CodexWriter mkdirs under SHARED_DIR and a
    # mock value materializes junk `MagicMock/...` dirs in the repo root.
    # AgentCore's first arg is ai_manager and it reads ai_manager.config, so
    # the nested attribute is the one that matters.
    config.SHARED_DIR = "/tmp/gaia_test_shared"
    config.config.SHARED_DIR = "/tmp/gaia_test_shared"

    config.constants = {
        "KNOWLEDGE_BASES": {},
        "COGNITIVE_AUDIT": {"enabled": False},
        "MODEL_CONFIGS": {"observer": {"enabled": False}},
        "LOOP_DETECTION_ENABLED": False,
        "MODELS": {
            "lite": {"name": "lite"},
            "gpu_prime": {"name": "prime"},
        },
    }

    with patch('gaia_common.utils.entity_validator.EntityValidator'), \
         patch('gaia_core.memory.session_manager.SessionManager'), \
         patch('gaia_core.main.AIManagerShim'):
        core = AgentCore(config)
        core.logger = MagicMock()
        core.model_pool = MagicMock()
        core.identity_guardian = MagicMock()
        core.identity_guardian.verify_prompt_stack.return_value = (True, [])
        # correct_text must pass the string through — the blast-shield regex
        # downstream operates on the returned value.
        core.entity_validator = MagicMock()
        core.entity_validator.correct_text.side_effect = lambda t: t
        core.timeline_store = None
        return core


def _drain_research_turn(core, user_input, session_id, calls):
    """Run run_turn with the research router forced to match, recording the
    order of history writes and the handler invocation into `calls`."""
    core.session_manager = MagicMock()
    core.session_manager.add_message.side_effect = (
        lambda sid, role, content: calls.append((role, content))
    )

    def fake_handler(topic, session_id, source, metadata):
        calls.append(("handler", topic))
        core.session_manager.add_message(session_id, "assistant", "brief")
        yield {"type": "token", "value": "brief"}

    with patch('gaia_core.cognition.agent_core.kb_approval') as mock_kb, \
         patch('gaia_core.cognition.agent_core.research_router') as mock_rr, \
         patch.object(core, '_handle_research_request', side_effect=fake_handler):
        mock_kb.get_pending_write.return_value = None
        mock_rr.detect_research_intent.return_value = "samvega"
        return list(core.run_turn(user_input, session_id))


def test_research_router_persists_user_message_before_assistant(mock_agent_core):
    session_id = "test_research_history"
    user_input = "What do you know about your Samvega system?"
    calls = []

    events = _drain_research_turn(mock_agent_core, user_input, session_id, calls)

    # The handler ran (early-return path was taken, not the normal pipeline)
    assert ("handler", "samvega") in calls
    assert any(e.get("type") == "token" for e in events)

    # The user message was persisted, and BEFORE the handler / assistant append
    user_writes = [c for c in calls if c[0] == "user"]
    assert user_writes == [("user", user_input)], (
        "user message must be persisted exactly once on the research path"
    )
    assert calls.index(("user", user_input)) < calls.index(("handler", "samvega"))
    assert calls.index(("user", user_input)) < calls.index(("assistant", "brief"))


def test_research_router_alternating_roles(mock_agent_core):
    """History written by the research path must alternate user → assistant."""
    calls = []
    _drain_research_turn(
        mock_agent_core, "research the gearbox lifecycle", "test_research_roles", calls,
    )
    roles = [role for role, _ in calls if role in ("user", "assistant")]
    assert roles == ["user", "assistant"]
