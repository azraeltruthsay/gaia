"""Unit tests for GoalDetector."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    Content,
    Context,
    Constraints,
    DetectedGoal,
    GoalConfidence,
    GoalState,
    Governance,
    Header,
    Intent,
    Metrics,
    Model,
    Persona,
    PersonaRole,
    Response,
    Reasoning,
    Routing,
    Safety,
    SessionHistoryRef,
    Status,
    PacketState,
    TargetEngine,
    SystemTask,
    TokenUsage,
)
from gaia_core.cognition.goal_detector import GoalDetector, MAX_CARRY_TURNS


# ── Helpers ──────────────────────────────────────────────────────────


def _make_packet(user_intent: str = "greeting", user_input: str = "Hello!") -> CognitionPacket:
    """Build a minimal CognitionPacket for testing."""
    return CognitionPacket(
        version="0.3",
        header=Header(
            datetime=datetime.now(timezone.utc).isoformat(),
            session_id="test-session",
            packet_id="pkt-001",
            sub_id="sub-001",
            persona=Persona(identity_id="Prime", persona_id="Default", role=PersonaRole.DEFAULT),
            origin="user",
            routing=Routing(target_engine=TargetEngine.PRIME),
            model=Model(name="test", provider="test", context_window_tokens=4096),
        ),
        intent=Intent(user_intent=user_intent, system_task=SystemTask.STREAM, confidence=0.9),
        context=Context(
            session_history_ref=SessionHistoryRef(type="ref", value="test"),
            cheatsheets=[],
            constraints=Constraints(max_tokens=1024, time_budget_ms=5000, safety_mode="standard"),
        ),
        content=Content(original_prompt=user_input),
        reasoning=Reasoning(),
        response=Response(candidate="", confidence=0.0, stream_proposal=False),
        governance=Governance(safety=Safety(execution_allowed=True, dry_run=True)),
        metrics=Metrics(token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), latency_ms=0),
        status=Status(finalized=False, state=PacketState.PROCESSING),
    )


def _make_session_manager(meta: dict | None = None):
    """Build a mock SessionManager with optional pre-loaded meta."""
    sm = MagicMock()
    store: dict = meta if meta is not None else {}
    sm.get_session_meta.side_effect = lambda sid, key, default=None: store.get(key, default)
    sm.set_session_meta.side_effect = lambda sid, key, value: store.__setitem__(key, value)
    return sm


# ── Fast-Path Tests ──────────────────────────────────────────────────


class TestFastPath:
    def test_greeting_maps_to_casual_conversation(self):
        packet = _make_packet(user_intent="greeting", user_input="Hey there!")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=None, session_id="s1")
        assert state.current_goal is not None
        assert state.current_goal.goal_id == "casual_conversation"
        assert state.current_goal.confidence == GoalConfidence.HIGH
        assert state.current_goal.source == "fast_path"

    def test_question_maps_to_information_seeking(self):
        packet = _make_packet(user_intent="question", user_input="What is quantum computing?")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=None, session_id="s1")
        assert state.current_goal.goal_id == "information_seeking"

    def test_tool_use_maps_to_task_execution(self):
        packet = _make_packet(user_intent="tool_use", user_input="Run the diagnostics")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=None, session_id="s1")
        assert state.current_goal.goal_id == "task_execution"

    def test_help_request_maps_to_task_assistance(self):
        packet = _make_packet(user_intent="help_request", user_input="Help me debug this")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=None, session_id="s1")
        assert state.current_goal.goal_id == "task_assistance"


# ── Session Carry Tests ──────────────────────────────────────────────


class TestSessionCarry:
    def test_carries_active_goal(self):
        stored_goal = {
            "current_goal": {
                "goal_id": "debug_server",
                "description": "Debugging a server crash",
                "confidence": "high",
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "source": "llm",
            },
            "previous_goals": [],
            "turn_count": 2,
            "goal_shifts": 0,
        }
        sm = _make_session_manager(meta={"goal_state": stored_goal})

        # Use an unknown intent so fast-path doesn't trigger
        packet = _make_packet(user_intent="unknown_complex", user_input="The logs show OOM errors")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=sm, session_id="s1")

        assert state.current_goal is not None
        assert state.current_goal.goal_id == "debug_server"
        assert state.current_goal.source == "session_carry"
        assert state.turn_count == 3  # was 2, incremented

    def test_carry_decays_after_max_turns(self):
        stored_goal = {
            "current_goal": {
                "goal_id": "old_goal",
                "description": "Something old",
                "confidence": "high",
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "source": "fast_path",
            },
            "previous_goals": [],
            "turn_count": MAX_CARRY_TURNS - 1,  # Will reach MAX_CARRY_TURNS on carry
            "goal_shifts": 0,
        }
        sm = _make_session_manager(meta={"goal_state": stored_goal})

        packet = _make_packet(user_intent="unknown_complex", user_input="Unrelated stuff")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=sm, session_id="s1")

        assert state.current_goal.confidence == GoalConfidence.LOW
        assert state.turn_count == MAX_CARRY_TURNS


# ── LLM Detection Tests ─────────────────────────────────────────────


class TestLLMDetect:
    def test_llm_detection_parses_response(self):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "GOAL_ID: debug_server_crash\nDESCRIPTION: User is debugging a server crash\nCONFIDENCE: high"}}]
        }
        mock_pool = MagicMock()
        mock_pool.get_model_for_role.return_value = mock_llm

        packet = _make_packet(user_intent="unknown_complex", user_input="My server keeps crashing with OOM errors")
        sm = _make_session_manager()
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=sm, session_id="s1", model_pool=mock_pool)

        assert state.current_goal is not None
        assert state.current_goal.goal_id == "debug_server_crash"
        assert state.current_goal.confidence == GoalConfidence.HIGH
        assert state.current_goal.source == "llm"

    def test_llm_detection_graceful_on_failure(self):
        mock_pool = MagicMock()
        mock_pool.get_model_for_role.side_effect = Exception("No model")

        packet = _make_packet(user_intent="unknown_complex", user_input="Help me please")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=None, session_id="s1", model_pool=mock_pool)

        # Should return empty state, not raise
        assert state.current_goal is None


# ── Goal Shift Tests ─────────────────────────────────────────────────


class TestGoalShift:
    def test_goal_shift_archives_previous(self):
        packet = _make_packet()
        old_goal = DetectedGoal(
            goal_id="old_goal",
            description="The old goal",
            confidence=GoalConfidence.HIGH,
            detected_at=datetime.now(timezone.utc).isoformat(),
            source="fast_path",
        )
        packet.goal_state = GoalState(current_goal=old_goal, turn_count=3, goal_shifts=0)

        sm = _make_session_manager()
        GoalDetector.handle_goal_shift("Debugging a new issue", packet, sm, "s1")

        assert packet.goal_state.current_goal.goal_id == "debugging_a_new_issue"
        assert packet.goal_state.goal_shifts == 1
        assert len(packet.goal_state.previous_goals) == 1
        assert packet.goal_state.previous_goals[0].goal_id == "old_goal"

    def test_goal_shift_without_previous_goal(self):
        packet = _make_packet()
        packet.goal_state = None

        sm = _make_session_manager()
        GoalDetector.handle_goal_shift("Brand new goal", packet, sm, "s1")

        assert packet.goal_state.current_goal is not None
        assert packet.goal_state.goal_shifts == 1
        assert len(packet.goal_state.previous_goals) == 0


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_goal_on_empty_input(self):
        packet = _make_packet(user_intent="unknown", user_input="")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=None, session_id="s1")
        assert state.current_goal is None

    def test_fast_path_takes_priority_over_carry(self):
        """Even with an active carried goal, fast-path wins if intent matches."""
        stored = {
            "current_goal": {
                "goal_id": "debug_server",
                "description": "Debugging",
                "confidence": "high",
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "source": "llm",
            },
            "previous_goals": [],
            "turn_count": 1,
            "goal_shifts": 0,
        }
        sm = _make_session_manager(meta={"goal_state": stored})

        packet = _make_packet(user_intent="greeting", user_input="Hey!")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=sm, session_id="s1")

        # Fast-path greeting should win
        assert state.current_goal.goal_id == "casual_conversation"
        assert state.current_goal.source == "fast_path"

    def test_parse_llm_response_handles_garbage(self):
        result = GoalDetector._parse_llm_response("totally unparseable garbage")
        assert result is None

    def test_persistence_round_trip(self):
        """Goal state persisted via set_session_meta and retrievable."""
        sm = _make_session_manager()
        packet = _make_packet(user_intent="question", user_input="What is AI?")
        detector = GoalDetector()
        state = detector.detect(packet, session_manager=sm, session_id="s1")

        # Verify set_session_meta was called
        sm.set_session_meta.assert_called()
        call_args = sm.set_session_meta.call_args
        assert call_args[0][1] == "goal_state"
        persisted = call_args[0][2]
        assert persisted["current_goal"]["goal_id"] == "information_seeking"
