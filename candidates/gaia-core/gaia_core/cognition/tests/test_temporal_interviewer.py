"""Tests for TemporalInterviewer — Prime interviews past-Lite via KV cache swapping."""

from __future__ import annotations

import json
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.temporal_interviewer import TemporalInterviewer


# ── Helpers ──────────────────────────────────────────────────────────


class FakeLlamaState:
    """Picklable stand-in for llama_cpp.LlamaState."""

    def __init__(self, label: str = "default"):
        self.data = b"\x00" * 1024
        self.label = label


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.SHARED_DIR = str(tmp_path)
    config.TEMPORAL_INTERVIEW_ROUNDS = 3
    config.TEMPORAL_INTERVIEW_ENABLED = True
    config.TEMPORAL_INTERVIEW_INTERVAL_TICKS = 6
    config.TEMPORAL_STATE_MAX_FILES = 5
    config.TEMPORAL_STATE_MAX_BYTES = 10_737_418_240
    config.TEMPORAL_STATE_BAKE_CONTEXT_TOKENS = 6000
    return config


@pytest.fixture
def mock_llm():
    """Mock Llama instance that tracks save_state / load_state calls."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "I was processing intent detection queries."}}]
    }
    llm.save_state.return_value = FakeLlamaState("current")
    llm.load_state.return_value = None
    return llm


@pytest.fixture
def mock_model_pool(mock_llm):
    pool = MagicMock()
    pool.get_model_for_role.return_value = mock_llm
    pool.forward_to_model.return_value = {
        "choices": [{"message": {"content": "What were you doing at that moment?"}}]
    }
    return pool


def _create_baked_state(state_dir: Path, ts_label: str) -> str:
    """Create a fake baked state (.bin + .json) and return the state_id."""
    state_id = f"lite_state_{ts_label}"
    bin_path = state_dir / f"{state_id}.bin"
    meta_path = state_dir / f"{state_id}.json"

    with open(bin_path, "wb") as f:
        pickle.dump(FakeLlamaState(ts_label), f)

    meta = {
        "timestamp": ts_label.replace("T", " ").replace("-", ":")[:-1],
        "state_id": state_id,
        "gaia_state": "active",
        "heartbeat_tick": 3,
        "state_size_bytes": bin_path.stat().st_size,
        "bake_duration_ms": 5000,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return state_id


@pytest.fixture
def tsm_with_states(mock_config, mock_model_pool, tmp_path):
    """Create a real TSM with 3 pre-baked states."""
    from gaia_core.cognition.temporal_state_manager import TemporalStateManager

    tsm = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)

    # Create 3 states: oldest, middle, newest (current)
    _create_baked_state(tsm.state_dir, "2026-02-18T10-00-00Z")
    _create_baked_state(tsm.state_dir, "2026-02-18T12-00-00Z")
    _create_baked_state(tsm.state_dir, "2026-02-18T14-00-00Z")

    return tsm


@pytest.fixture
def mock_journal():
    journal = MagicMock()
    journal.load_recent_entries.return_value = [
        "## Entry: 2026-02-18T10:30:00Z\n**State:** ACTIVE for 2h | **Heartbeat:** #5\n"
        "I've been handling intent detection.  The pattern of user messages has been consistent.",
        "## Entry: 2026-02-18T12:30:00Z\n**State:** ACTIVE for 4h | **Heartbeat:** #11\n"
        "Switched focus to conversation about consciousness frameworks.",
    ]
    journal.get_entry_count.return_value = 2
    return journal


@pytest.fixture
def mock_timeline():
    return MagicMock()


@pytest.fixture
def interviewer(mock_config, mock_model_pool, tsm_with_states, mock_journal, mock_timeline):
    return TemporalInterviewer(
        config=mock_config,
        model_pool=mock_model_pool,
        temporal_state_manager=tsm_with_states,
        lite_journal=mock_journal,
        timeline_store=mock_timeline,
    )


# ── TestInterviewTargetSelection ─────────────────────────────────────


class TestInterviewTargetSelection:
    def test_selects_oldest_uninterviewed_state(self, interviewer):
        target = interviewer._select_interview_target()
        assert target is not None
        # Should pick the oldest (first) state, not the newest (current)
        assert "2026-02-18T10-00-00Z" in target["state_id"]

    def test_skips_most_recent_state(self, mock_config, mock_model_pool):
        """With only 1 state (the current one), nothing to interview."""
        from gaia_core.cognition.temporal_state_manager import TemporalStateManager

        tsm = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        _create_baked_state(tsm.state_dir, "2026-02-18T14-00-00Z")

        iv = TemporalInterviewer(
            config=mock_config, model_pool=mock_model_pool,
            temporal_state_manager=tsm,
        )
        assert iv._select_interview_target() is None

    def test_falls_back_to_already_interviewed(self, interviewer, tsm_with_states):
        """When all non-current states have transcripts, fall back to oldest."""
        # Create transcript files for both non-current states
        for ts in ("2026-02-18T10-00-00Z", "2026-02-18T12-00-00Z"):
            sid = f"lite_state_{ts}"
            transcript_path = (
                interviewer.transcript_dir / f"interview_{sid}_2026-02-18T15-00-00Z.json"
            )
            transcript_path.write_text("{}", encoding="utf-8")

        target = interviewer._select_interview_target()
        assert target is not None
        # Falls back to oldest (already interviewed)
        assert "2026-02-18T10-00-00Z" in target["state_id"]

    def test_returns_none_with_no_states(self, mock_config, mock_model_pool):
        from gaia_core.cognition.temporal_state_manager import TemporalStateManager

        tsm = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        iv = TemporalInterviewer(
            config=mock_config, model_pool=mock_model_pool,
            temporal_state_manager=tsm,
        )
        assert iv._select_interview_target() is None


# ── TestInterviewFlow ────────────────────────────────────────────────


class TestInterviewFlow:
    def test_full_interview_cycle(self, interviewer, mock_llm, mock_model_pool):
        """Happy path: save -> load past -> interview -> restore -> transcript."""
        transcript = interviewer.conduct_interview()

        assert transcript is not None
        assert transcript["round_count"] == 3
        assert len(transcript["rounds"]) == 3

        # Verify state save/restore: save_state once (start) + load_state twice
        # (load past + restore current)
        assert mock_llm.save_state.call_count == 1
        assert mock_llm.load_state.call_count == 2

        # Verify Prime was called for questions (3 rounds + 1 coherence)
        assert mock_model_pool.forward_to_model.call_count == 4

        # Verify Lite was called for answers (3 rounds)
        assert mock_llm.create_chat_completion.call_count == 3

        # Verify transcript file saved
        transcripts = list(interviewer.transcript_dir.glob("interview_*.json"))
        assert len(transcripts) == 1

        saved = json.loads(transcripts[0].read_text())
        assert saved["round_count"] == 3
        assert "coherence" in saved

    def test_state_restored_on_interview_error(self, interviewer, mock_llm):
        """If Lite throws during interview, current state must still be restored."""
        # Make Lite fail on every answer
        mock_llm.create_chat_completion.side_effect = RuntimeError("boom")

        # Should not crash
        transcript = interviewer.conduct_interview()

        # State restore must have been called (load_state for past + restore)
        assert mock_llm.load_state.call_count >= 2

    def test_returns_none_without_model_pool(self, mock_config, tsm_with_states):
        iv = TemporalInterviewer(
            config=mock_config,
            model_pool=None,
            temporal_state_manager=tsm_with_states,
        )
        assert iv.conduct_interview() is None

    def test_returns_none_without_tsm(self, mock_config, mock_model_pool):
        iv = TemporalInterviewer(
            config=mock_config,
            model_pool=mock_model_pool,
            temporal_state_manager=None,
        )
        assert iv.conduct_interview() is None


# ── TestLockBehavior ─────────────────────────────────────────────────


class TestLockBehavior:
    def test_interview_holds_lite_lock(self, interviewer, mock_llm):
        """During interview, _LITE_LOCK should be held (blocking other Lite access)."""
        from gaia_core.cognition.temporal_state_manager import _LITE_LOCK

        lock_was_held = threading.Event()
        lock_check_done = threading.Event()

        # Slow down Lite answers so we can check the lock
        original_side_effect = mock_llm.create_chat_completion.return_value

        call_count = 0

        def slow_answer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Signal that we're in the interview
                lock_was_held.set()
                # Wait for the lock check to complete
                lock_check_done.wait(timeout=5)
            return original_side_effect

        mock_llm.create_chat_completion.side_effect = slow_answer

        # Start interview in a thread
        result_holder = [None]

        def run_interview():
            result_holder[0] = interviewer.conduct_interview()

        t = threading.Thread(target=run_interview)
        t.start()

        # Wait for the interview to be running
        lock_was_held.wait(timeout=5)

        # Try to acquire the lock (non-blocking)
        acquired = _LITE_LOCK.acquire(blocking=False)
        if acquired:
            _LITE_LOCK.release()

        lock_check_done.set()
        t.join(timeout=10)

        # Lock should NOT have been acquirable during the interview
        assert not acquired, "_LITE_LOCK was not held during interview"

    def test_lock_released_after_interview(self, interviewer):
        """After interview completes, the lock must be released."""
        from gaia_core.cognition.temporal_state_manager import _LITE_LOCK

        interviewer.conduct_interview()

        # Lock should be acquirable now
        acquired = _LITE_LOCK.acquire(blocking=False)
        assert acquired, "_LITE_LOCK was not released after interview"
        _LITE_LOCK.release()


# ── TestNarrativeCoherence ───────────────────────────────────────────


class TestNarrativeCoherence:
    def test_coherence_analysis_called(self, interviewer, mock_model_pool):
        """Prime should be called to compare journal vs interview."""
        interviewer.conduct_interview()

        # forward_to_model should be called for coherence (last call)
        calls = mock_model_pool.forward_to_model.call_args_list
        assert len(calls) >= 1
        # The last call should be the coherence analysis
        last_call = calls[-1]
        messages = last_call[1].get("messages") or last_call[0][1]
        # System message should mention coherence
        assert "coherence" in messages[0]["content"].lower()

    def test_coherence_parsing(self, interviewer):
        """Known analysis text should be parsed into structured fields."""
        analysis_text = (
            "TOPIC_OVERLAP: 0.85 Good alignment on intent detection topics\n"
            "TONE_CONSISTENCY: 0.70 Slightly more reflective in interview\n"
            "INFO_LOSS: temporal context, world state details\n"
            "INFO_GAIN: emotional nuance, unfinished thread about caching\n"
            "OVERALL: 0.78 Solid coherence with expected KV compression effects"
        )

        result = interviewer._parse_coherence(analysis_text)

        assert abs(result["topic_overlap"] - 0.85) < 0.01
        assert abs(result["tone_consistency"] - 0.70) < 0.01
        assert "temporal context" in result["information_loss"]
        assert "world state details" in result["information_loss"]
        assert "emotional nuance" in result["information_gain"]
        assert abs(result["overall_coherence"] - 0.78) < 0.01
        assert "coherence" in result["narrative"].lower()

    def test_coherence_graceful_on_parse_failure(self, interviewer):
        """Malformed analysis text should produce default scores, not crash."""
        result = interviewer._parse_coherence("This is not structured output at all!")

        assert result["topic_overlap"] == -1.0
        assert result["overall_coherence"] == -1.0
        assert result["information_loss"] == []
        assert result["information_gain"] == []


# ── TestTranscriptStorage ────────────────────────────────────────────


class TestTranscriptStorage:
    def test_transcript_saved_as_json(self, interviewer):
        interviewer.conduct_interview()

        transcripts = list(interviewer.transcript_dir.glob("interview_*.json"))
        assert len(transcripts) == 1

        data = json.loads(transcripts[0].read_text())
        assert "state_id" in data
        assert "interview_timestamp" in data
        assert "rounds" in data
        assert "coherence" in data
        assert "duration_ms" in data

    def test_transcript_contains_all_rounds(self, interviewer):
        transcript = interviewer.conduct_interview()

        assert transcript is not None
        assert len(transcript["rounds"]) == 3
        for r in transcript["rounds"]:
            assert "question" in r
            assert "answer" in r

    def test_transcript_dir_created(self, mock_config, mock_model_pool, tsm_with_states):
        """Transcript directory should be auto-created."""
        iv = TemporalInterviewer(
            config=mock_config,
            model_pool=mock_model_pool,
            temporal_state_manager=tsm_with_states,
        )
        assert iv.transcript_dir.exists()
        assert iv.transcript_dir.name == "interviews"


# ── TestHeartbeatIntegration ─────────────────────────────────────────


class FakeConfig:
    HEARTBEAT_INTERVAL_SECONDS = 5
    HEARTBEAT_ENABLED = True
    LITE_JOURNAL_ENABLED = False
    TEMPORAL_STATE_ENABLED = False
    TEMPORAL_INTERVIEW_ENABLED = True
    TEMPORAL_INTERVIEW_INTERVAL_TICKS = 6
    TEMPORAL_INTERVIEW_ROUNDS = 3
    TEMPORAL_BAKE_INTERVAL_TICKS = 3


class TestHeartbeatIntegration:
    def test_interview_triggered_on_interval(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        mock_interviewer = MagicMock()
        mock_interviewer.conduct_interview.return_value = {
            "coherence": {"overall_coherence": 0.8},
        }

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._temporal_interviewer = mock_interviewer
        hb._interview_interval = 6

        # Tick 6 times (interview should trigger on tick 6)
        for _ in range(6):
            hb._run_temporal_tasks()

        mock_interviewer.conduct_interview.assert_called_once()

    def test_interview_not_triggered_off_interval(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        mock_interviewer = MagicMock()

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._temporal_interviewer = mock_interviewer
        hb._interview_interval = 6

        # Only 5 ticks — should not trigger
        for _ in range(5):
            hb._run_temporal_tasks()

        mock_interviewer.conduct_interview.assert_not_called()

    def test_interview_skipped_when_sleeping(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat
        from gaia_core.cognition.sleep_wake_manager import GaiaState

        mock_interviewer = MagicMock()
        swm = MagicMock()
        swm.get_state.return_value = GaiaState.ASLEEP

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._temporal_interviewer = mock_interviewer
        hb._interview_interval = 6
        hb.sleep_wake_manager = swm

        for _ in range(6):
            hb._run_temporal_tasks()

        mock_interviewer.conduct_interview.assert_not_called()

    def test_interview_failure_doesnt_crash_heartbeat(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        mock_interviewer = MagicMock()
        mock_interviewer.conduct_interview.side_effect = RuntimeError("interview exploded")

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._temporal_interviewer = mock_interviewer
        hb._interview_interval = 6

        # Should not raise
        for _ in range(6):
            hb._run_temporal_tasks()

        # Heartbeat should continue despite failure
        assert hb._tick_count == 6
