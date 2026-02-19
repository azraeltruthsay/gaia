"""Tests for TemporalStateManager — KV cache state baking and restoration."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.temporal_state_manager import TemporalStateManager


# ── Fixtures ────────────────────────────────────────────────────────


class FakeLlamaState:
    """Picklable stand-in for llama_cpp.LlamaState."""
    def __init__(self):
        self.data = b"\x00" * 1024  # 1KB fake KV cache


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.SHARED_DIR = str(tmp_path)
    config.TEMPORAL_STATE_MAX_FILES = 5
    config.TEMPORAL_STATE_MAX_BYTES = 10_737_418_240
    config.TEMPORAL_STATE_BAKE_CONTEXT_TOKENS = 6000
    return config


@pytest.fixture
def mock_llm():
    """Mock Llama instance with save_state/load_state."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "I am reflecting on temporal context..."}}]
    }
    llm.save_state.return_value = FakeLlamaState()
    llm.load_state.return_value = None
    return llm


@pytest.fixture
def mock_model_pool(mock_llm):
    pool = MagicMock()
    pool.get_model_for_role.return_value = mock_llm
    return pool


@pytest.fixture
def mock_timeline():
    timeline = MagicMock()
    timeline.events_by_type.return_value = []
    timeline.recent_events.return_value = []
    timeline.last_event_of_type.return_value = None
    return timeline


@pytest.fixture
def mock_session_manager():
    sm = MagicMock()
    sm.sessions = {}
    sm.get_history.return_value = []
    return sm


@pytest.fixture
def mock_journal():
    journal = MagicMock()
    journal.load_recent_entries.return_value = ["## Entry: test\nSome content."]
    journal.get_entry_count.return_value = 1
    return journal


# ── TestStateDirectory ──────────────────────────────────────────────


class TestStateDirectory:
    def test_creates_state_dir(self, mock_config, mock_model_pool):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        assert mgr.state_dir.exists()
        assert mgr.state_dir.name == "temporal_states"

    def test_list_states_empty(self, mock_config, mock_model_pool):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        assert mgr.list_states() == []


# ── TestBakeState ───────────────────────────────────────────────────


class TestBakeState:
    def test_bake_creates_bin_file(self, mock_config, mock_model_pool, mock_llm):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        path = mgr.bake_state()

        assert path is not None
        assert path.exists()
        assert path.suffix == ".bin"
        assert path.name.startswith("lite_state_")
        mock_llm.save_state.assert_called_once()

    def test_bake_creates_json_sidecar(self, mock_config, mock_model_pool):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        path = mgr.bake_state()

        meta_path = path.with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "timestamp" in meta
        assert "state_id" in meta
        assert "state_size_bytes" in meta
        assert "bake_duration_ms" in meta

    def test_bake_returns_none_without_llm(self, mock_config):
        pool = MagicMock()
        pool.get_model_for_role.return_value = None
        mgr = TemporalStateManager(config=mock_config, model_pool=pool)

        assert mgr.bake_state() is None

    def test_bake_returns_none_without_model_pool(self, mock_config):
        mgr = TemporalStateManager(config=mock_config, model_pool=None)
        assert mgr.bake_state() is None

    def test_bake_state_is_picklable(self, mock_config, mock_model_pool):
        """Verify the saved .bin file can be unpickled."""
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        path = mgr.bake_state()

        with open(path, "rb") as f:
            state = pickle.load(f)
        assert isinstance(state, FakeLlamaState)

    def test_bake_includes_journal_content(
        self, mock_config, mock_model_pool, mock_llm, mock_journal,
    ):
        mgr = TemporalStateManager(
            config=mock_config, model_pool=mock_model_pool,
            lite_journal=mock_journal,
        )
        mgr.bake_state()

        # Verify the LLM was called and the context included journal content
        call_args = mock_llm.create_chat_completion.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        user_msg = messages[-1]["content"]
        assert "test" in user_msg or "Some content" in user_msg


# ── TestLoadState ───────────────────────────────────────────────────


class TestLoadState:
    def test_load_existing_state(self, mock_config, mock_model_pool, mock_llm):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        path = mgr.bake_state()
        state_id = path.stem

        # Reset mock to verify load is called
        mock_llm.load_state.reset_mock()

        result = mgr.load_state(state_id)
        assert result is True
        mock_llm.load_state.assert_called_once()

    def test_load_nonexistent_returns_false(self, mock_config, mock_model_pool):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        assert mgr.load_state("nonexistent_state") is False

    def test_corrupt_state_renamed(self, mock_config, mock_model_pool, mock_llm):
        """When load_state fails, the .bin should be renamed to .corrupt."""
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        path = mgr.bake_state()
        state_id = path.stem

        # Make load_state raise
        mock_llm.load_state.side_effect = RuntimeError("corrupt KV cache")

        result = mgr.load_state(state_id)
        assert result is False

        # Original .bin should be gone, .corrupt should exist
        assert not path.exists()
        corrupt = path.with_suffix(".bin.corrupt")
        assert corrupt.exists()

    def test_restore_current_loads_latest(self, mock_config, mock_model_pool, mock_llm):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)
        mgr.bake_state()

        mock_llm.load_state.reset_mock()
        result = mgr.restore_current()
        assert result is True
        mock_llm.load_state.assert_called_once()


# ── TestRotation ────────────────────────────────────────────────────


class TestRotation:
    def test_cleanup_enforces_max_files(self, mock_config, mock_model_pool):
        mock_config.TEMPORAL_STATE_MAX_FILES = 2
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)

        # Create 4 fake state files
        for i in range(4):
            ts = f"2026-02-18T{10+i:02d}-00-00Z"
            (mgr.state_dir / f"lite_state_{ts}.bin").write_bytes(b"\x00" * 100)
            (mgr.state_dir / f"lite_state_{ts}.json").write_text("{}")

        deleted = mgr.cleanup_old_states()
        assert deleted == 2

        remaining_bins = list(mgr.state_dir.glob("*.bin"))
        assert len(remaining_bins) == 2

    def test_cleanup_enforces_max_bytes(self, mock_config, mock_model_pool):
        mock_config.TEMPORAL_STATE_MAX_FILES = 10  # Won't trigger file limit
        mock_config.TEMPORAL_STATE_MAX_BYTES = 250  # 250 bytes total budget
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)

        # Create 3 files of 100 bytes each (300 > 250 budget)
        for i in range(3):
            ts = f"2026-02-18T{10+i:02d}-00-00Z"
            (mgr.state_dir / f"lite_state_{ts}.bin").write_bytes(b"\x00" * 100)
            (mgr.state_dir / f"lite_state_{ts}.json").write_text("{}")

        deleted = mgr.cleanup_old_states()
        assert deleted >= 1

        total_size = sum(
            p.stat().st_size for p in mgr.state_dir.glob("*.bin")
        )
        assert total_size <= 250

    def test_cleanup_deletes_sidecar_too(self, mock_config, mock_model_pool):
        mock_config.TEMPORAL_STATE_MAX_FILES = 1
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)

        # Create 2 states
        for i in range(2):
            ts = f"2026-02-18T{10+i:02d}-00-00Z"
            (mgr.state_dir / f"lite_state_{ts}.bin").write_bytes(b"\x00" * 100)
            (mgr.state_dir / f"lite_state_{ts}.json").write_text('{"test": true}')

        mgr.cleanup_old_states()

        remaining_jsons = list(mgr.state_dir.glob("*.json"))
        assert len(remaining_jsons) == 1


# ── TestContextReconstruction ───────────────────────────────────────


class TestContextReconstruction:
    def test_reconstruct_timeline_context(
        self, mock_config, mock_model_pool, mock_timeline,
    ):
        # Create mock events
        mock_event = MagicMock()
        mock_event.ts = "2026-02-18T14:30:00Z"
        mock_event.event = "state_change"
        mock_event.data = {"from": "active", "to": "drowsy"}
        mock_timeline.recent_events.return_value = [mock_event]

        mgr = TemporalStateManager(
            config=mock_config, model_pool=mock_model_pool,
            timeline_store=mock_timeline,
        )
        context = mgr._reconstruct_timeline_context()

        assert "state_change" in context
        assert "2026-02-18" in context

    def test_reconstruct_conversation_context(
        self, mock_config, mock_model_pool, mock_session_manager,
    ):
        # Create a mock session with messages
        session = MagicMock()
        session.last_message_timestamp.return_value = datetime.now(timezone.utc)
        mock_session_manager.sessions = {"discord_dm_123": session}
        mock_session_manager.get_history.return_value = [
            {"role": "user", "content": "Hello GAIA"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]

        mgr = TemporalStateManager(
            config=mock_config, model_pool=mock_model_pool,
            session_manager=mock_session_manager,
        )
        context = mgr._reconstruct_conversation_context()

        assert "discord_dm_123" in context
        assert "Hello GAIA" in context

    def test_list_states_with_metadata(self, mock_config, mock_model_pool):
        mgr = TemporalStateManager(config=mock_config, model_pool=mock_model_pool)

        # Create a state with sidecar
        ts = "2026-02-18T14-30-00Z"
        (mgr.state_dir / f"lite_state_{ts}.bin").write_bytes(b"\x00" * 100)
        (mgr.state_dir / f"lite_state_{ts}.json").write_text(
            json.dumps({"timestamp": "2026-02-18T14:30:00Z", "gaia_state": "active"})
        )

        states = mgr.list_states()
        assert len(states) == 1
        assert states[0]["state_id"] == f"lite_state_{ts}"
        assert states[0]["timestamp"] == "2026-02-18T14:30:00Z"
        assert states[0]["gaia_state"] == "active"
