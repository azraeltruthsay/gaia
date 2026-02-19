"""Tests for ThoughtSeedHeartbeat — GAIA's thought seed triage daemon."""

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.thought_seed import (
    SEEDS_ARCHIVE_DIR,
    SEEDS_DIR,
    SEEDS_PENDING_DIR,
    archive_seed,
    defer_seed,
    list_pending_seeds_due,
    list_unreviewed_seeds,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_seeds_dirs(tmp_path, monkeypatch):
    """Redirect all seed directories to a temp path for test isolation."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    monkeypatch.setattr("gaia_core.cognition.thought_seed.SEEDS_DIR", seeds)
    monkeypatch.setattr("gaia_core.cognition.thought_seed.SEEDS_ARCHIVE_DIR", seeds / "archive")
    monkeypatch.setattr("gaia_core.cognition.thought_seed.SEEDS_PENDING_DIR", seeds / "pending")
    return seeds


def _write_seed(seeds_dir: Path, filename: str, **overrides) -> Path:
    """Helper to create a seed JSON file."""
    data = {
        "created": datetime.now(timezone.utc).isoformat(),
        "context": {"prompt": "test", "packet_id": "pkt1", "persona": "prime"},
        "seed": "Consider implementing a caching layer",
        "reviewed": False,
        "action_taken": False,
        "result": None,
    }
    data.update(overrides)
    path = seeds_dir / filename
    path.write_text(json.dumps(data, indent=2))
    return path


# ── Seed Directory Operations ────────────────────────────────────────────


class TestSeedDirectoryOps:
    def test_archive_seed_moves_file(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_test_001.json")

        result = archive_seed("seed_test_001.json")

        assert result is True
        assert not (seeds / "seed_test_001.json").exists()
        archived = seeds / "archive" / "seed_test_001.json"
        assert archived.exists()
        data = json.loads(archived.read_text())
        assert data["archived"] is True
        assert "archived_at" in data

    def test_archive_nonexistent_returns_false(self, _patch_seeds_dirs):
        assert archive_seed("nonexistent.json") is False

    def test_defer_seed_moves_file(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_test_002.json")

        result = defer_seed("seed_test_002.json")

        assert result is True
        assert not (seeds / "seed_test_002.json").exists()
        pending = seeds / "pending" / "seed_test_002.json"
        assert pending.exists()
        data = json.loads(pending.read_text())
        assert data["pending"] is True
        assert "deferred_at" in data

    def test_defer_seed_with_revisit_after(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_test_003.json")
        revisit = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()

        defer_seed("seed_test_003.json", revisit_after=revisit)

        pending = seeds / "pending" / "seed_test_003.json"
        data = json.loads(pending.read_text())
        assert data["revisit_after"] == revisit

    def test_pending_due_promotes_back(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        pending_dir = seeds / "pending"
        pending_dir.mkdir(parents=True)

        # Seed deferred 8 days ago with no revisit_after — should be due
        old_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        _write_seed(
            pending_dir, "seed_old.json",
            pending=True, deferred_at=old_time,
        )

        promoted = list_pending_seeds_due()

        assert len(promoted) == 1
        assert (seeds / "seed_old.json").exists()
        assert not (pending_dir / "seed_old.json").exists()
        # Promoted seed should have reviewed reset
        data = json.loads((seeds / "seed_old.json").read_text())
        assert data["reviewed"] is False
        assert "pending" not in data

    def test_pending_not_due_stays(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        pending_dir = seeds / "pending"
        pending_dir.mkdir(parents=True)

        # Seed deferred 2 days ago — not due yet
        recent_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        _write_seed(
            pending_dir, "seed_recent.json",
            pending=True, deferred_at=recent_time,
        )

        promoted = list_pending_seeds_due()

        assert len(promoted) == 0
        assert (pending_dir / "seed_recent.json").exists()

    def test_pending_with_future_revisit_stays(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        pending_dir = seeds / "pending"
        pending_dir.mkdir(parents=True)

        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        _write_seed(
            pending_dir, "seed_future.json",
            pending=True,
            deferred_at=datetime.now(timezone.utc).isoformat(),
            revisit_after=future,
        )

        promoted = list_pending_seeds_due()
        assert len(promoted) == 0

    def test_pending_with_past_revisit_promotes(self, _patch_seeds_dirs):
        seeds = _patch_seeds_dirs
        pending_dir = seeds / "pending"
        pending_dir.mkdir(parents=True)

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        _write_seed(
            pending_dir, "seed_past.json",
            pending=True,
            deferred_at=datetime.now(timezone.utc).isoformat(),
            revisit_after=past,
        )

        promoted = list_pending_seeds_due()
        assert len(promoted) == 1


# ── Triage Decisions ─────────────────────────────────────────────────────


def _mock_llm(response_text: str) -> MagicMock:
    """Create a mock LLM that returns the given text."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response_text}}]
    }
    return llm


class TestTriageSeed:
    def test_archive_decision(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        llm = _mock_llm("ARCHIVE\nThis seed is too vague to act on.")

        decision, reason = hb._triage_seed(llm, {"seed": "test", "context": {}})

        assert decision == "archive"
        assert "vague" in reason

    def test_pending_decision(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        llm = _mock_llm("PENDING\nRevisit after the refactoring is done.")

        decision, reason = hb._triage_seed(llm, {"seed": "test", "context": {}})

        assert decision == "pending"

    def test_act_decision(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        llm = _mock_llm("ACT\nThis is immediately actionable.")

        decision, reason = hb._triage_seed(llm, {"seed": "test", "context": {}})

        assert decision == "act"

    def test_unparseable_defaults_to_pending(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        llm = _mock_llm("I'm not sure what to do with this seed.")

        decision, reason = hb._triage_seed(llm, {"seed": "test", "context": {}})

        assert decision == "pending"

    def test_llm_failure_defaults_to_pending(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        llm = MagicMock()
        llm.create_chat_completion.side_effect = RuntimeError("model crashed")

        decision, reason = hb._triage_seed(llm, {"seed": "test", "context": {}})

        assert decision == "pending"


# ── Act on Seed ──────────────────────────────────────────────────────────


class TestActOnSeed:
    def test_act_when_active(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat
        from gaia_core.cognition.sleep_wake_manager import GaiaState

        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_act.json")

        swm = MagicMock()
        swm.get_state.return_value = GaiaState.ACTIVE

        agent_core = MagicMock()
        agent_core.run_turn.return_value = iter([{"type": "token", "value": "done"}])

        llm = _mock_llm("Expanded prompt: investigate the caching layer")

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        hb.sleep_wake_manager = swm
        hb.agent_core = agent_core
        hb._timeline = None

        hb._act_on_seed(llm, "seed_act.json", {"seed": "caching", "context": {}})

        agent_core.run_turn.assert_called_once()
        call_kwargs = agent_core.run_turn.call_args
        assert call_kwargs.kwargs.get("source") == "heartbeat" or call_kwargs[1].get("source") == "heartbeat"

    def test_act_defers_when_dreaming(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat
        from gaia_core.cognition.sleep_wake_manager import GaiaState

        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_dream.json")

        swm = MagicMock()
        swm.get_state.return_value = GaiaState.DREAMING

        llm = _mock_llm("Expanded prompt")

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        hb.sleep_wake_manager = swm
        hb.agent_core = MagicMock()
        hb._timeline = None

        hb._act_on_seed(llm, "seed_dream.json", {"seed": "test", "context": {}})

        # Should be deferred, not acted on
        hb.agent_core.run_turn.assert_not_called()
        pending = seeds / "pending" / "seed_dream.json"
        assert pending.exists()

    def test_seed_archived_after_act(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat
        from gaia_core.cognition.sleep_wake_manager import GaiaState

        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_done.json")

        swm = MagicMock()
        swm.get_state.return_value = GaiaState.ACTIVE

        agent_core = MagicMock()
        agent_core.run_turn.return_value = iter([])

        llm = _mock_llm("Expanded prompt")

        hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
        hb.sleep_wake_manager = swm
        hb.agent_core = agent_core
        hb._timeline = None

        hb._act_on_seed(llm, "seed_done.json", {"seed": "test", "context": {}})

        # Seed should be archived after acting
        assert not (seeds / "seed_done.json").exists()
        assert (seeds / "archive" / "seed_done.json").exists()


# ── Heartbeat Lifecycle ──────────────────────────────────────────────────


class FakeConfig:
    HEARTBEAT_INTERVAL_SECONDS = 5
    HEARTBEAT_ENABLED = True


class TestHeartbeatLifecycle:
    def test_start_creates_thread(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb.start()
        try:
            assert hb._thread is not None
            assert hb._thread.is_alive()
            assert hb._thread.daemon is True
            assert hb._thread.name == "ThoughtSeedHeartbeat"
        finally:
            hb.stop()

    def test_stop_terminates_thread(self):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb.start()
        hb.stop()

        assert hb._thread is None
        assert hb._running is False

    def test_tick_emits_timeline_event(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        timeline = MagicMock()
        hb = ThoughtSeedHeartbeat(config=FakeConfig(), timeline_store=timeline)

        # _tick with no seeds should emit a tick with 0 counts
        hb._tick()

        timeline.append.assert_called_once()
        call_args = timeline.append.call_args
        assert call_args[0][0] == "heartbeat_tick"
        data = call_args[0][1]
        assert data["seeds_found"] == 0
        assert data["archived"] == 0
        assert data["deferred"] == 0
        assert data["acted"] == 0
        assert "tick_number" in data
        assert "interview_conducted" in data

    def test_tick_triages_seeds(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        seeds = _patch_seeds_dirs
        _write_seed(seeds, "seed_a.json")
        _write_seed(seeds, "seed_b.json")

        llm = _mock_llm("ARCHIVE\nNot relevant.")
        model_pool = MagicMock()
        model_pool.get_model_for_role.return_value = llm

        timeline = MagicMock()
        hb = ThoughtSeedHeartbeat(
            config=FakeConfig(),
            model_pool=model_pool,
            timeline_store=timeline,
        )

        hb._tick()

        # Both seeds should have been archived
        assert not (seeds / "seed_a.json").exists()
        assert not (seeds / "seed_b.json").exists()
        assert (seeds / "archive" / "seed_a.json").exists()
        assert (seeds / "archive" / "seed_b.json").exists()


# ── Temporal Awareness Integration ──────────────────────────────────


class TestTemporalIntegration:
    def test_tick_writes_journal_entry(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        mock_journal = MagicMock()
        mock_journal.write_entry.return_value = "Test journal entry."

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._lite_journal = mock_journal

        hb._tick()

        mock_journal.write_entry.assert_called_once()
        assert mock_journal.tick_count == 1

    def test_tick_bakes_state_on_interval(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        mock_tsm = MagicMock()
        mock_tsm.bake_state.return_value = Path("/tmp/fake_state.bin")

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._temporal_state_manager = mock_tsm
        hb._bake_interval = 3

        # Tick 3 times — bake should happen on tick 3
        hb._tick()  # tick_count=1, no bake
        hb._tick()  # tick_count=2, no bake
        hb._tick()  # tick_count=3, BAKE

        assert mock_tsm.bake_state.call_count == 1

    def test_tick_skips_bake_off_interval(self, _patch_seeds_dirs):
        from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat

        mock_tsm = MagicMock()

        hb = ThoughtSeedHeartbeat(config=FakeConfig())
        hb._temporal_state_manager = mock_tsm
        hb._bake_interval = 3

        # Only 2 ticks — shouldn't bake
        hb._tick()
        hb._tick()

        mock_tsm.bake_state.assert_not_called()
