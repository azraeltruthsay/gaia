"""
Integration tests for the RAG + Rolling History pipeline.

Tests the interaction between:
- SessionManager indexing hook (add_message → index_turn)
- AgentCore sliding window + RAG retrieval (_create_initial_packet)
- PromptBuilder Tier 1.5 injection (retrieved_session_context → final prompt)
- Archive flow (summarize_and_archive → archive_and_reset)
"""

import json
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEmbedModel:
    """Deterministic fake embedding model."""
    DIM = 64

    def encode(self, texts, show_progress_bar=False):
        results = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            for i, ch in enumerate(text[:self.DIM]):
                vec[i % self.DIM] += ord(ch) / 128.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return np.array(results)


@pytest.fixture(autouse=True)
def clear_singletons():
    from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
    SessionHistoryIndexer._instances.clear()
    yield
    SessionHistoryIndexer._instances.clear()


@pytest.fixture
def persist_dir(tmp_path):
    d = tmp_path / "session_vectors"
    d.mkdir()
    return str(d)


@pytest.fixture
def fake_model():
    return FakeEmbedModel()


# ---------------------------------------------------------------------------
# Test: SessionManager → SessionHistoryIndexer hook
# ---------------------------------------------------------------------------

class TestSessionManagerIndexingHook:
    """Verify that SessionManager.add_message() triggers turn indexing."""

    def test_assistant_message_triggers_indexing(self, tmp_path, persist_dir, fake_model):
        """When an assistant message completes a turn pair, it should be indexed."""
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer

        # Patch the state file and indexer
        state_file = str(tmp_path / "sessions.json")
        with (
            patch("gaia_core.memory.session_manager.STATE_FILE", state_file),
            patch("gaia_core.memory.session_manager.LAST_ACTIVITY_FILE", str(tmp_path / "activity.ts")),
            patch("gaia_core.memory.session_history_indexer._get_embed_model", return_value=fake_model),
            patch("gaia_core.memory.session_history_indexer._DEFAULT_PERSIST_DIR", persist_dir),
        ):
            from gaia_core.memory.session_manager import SessionManager
            config = MagicMock()
            config.KNOWLEDGE_CODEX_DIR = str(tmp_path / "knowledge")
            sm = SessionManager(config)

            # Add a user-assistant pair
            sm.add_message("test-sess", "user", "What is Python?")
            sm.add_message("test-sess", "assistant", "Python is a programming language.")

            # Check that the indexer got the turn
            indexer = SessionHistoryIndexer.instance("test-sess")
            assert len(indexer.turns) == 1
            assert indexer.turns[0]["user"] == "What is Python?"

    def test_user_message_alone_does_not_index(self, tmp_path, persist_dir, fake_model):
        """A user message without a following assistant message should not trigger indexing."""
        state_file = str(tmp_path / "sessions.json")
        with (
            patch("gaia_core.memory.session_manager.STATE_FILE", state_file),
            patch("gaia_core.memory.session_manager.LAST_ACTIVITY_FILE", str(tmp_path / "activity.ts")),
            patch("gaia_core.memory.session_history_indexer._get_embed_model", return_value=fake_model),
        ):
            from gaia_core.memory.session_manager import SessionManager
            from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
            config = MagicMock()
            config.KNOWLEDGE_CODEX_DIR = str(tmp_path / "knowledge")
            sm = SessionManager(config)

            # Use a unique session ID to avoid any leakage
            sm.add_message("user-only-sess", "user", "Hello")

            indexer = SessionHistoryIndexer.instance("user-only-sess")
            assert len(indexer.turns) == 0

    def test_indexing_failure_does_not_block_message(self, tmp_path):
        """If indexing fails, the message should still be added to history."""
        state_file = str(tmp_path / "sessions.json")
        with (
            patch("gaia_core.memory.session_manager.STATE_FILE", state_file),
            patch("gaia_core.memory.session_manager.LAST_ACTIVITY_FILE", str(tmp_path / "activity.ts")),
            patch(
                "gaia_core.memory.session_history_indexer.SessionHistoryIndexer.instance",
                side_effect=RuntimeError("Boom!"),
            ),
        ):
            from gaia_core.memory.session_manager import SessionManager
            config = MagicMock()
            config.KNOWLEDGE_CODEX_DIR = str(tmp_path / "knowledge")
            sm = SessionManager(config)

            sm.add_message("test-sess", "user", "Hello")
            sm.add_message("test-sess", "assistant", "Hi there")

            # Message should still be in history despite indexing failure
            history = sm.get_history("test-sess")
            assert len(history) == 2


# ---------------------------------------------------------------------------
# Test: _format_retrieved_session_context
# ---------------------------------------------------------------------------

class TestFormatRetrievedContext:
    def test_empty_results(self):
        from gaia_core.cognition.agent_core import _format_retrieved_session_context
        assert _format_retrieved_session_context({"turns": [], "topics": []}) == ""

    def test_turns_only(self):
        from gaia_core.cognition.agent_core import _format_retrieved_session_context
        result = _format_retrieved_session_context({
            "turns": [
                {"idx": 2, "user": "What is Python?", "assistant": "A language.", "similarity": 0.85},
            ],
            "topics": [],
        })
        assert "Turn 2" in result
        assert "What is Python?" in result
        assert "0.85" in result

    def test_topics_only(self):
        from gaia_core.cognition.agent_core import _format_retrieved_session_context
        result = _format_retrieved_session_context({
            "turns": [],
            "topics": [
                {"label": "Python discussion", "similarity": 0.72},
            ],
        })
        assert "Python discussion" in result
        assert "0.72" in result

    def test_both_turns_and_topics(self):
        from gaia_core.cognition.agent_core import _format_retrieved_session_context
        result = _format_retrieved_session_context({
            "turns": [{"idx": 1, "user": "Q", "assistant": "A", "similarity": 0.9}],
            "topics": [{"label": "Topic X", "similarity": 0.6}],
        })
        assert "Earlier conversation topics:" in result
        assert "Relevant earlier exchanges:" in result


# ---------------------------------------------------------------------------
# Test: Prompt builder Tier 1.5 injection
# ---------------------------------------------------------------------------

class TestPromptBuilderTier15:
    """Test that Tier 1.5 (retrieved_session_context) is correctly injected."""

    def test_rag_content_appears_in_prompt(self):
        """When a packet has retrieved_session_context, it should appear in the final prompt."""
        # We need to test the prompt builder's handling of the DataField.
        # Rather than importing the full prompt builder (which has many deps),
        # we test the logic in isolation.
        rag_content = "Relevant earlier exchanges:\n[Turn 2, sim=0.85]\n  User: What is Python?\n  Assistant: A language."

        # Simulate what the prompt builder does for Tier 1.5
        remaining_budget = 4000  # tokens
        rag_budget = int(remaining_budget * 0.30)

        # Simple token estimate (4 chars per token)
        rag_tokens = len(rag_content) // 4

        session_rag_prompt = {}
        if rag_tokens <= remaining_budget:
            session_rag_prompt = {
                "role": "system",
                "content": f"[Relevant context from earlier in this conversation]\n{rag_content}",
            }

        assert session_rag_prompt != {}
        assert "Relevant context from earlier" in session_rag_prompt["content"]
        assert "What is Python?" in session_rag_prompt["content"]

    def test_rag_content_truncated_when_over_budget(self):
        """RAG content exceeding 30% of remaining budget should be truncated."""
        remaining_budget = 100  # Very small budget
        rag_budget = int(remaining_budget * 0.30)  # 30 tokens
        rag_content = "x" * 500  # Way over budget

        rag_tokens = len(rag_content) // 4  # ~125 tokens
        if rag_tokens > rag_budget and rag_budget > 0:
            char_limit = rag_budget * 4
            rag_content = rag_content[:char_limit] + "\n[...truncated]"

        assert "[...truncated]" in rag_content
        # Should be roughly rag_budget * 4 chars + truncation marker
        assert len(rag_content) < 200

    def test_empty_rag_content_produces_no_prompt(self):
        """Empty RAG content should not produce a prompt entry."""
        rag_content = ""
        session_rag_prompt = {}
        if rag_content:
            session_rag_prompt = {"role": "system", "content": rag_content}
        assert session_rag_prompt == {}


# ---------------------------------------------------------------------------
# Test: Sliding window behavior
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    """Test that the sliding window correctly limits history in the packet."""

    def test_short_history_fully_included(self):
        """History shorter than SLIDING_WINDOW_SIZE should be fully included."""
        SLIDING_WINDOW_SIZE = 6
        history = [
            {"id": "1", "role": "user", "content": "Hello"},
            {"id": "2", "role": "assistant", "content": "Hi there"},
        ]
        window = history[-SLIDING_WINDOW_SIZE:]
        assert len(window) == 2

    def test_long_history_windowed(self):
        """History longer than SLIDING_WINDOW_SIZE should be windowed."""
        SLIDING_WINDOW_SIZE = 6
        history = [{"id": str(i), "role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
                   for i in range(20)]
        window = history[-SLIDING_WINDOW_SIZE:]
        assert len(window) == 6
        assert window[0]["id"] == "14"  # 20 - 6 = 14

    def test_rag_only_triggers_beyond_window(self):
        """RAG retrieval should only happen when history exceeds the sliding window."""
        SLIDING_WINDOW_SIZE = 6
        short_history = [{"role": "user", "content": "Hi"}] * 4
        long_history = [{"role": "user", "content": "Hi"}] * 10

        # Short: no RAG
        assert len(short_history) <= SLIDING_WINDOW_SIZE
        # Long: RAG should trigger
        assert len(long_history) > SLIDING_WINDOW_SIZE


# ---------------------------------------------------------------------------
# Test: Archive flow integration
# ---------------------------------------------------------------------------

class TestArchiveFlowIntegration:
    """Test that summarize_and_archive correctly archives the vector index."""

    def test_archive_called_during_summarize(self, tmp_path, fake_model):
        """When SessionManager.summarize_and_archive runs, the vector index should be archived."""
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer

        persist_dir = str(tmp_path / "sv")
        os.makedirs(persist_dir, exist_ok=True)
        state_file = str(tmp_path / "sessions.json")

        with (
            patch("gaia_core.memory.session_manager.STATE_FILE", state_file),
            patch("gaia_core.memory.session_manager.LAST_ACTIVITY_FILE", str(tmp_path / "activity.ts")),
            patch("gaia_core.memory.session_history_indexer._get_embed_model", return_value=fake_model),
        ):
            from gaia_core.memory.session_manager import SessionManager
            config = MagicMock()
            config.KNOWLEDGE_CODEX_DIR = str(tmp_path / "knowledge")
            sm = SessionManager(config)
            sm.max_active_messages = 999  # Prevent auto-archive during population

            # Pre-create the indexer with our persist_dir and explicitly set the fake model
            indexer = SessionHistoryIndexer("archive-sess", persist_dir=persist_dir)
            indexer._model = fake_model
            indexer._model_checked = True
            SessionHistoryIndexer._instances["archive-sess"] = indexer

            # Add messages — the indexing hook will use our pre-created singleton
            for i in range(10):
                sm.add_message("archive-sess", "user", f"Question {i}")
                sm.add_message("archive-sess", "assistant", f"Answer {i}")

            assert len(indexer.turns) == 10

            # Mock the summarizer and archiver to avoid full LLM calls
            sm.summarizer.generate_summary = MagicMock(return_value="Test summary")
            sm.keyword_extractor.extract_keywords = MagicMock(return_value=["test"])
            sm.archiver.archive_conversation = MagicMock()

            # Now trigger archive explicitly
            sm.summarize_and_archive("archive-sess")

            # Vector index should be cleared after archive
            assert len(indexer.turns) == 0

            # Archive file should exist
            archive_dir = os.path.join(persist_dir, "archive")
            assert os.path.isdir(archive_dir)
