"""
Tests for SessionHistoryIndexer — the per-session vector index that powers
the RAG component of the rolling history feature.

Tests cover:
1. Instantiation and singleton pattern
2. Turn indexing (with and without embedding model)
3. Retrieval (semantic similarity, exclusion window, thresholds)
4. Topic summary generation
5. Persistence (save/load cycle)
6. Archive and reset
7. Graceful degradation when no embedding model is available
"""

import json
import os
import shutil
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_singletons():
    """Clear the class-level singleton cache between tests."""
    from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
    SessionHistoryIndexer._instances.clear()
    yield
    SessionHistoryIndexer._instances.clear()


@pytest.fixture
def persist_dir(tmp_path):
    """Provide a temporary directory for vector index persistence."""
    d = tmp_path / "session_vectors"
    d.mkdir()
    return str(d)


class FakeEmbedModel:
    """A deterministic fake embedding model for testing.

    Encodes text by hashing characters into a fixed-dim vector.
    This gives us stable, reproducible similarity scores.
    """
    DIM = 64

    def encode(self, texts, show_progress_bar=False):
        results = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            for i, ch in enumerate(text[:self.DIM]):
                vec[i % self.DIM] += ord(ch) / 128.0
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return np.array(results)


@pytest.fixture
def fake_model():
    return FakeEmbedModel()


@pytest.fixture
def patched_model(fake_model):
    """Patch the model loader to return our fake model."""
    with patch(
        "gaia_core.memory.session_history_indexer._get_embed_model",
        return_value=fake_model,
    ):
        yield fake_model


@pytest.fixture
def no_model():
    """Patch the model loader to return None (no embedding model available)."""
    with patch(
        "gaia_core.memory.session_history_indexer._get_embed_model",
        return_value=None,
    ):
        yield


@pytest.fixture
def indexer(persist_dir, patched_model):
    """A fresh SessionHistoryIndexer with a fake embedding model."""
    from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
    return SessionHistoryIndexer("test-session-001", persist_dir=persist_dir)


@pytest.fixture
def indexer_no_model(persist_dir, no_model):
    """A SessionHistoryIndexer with no embedding model (degradation mode)."""
    from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
    return SessionHistoryIndexer("test-session-no-model", persist_dir=persist_dir)


# ---------------------------------------------------------------------------
# 1. Instantiation and singleton pattern
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_creates_empty_index(self, indexer):
        assert indexer.turns == []
        assert indexer.turn_embeddings == []
        assert indexer.topics == []
        assert indexer.topic_embeddings == []
        assert indexer._last_topic_turn_idx == -1

    def test_singleton_returns_same_instance(self, persist_dir, patched_model):
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
        a = SessionHistoryIndexer.instance("singleton-test")
        b = SessionHistoryIndexer.instance("singleton-test")
        assert a is b

    def test_singleton_different_sessions_are_different(self, persist_dir, patched_model):
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
        a = SessionHistoryIndexer.instance("session-A")
        b = SessionHistoryIndexer.instance("session-B")
        assert a is not b


# ---------------------------------------------------------------------------
# 2. Turn indexing
# ---------------------------------------------------------------------------

class TestTurnIndexing:
    def test_index_one_turn(self, indexer):
        indexer.index_turn(0, "Hello, how are you?", "I'm doing well, thank you!")
        assert len(indexer.turns) == 1
        assert len(indexer.turn_embeddings) == 1
        assert indexer.turns[0]["idx"] == 0
        assert indexer.turns[0]["user"] == "Hello, how are you?"
        assert indexer.turns[0]["assistant"] == "I'm doing well, thank you!"
        assert "timestamp" in indexer.turns[0]

    def test_index_multiple_turns(self, indexer):
        for i in range(5):
            indexer.index_turn(i, f"Question {i}", f"Answer {i}")
        assert len(indexer.turns) == 5
        assert len(indexer.turn_embeddings) == 5

    def test_duplicate_turn_idx_is_skipped(self, indexer):
        indexer.index_turn(0, "First", "Response")
        indexer.index_turn(0, "Duplicate", "Should be skipped")
        assert len(indexer.turns) == 1
        assert indexer.turns[0]["user"] == "First"

    def test_long_messages_are_truncated(self, indexer):
        long_msg = "x" * 5000
        indexer.index_turn(0, long_msg, long_msg)
        assert len(indexer.turns[0]["user"]) == 2000
        assert len(indexer.turns[0]["assistant"]) == 2000

    def test_embedding_is_numpy_array(self, indexer):
        indexer.index_turn(0, "Test", "Response")
        assert isinstance(indexer.turn_embeddings[0], np.ndarray)
        assert indexer.turn_embeddings[0].dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def _populate(self, indexer, n=10):
        """Populate indexer with n diverse turns."""
        topics = [
            ("What is Python?", "Python is a programming language."),
            ("Tell me about cats", "Cats are domesticated felines."),
            ("How does gravity work?", "Gravity is a fundamental force."),
            ("What is photosynthesis?", "Plants convert sunlight to energy."),
            ("Explain recursion", "A function that calls itself."),
            ("What is the capital of France?", "Paris is the capital."),
            ("How to bake bread?", "Mix flour, water, yeast..."),
            ("What is machine learning?", "ML is a subset of AI."),
            ("Tell me about the ocean", "The ocean covers 71% of Earth."),
            ("How do computers work?", "Computers process binary instructions."),
        ]
        for i in range(n):
            user, assistant = topics[i % len(topics)]
            indexer.index_turn(i, user, assistant)

    def test_retrieve_returns_dict_with_turns_and_topics(self, indexer):
        self._populate(indexer, 3)
        result = indexer.retrieve("programming")
        assert "turns" in result
        assert "topics" in result

    def test_retrieve_empty_index_returns_empty(self, indexer):
        result = indexer.retrieve("anything")
        assert result == {"turns": [], "topics": []}

    def test_retrieve_excludes_recent_turns(self, indexer):
        self._populate(indexer, 10)
        # With exclude_recent_n=6, we exclude the last 3 turn-pairs (6 messages / 2)
        result = indexer.retrieve("Python programming", exclude_recent_n=6)
        returned_idxs = [t["idx"] for t in result["turns"]]
        # None of the returned turns should be from the last 3 turn-pairs (idx 7, 8, 9)
        for idx in returned_idxs:
            assert idx < 7, f"Turn {idx} should have been excluded by sliding window"

    def test_retrieve_returns_similarity_scores(self, indexer):
        self._populate(indexer, 5)
        result = indexer.retrieve("Python", exclude_recent_n=0)
        if result["turns"]:
            for turn in result["turns"]:
                assert "similarity" in turn
                assert 0.0 <= turn["similarity"] <= 1.0

    def test_retrieve_respects_top_k(self, indexer):
        self._populate(indexer, 10)
        result = indexer.retrieve("test", top_k_turns=2, exclude_recent_n=0)
        assert len(result["turns"]) <= 2

    def test_retrieve_filters_by_minimum_threshold(self, indexer):
        """All returned turns should meet the 0.15 minimum similarity threshold."""
        self._populate(indexer, 10)
        result = indexer.retrieve("Python programming language", exclude_recent_n=0)
        for turn in result["turns"]:
            assert turn["similarity"] >= 0.15


# ---------------------------------------------------------------------------
# 4. Topic summary generation
# ---------------------------------------------------------------------------

class TestTopicSummaries:
    def test_no_topic_before_interval(self, indexer):
        """Topics should not be generated before _TOPIC_INTERVAL turns."""
        for i in range(5):
            indexer.index_turn(i, f"Question {i}", f"Answer {i}")
        assert len(indexer.topics) == 0

    def test_topic_generated_at_interval(self, indexer):
        """A topic summary should appear after _TOPIC_INTERVAL (6) turns."""
        from gaia_core.memory.session_history_indexer import _TOPIC_INTERVAL
        for i in range(_TOPIC_INTERVAL):
            indexer.index_turn(i, f"Question about topic {i}", f"Answer for topic {i}")
        assert len(indexer.topics) == 1
        assert "label" in indexer.topics[0]
        assert "summary" in indexer.topics[0]
        assert "turn_range" in indexer.topics[0]

    def test_topic_has_correct_turn_range(self, indexer):
        from gaia_core.memory.session_history_indexer import _TOPIC_INTERVAL
        for i in range(_TOPIC_INTERVAL):
            indexer.index_turn(i, f"Q{i}", f"A{i}")
        topic = indexer.topics[0]
        assert topic["turn_range"] == [0, _TOPIC_INTERVAL - 1]

    def test_multiple_topics_generated(self, indexer):
        from gaia_core.memory.session_history_indexer import _TOPIC_INTERVAL
        for i in range(_TOPIC_INTERVAL * 2):
            indexer.index_turn(i, f"Question {i}", f"Answer {i}")
        assert len(indexer.topics) == 2

    def test_topic_embedding_stored(self, indexer):
        from gaia_core.memory.session_history_indexer import _TOPIC_INTERVAL
        for i in range(_TOPIC_INTERVAL):
            indexer.index_turn(i, f"Q{i}", f"A{i}")
        assert len(indexer.topic_embeddings) == 1
        assert isinstance(indexer.topic_embeddings[0], np.ndarray)

    def test_topics_retrievable(self, indexer):
        from gaia_core.memory.session_history_indexer import _TOPIC_INTERVAL
        for i in range(_TOPIC_INTERVAL):
            indexer.index_turn(i, f"Tell me about Python {i}", f"Python is great {i}")
        result = indexer.retrieve("Python programming", exclude_recent_n=0, top_k_topics=5)
        # Should have at least one topic
        assert len(result["topics"]) >= 1


# ---------------------------------------------------------------------------
# 5. Persistence (save/load cycle)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_json_file(self, indexer, persist_dir):
        indexer.index_turn(0, "Hello", "Hi there")
        expected_path = os.path.join(persist_dir, "test-session-001.json")
        assert os.path.exists(expected_path)

    def test_load_restores_turns(self, persist_dir, patched_model):
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
        # Create and populate
        idx1 = SessionHistoryIndexer("persist-test", persist_dir=persist_dir)
        idx1.index_turn(0, "Question A", "Answer A")
        idx1.index_turn(1, "Question B", "Answer B")

        # Create a new instance from the same persist dir (simulating restart)
        SessionHistoryIndexer._instances.clear()
        idx2 = SessionHistoryIndexer("persist-test", persist_dir=persist_dir)

        assert len(idx2.turns) == 2
        assert idx2.turns[0]["user"] == "Question A"
        assert idx2.turns[1]["user"] == "Question B"
        assert len(idx2.turn_embeddings) == 2
        assert isinstance(idx2.turn_embeddings[0], np.ndarray)

    def test_load_restores_topics(self, persist_dir, patched_model):
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer, _TOPIC_INTERVAL
        idx1 = SessionHistoryIndexer("persist-topics", persist_dir=persist_dir)
        for i in range(_TOPIC_INTERVAL):
            idx1.index_turn(i, f"Q{i}", f"A{i}")
        assert len(idx1.topics) == 1

        SessionHistoryIndexer._instances.clear()
        idx2 = SessionHistoryIndexer("persist-topics", persist_dir=persist_dir)
        assert len(idx2.topics) == 1
        assert len(idx2.topic_embeddings) == 1

    def test_load_restores_last_topic_turn_idx(self, persist_dir, patched_model):
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer, _TOPIC_INTERVAL
        idx1 = SessionHistoryIndexer("persist-idx", persist_dir=persist_dir)
        for i in range(_TOPIC_INTERVAL):
            idx1.index_turn(i, f"Q{i}", f"A{i}")
        saved_idx = idx1._last_topic_turn_idx

        SessionHistoryIndexer._instances.clear()
        idx2 = SessionHistoryIndexer("persist-idx", persist_dir=persist_dir)
        assert idx2._last_topic_turn_idx == saved_idx

    def test_corrupt_file_starts_fresh(self, persist_dir, patched_model):
        from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
        # Write corrupt JSON
        path = os.path.join(persist_dir, "corrupt-session.json")
        with open(path, 'w') as f:
            f.write("{{{invalid json")

        idx = SessionHistoryIndexer("corrupt-session", persist_dir=persist_dir)
        assert idx.turns == []
        assert idx.turn_embeddings == []


# ---------------------------------------------------------------------------
# 6. Archive and reset
# ---------------------------------------------------------------------------

class TestArchiveAndReset:
    def test_archive_creates_archive_file(self, indexer, persist_dir):
        indexer.index_turn(0, "Hello", "Hi")
        indexer.archive_and_reset()

        archive_dir = os.path.join(persist_dir, "archive")
        assert os.path.isdir(archive_dir)
        archive_files = os.listdir(archive_dir)
        assert len(archive_files) == 1
        assert archive_files[0].startswith("test-session-001_")
        assert archive_files[0].endswith(".json")

    def test_archive_clears_index(self, indexer):
        indexer.index_turn(0, "Hello", "Hi")
        indexer.index_turn(1, "World", "Earth")
        assert len(indexer.turns) == 2

        indexer.archive_and_reset()

        assert indexer.turns == []
        assert indexer.turn_embeddings == []
        assert indexer.topics == []
        assert indexer.topic_embeddings == []
        assert indexer._last_topic_turn_idx == -1

    def test_archive_preserves_data_in_archive_file(self, indexer, persist_dir):
        indexer.index_turn(0, "Important question", "Critical answer")
        indexer.archive_and_reset()

        archive_dir = os.path.join(persist_dir, "archive")
        archive_file = os.listdir(archive_dir)[0]
        with open(os.path.join(archive_dir, archive_file), 'r') as f:
            data = json.load(f)
        assert len(data["turns"]) == 1
        assert data["turns"][0]["user"] == "Important question"

    def test_archive_empty_index_is_noop(self, indexer, persist_dir):
        indexer.archive_and_reset()
        archive_dir = os.path.join(persist_dir, "archive")
        assert not os.path.exists(archive_dir)

    def test_new_turns_after_archive(self, indexer):
        indexer.index_turn(0, "Before archive", "Response")
        indexer.archive_and_reset()
        indexer.index_turn(0, "After archive", "New response")
        assert len(indexer.turns) == 1
        assert indexer.turns[0]["user"] == "After archive"


# ---------------------------------------------------------------------------
# 7. Graceful degradation (no embedding model)
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_index_turn_is_noop_without_model(self, indexer_no_model):
        indexer_no_model.index_turn(0, "Hello", "Hi")
        assert len(indexer_no_model.turns) == 0
        assert len(indexer_no_model.turn_embeddings) == 0

    def test_retrieve_returns_empty_without_model(self, indexer_no_model):
        result = indexer_no_model.retrieve("anything")
        assert result == {"turns": [], "topics": []}

    def test_archive_works_without_model(self, indexer_no_model):
        # Should not crash even with empty index
        indexer_no_model.archive_and_reset()

    def test_no_topics_generated_without_model(self, indexer_no_model):
        for i in range(10):
            indexer_no_model.index_turn(i, f"Q{i}", f"A{i}")
        assert len(indexer_no_model.topics) == 0


# ---------------------------------------------------------------------------
# 8. Cosine similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from gaia_core.memory.session_history_indexer import _cosine_similarity
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        from gaia_core.memory.session_history_indexer import _cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        from gaia_core.memory.session_history_indexer import _cosine_similarity
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        from gaia_core.memory.session_history_indexer import _cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6
