"""
Per-session vector index for conversation turns and topic summaries.

Provides semantic retrieval over session history so that older turns
can be recalled by relevance rather than recency. Designed to complement
the sliding-window approach in agent_core._create_initial_packet().

Graceful degradation: if no embedding model is available, all operations
silently no-op and retrieve() returns empty results.
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("GAIA.SessionHistoryIndexer")

# Persistence root for per-session vector indexes
_DEFAULT_PERSIST_DIR = "data/shared/session_vectors"

# How many turns between topic summary generations
_TOPIC_INTERVAL = 6

_lock = threading.Lock()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _get_embed_model():
    """Lazy-load embedding model from ModelPool, falling back to direct load."""
    # Try 1: ModelPool (already-loaded shared instance)
    try:
        from gaia_core.models.model_pool import get_model_pool
        pool = get_model_pool()
        if pool is not None:
            model = pool.get_embed_model(timeout=0, lazy_load=True)
            if model is not None:
                return model
    except Exception:
        pass

    # Try 2: Direct SentenceTransformer load from known paths
    try:
        from sentence_transformers import SentenceTransformer
        import os
        for path in [
            os.environ.get("EMBEDDING_MODEL_PATH", ""),
            "/models/all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",  # Downloads from HuggingFace if not local
        ]:
            if path and (not path.startswith("/") or os.path.isdir(path)):
                model = SentenceTransformer(path, device="cpu")
                logger.info(f"Loaded embedding model directly from {path}")
                return model
    except Exception as e:
        logger.debug(f"Direct SentenceTransformer load failed: {e}")

    return None


class SessionHistoryIndexer:
    """Per-session vector index for conversation turns and topic summaries."""

    _instances: Dict[str, "SessionHistoryIndexer"] = {}

    def __init__(self, session_id: str, persist_dir: str = _DEFAULT_PERSIST_DIR):
        self.session_id = session_id
        self.persist_dir = persist_dir
        self._model = None  # Lazy-loaded
        self._model_checked = False

        # Turn index: individual user-assistant pairs
        self.turns: List[Dict] = []
        self.turn_embeddings: List[np.ndarray] = []

        # Topic index: summaries of turn clusters
        self.topics: List[Dict] = []
        self.topic_embeddings: List[np.ndarray] = []

        # Track how many turns have been summarized into topics
        self._last_topic_turn_idx = -1

        self._load()

    @classmethod
    def instance(cls, session_id: str) -> "SessionHistoryIndexer":
        if session_id not in cls._instances:
            cls._instances[session_id] = cls(session_id)
        return cls._instances[session_id]

    def _get_model(self):
        """Get embedding model, caching the result."""
        if not self._model_checked:
            self._model = _get_embed_model()
            self._model_checked = True
        return self._model

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into an embedding vector. Returns None if no model."""
        model = self._get_model()
        if model is None:
            return None
        try:
            embedding = model.encode([text], show_progress_bar=False)
            if hasattr(embedding, 'numpy'):
                embedding = embedding.numpy()
            return np.array(embedding[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def index_turn(self, turn_idx: int, user_msg: str, assistant_msg: str):
        """Embed a completed user-assistant pair. Called after assistant response."""
        # Skip if already indexed
        if any(t["idx"] == turn_idx for t in self.turns):
            return

        combined = f"User: {user_msg[:1000]}\nAssistant: {assistant_msg[:500]}"
        embedding = self._encode(combined)
        if embedding is None:
            return  # Graceful degradation

        self.turns.append({
            "idx": turn_idx,
            "user": user_msg[:2000],
            "assistant": assistant_msg[:2000],
            "timestamp": datetime.utcnow().isoformat()
        })
        self.turn_embeddings.append(embedding)

        self._maybe_generate_topic_summary()
        self._save()

    def retrieve(self, query: str, top_k_turns: int = 3, top_k_topics: int = 2,
                 exclude_recent_n: int = 6) -> Dict:
        """Retrieve relevant historical turns and topic summaries.

        Args:
            query: Current user input to match against.
            top_k_turns: Max turn-pairs to return.
            top_k_topics: Max topic summaries to return.
            exclude_recent_n: Skip the last N turns (they're in the sliding window).

        Returns:
            {"turns": [...], "topics": [...]}
        """
        result: Dict[str, list] = {"turns": [], "topics": []}

        query_embedding = self._encode(query)
        if query_embedding is None:
            return result

        # --- Turn retrieval (exclude sliding window) ---
        if self.turns and self.turn_embeddings:
            # Calculate how many turns to exclude from the end
            n_turns = len(self.turns)
            # exclude_recent_n is in messages (user+assistant pairs = turns)
            exclude_turns = exclude_recent_n // 2  # convert messages to turn-pairs
            searchable = max(0, n_turns - exclude_turns)

            if searchable > 0:
                similarities = []
                for i in range(searchable):
                    sim = _cosine_similarity(query_embedding, self.turn_embeddings[i])
                    similarities.append((sim, i))
                similarities.sort(reverse=True)
                for sim, idx in similarities[:top_k_turns]:
                    if sim > 0.15:  # Minimum relevance threshold
                        result["turns"].append({
                            **self.turns[idx],
                            "similarity": round(sim, 3)
                        })

        # --- Topic retrieval ---
        if self.topics and self.topic_embeddings:
            similarities = []
            for i, emb in enumerate(self.topic_embeddings):
                sim = _cosine_similarity(query_embedding, emb)
                similarities.append((sim, i))
            similarities.sort(reverse=True)
            for sim, idx in similarities[:top_k_topics]:
                if sim > 0.10:  # Lower threshold for broader topic context
                    result["topics"].append({
                        **self.topics[idx],
                        "similarity": round(sim, 3)
                    })

        return result

    def _maybe_generate_topic_summary(self):
        """Every ~_TOPIC_INTERVAL turns, generate a topic summary for unsummarized turns."""
        unsummarized_start = self._last_topic_turn_idx + 1
        unsummarized_count = len(self.turns) - unsummarized_start

        if unsummarized_count < _TOPIC_INTERVAL:
            return

        # Gather user messages from unsummarized turns
        batch = self.turns[unsummarized_start:unsummarized_start + _TOPIC_INTERVAL]
        user_messages = [t["user"] for t in batch]

        # Extractive fallback: first sentence of each user message
        summary_parts = []
        for msg in user_messages:
            first_sentence = msg.split('.')[0].strip()
            if first_sentence:
                summary_parts.append(first_sentence)
        summary = ". ".join(summary_parts[:3]) + "." if summary_parts else "General conversation."

        topic_label = summary[:80]

        # Embed the summary
        embedding = self._encode(summary)
        if embedding is None:
            return

        self.topics.append({
            "label": topic_label,
            "summary": summary,
            "turn_range": [batch[0]["idx"], batch[-1]["idx"]],
            "timestamp": datetime.utcnow().isoformat()
        })
        self.topic_embeddings.append(embedding)
        self._last_topic_turn_idx = unsummarized_start + _TOPIC_INTERVAL - 1

    def archive_and_reset(self):
        """Move current index to archive and start fresh. Called before history clear."""
        if not self.turns and not self.topics:
            return

        archive_path = os.path.join(self.persist_dir, "archive", f"{self.session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            self._save_to(archive_path)
            logger.info(f"Session vector index archived: {archive_path}")
        except Exception as e:
            logger.warning(f"Failed to archive session vector index: {e}")

        # Reset
        self.turns.clear()
        self.turn_embeddings.clear()
        self.topics.clear()
        self.topic_embeddings.clear()
        self._last_topic_turn_idx = -1
        self._save()

    def _save(self):
        """Persist index to JSON."""
        path = os.path.join(self.persist_dir, f"{self.session_id}.json")
        self._save_to(path)

    def _save_to(self, path: str):
        """Persist index to a specific path."""
        with _lock:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                data = {
                    "session_id": self.session_id,
                    "last_topic_turn_idx": self._last_topic_turn_idx,
                    "turns": self.turns,
                    "turn_embeddings": [e.tolist() for e in self.turn_embeddings],
                    "topics": self.topics,
                    "topic_embeddings": [e.tolist() for e in self.topic_embeddings],
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save session vector index to {path}: {e}")

    def _load(self):
        """Load index from JSON if it exists."""
        path = os.path.join(self.persist_dir, f"{self.session_id}.json")
        with _lock:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.turns = data.get("turns", [])
                    self.turn_embeddings = [np.array(e, dtype=np.float32) for e in data.get("turn_embeddings", [])]
                    self.topics = data.get("topics", [])
                    self.topic_embeddings = [np.array(e, dtype=np.float32) for e in data.get("topic_embeddings", [])]
                    self._last_topic_turn_idx = data.get("last_topic_turn_idx", -1)
                    logger.debug(f"Loaded session index for {self.session_id}: {len(self.turns)} turns, {len(self.topics)} topics")
            except Exception as e:
                logger.warning(f"Failed to load session vector index: {e}")
