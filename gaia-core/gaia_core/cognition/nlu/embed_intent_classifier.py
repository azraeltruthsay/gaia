"""
Embedding-based intent classifier for GAIA.

Replaces the keyword-heuristic fallback in intent_detection.py with
cosine-similarity classification against a bank of labeled exemplar
phrases.  Uses the same MiniLM-L6-v2 model already loaded by ModelPool
for semantic probe and session history indexing.

Design:
  1. On first call, loads intent_exemplars.json and encodes all
     exemplar phrases into a matrix of embeddings (cached for session).
  2. At inference, encodes the user query, computes cosine similarity
     against all exemplar embeddings, and returns the intent label
     of the nearest match (if above the confidence threshold).
  3. Falls back to "other" if no exemplar exceeds the threshold.

Thread-safety: the classifier is a singleton initialised under a lock.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("GAIA.EmbedIntentClassifier")

_EXEMPLARS_FILE = Path(__file__).parent / "intent_exemplars.json"


class EmbedIntentClassifier:
    """Singleton embedding-based intent classifier."""

    _instance: Optional["EmbedIntentClassifier"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._ready = False
        self._labels: list[str] = []          # one label per exemplar row
        self._exemplar_matrix = None           # (N, dim) numpy array
        self._embed_model = None

    @classmethod
    def instance(cls) -> "EmbedIntentClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, embed_model, config: Optional[dict] = None) -> bool:
        """Encode all exemplar phrases.  Returns True on success.

        Args:
            embed_model: A SentenceTransformer (or compatible) instance
                         that exposes `.encode(texts, ...)`.
            config: Optional EMBED_INTENT config dict from gaia_constants.
        """
        if self._ready:
            return True

        cfg = config or {}
        self._other_penalty = cfg.get("other_penalty", 0.10)
        exemplars_path = cfg.get("exemplars_path") or str(_EXEMPLARS_FILE)

        try:
            with open(exemplars_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception:
            logger.error("Failed to load intent exemplars from %s", exemplars_path)
            return False

        labels: list[str] = []
        phrases: list[str] = []
        for intent_label, examples in bank.items():
            if intent_label.startswith("_"):
                continue  # skip _comment etc.
            for phrase in examples:
                labels.append(intent_label)
                phrases.append(phrase)

        if not phrases:
            logger.warning("Intent exemplar bank is empty; classifier disabled.")
            return False

        try:
            self._embed_model = embed_model
            vectors = embed_model.encode(
                phrases,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self._exemplar_matrix = np.asarray(vectors, dtype=np.float32)
            self._labels = labels
            self._ready = True
            logger.warning(
                "EmbedIntentClassifier ready: %d exemplars across %d intents",
                len(phrases),
                len(set(labels)),
            )
            return True
        except Exception:
            logger.exception("Failed to encode intent exemplars")
            return False

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        confidence_threshold: float = 0.45,
        top_k: int = 3,
    ) -> Tuple[str, float]:
        """Classify a user query by embedding similarity.

        Returns:
            (intent_label, confidence_score).
            Falls back to ("other", 0.0) if not ready or below threshold.
        """
        if not self._ready or self._embed_model is None:
            return ("other", 0.0)

        try:
            query_vec = self._embed_model.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            query_vec = np.asarray(query_vec, dtype=np.float32)

            # Cosine similarity (vectors are already L2-normalised)
            similarities = (self._exemplar_matrix @ query_vec.T).squeeze()

            # Aggregate: for each intent, take the mean of top-k similarities
            intent_scores: dict[str, list[float]] = {}
            for idx, score in enumerate(similarities):
                label = self._labels[idx]
                intent_scores.setdefault(label, []).append(float(score))

            best_intent = "other"
            best_score = 0.0

            other_penalty = getattr(self, '_other_penalty', 0.10)
            for label, scores in intent_scores.items():
                # Use the mean of the top-k scores for this intent
                top_scores = sorted(scores, reverse=True)[:top_k]
                avg = sum(top_scores) / len(top_scores)
                # Penalize "other" so close calls prefer a specific intent
                if label == "other":
                    avg -= other_penalty
                if avg > best_score:
                    best_score = avg
                    best_intent = label

            if best_score < confidence_threshold:
                logger.info(
                    "Embed intent below threshold: best=%s score=%.3f (threshold=%.2f)",
                    best_intent, best_score, confidence_threshold,
                )
                return ("other", best_score)

            logger.info(
                "Embed intent classified: %s (score=%.3f)",
                best_intent, best_score,
            )
            return (best_intent, best_score)

        except Exception:
            logger.exception("Embed intent classification failed")
            return ("other", 0.0)

    @property
    def ready(self) -> bool:
        return self._ready
