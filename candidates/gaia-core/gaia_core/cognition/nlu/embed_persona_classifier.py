"""
Embedding-based persona classifier for GAIA.

Replaces the keyword fallback in `persona_switcher.get_persona_for_request`
with cosine-similarity classification against a bank of persona-typed
exemplar phrases. Uses the same MiniLM-L6-v2 model already loaded by
ModelPool for semantic probe and intent classification.

Design mirrors EmbedIntentClassifier:
  1. On first call, loads persona_exemplars.json and encodes all phrases
     into a matrix of embeddings (cached singleton).
  2. At inference, encodes the user query, computes cosine similarity
     against all persona exemplars, and returns the persona label of
     the strongest aggregate match (mean of top-k per persona).
  3. Returns (None, score) if no persona exceeds the confidence threshold.

Why a separate classifier from the intent one: intent and persona are
orthogonal axes. A user can be in the dnd_player_assistant persona while
expressing a write_file intent ("update Rupert's character sheet"), and
mixing them into one classifier muddles both signals.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("GAIA.EmbedPersonaClassifier")

_EXEMPLARS_FILE = Path(__file__).parent / "persona_exemplars.json"


class EmbedPersonaClassifier:
    """Singleton embedding-based persona classifier."""

    _instance: Optional["EmbedPersonaClassifier"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._ready = False
        self._labels: list[str] = []
        self._exemplar_matrix = None  # (N, dim) numpy array
        self._embed_model = None

    @classmethod
    def instance(cls) -> "EmbedPersonaClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialise(self, embed_model, config: Optional[dict] = None) -> bool:
        """Encode all persona exemplar phrases. Returns True on success.

        Args:
            embed_model: A SentenceTransformer (or compatible) instance
                exposing `.encode(texts, ...)`.
            config: Optional EMBED_PERSONA config dict from gaia_constants.
                Recognised keys: ``exemplars_path`` (override file).
        """
        if self._ready:
            return True

        cfg = config or {}
        exemplars_path = cfg.get("exemplars_path") or str(_EXEMPLARS_FILE)

        try:
            with open(exemplars_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception:
            logger.error("Failed to load persona exemplars from %s", exemplars_path)
            return False

        labels: list[str] = []
        phrases: list[str] = []
        for persona_label, examples in bank.items():
            if persona_label.startswith("_"):
                continue
            if not isinstance(examples, list):
                continue
            for phrase in examples:
                labels.append(persona_label)
                phrases.append(phrase)

        if not phrases:
            logger.warning("Persona exemplar bank is empty; classifier disabled.")
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
                "EmbedPersonaClassifier ready: %d exemplars across %d personas",
                len(phrases),
                len(set(labels)),
            )
            return True
        except Exception:
            logger.exception("Failed to encode persona exemplars")
            return False

    def classify(
        self,
        text: str,
        confidence_threshold: float = 0.45,
        top_k: int = 3,
    ) -> Tuple[Optional[str], float]:
        """Classify a user query by embedding similarity to persona banks.

        Args:
            text: User query.
            confidence_threshold: Minimum aggregated cosine score for a
                positive match.
            top_k: Aggregate score per persona = mean of top_k similarities.
                Higher k = more conservative (requires broader fit).

        Returns:
            (persona_label, score) on match, (None, best_score) on miss.
        """
        if not self._ready or self._embed_model is None:
            return (None, 0.0)

        if not text or not text.strip():
            return (None, 0.0)

        try:
            query_vec = self._embed_model.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            query_vec = np.asarray(query_vec, dtype=np.float32)

            similarities = (self._exemplar_matrix @ query_vec.T).squeeze()
            if similarities.ndim == 0:
                similarities = np.atleast_1d(similarities)

            persona_scores: dict[str, list[float]] = {}
            for idx, score in enumerate(similarities):
                label = self._labels[idx]
                persona_scores.setdefault(label, []).append(float(score))

            best_persona: Optional[str] = None
            best_score = 0.0
            for label, scores in persona_scores.items():
                top_scores = sorted(scores, reverse=True)[:top_k]
                avg = sum(top_scores) / len(top_scores)
                if avg > best_score:
                    best_score = avg
                    best_persona = label

            if best_persona is None or best_score < confidence_threshold:
                logger.info(
                    "Embed persona below threshold: best=%s score=%.3f (threshold=%.2f)",
                    best_persona, best_score, confidence_threshold,
                )
                return (None, best_score)

            logger.info(
                "Embed persona classified: %s (score=%.3f)",
                best_persona, best_score,
            )
            return (best_persona, best_score)

        except Exception:
            logger.exception("Embed persona classification failed")
            return (None, 0.0)

    @property
    def ready(self) -> bool:
        return self._ready
