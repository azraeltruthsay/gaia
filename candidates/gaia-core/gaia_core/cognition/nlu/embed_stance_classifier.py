"""
Embedding-based stance classifier for GAIA.

Stance = how the user wants the reply framed (advisor voice vs in-character
vs narrative observation, etc.) — orthogonal to persona (which topic) and
intent (which action). Detects the user's framing so the response generator
can match it (e.g. emit `(As in-universe GAIA)` voice tag for in-character
invitations).

Phase 2 scope: a single stance class — `in_character_invitation`. Anything
below threshold falls through to default advisor voice. Future phases can
add narrative_description, casual, etc. without changing this class — just
add keys to stance_exemplars.json.

Mirrors EmbedPersonaClassifier and EmbedIntentClassifier in shape — same
singleton pattern, same MiniLM-L6-v2 model reuse, same mean-of-top-k
aggregation.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("GAIA.EmbedStanceClassifier")

_EXEMPLARS_FILE = Path(__file__).parent / "stance_exemplars.json"


class EmbedStanceClassifier:
    """Singleton embedding-based stance classifier."""

    _instance: Optional["EmbedStanceClassifier"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._ready = False
        self._labels: list[str] = []
        self._exemplar_matrix = None
        self._embed_model = None

    @classmethod
    def instance(cls) -> "EmbedStanceClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialise(self, embed_model, config: Optional[dict] = None) -> bool:
        """Encode all stance exemplar phrases. Returns True on success."""
        if self._ready:
            return True

        cfg = config or {}
        exemplars_path = cfg.get("exemplars_path") or str(_EXEMPLARS_FILE)

        try:
            with open(exemplars_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception:
            logger.error("Failed to load stance exemplars from %s", exemplars_path)
            return False

        labels: list[str] = []
        phrases: list[str] = []
        for stance_label, examples in bank.items():
            if stance_label.startswith("_"):
                continue
            if not isinstance(examples, list):
                continue
            for phrase in examples:
                labels.append(stance_label)
                phrases.append(phrase)

        if not phrases:
            logger.warning("Stance exemplar bank is empty; classifier disabled.")
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
                "EmbedStanceClassifier ready: %d exemplars across %d stances",
                len(phrases),
                len(set(labels)),
            )
            return True
        except Exception:
            logger.exception("Failed to encode stance exemplars")
            return False

    def classify(
        self,
        text: str,
        confidence_threshold: float = 0.40,
        top_k: int = 3,
    ) -> Tuple[Optional[str], float]:
        """Classify a user query by embedding similarity to stance banks.

        Returns:
            (stance_label, score) on match, (None, best_score) on miss.
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

            stance_scores: dict[str, list[float]] = {}
            for idx, score in enumerate(similarities):
                label = self._labels[idx]
                stance_scores.setdefault(label, []).append(float(score))

            best_stance: Optional[str] = None
            best_score = 0.0
            for label, scores in stance_scores.items():
                top_scores = sorted(scores, reverse=True)[:top_k]
                avg = sum(top_scores) / len(top_scores)
                if avg > best_score:
                    best_score = avg
                    best_stance = label

            if best_stance is None or best_score < confidence_threshold:
                logger.info(
                    "Embed stance below threshold: best=%s score=%.3f (threshold=%.2f)",
                    best_stance, best_score, confidence_threshold,
                )
                return (None, best_score)

            logger.info(
                "Embed stance classified: %s (score=%.3f)",
                best_stance, best_score,
            )
            return (best_stance, best_score)

        except Exception:
            logger.exception("Embed stance classification failed")
            return (None, 0.0)

    @property
    def ready(self) -> bool:
        return self._ready
