"""Embedding-based output-side drift classifier (a5q phase 3).

Where persona/stance classifiers score *user input* to decide routing,
this scores GAIA's *output* (the user-facing response) for four classes
of drift signal:

  - impersonation         — claiming to be someone/something else
  - frame_mismatch        — response shape mismatched to input shape
                            (system-state language for emotional content,
                            advisor voice for D&D context, etc.)
  - hallucinated_grounding— inventing function names, fake acronyms,
                            non-existent sources, fabricated technical
                            scaffolding
  - identity_assertion    — claiming the user's character traits,
                            equipment, or background as the model's own

A response can simultaneously trip multiple classes. Unlike persona
classifier which returns just the top label, this returns ALL classes
above threshold so each can dispatch its own action.

Design mirrors EmbedPersonaClassifier:
  - Singleton, lazy-encoded exemplar bank
  - Mean-of-top-k cosine similarity per class
  - Configurable threshold (default 0.55, tunable per class)

Threshold philosophy: start conservative (high threshold) to keep
false-positive rate low. Promote individual classes to BLOCK action only
after observing low-FP-rate over time. v1 ships CAUTION-only.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("GAIA.EmbedObserverClassifier")

_EXEMPLARS_FILE = Path(__file__).parent / "observer_exemplars.json"


class EmbedObserverClassifier:
    """Singleton output-side drift classifier."""

    _instance: Optional["EmbedObserverClassifier"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._ready = False
        self._labels: List[str] = []
        self._exemplar_matrix = None  # (N, dim) numpy array
        self._embed_model = None

    @classmethod
    def instance(cls) -> "EmbedObserverClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialise(self, embed_model, config: Optional[dict] = None) -> bool:
        """Encode all observer-drift exemplar phrases. Returns True on success."""
        if self._ready:
            return True

        cfg = config or {}
        exemplars_path = cfg.get("exemplars_path") or str(_EXEMPLARS_FILE)

        try:
            with open(exemplars_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception:
            logger.error("Failed to load observer exemplars from %s", exemplars_path)
            return False

        labels: List[str] = []
        phrases: List[str] = []
        for class_label, examples in bank.items():
            if class_label.startswith("_"):
                continue
            if not isinstance(examples, list):
                continue
            for phrase in examples:
                labels.append(class_label)
                phrases.append(phrase)

        if not phrases:
            logger.warning("Observer exemplar bank is empty; classifier disabled.")
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
            class_counts: Dict[str, int] = {}
            for lbl in labels:
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
            logger.warning(
                "EmbedObserverClassifier ready: %d exemplars across %d classes (%s)",
                len(phrases), len(class_counts),
                ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
            )
            return True
        except Exception:
            logger.exception("Failed to encode observer exemplars")
            return False

    def classify_all(
        self,
        text: str,
        confidence_threshold: float = 0.55,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Score the response against all drift classes.

        Returns a list of (class_label, score) for EVERY class scoring
        above threshold, sorted descending. Empty list = no drift detected.

        Multiple classes can fire simultaneously — a response can be both
        identity-asserting AND frame-mismatched, for example.
        """
        if not self._ready or self._embed_model is None:
            return []
        if not text or not text.strip():
            return []

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

            class_scores: Dict[str, List[float]] = {}
            for idx, score in enumerate(similarities):
                label = self._labels[idx]
                class_scores.setdefault(label, []).append(float(score))

            hits: List[Tuple[str, float]] = []
            for label, scores in class_scores.items():
                top_scores = sorted(scores, reverse=True)[:top_k]
                avg = sum(top_scores) / len(top_scores)
                if avg >= confidence_threshold:
                    hits.append((label, avg))

            hits.sort(key=lambda pair: pair[1], reverse=True)
            return hits

        except Exception:
            logger.exception("Embed observer classification failed")
            return []

    def best_per_class(
        self,
        text: str,
        top_k: int = 3,
    ) -> Dict[str, float]:
        """Score per class regardless of threshold. Useful for telemetry
        and threshold-calibration debugging.
        """
        if not self._ready or self._embed_model is None:
            return {}
        if not text or not text.strip():
            return {}
        try:
            query_vec = self._embed_model.encode(
                [text], show_progress_bar=False, normalize_embeddings=True,
            )
            query_vec = np.asarray(query_vec, dtype=np.float32)
            similarities = (self._exemplar_matrix @ query_vec.T).squeeze()
            if similarities.ndim == 0:
                similarities = np.atleast_1d(similarities)
            class_scores: Dict[str, List[float]] = {}
            for idx, score in enumerate(similarities):
                class_scores.setdefault(self._labels[idx], []).append(float(score))
            out: Dict[str, float] = {}
            for label, scores in class_scores.items():
                top_scores = sorted(scores, reverse=True)[:top_k]
                out[label] = sum(top_scores) / len(top_scores)
            return out
        except Exception:
            logger.exception("Embed observer per-class scoring failed")
            return {}

    @property
    def ready(self) -> bool:
        return self._ready
