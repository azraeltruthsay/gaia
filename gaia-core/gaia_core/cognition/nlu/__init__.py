"""
gaia_core.cognition.nlu - Natural Language Understanding modules.

This package provides:
- intent_detection: Fast reflex, LLM-powered, and embedding-based intent detection
- embed_intent_classifier: Cosine-similarity intent classification via embeddings
"""

from .intent_detection import detect_intent, Plan, fast_intent_check
from .embed_intent_classifier import EmbedIntentClassifier

__all__ = [
    "detect_intent",
    "Plan",
    "fast_intent_check",
    "EmbedIntentClassifier",
]
