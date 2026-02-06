"""
gaia_core.cognition.nlu - Natural Language Understanding modules.

This package provides:
- intent_detection: Fast reflex and LLM-powered intent detection
"""

from .intent_detection import detect_intent, Plan, fast_intent_check

__all__ = [
    "detect_intent",
    "Plan",
    "fast_intent_check",
]
