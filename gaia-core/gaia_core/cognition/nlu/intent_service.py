# gaia_core/cognition/nlu/intent_service.py
"""
Public façade for intent detection.

Other code should import ONLY from this file:
    from gaia_core.cognition.nlu.intent_service import detect_intent

Behind the curtain we forward to the real detector implementation
(intent_detection.py).  If you ever swap in a different model, you only
touch this file.
"""

from .intent_detection import detect_intent  # main entrypoint

# Optional shorthand alias if you prefer detect(...)
detect = detect_intent
