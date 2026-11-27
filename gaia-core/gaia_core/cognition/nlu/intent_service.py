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
# d3px: both minds' reading of a message (act / source / depth), from their
# own hidden states. Shadow-logged for now; the gates move onto it next.
from .reading import read_message  # noqa: F401

# Optional shorthand alias if you prefer detect(...)
detect = detect_intent
