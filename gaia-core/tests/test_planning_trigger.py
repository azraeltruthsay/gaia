"""A question about her architecture is not a request for a plan (2026-11-05)."""
from gaia_core.cognition.nlu.intent_detection import _keyword_intent_classify


def test_architecture_question_is_not_planning():
    q = ("Hmm, that's a description of the bicameral mind, but there's a lot more complexity to the system. "
         "For instance, do you know how many containers are used to run the architecture and how they interact?")
    assert _keyword_intent_classify(q) != "planning"


def test_plan_requests_still_are():
    assert _keyword_intent_classify("How would you architect a caching layer?") == "planning"
    assert _keyword_intent_classify("Create a detailed implementation plan for adding OAuth") == "planning"
