"""A question about her architecture is not a request for a plan (2026-11-05)."""
from gaia_core.cognition.nlu.intent_detection import _keyword_intent_classify


def test_architecture_question_is_not_planning():
    q = ("Hmm, that's a description of the bicameral mind, but there's a lot more complexity to the system. "
         "For instance, do you know how many containers are used to run the architecture and how they interact?")
    assert _keyword_intent_classify(q) != "planning"


def test_plan_requests_still_are():
    assert _keyword_intent_classify("How would you architect a caching layer?") == "planning"
    assert _keyword_intent_classify("Create a detailed implementation plan for adding OAuth") == "planning"


def test_the_architect_and_her_blueprints_are_not_plans():
    for q in ("Hi GAIA, I'm Azrael, your architect.",
              "Does that apply to me as well though? I'm Azrael, and I've been the sole human architect "
              "and creator of your system and all reference documents.",
              "Can you open your blueprint for the doctor and tell me what it does?"):
        assert _keyword_intent_classify(q) != "planning", q


def test_asking_for_one_still_is():
    assert _keyword_intent_classify("Can you architect a retry queue for the web service?") == "planning"
    assert _keyword_intent_classify("Draft a blueprint for the new audio pipeline") == "planning"
