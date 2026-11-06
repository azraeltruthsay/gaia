"""Asked about her blueprints or services, she reads Your Services (albm, 2026-11-05)."""
from gaia_common.utils.founding_docs import named_document


def title(q):
    got = named_document(q)
    return got[1] if got else None


def test_her_services_and_blueprints():
    for q in ("Can you review your blueprints?",
              "Ok, can you review your blueprints and get an understanding of how your architecture works?",
              "Do you know how many containers are used to run the architecture and how they interact?",
              "What services make up your body?", "Tell me about GAIA's services"):
        assert title(q) == "Your Services", q


def test_others_left_alone():
    assert title("How does your mind work?") == "How Your Mind Works"
    assert title("What is your architecture?") == "How Your Mind Works"
    for q in ("How many services does Netflix run?", "Tell me about Docker containers"):
        assert title(q) is None, q


def test_review_means_the_whole_document():
    from gaia_common.utils.founding_docs import wants_whole
    assert wants_whole("Can you review your blueprints?")
    assert not wants_whole("Review Article II of your constitution")


def test_a_request_asked_as_a_question_is_put_as_the_request():
    from gaia_common.utils.founding_docs import as_request
    assert as_request("Can you review your blueprints?") == "Review your blueprints."
    assert as_request("Ok, could you please read your constitution and tell me what you think?") == \
        "Read your constitution and tell me what you think."
    assert as_request("How does your mind work?") == "How does your mind work?"
    assert as_request("Do you know how many containers you run?") == "Do you know how many containers you run?"
