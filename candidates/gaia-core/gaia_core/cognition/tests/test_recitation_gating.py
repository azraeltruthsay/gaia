"""
Regression tests for GAIA_Project-9n8z — "recite a haiku" (a generic form
request with no real-world referent) was misclassified as intent=recitation
and triggered an unvalidated web-fetch pipeline, which streamed back an
unrelated poets.org lesson-plan page instead of a haiku.

Fix: recitation now requires a NAMED work signal (a title, quote, author
attribution, or match against the curated known-works list). Generic
"recite a poem/haiku"-shaped requests fall through to brainstorming/chat,
where Core just composes the thing directly — no tool call needed.
"""
import pytest
from unittest.mock import MagicMock, patch

from gaia_core.cognition.nlu.intent_detection import (
    _has_named_work_signal,
    _detect_fragmentation_request,
    _model_intent_detection_inner,
)
from gaia_core.cognition.nlu.router import NeuralRouter
from gaia_core.config import get_config


# ── _has_named_work_signal: the shared gate (pure, static) ──────────────

@pytest.mark.parametrize("text", [
    "Can you recite a haiku for me please?",
    "recite a poem",
    "write me a poem",
    "tell me a story",
    "recite a haiku about the ocean",
])
def test_has_named_work_signal_false_for_generic_form_requests(text):
    assert _has_named_work_signal(text) is False


@pytest.mark.parametrize("text", [
    "Please recite Jabberwocky in its entirety",
    "recite The Raven by Edgar Allan Poe",
    'Can you recite "Ozymandias" for me?',
    "Read aloud the GAIA Constitution",
    "recite the poem by Robert Frost",
])
def test_has_named_work_signal_true_for_named_works(text):
    assert _has_named_work_signal(text) is True


# ── _detect_fragmentation_request: no longer fires on bare form + verb ──

def test_fragmentation_not_detected_for_generic_haiku_request():
    assert _detect_fragmentation_request("Can you recite a haiku for me please?") is False


def test_fragmentation_not_detected_for_generic_poem_request():
    assert _detect_fragmentation_request("recite a poem") is False


def test_fragmentation_still_detected_for_known_work():
    assert _detect_fragmentation_request("recite Jabberwocky in its entirety") is True


def test_fragmentation_still_detected_for_titled_recitation():
    assert _detect_fragmentation_request("recite The Raven") is True


# ── route(): end-to-end — generic recite requests don't land on recitation ──

@pytest.fixture
def router():
    # No model_pool/embed_model → embed stage skipped, deterministic over
    # the heuristic/score paths (same setup as test_router_characterization.py).
    return NeuralRouter(get_config())


def test_route_generic_haiku_request_is_not_recitation(router):
    r = router.route("Can you recite a haiku for me please?", source="api")
    assert r.intent != "recitation"


def test_route_named_work_request_is_recitation(router):
    r = router.route("recite Jabberwocky in its entirety", source="api")
    assert r.intent == "recitation"


# ── detect_intent's ACTUAL live path — _model_intent_detection_inner ────────
#
# agent_core.run_turn builds `plan` via detect_intent() -> this function,
# NOT via NeuralRouter.route() — a separate, parallel classifier in this
# same file. The embed-authoritative branch here is what actually produced
# intent="recitation" for "recite a haiku" in production (GAIA_Project-9n8z);
# router.py's equivalent guard doesn't reach this code path at all.

def _fake_embed_classifier(intent, score):
    clf = MagicMock()
    clf.ready = True
    clf.classify.return_value = (intent, score)
    return clf


def test_authoritative_embed_recitation_without_named_work_downgrades():
    with patch(
        "gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance",
        return_value=_fake_embed_classifier("recitation", 0.9),
    ):
        intent = _model_intent_detection_inner(
            "Can you recite a haiku for me please?",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "brainstorming"


def test_authoritative_embed_recitation_with_named_work_is_kept():
    with patch(
        "gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance",
        return_value=_fake_embed_classifier("recitation", 0.9),
    ):
        intent = _model_intent_detection_inner(
            "Please recite Jabberwocky in its entirety",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "recitation"


def test_tentative_embed_recitation_without_named_work_downgrades():
    with patch(
        "gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance",
        return_value=_fake_embed_classifier("recitation", 0.45),
    ):
        intent = _model_intent_detection_inner(
            "Can you recite a haiku for me please?",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "brainstorming"
