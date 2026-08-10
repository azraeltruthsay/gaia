"""
Regression tests for GAIA_Project-akmh — infra self-knowledge questions
dodging the system-keyword override.

The override (c0c987c) upgrades intent "chat" -> "other" when the text
contains system keywords ("docker", "container", "mcp", ...), so awareness
grounding fires instead of GAIA deflecting the question back to the
operator. But two real turns showed the gap: "how many docker containers"
classified as "greeting" (the embed classifier matched it closer to greeting
exemplars than chat), and a follow-up classified "list_tools" (a
low-confidence embed miss fell through to the LLM stage, which guessed
wrong). Neither "chat"-only check caught them.

Fix: the override now also applies to "greeting" and "list_tools", with a
confidence gate on "list_tools" specifically — a genuine, confidently
matched request like "what MCP tools do you have?" must survive even
though it contains the "mcp" keyword.
"""
from unittest.mock import MagicMock, patch

from gaia_core.cognition.nlu.intent_detection import model_intent_detection
from gaia_core.config import get_config


def _fake_embed_classifier(intent, score):
    clf = MagicMock()
    clf.ready = True
    clf.classify.return_value = (intent, score)
    return clf


def test_greeting_with_docker_keyword_upgrades_to_other():
    # "Good morning GAIA, how many docker containers are you running?"
    # scored closer to the greeting exemplars than chat/list_tools.
    with patch(
        "gaia_core.cognition.nlu.intent_detection._model_intent_detection_inner",
        return_value="greeting",
    ):
        intent = model_intent_detection(
            "Good morning GAIA, how many docker containers are you running?",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "other"


def test_greeting_without_system_keyword_is_unaffected():
    with patch(
        "gaia_core.cognition.nlu.intent_detection._model_intent_detection_inner",
        return_value="greeting",
    ):
        intent = model_intent_detection(
            "Good morning GAIA, how are you today?",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "greeting"


def test_low_confidence_list_tools_with_system_keyword_upgrades_to_other():
    # LLM stage (stage 4) guessed "list_tools" for a system question after a
    # weak embed pass (score 0.207, below the 0.42 threshold). The override's
    # confidence re-check should agree it's not a real list_tools match and
    # let the upgrade to "other" through.
    with patch(
        "gaia_core.cognition.nlu.intent_detection._model_intent_detection_inner",
        return_value="list_tools",
    ), patch(
        "gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance",
        return_value=_fake_embed_classifier("list_tools", 0.207),
    ):
        intent = model_intent_detection(
            "how many docker containers does GAIA run",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "other"


def test_confident_list_tools_with_system_keyword_is_preserved():
    # "What MCP tools do you have?" legitimately contains "mcp" but is a
    # genuine, confidently-matched list_tools request — must not be
    # clobbered into "other".
    with patch(
        "gaia_core.cognition.nlu.intent_detection._model_intent_detection_inner",
        return_value="list_tools",
    ), patch(
        "gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance",
        return_value=_fake_embed_classifier("list_tools", 0.9),
    ):
        intent = model_intent_detection(
            "What MCP tools do you have?",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "list_tools"


def test_chat_with_docker_keyword_still_upgrades_to_other():
    # Pre-existing behaviour (c0c987c) — must not regress.
    with patch(
        "gaia_core.cognition.nlu.intent_detection._model_intent_detection_inner",
        return_value="chat",
    ):
        intent = model_intent_detection(
            "how many docker containers does GAIA run",
            get_config(),
            embed_model=MagicMock(),
        )
    assert intent == "other"
