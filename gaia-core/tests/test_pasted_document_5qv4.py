"""5qv4: a pasted dev journal fired five keyword detectors and she never answered it.

The lines below are the ones that fired, from the 2026-11-15 journal message.
"""
from unittest.mock import MagicMock, patch

from gaia_core.behavior.persona_switcher import get_persona_for_request
from gaia_core.cognition.nlu.intent_detection import detect_intent
from gaia_core.cognition.stakes_classifier import classify_stakes
from gaia_common.utils.host_facts import asks_about_host
from gaia_core.config import get_config

JOURNAL = "\n\n".join([
    "# 2026-11-15 — Dev journal: delta since 2026-11-11",
    "Gear 1 now runs Core on the GPU at 70 tok/s.",                       # host facts
    'The post line is now the last thing in the system prompt; any spelling of "Grotto" resolves.',  # persona
    "**`self_review` sleep task**: she reads her own turns, see `knowledge/blueprints/SELF_REPAIR_LOOP.md`.",  # read_file
    "Wake from park broke after a restart; each section is re-rolled up to 2 times.",  # stakes
    "`mjcg` fixed by hand. A message to write a note reached file(write).",  # tool routing
] * 4)
MESSAGE = ("Hi GAIA. Azrael asked me to show you this week's dev journal and ask what you think.\n\n"
           "1. Which of these changes can you notice from the inside?\n"
           "2. Is anything in it wrong?\n\n"
           "----- JOURNAL -----\n" + JOURNAL)


def test_not_the_dnd_persona():
    persona, _kb = get_persona_for_request(MESSAGE)
    assert persona != "dnd_player_assistant"


def test_spell_is_still_a_dnd_word():
    assert get_persona_for_request("Which spell should Rupert prepare?")[0] == "dnd_player_assistant"
    assert get_persona_for_request("How many spells does a wizard get?")[0] == "dnd_player_assistant"


def test_spelling_is_not():
    assert get_persona_for_request("What's the right spelling of Grotto?")[0] != "dnd_player_assistant"


def test_not_a_hardware_question():
    assert not asks_about_host(MESSAGE)


def test_no_real_world_or_game_question():
    assert not classify_stakes(MESSAGE).requires_clarification


def test_not_a_file_read():
    fake = MagicMock()
    fake.classify.return_value = ("other", 0.1)
    fake.ready = True
    with patch("gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance", return_value=fake):
        plan = detect_intent(MESSAGE, get_config())
    assert plan.intent not in {"read_file", "write_file", "find_file"}


def test_a_short_read_request_still_reads():
    fake = MagicMock()
    fake.classify.return_value = ("other", 0.1)
    fake.ready = True
    with patch("gaia_core.cognition.nlu.intent_detection.EmbedIntentClassifier.instance", return_value=fake):
        plan = detect_intent("read /shared/CLAUDE.md", get_config())
    assert plan.intent == "read_file"


def test_just_say_so_is_not_a_recitation():
    from gaia_core.cognition.nlu.intent_detection import _detect_fragmentation_request
    ask = ("Hi GAIA. This is Claude. Azrael asked me to show you this week's dev journal. "
           "The full journal is below. If you don't remember a conversation, just say so.")
    assert not _detect_fragmentation_request(ask)


def test_recite_a_named_work_still_is():
    from gaia_core.cognition.nlu.intent_detection import _detect_fragmentation_request
    assert _detect_fragmentation_request("Recite Jabberwocky for me")


def test_router_reflex_reads_the_ask():
    from gaia_core.cognition.nlu.router import NeuralRouter
    result = NeuralRouter(get_config()).route(MESSAGE, source="web")
    assert getattr(result, "intent", None) not in {"read_file", "write_file"}
