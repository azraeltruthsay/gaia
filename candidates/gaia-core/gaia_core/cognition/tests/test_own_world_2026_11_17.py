"""2026-11-17 Discord review: her campaign world is Azrael's, not the web's,
and the campaign persona lasts the conversation."""
from datetime import datetime, timedelta, timezone

import pytest

from gaia_core.behavior import persona_switcher as ps
from gaia_core.cognition.nlu.intent_detection import (
    is_own_name, mentions_own_name, names_introduced, own_names)

LLASAK = ("Oh there are native humanoids all over the place. They live in small villages "
          "and hamlets dotting the countryside. The nation is called Llasak and the capital "
          "city is called Leshen. The ruling family are the Neraly family.")


def test_names_introduced_from_the_campaign_message():
    assert set(names_introduced(LLASAK)) == {"Llasak", "Leshen", "Neraly"}


@pytest.mark.parametrize("text, names", [
    ("Their seat is House Varrow, east of the river", {"Varrow"}),
    ("my dog is named Biscuit", {"Biscuit"}),
    ("What is the capital of Australia?", set()),
    ("The Witcher has a monster called a leshen", set()),   # not capitalized: not a name
])
def test_names_introduced_cases(text, names):
    assert set(names_introduced(text)) == names


def test_own_names_collects_the_conversation():
    history = [{"role": "user", "content": LLASAK},
               {"role": "assistant", "content": "The Neraly family — called the Night Court?"}]
    names = own_names(history, "and House Ostrel rules the south")
    assert {"Llasak", "Leshen", "Neraly", "Ostrel"} <= set(names)
    assert "Night" not in names            # her words aren't theirs


def test_plural_and_possessive_still_theirs():
    names = ["Neraly", "Leshen"]
    assert is_own_name("Neralys", names)
    assert is_own_name("Neraly's", names)
    assert is_own_name("The Neralys", names)
    assert not is_own_name("Canberra", names)
    assert mentions_own_name("The Neralys keep the peasants scared", names)
    assert not mentions_own_name("what is the capital of Australia?", names)


class _Clf:
    ready = True

    def __init__(self, label):
        self.label = label

    def classify(self, text, confidence_threshold=0.0):
        return self.label, 0.2


@pytest.fixture
def clf(monkeypatch):
    from gaia_core.cognition.nlu import embed_persona_classifier as epc
    holder = {"clf": _Clf("dnd_player_assistant")}
    monkeypatch.setattr(epc.EmbedPersonaClassifier, "instance", classmethod(lambda cls: holder["clf"]))
    return holder


def _prev(minutes_ago=5):
    at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"persona": "dnd_player_assistant", "kb": "dnd_campaign", "at": at.isoformat()}


def test_campaign_persona_is_kept(clf):
    assert ps.sticky_persona(_prev(), LLASAK, embed_model=object()) == (
        "dnd_player_assistant", "dnd_campaign")


def test_other_topic_is_not_kept(clf):
    clf["clf"] = _Clf("conversational")
    assert ps.sticky_persona(_prev(), "how is the GPU doing?", embed_model=object()) is None


def test_kept_only_for_a_while(clf):
    assert ps.sticky_persona(_prev(minutes_ago=ps.STICKY_PERSONA_MINUTES + 1), LLASAK,
                             embed_model=object()) is None


def test_nothing_to_keep(clf):
    assert ps.sticky_persona(None, LLASAK, embed_model=object()) is None
