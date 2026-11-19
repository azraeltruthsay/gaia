"""She @mentions the people she addresses by name, never someone she only talks about."""
from gaia_web.mentions import channel_people, mention_names

SID = "discord_channel_148"
PEOPLE = [
    {"names": ["schwiftyboi", "Nick"], "name_sources": {"schwiftyboi": f"Discord display name in {SID}"},
     "accounts": [{"kind": "discord", "id": "111", "source": f"spoke in {SID}"}]},
    {"names": ["ShinyAndreon", "Andy"], "name_sources": {"ShinyAndreon": f"Discord display name in {SID}"},
     "accounts": [{"kind": "discord", "id": "222", "source": "seeded"}]},
    {"names": ["Azrael"], "name_sources": {"Azrael": "Discord display name in discord_channel_999"},
     "accounts": [{"kind": "discord", "id": "333", "source": "seeded"}]},
]
NAMES = channel_people(PEOPLE, SID)


def test_only_people_seen_here():
    assert NAMES == {"schwiftyboi": "111", "Nick": "111", "ShinyAndreon": "222", "Andy": "222"}


def test_name_two_people_share_is_dropped():
    twin = {"names": ["Andy"], "accounts": [{"kind": "discord", "id": "444", "source": f"spoke in {SID}"}]}
    assert "Andy" not in channel_people(PEOPLE + [twin], SID)


def test_opening_address():
    assert mention_names("Nick, it's yours.", NAMES) == "<@111>, it's yours."


def test_after_sentence_and_trailing_address():
    assert mention_names("Sure. Andy, welcome.", NAMES) == "Sure. <@222>, welcome."
    assert mention_names("What do you think, Andy?", NAMES) == "What do you think, <@222>?"


def test_at_name():
    assert mention_names("Ask @Nick about it.", NAMES) == "Ask <@111> about it."


def test_passing_mention_is_text():
    for t in ("Nick's handle is fine.", "I told Andy yesterday.", "Nickel, please."):
        assert mention_names(t, NAMES) == t


def test_once_per_person_and_skip_author():
    assert mention_names("Nick, hi. Sure, Nick.", NAMES) == "<@111>, hi. Sure, Nick."
    assert mention_names("Nick, hi.", NAMES, skip_ids=["111"]) == "Nick, hi."
