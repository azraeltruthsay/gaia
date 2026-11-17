"""DM-only campaign entries open only in a conversation that's Azrael's alone,
decided from the session, never from what the model wrote (2026-11-17)."""

from gaia_core.cognition import campaign_scope as cs


class Reg:
    def __init__(self, people):
        self.people = people

    def for_session(self, sid):
        return self.people.get(sid)


def _patch(monkeypatch, people):
    import gaia_common.utils.people as people_mod
    monkeypatch.setattr(people_mod, "PeopleRegistry", lambda: Reg(people))


AZ = {"names": ["Azrael"], "relationship": "who created you"}
NICK = {"names": ["Nick"], "relationship": "Azrael's friend"}


def test_creator_dm_gets_dm_view(monkeypatch):
    _patch(monkeypatch, {"web_9868c2b7": AZ})
    assert cs.scope("worldbuild", {"action": "get"}, "web_9868c2b7")["_dm_view"] is True


def test_friend_and_unknown_do_not(monkeypatch):
    _patch(monkeypatch, {"web_3132edcd": NICK})
    assert cs.scope("worldbuild", {}, "web_3132edcd")["_dm_view"] is False
    assert cs.scope("worldbuild", {}, "web_00000000")["_dm_view"] is False
    assert cs.scope("worldbuild", {}, "")["_dm_view"] is False


def test_model_cannot_grant_itself(monkeypatch):
    _patch(monkeypatch, {})
    out = cs.scope("worldbuild", {"action": "get", "_dm_view": True}, "web_3132edcd")
    assert out["_dm_view"] is False
    assert cs.scope("kanka_get_entity", {"_dm_view": True}, "")["_dm_view"] is False


def test_never_in_a_channel_even_for_azrael(monkeypatch):
    import gaia_common.utils.people as people_mod
    monkeypatch.setattr(people_mod, "is_channel", lambda sid: True)
    _patch(monkeypatch, {"discord_channel_1": AZ})
    assert cs.scope("worldbuild", {}, "discord_channel_1")["_dm_view"] is False


def test_other_tools_untouched():
    p = {"action": "read", "_dm_view": True}
    assert cs.scope("file", p, "x") == p


def test_reads_turn_session_when_none_given(monkeypatch):
    from gaia_core.models import turn_context
    _patch(monkeypatch, {"web_9868c2b7": AZ})
    turn_context.set_turn("web_9868c2b7", "t")
    try:
        assert cs.scope("worldbuild", {})["_dm_view"] is True
    finally:
        turn_context.clear_turn()
    assert cs.scope("worldbuild", {})["_dm_view"] is False


def _names(monkeypatch, names):
    import time
    hashes = frozenset(cs._name_hash(n) for n in names)
    monkeypatch.setattr(cs, "_names_cache", {"at": time.time(), "hashes": hashes,
                                             "max_words": max(len(n.split()) for n in names)})


def test_campaign_names_found_by_hash(monkeypatch):
    _names(monkeypatch, ["Leshen", "Dusk in Leshen", "Llasak"])
    assert cs.campaign_names_in("Can you look up Leshen in the Kanka campaign?") == ["Kanka", "Leshen"]
    assert cs.campaign_names_in("What happens at Dusk in Leshen?") == ["Dusk", "Leshen"]
    assert cs.campaign_names_in("Who rules Llasak's capital?") == ["Llasak"]


def test_ordinary_words_are_not_campaign_names(monkeypatch):
    _names(monkeypatch, ["Leshen"])
    assert cs.campaign_names_in("What's the capital of Australia?") == []
    assert cs.campaign_names_in("") == []


def test_no_campaign_reachable_means_no_names(monkeypatch):
    import time
    monkeypatch.setattr(cs, "_names_cache", {"at": time.time(), "hashes": frozenset(), "max_words": 1})
    assert cs.campaign_names_in("Tell me about Leshen") == []


def test_own_names_keep_campaign_names_off_the_web(monkeypatch):
    from gaia_core.cognition.nlu.intent_detection import is_own_name
    _names(monkeypatch, ["Leshen"])
    own = cs.campaign_names_in("look up Leshen in the Kanka campaign")
    assert is_own_name("Leshen", own) and is_own_name("Kanka campaign", own)


def test_entries_as_written_longest_first(monkeypatch):
    _names(monkeypatch, ["Leshen", "Dusk in Leshen", "Llasak"])
    assert cs.campaign_entries_in("What happens at Dusk in Leshen?") == ["Dusk in Leshen"]
    assert cs.campaign_entries_in("What is Leshen, in the campaign?") == ["Leshen"]
    assert cs.campaign_entries_in("Is Leshen in Llasak's south?") == ["Leshen", "Llasak"]
    assert cs.campaign_entries_in("what is the weather") == []


def test_change_requests_go_to_routing():
    assert cs.asks_for_change("Add to Leshen's entry that the lords meet at dusk")
    assert cs.asks_for_change("Can you make a note about Llasak")
    assert not cs.asks_for_change("What is Leshen?")


class FakeModel:
    def __init__(self, reply):
        self.reply = reply

    def create_chat_completion(self, **kw):
        self.kw = kw
        return {"choices": [{"message": {"content": self.reply}}]}


def test_change_params_from_a_narrow_answer():
    m = FakeModel('```json\n{"entity": "leshen", "append_entry": "The Lords of Llasak meet at dusk."}\n```')
    p = cs.change_params(m, "add to Leshen's entry that the Lords meet at dusk", ["Leshen", "Llasak"])
    assert p == {"action": "update", "entity": "Leshen", "append_entry": "The Lords of Llasak meet at dusk."}
    assert "Leshen, Llasak" in m.kw["messages"][0]["content"]


def test_change_params_rejects_an_entry_not_named_or_no_text():
    assert cs.change_params(FakeModel('{"entity": "Oakhaven", "append_entry": "x"}'), "r", ["Leshen"]) is None
    assert cs.change_params(FakeModel('{"entity": "Leshen", "append_entry": ""}'), "r", ["Leshen"]) is None
    assert cs.change_params(FakeModel("I can't"), "r", ["Leshen"]) is None
