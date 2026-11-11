

def test_a_held_back_reply_is_recalled_as_held_back(tmp_path):
    from gaia_common.utils.conversation_memory import HELD_BACK_REPLY, ConversationMemory
    m = ConversationMemory(tmp_path)
    m.record_exchange("discord_dm_596925786208993283", "You spoke with Andy on Wednesday", "I do recall that conversation now",
                      ts="2026-11-11T06:55:00+00:00")
    rec = next(iter(m._read(m._path("discord_dm_596925786208993283"))))
    assert m.hold_back("discord_dm_596925786208993283", rec["id"], "claimed a memory after a failed lookup")
    assert m._checked(m._path("discord_dm_596925786208993283"))[0]["reply"] == HELD_BACK_REPLY
    assert not m.hold_back("discord_dm_596925786208993283", "nope", "x")


def test_putting_her_right_about_who_did_it_is_a_correction():
    from gaia_common.utils.conversation_memory import _CORRECTED_RE
    assert _CORRECTED_RE.search("Oh Andy isn't building you, I am. He's just my friend")
    assert not _CORRECTED_RE.search("I am glad it isn't raining today")


def test_a_known_person_named_in_passing_is_described(tmp_path):
    from gaia_common.utils.people import PeopleRegistry, mentioned_line
    reg = PeopleRegistry(tmp_path)
    reg.create("Andy", [{"kind": "session", "id": "web_3132edcd"}], relationship="Azrael's friend")
    reg.create("Azrael", [{"kind": "discord", "id": "596925786208993283"}], relationship="who created you")
    line = mentioned_line("Tomorrow I'm going to see Andy again", "discord_dm_596925786208993283", reg)
    assert line.startswith("Andy, whom they mention, is Azrael's friend; you have spoken with them before, in a web chat")
    assert mentioned_line("I'm Azrael, as you know", "discord_dm_596925786208993283", reg) == ""
    assert mentioned_line("Nobody Known here", "", reg) == ""
