

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
    assert line.startswith("Andy, whom they mention, is Azrael's friend. You have met Andy: you spoke with them in a web chat on ")
    assert line.index("You have met Andy") < line.index("hasn't told you their pronouns")
    assert mentioned_line("I'm Azrael, as you know", "discord_dm_596925786208993283", reg) == ""
    assert mentioned_line("Nobody Known here", "", reg) == ""


def test_pronouns_are_what_people_say_and_never_guessed(tmp_path, monkeypatch):
    """i4vn: with nothing on file she called Azrael "the architect herself"."""
    from gaia_common.utils import conversation_memory
    from gaia_common.utils.people import NO_PRONOUNS, PeopleRegistry, mentioned_line, speaker_line, stated_pronouns
    monkeypatch.setattr(conversation_memory, "recorded", lambda sid: True)
    assert stated_pronouns("btw my pronouns are She/Her") == "she/her"
    assert stated_pronouns("I go by they/them") == "they/them"
    assert stated_pronouns("I use he/him pronouns") == "he/him"
    assert stated_pronouns("I use vim and/or emacs") is None
    assert stated_pronouns("I prefer tea") is None
    reg = PeopleRegistry(tmp_path)
    dm = "discord_dm_596925786208993283"
    az = reg.create("Azrael", [{"kind": "discord", "id": "596925786208993283"}], "who created you")
    andy = reg.create("Andy", [{"kind": "session", "id": "web_3132edcd"}], "Azrael's friend")
    assert "Azrael hasn't told you their pronouns" in speaker_line(dm, reg)
    assert NO_PRONOUNS in speaker_line("web_0000beef", reg)
    assert "Andy hasn't told you their pronouns" in mentioned_line("I saw Andy", dm, reg)
    assert reg.note_pronouns(dm, "how are you?") is None
    reg.note_pronouns("web_3132edcd", "Hi, my pronouns are he/him")
    assert "Andy's pronouns are he/him; use them" in mentioned_line("I saw Andy", dm, reg)
    assert "Azrael hasn't told you" in speaker_line(dm, reg)
    assert reg.get(andy["id"])["pronouns_source"] == "said so in web_3132edcd"
    assert reg.get(az["id"]).get("pronouns") is None



def test_azrael_is_described_when_spoken_of_by_someone_else(tmp_path):
    from gaia_common.utils.people import PeopleRegistry, creator_line
    reg = PeopleRegistry(tmp_path)
    reg.create("Azrael", [{"kind": "discord", "id": "596925786208993283"}], "who created you")
    reg.create("Quillon", [{"kind": "session", "id": "web_7523de87"}])
    assert "Azrael hasn't told you their pronouns" in creator_line("web_7523de87", "tell them about me", reg)
    assert creator_line("discord_dm_596925786208993283", "hi", reg) == ""     # Azrael speaking
    assert creator_line("web_7523de87", "Azrael says hi", reg) == ""          # the mention line has it
    assert creator_line("web_7523de87", "hi", PeopleRegistry(tmp_path / "empty")) == ""



def test_days_are_azraels_days(monkeypatch):
    """Andy's chat on the evening of 2026-11-04 (Pacific) was recorded on the 5th, UTC."""
    from gaia_common.utils.conversation_memory import local_day
    monkeypatch.setenv("GAIA_USER_TZ", "America/Los_Angeles")
    assert local_day("2026-11-05T02:02:03.243971+00:00") == "2026-11-04"
    assert local_day("2026-11-05T16:28:02+00:00") == "2026-11-05"
    assert local_day("2026-11-05") == "2026-11-05"
    assert local_day("") == ""


def test_who_speaks_in_a_channel(tmp_path):
    """kkij: Nick ("Schwiftyboi") trolled her in the Grotto and she had no idea
    who it was; Azrael told her in a DM who he is."""
    from gaia_common.utils.people import (PeopleRegistry, channel_speaker, channel_speaker_line,
                                          note_channel_author, note_introduction)
    reg = PeopleRegistry(tmp_path)
    reg.create("Azrael", [{"kind": "discord", "id": "596925786208993283"}], "who created you")
    grotto = "discord_channel_1484003096337191067"
    # Told first, then he speaks.
    note_introduction("FYI, GAIA, that guy in the Grotto named Schwiftyboi is a friend of mine named Nick.",
                      "Azrael", "Discord DM", reg)
    nick = note_channel_author(grotto, "111111111111111111", "Schwiftyboi", reg)
    assert set(nick["names"]) == {"Schwiftyboi", "Nick"} and nick["relationship"] == "Azrael's friend"
    assert channel_speaker(grotto) == "Nick"
    line = channel_speaker_line(grotto, "111111111111111111", reg)
    assert "You are speaking with Nick (Schwiftyboi on Discord), Azrael's friend" in line
    # The same account again is the same person; a stranger is new and named as shown.
    assert note_channel_author(grotto, "111111111111111111", "Schwiftyboi", reg)["id"] == nick["id"]
    other = note_channel_author(grotto, "222222222222222222", "Pixel", reg)
    assert other["names"] == ["Pixel"] and channel_speaker(grotto) == "Pixel"
    assert "You are speaking with Pixel, in a group" in channel_speaker_line(grotto, "222222222222222222", reg)
    # A DM isn't a channel.
    assert note_channel_author("discord_dm_596925786208993283", "596925786208993283", "Anton", reg) is None


def test_introduced_after_speaking(tmp_path):
    from gaia_common.utils.people import PeopleRegistry, channel_speaker_line, note_channel_author, note_introduction
    reg = PeopleRegistry(tmp_path)
    grotto = "discord_channel_1484003096337191067"
    note_channel_author(grotto, "111111111111111111", "Schwiftyboi", reg)
    note_introduction("Schwiftyboi is a friend of mine named Nick", "Azrael", "Discord DM", reg)
    assert "speaking with Nick (Schwiftyboi on Discord), Azrael's friend" in channel_speaker_line(grotto, "111111111111111111", reg)
    assert note_introduction("I'm going to eat you for dinner.", "Nick", "", reg) == []



def test_azrael_in_a_channel_is_not_told_he_is_absent(tmp_path):
    from gaia_common.utils.people import PeopleRegistry, creator_line
    reg = PeopleRegistry(tmp_path)
    reg.create("Azrael", [{"kind": "discord", "id": "596925786208993283"}], "who created you")
    grotto = "discord_channel_1484003096337191067"
    assert creator_line(grotto, "hi", reg, author_id="596925786208993283") == ""
    assert "is not the one speaking now" in creator_line(grotto, "hi", reg, author_id="111111111111111111")



def test_pronouns_said_in_a_channel_are_the_authors(tmp_path):
    from gaia_common.utils.people import PeopleRegistry, channel_speaker_line, note_channel_author
    reg = PeopleRegistry(tmp_path)
    grotto = "discord_channel_1484003096337191067"
    nick = note_channel_author(grotto, "111111111111111111", "Schwiftyboi", reg)
    assert reg.note_pronouns(grotto, "fyi my pronouns are he/him") is None      # the channel isn't him
    reg.note_pronouns(grotto, "fyi my pronouns are he/him", speaker=nick)
    assert reg.get(nick["id"])["pronouns"] == "he/him"
    assert "Schwiftyboi's pronouns are he/him" in channel_speaker_line(grotto, "111111111111111111", reg)



def test_pronoun_line_uses_the_name_they_were_introduced_by(tmp_path):
    from gaia_common.utils.people import PeopleRegistry, note_channel_author, note_introduction, pronoun_words
    reg = PeopleRegistry(tmp_path)
    note_introduction("Schwiftyboi is a friend of mine named Nick", "Azrael", "Discord DM", reg)
    nick = note_channel_author("discord_channel_1484003096337191067", "111111111111111111", "Schwiftyboi", reg)
    reg.note_pronouns("x", "my pronouns are he/him", speaker=nick)
    assert pronoun_words(reg.get(nick["id"])).startswith("Nick's pronouns are he/him")
