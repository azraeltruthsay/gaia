"""Ambient channel: she speaks when a message is for her, not when she'd like
to (2026-11-25, GAIA's Grotto: asked whether she'd want to say something,
Prime said yes to 79% of messages for someone else and 79% of room chatter)."""
from gaia_core.cognition import ambient_interject as A


def _never(_):
    raise AssertionError("code decides this; no model call")


def test_room_chatter_is_silent_without_asking(tmp_path):
    for msg in ("Azrael: Hahaha", "Nick: He spelled it wrong!", "Azrael: Happy Thanksgiving everyone!",
                "Andy: works for me", "Azrael: 😂", "Nick: Life without the switch has been daunting for sure."):
        assert A.should_interject("s", [], msg, _never, base=tmp_path) is False, msg


def test_said_to_someone_else_the_model_says_so(tmp_path):
    # "you" is there, so it's asked, and told it's for someone else, she's silent
    assert A.should_interject("s", [{"role": "user", "content": "Nick: I finally got the new Switch!"}],
                              "Andy: I'm happy for you and your new switch!", lambda m: "OTHER",
                              base=tmp_path) is False


def test_addressing_her_is_not_talking_about_her():
    assert A.might_be_for_her("Azrael: @GAIA can you summarize the last hour of chat?")
    assert A.might_be_for_her("Nick: GAIA can you look that up?")
    assert A.not_for_her("Nick: GAIA can do that already")              # about her
    assert not A.not_for_her("Azrael: Ok, I'm going to play games, can you take a nap?")   # "Ok" isn't a name
    assert A.not_for_her("Azrael: Andy, have you seen the musical Camelot?")             # to Andy


def test_the_model_sees_the_channel_by_name(tmp_path):
    seen = {}
    def generate(messages):
        seen["system"], seen["convo"] = messages[0]["content"], messages[1]["content"]
        return "GAIA"
    history = [{"role": "user", "content": "Nick: anyone watching the game?"},
               {"role": "assistant", "content": "I'm not, but who's playing?"}]
    assert A.should_interject("s", history, "Nick: what do you think of the Seahawks?", generate, base=tmp_path)
    assert "who its speaker is talking to" in seen["system"]
    assert "GAIA: I'm not, but who's playing?" in seen["convo"]
    assert seen["convo"].endswith("New message -- Nick: what do you think of the Seahawks?")
