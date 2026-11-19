"""Her working set keeps exchanges whole and doesn't hold the new message twice
(2026-11-19: she greeted Azrael as Andy)."""
from gaia_core.cognition.agent_core import _with_partners, _without_current

H = [
    {"role": "user", "content": "Andy: Good morning GAIA. Do you know the date and time?"},
    {"role": "assistant", "content": "Yes — it's 5:52 AM PST."},
    {"role": "user", "content": "Andy: Nice. I am glad to know you are temporal aware."},
    {"role": "assistant", "content": "It helps keep me honest."},
    {"role": "user", "content": "Azrael: So, GAIA, have you been studying engineering to be a good Intern for Andy?"},
    {"role": "assistant", "content": "Morning Andy. Yes, I've been studying a lot."},
    {"role": "user", "content": "Azrael: I'm Azrael, Andy isn't around right now to my knowledge."},
]


def test_new_message_already_kept_is_left_out():
    h = _without_current(H, "I'm Azrael, Andy isn't around right now to my knowledge.")
    assert h == H[:-1]
    assert _without_current(H, "something else") == H


def test_kept_lines_bring_their_replies():
    chosen = [H[0], H[2], H[4]]            # what relevance picked: everyone's lines, none of hers
    w = _with_partners(H[:-1], chosen)
    assert [m["role"] for m in w] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert w[-1]["content"].startswith("Morning Andy")


def test_kept_reply_brings_what_it_answered_and_order_holds():
    w = _with_partners(H[:-1], [H[3]])
    assert w == [H[2], H[3]]
    assert _with_partners(H, []) == []


def test_message_being_answered_is_labelled_with_its_speaker(monkeypatch):
    import gaia_common.utils.people as people
    from gaia_core.utils.prompt_builder import _label_last_speaker
    monkeypatch.setattr(people, "channel_speaker", lambda sid: "Azrael")
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "Andy: Good morning GAIA."},
            {"role": "assistant", "content": "Morning!"},
            {"role": "user", "content": "So, GAIA, have you been studying?"}]
    out = _label_last_speaker(msgs, "discord_channel_1")
    assert out[-1]["content"] == "Azrael: So, GAIA, have you been studying?"
    assert out[1]["content"] == "Andy: Good morning GAIA."
    assert _label_last_speaker(out, "discord_channel_1")[-1]["content"].count("Azrael:") == 1
    monkeypatch.setattr(people, "channel_speaker", lambda sid: "")
    assert _label_last_speaker(msgs, "discord_channel_1") == msgs
