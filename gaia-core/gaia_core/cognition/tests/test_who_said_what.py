"""who_said_what: speakers of a channel's last lines (2026-11-20)."""
from gaia_core.utils.prompt_builder import who_said_what


def test_recap_names_the_speaker_not_the_person_mentioned():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "Azrael: So with GAIA, we've invented the CFR system. Contextual Focus Resolution."},
            {"role": "assistant", "content": "[Prime]\n\nYou've got such an elegant mind for bridging concepts."},
            {"role": "user", "content": "Azrael: That wasn't for you, silly. I was talking to Andy."}]
    out = who_said_what(msgs)
    lines = out.splitlines()
    assert lines[0].startswith("Who said what")
    assert lines[1].startswith('Azrael: "So with GAIA') and lines[2].startswith('You: "You\'ve got')
    assert lines[3].startswith('Azrael: "That wasn\'t for you')


def test_nothing_for_a_one_to_one_exchange():
    assert who_said_what([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]) == ""
