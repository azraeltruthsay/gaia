"""Her inner voice in the chat: switches, trace, thinking (2026-11-05)."""
from gaia_core.cognition import inner_voice as iv


class Sessions:
    def __init__(self):
        self.meta = {}

    def get_session_meta(self, sid, key, default=None):
        return self.meta.get((sid, key), default)

    def set_session_meta(self, sid, key, value):
        self.meta[(sid, key)] = value


def test_commands_are_recognised_and_nothing_else_is():
    assert iv.parse_command("!thinking on") == ("thinking", "on")
    assert iv.parse_command("/Thoughts AUTO") == ("thoughts", "auto")
    assert iv.parse_command("!thinking") == ("thinking", None)
    assert iv.parse_command("I was thinking on the bus") is None
    assert iv.parse_command("!thinking about you") is None


def test_a_switch_is_kept_per_conversation():
    s = Sessions()
    assert iv.mode(s, "a", "thinking") == "auto"
    assert "Thinking: on." in iv.handle_command(s, "a", "!thinking on")
    assert iv.mode(s, "a", "thinking") == "on" and iv.mode(s, "b", "thinking") == "auto"
    assert "Thinking is on." in iv.handle_command(s, "a", "!thinking")
    assert iv.handle_command(s, "a", "hello") is None


def test_thinking_modes():
    assert iv.wants_deep_focus("on", asked=False)
    assert iv.wants_deep_focus("auto", asked=True) and not iv.wants_deep_focus("auto", asked=False)
    assert iv.wants_deep_focus("off", asked=True)  # asked to think hard: she still does, privately
    assert not iv.show_thinking("off", "some thought") and iv.show_thinking("auto", "some thought")
    assert not iv.show_thinking("on", "  ")


def test_trace_in_auto_only_when_she_did_more_than_answer():
    plain = iv.trace(model="prime")
    assert plain == ["answered as Prime"]
    assert not iv.show_trace("auto", plain) and iv.show_trace("on", plain)
    busy = iv.trace(model="prime", conv_recall=True, sources=["mempalace", "web", "web"])
    assert iv.show_trace("auto", busy) and not iv.show_trace("off", busy)
    assert iv.render_trace(busy) == ("*[What I did]* answered as Prime · recalled our earlier conversations"
                                     " · checked my memory palace · searched the web")


def test_thinking_is_quoted_and_capped():
    out = iv.render_thinking("first line\n\nsecond " + "word " * 600)
    assert out.startswith("*[Thinking]*\n> first line\n>\n> second")
    assert len(out) < 1700 and out.endswith("…")
