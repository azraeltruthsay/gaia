"""Her inner monologue: her own question, in her own voice; only a real SAY speaks."""
import json

from gaia_core.cognition import inner_monologue as im


def _ask(tmp_path, monkeypatch, answer):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    seen = []
    out = im.ask_myself("I finished reading it. Anything to add?",
                        lambda m: seen.append(m) or answer, about="test", session_id="s1")
    return out, seen


def test_framed_as_her_own_thought_not_a_message(tmp_path, monkeypatch):
    out, seen = _ask(tmp_path, monkeypatch, "THOUGHT: yes\nSAY: Hi")
    system, user = seen[0]
    assert "inner monologue" in system["content"] and "No one wrote it to you" in system["content"]
    assert user["content"].startswith("(My own thought. Nobody sent this")


def test_say_speaks_nothing_stays_quiet(tmp_path, monkeypatch):
    out, _ = _ask(tmp_path, monkeypatch, "THOUGHT: I can answer now.\nSAY: The seal shows the number.")
    assert out["say"] == "The seal shows the number." and out["thought"] == "I can answer now."
    for quiet in ("THOUGHT: no\nSAY: NOTHING", "THOUGHT: no\nSAY: nothing.", "SAY:", "THOUGHT: hmm",
                  "Sure, here's what I'd say: hello"):
        out, _ = _ask(tmp_path, monkeypatch, quiet)
        assert out["say"] is None, quiet


def test_every_monologue_is_logged(tmp_path, monkeypatch):
    _ask(tmp_path, monkeypatch, "THOUGHT: nothing new\nSAY: NOTHING")
    _ask(tmp_path, monkeypatch, "no format at all")
    rows = [json.loads(l) for l in (tmp_path / "inner_monologue.jsonl").read_text().splitlines()]
    assert rows[0]["thought"] == "nothing new" and rows[0]["say"] is None
    assert rows[1]["unparsed"] == "no format at all"


def test_short_refusals_are_silence_and_format_is_restated_last(tmp_path, monkeypatch):
    for quiet in ("THOUGHT: x\nSAY: Nothing to add.", "THOUGHT: x\nSAY: None - nothing new here",
                  "THOUGHT: x\nSAY: No."):
        out, _ = _ask(tmp_path, monkeypatch, quiet)
        assert out["say"] is None, quiet
    out, seen = _ask(tmp_path, monkeypatch, "THOUGHT: x\nSAY: Nothing in the code covers fees, but part 2 does.")
    assert out["say"]                       # a real sentence that starts with 'Nothing'
    assert seen[0][1]["content"].rstrip().endswith("SAY: NOTHING.)")


def test_inner_voice_turn_is_marked_and_framed():
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField, Intent
    from gaia_core.utils.prompt_builder import build_from_packet
    assert im.mark("I wonder.").startswith(im.MARK) and im.mark(im.mark("x")) == im.mark("x")
    pkt = CognitionPacket(content=Content(original_prompt=im.mark("My top topic is the garden.")),
                          intent=Intent(user_intent="chat"))
    pkt.content.data_fields.append(DataField(key="inner_voice", value=True))
    msgs = build_from_packet(pkt)
    assert msgs[0]["role"] == "system" and im.TURN_FRAME in msgs[0]["content"]
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith(im.MARK)
    plain = CognitionPacket(content=Content(original_prompt="Hi"), intent=Intent(user_intent="chat"))
    assert im.TURN_FRAME not in build_from_packet(plain)[0]["content"]
