"""rlx8: "Remember this: my favorite color is cerulean." was saved as
'untitled' and its reply never reached the session history, so the next
turn ("What is my favorite color?") followed another user turn and Core
answered about herself."""

from gaia_core.cognition.knowledge_ingestion import detect_save_command


def test_untitled_save_is_named_after_what_it_says():
    r = detect_save_command("Remember this: my favorite color is cerulean.")
    assert r["subject"] == "my favorite color is cerulean"
    assert r["raw_content"] == "my favorite color is cerulean."


def test_explicit_subject_is_kept():
    r = detect_save_command("Remember this about Excalibur: it came from the Lady of the Lake")
    assert r["subject"] == "Excalibur"


def test_bare_save_without_content_stays_untitled():
    assert detect_save_command("Save this")["subject"] == "untitled"


def test_ordinary_questions_are_not_saves():
    assert detect_save_command("What is my favorite color?") is None


def _packet(history, prompt="What is my favorite color?"):
    from gaia_common.protocols.cognition_packet import (
        CognitionPacket, Content, Context, RelevantHistorySnippet, SessionHistoryRef)
    snips = [RelevantHistorySnippet(id=str(i), role=r, summary=t) for i, (r, t) in enumerate(history)]
    return CognitionPacket(
        context=Context(session_history_ref=SessionHistoryRef(type="hash", value=""),
                        cheatsheets=[], constraints=None, relevant_history_snippet=snips),
        content=Content(original_prompt=prompt, data_fields=[]))


def test_slim_prompt_carries_the_conversation():
    from gaia_core.utils.prompt_builder import build_from_packet
    msgs = build_from_packet(_packet([
        ("user", "Remember this: my favorite color is cerulean."),
        ("assistant", "Acknowledged. I've documented it."),
    ]), slim_mode=True)
    assert msgs[-1] == {"role": "user", "content": "What is my favorite color?"}
    assert msgs[-3] == {"role": "user", "content": "Remember this: my favorite color is cerulean."}
    assert msgs[-2]["role"] == "assistant"
    roles = [m["role"] for m in msgs[1:]]
    assert all(a != b for a, b in zip(roles, roles[1:]))  # strictly alternating


def test_slim_history_is_trimmed_and_alternates():
    from gaia_core.utils.prompt_builder import _slim_history
    hist = [("assistant", "stray opener")] + [("user", f"u{i}") if i % 2 == 0 else ("assistant", f"a{i}")
                                              for i in range(10)] + [("user", "x" * 2000)]
    msgs = _slim_history(_packet(hist))
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert len(msgs) <= 7 and all(len(m["content"]) <= 600 for m in msgs)
    assert _slim_history(_packet([])) == []
