"""A failed memory lookup and what she says after it (2026-11-11, Discord)."""
from gaia_core.cognition.reply_scrub import claims_memory, honest_about_failure, palace_recall_params
from gaia_core.cognition.semantic_probe import extract_candidate_phrases


def _said(events):
    return "".join(e["value"] for e in events if e.get("type") == "token")


def test_a_near_miss_palace_action_is_a_recall_of_what_she_was_told():
    assert palace_recall_params({"action": "recalling"}, "the chat with Andy") == \
        {"action": "recall", "query": "the chat with Andy"}
    assert palace_recall_params({"action": "recall", "query": "Andy"}, "x")["query"] == "Andy"
    assert palace_recall_params({"action": "store", "content": "c"}, "x") == {"action": "store", "content": "c"}


def test_saying_she_remembers_after_a_failed_lookup_is_not_kept():
    reply = [{"type": "token", "value": "I do recall that conversation now, thanks for the context."}]
    out = _said(honest_about_failure(iter(reply), "palace(recall)", memory=True))
    assert "won't claim to remember" in out


def test_owning_that_she_doesnt_remember_is_kept():
    reply = [{"type": "token", "value": "I don't recall that one, can you tell me more?"}]
    assert _said(honest_about_failure(iter(reply), "palace(recall)", memory=True)).startswith("I don't recall")
    assert not claims_memory("I'm not sure I remember if we did")


def test_apostrophes_and_sentence_ends_dont_make_phrases():
    got = extract_candidate_phrases("Oh Andy isn't building you, I am. He's just my friend")
    assert not any("building you" in p for p in got)
    got = extract_candidate_phrases("We used the Web Chat on Mission Control. It was fine, then Discord. Andy came")
    assert "Mission Control" in got and not any(". " in p for p in got)
    assert "R.S.S. Alice" in extract_candidate_phrases("aboard the R.S.S. Alice today")


class _Memory:
    RECS = [
        {"session_id": "discord_az", "user": "So, last week on Wednesday, you spoke with Andy through the "
         "Web Chat interface on Mission Control.", "reply": "I do recall that conversation now", "ts": "2026-11-11"},
        {"session_id": "web_andy", "user": "Hello GAIA. I am Andy. I am Azrael's friend.",
         "reply": "Hello, Andy.", "ts": "2026-11-05"},
    ]

    def recall_elsewhere(self, session_id, topic, top_k=3):
        t = set(topic.lower().split())
        return [r for r in self.RECS if t & set(r["user"].lower().replace(".", " ").split())][:top_k]


def test_his_own_conversation_comes_back_not_a_retelling(monkeypatch):
    from gaia_core.cognition import knowledge_router as kr
    monkeypatch.setattr(kr, "_people_named", lambda topic: [
        {"id": "p1", "names": ["Andy"], "accounts": [{"kind": "session", "id": "web_andy"}]}])
    got = kr._elsewhere(_Memory(), "new", "Wednesday Andy Web Chat Mission Control",
                        "Last week on Wednesday you spoke with Andy through the Web Chat interface on "
                        "Mission Control. Do you remember?")
    assert [h["session_id"] for h in got] == ["web_andy"]


def test_a_correction_shows_her_what_she_last_said():
    from gaia_core.utils.prompt_builder import correction_line
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "Do you know Andy?"},
            {"role": "assistant", "content": "He set out to build me."},
            {"role": "user", "content": "Oh Andy isn't building you, I am."}]
    line = correction_line("Oh Andy isn't building you, I am.", msgs)
    assert 'Your last reply was: "He set out to build me."' in line
    assert correction_line("Tell me about Andy", msgs) == ""


def test_messages_about_them_and_her_are_not_looked_up_on_the_web():
    from gaia_core.cognition.agent_core import is_relational
    for t in ["Does that apply to me as well though? I'm Azrael, and I've been the sole human architect. "
              "Would you share your ideas and plans with me?",
              "Do you have anything you'd like to change about yourself?",
              "I am your creator, you know.", "Oh Andy isn't building you, I am. He's just my friend"]:
        assert is_relational(t), t
    for t in ["How many sides does a heptagon have?", "Who was the most well known Roman emperor?",
              "What's the weather in Paris with the storm?"]:
        assert not is_relational(t), t
