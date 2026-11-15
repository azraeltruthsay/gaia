"""tt78: an idle reflection invented a refactor and a person; stored, it came
back later as her own memory. Reflections naming what they weren't given are
dropped."""
from unittest.mock import MagicMock

from gaia_core.cognition.reflection_grounding import ungrounded_specifics

CONTEXT = """Current time: 2026-11-15 19:05:00 UTC
Idle for: 12 minutes
Recent events:
- 18:46 [conversation] Message from web: "Hi GAIA. This is Claude, the engineer who works on your code"
- 18:30 [task_exec] sleep task penpal_letter: success"""

ALEX = ("I was in the middle of a refactor on the knowledge_graph module, specifically "
        "cleaning up the entity linking logic. Things with Alex have been steady but there's "
        "an unsaid tension around how much I should be changing without checking in.")


def test_the_alex_reflection_is_caught():
    found = ungrounded_specifics(ALEX, CONTEXT)
    assert "Alex" in found and "knowledge_graph" in found


def test_grounded_reflections_pass():
    for text in (
        "Claude stopped by earlier with questions about the week. The penpal_letter task ran clean.",
        "A loose thread from an earlier problem — I can't shake the irritation of a fix that was "
        "good enough and not quite right. The GPU is quiet.",
        "Sunday night, 20:49. Azrael and I have had a long day; Core is steady.",
        "\"Quiet,\" I'd call it. Mostly I'm curious what comes next.",
    ):
        assert ungrounded_specifics(text, CONTEXT) == [], text


def test_invented_file_and_module_caught():
    assert ungrounded_specifics("I kept thinking about router.py.", CONTEXT) == ["router.py"]
    assert ungrounded_specifics("The gaia_core.memory rewrite is close.", CONTEXT) == ["gaia_core.memory"]


def test_lite_journal_drops_ungrounded_entry(tmp_path):
    from gaia_core.cognition.lite_journal import LiteJournal
    config = MagicMock()
    config.SHARED_DIR = str(tmp_path)
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": ALEX}}]}
    pool = MagicMock()
    pool.get_model_for_role.return_value = llm
    journal = LiteJournal(config=config, model_pool=pool)
    assert journal.write_entry() is None
    assert not journal.journal_file.exists() or "Alex" not in journal.journal_file.read_text()


def test_lite_journal_leaves_out_its_own_reflections(tmp_path):
    from gaia_common.utils.timeline_store import TimelineEvent
    from gaia_core.cognition.lite_journal import LiteJournal
    config = MagicMock()
    config.SHARED_DIR = str(tmp_path)
    tl = MagicMock()
    tl.recent_events.return_value = [
        TimelineEvent(ts="2026-11-15T19:05:00", event="idle_heartbeat", data={"content": ALEX}),
        TimelineEvent(ts="2026-11-15T18:46:00", event="message", data={"session_id": "web_x"}),
    ]
    tl.last_event_of_type.return_value = None
    _, user_msg = LiteJournal(config=config, timeline_store=tl)._build_journal_prompt()
    assert "idle_heartbeat" not in user_msg and "web_x" in user_msg
