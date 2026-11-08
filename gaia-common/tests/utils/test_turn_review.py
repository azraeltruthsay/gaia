"""a0aw.6: turn-level view of gaia-core's log over a window."""

from datetime import datetime, timezone

from gaia_common.utils import turn_review as T

NOW = datetime(2026, 11, 8, 16, 40, tzinfo=timezone.utc)


def _l(ts, level, logger, msg):
    return f"2026-11-08T{ts}.000000+00:00 [gaia-core] {level}:{logger}:{msg}"


LOG = "\n".join([
    _l("15:00:00", "INFO", "GAIA.Core.API", "Processing packet web_old: 'Too old...' from web"),
    _l("16:27:12", "INFO", "GAIA.Core.API", "Processing packet web_a: 'Write a two-line poem about rain on a ...' from web"),
    _l("16:27:13", "INFO", "GAIA.IntentDetection", "Intent detection: done intent=brainstorming complexity=0.3 routing=TRIAGE"),
    _l("16:27:13", "INFO", "GAIA.PromptBuilder", "CognitionPacket(header=Header(session_id='s1', packet_id='p1', " + "x" * 5000),
    _l("16:27:15", "INFO", "GAIA.AgentCore", "[DELIBERATION] cognition-1 | model=core forbidden=0 confab=0 retried=False elapsed=4600ms"),
    _l("16:27:15", "WARNING", "GAIA.VLLMRemote", "vLLM POST http://localhost:8092/v1 payload=1307B"),
    _l("16:27:21", "INFO", "GAIA.Core.API", "Processing packet web_b: 'Write a haiku about a cat watching snow....' from web"),
    _l("16:27:22", "INFO", "GAIA.IntentDetection", "LLM intent 'recitation' has no named-work signal; downgrading to 'brainstorming'."),
    _l("16:27:22", "INFO", "GAIA.IntentDetection", "Intent detection: done intent=brainstorming complexity=0.357 routing=TRIAGE"),
    _l("16:27:22", "INFO", "GAIA.AgentCore", "Tool routing triggered for intent: brainstorming"),
    _l("16:27:24", "INFO", "GAIA.ToolSelector", "Tool selector determined no tool needed: general knowledge."),
    _l("16:27:24", "INFO", "GAIA.AgentCore", "Tool selector determined no tool needed"),
    _l("16:27:24", "WARNING", "GAIA.AgentCore", "[MODEL_SELECT DEBUG] acquired selected_model_name=nano -> <obj at 0x7a2f>"),
    _l("16:27:25", "WARNING", "GAIA.VLLMRemote", "vLLM POST http://localhost:8092/v1 payload=1310B"),
    _l("16:30:00", "INFO", "GAIA.Core.API", "Processing packet web_r: 'Look back over your turns...' from web"),
    _l("16:30:02", "INFO", "GAIA.ToolCallParser", "Tool call detected: introspect(action=trace, {})"),
    _l("16:35:12", "INFO", "GAIA.Core.API", "Processing packet web_c: 'GAIA, please use your introspect tool...' from web"),
    _l("16:35:14", "INFO", "GAIA.AgentCore", "Tool routing: executing introspect"),
]) + "\n"

HISTORY = """# Session History: s1

### USER (2026-11-08 16:27:13 UTC)

Write a two-line poem about rain on a tin roof.

---

### ASSISTANT (2026-11-08 16:27:21 UTC)

A rhythmic drum upon the metal sheet.

---

### USER (2026-11-08 16:27:23 UTC)

Write a haiku about a cat watching snow.

---

### ASSISTANT (2026-11-08 16:27:31 UTC)

Snowflakes drift by.

---
"""


def _setup(tmp_path):
    log = tmp_path / "gaia-core.log"
    log.write_text(LOG)
    hist = tmp_path / "chat_history"
    hist.mkdir()
    (hist / "s1.md").write_text(HISTORY)
    return log, hist


def test_turns_in_window_with_routing_and_replies(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW)
    text = r["text"]
    assert r["count"] == 2  # old turn outside window, current turn left out
    assert "this turn left out" in text and "introspect tool" not in text
    assert "Too old" not in text
    poem, haiku = text.split("[gaia-core.log:")[1:3]
    assert "tool routing did not fire" in poem and "answered by core in 4.6s" in poem
    assert 'reply: "A rhythmic drum' in poem
    assert "tool routing fired (intent brainstorming)" in haiku
    assert haiku.count("no tool needed") == 1
    assert "answered by core" in haiku  # nano is Core's alias
    assert "recitation downgraded to brainstorming" in haiku
    assert 'reply: "Snowflakes drift by."' in haiku
    assert "session s1" in haiku  # matched by text: the fast path logs no session


def test_earlier_reviews_left_out_unless_asked(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW)
    assert "1 turn(s) from earlier reviews of yourself left out" in r["text"] and "Look back" not in r["text"]
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW, include_reviews=True)
    assert r["count"] == 3 and "Look back" in r["text"]
    assert "introspect(trace) called mid-reply" in r["text"]


def test_follow_up_in_review_session_left_out(tmp_path):
    log, hist = _setup(tmp_path)
    log.write_text(LOG.replace(
        _l("16:35:12", "INFO", "GAIA.Core.API", "Processing packet web_c"),
        _l("16:30:00", "INFO", "GAIA.PromptBuilder", "Final prompt assembled for session 'rev1'. Messages: 2") + "\n"
        + _l("16:31:00", "INFO", "GAIA.Core.API", "Processing packet web_f: 'Which turns exactly?...' from web") + "\n"
        + _l("16:31:01", "INFO", "GAIA.PromptBuilder", "Final prompt assembled for session 'rev1'. Messages: 4") + "\n"
        + _l("16:35:12", "INFO", "GAIA.Core.API", "Processing packet web_c")))
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW)
    assert "Which turns exactly" not in r["text"] and "2 turn(s) from earlier reviews" in r["text"]


def test_recurring_warnings_left_out(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW)
    assert "vLLM POST" not in r["text"]
    assert "MODEL_SELECT" in r["text"]  # only on the haiku turn


def test_search_until_and_budget(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW, search="haiku")
    assert r["count"] == 1 and "poem" not in r["text"].split("\n", 1)[1]
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW,
                       until=T.parse_window_bound("2026-11-08T16:27:15", NOW))
    assert r["count"] == 1
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW, text_budget=50)
    assert "the last 1 shown" in r["text"]


def test_raw_lines_capped(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW)
    assert r["_raw_lines"] and max(len(x) for x in r["_raw_lines"]) <= 300


def test_parse_window_bound():
    assert T.parse_window_bound(30, NOW) == datetime(2026, 11, 8, 16, 10, tzinfo=timezone.utc)
    assert T.parse_window_bound("2026-11-08T16:00:00Z", NOW).hour == 16
    assert T.parse_window_bound("2026-11-08 16:00", NOW).minute == 0
    assert T.parse_window_bound("", NOW) is None and T.parse_window_bound("nonsense", NOW) is None


PLAYBOOK = """---
name: ts-misrouted-request
description: x
status: active
domain: troubleshooting
seen_in: [turns, logs]
---
# A request took the wrong path

## Symptoms
- A turn shows `tool routing fired` for an intent that needs no tool,
  especially when the selector then says `no tool needed`.
- You were asked to make something and talked *about* it.

## Find it
1. not a symptom
"""


def _skills(tmp_path):
    d = tmp_path / "skills"
    (d / "ts-misrouted-request").mkdir(parents=True)
    (d / "ts-misrouted-request" / "SKILL.md").write_text(PLAYBOOK)
    (d / "study").mkdir()
    (d / "study" / "SKILL.md").write_text(PLAYBOOK.replace("troubleshooting", "training").replace("ts-misrouted-request", "study"))
    (d / "ts-old").mkdir()
    (d / "ts-old" / "SKILL.md").write_text(PLAYBOOK.replace("active", "retired").replace("ts-misrouted-request", "ts-old"))
    (d / "ts-gear").mkdir()
    (d / "ts-gear" / "SKILL.md").write_text(PLAYBOOK.replace("seen_in: [turns, logs]", "seen_in: [gear, logs]").replace("ts-misrouted-request", "ts-gear"))
    return d


def test_playbook_symptoms_only_active_troubleshooting(tmp_path):
    got = T.playbook_symptoms(_skills(tmp_path))
    assert [n for n, _ in got] == ["ts-misrouted-request"]
    bullets = got[0][1]
    assert bullets[0] == "A turn shows `tool routing fired` for an intent that needs no tool, especially when the selector then says `no tool needed`."
    assert bullets[1] == "You were asked to make something and talked about it."
    assert not any("not a symptom" in b for b in bullets)


def test_review_lists_criteria_without_verdicts(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW, skills_dir=_skills(tmp_path))
    text = r["text"]
    assert "What your troubleshooting playbooks say to look for:" in text
    assert text.index("Write a haiku") < text.index("What your troubleshooting playbooks")
    assert r["playbooks"] == ["ts-misrouted-request"]
    # the turn is described, not judged
    assert "wrong" not in text.split("What your troubleshooting playbooks")[0].lower()


def test_criteria_fit_inside_budget(tmp_path):
    log, hist = _setup(tmp_path)
    r = T.review_turns(log, hist, since=T.parse_window_bound(60, NOW), now=NOW,
                       skills_dir=_skills(tmp_path), text_budget=700)
    assert len(r["text"]) < 1400 and "playbooks say" in r["text"]
