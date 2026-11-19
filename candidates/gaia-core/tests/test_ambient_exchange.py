"""A reply to her is not ambient: the person she just answered, talking
back with no one in between, gets a reply without naming her (2026-11-19,
Andy answering her intern question)."""
from gaia_core.cognition import ambient_interject as ai

T0 = "2026-11-19T03:21:40+00:00"
T1 = "2026-11-19T03:21:48+00:00"
NOW = 1795058746.0  # 2026-11-19T03:25:46Z


def _h(*rows):
    return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]


ANDY = _h(("user", "ShinyAndreon: are you willing to be an intern?", T0),
          ("assistant", "What would you expect from an intern?", T1),
          ("user", "ShinyAndreon: I would expect an intern to assist with tasks.", "2026-11-19T03:25:46+00:00"))
NEW = "I would expect an intern to assist with tasks."


def test_andy_answering_her_is_in_exchange():
    assert ai.in_exchange(ANDY, NEW, now=NOW)


def test_someone_else_is_not():
    h = ANDY[:-1] + _h(("user", "Nick: lol interns", "2026-11-19T03:25:46+00:00"))
    assert not ai.in_exchange(h, "lol interns", now=NOW)


def test_someone_else_in_between_is_not():
    h = ANDY[:2] + _h(("user", "Nick: hey", "2026-11-19T03:23:00+00:00")) + ANDY[2:]
    assert not ai.in_exchange(h, NEW, now=NOW)


def test_their_own_messages_in_between_still_count():
    h = ANDY[:2] + _h(("user", "ShinyAndreon: hmm", "2026-11-19T03:23:00+00:00")) + ANDY[2:]
    assert ai.in_exchange(h, NEW, now=NOW)


def test_stale_reply_is_not():
    assert not ai.in_exchange(ANDY, NEW, now=NOW + 3600)


def test_should_interject_skips_judgment(tmp_path, monkeypatch):
    monkeypatch.setattr(ai, "in_exchange", lambda h, u: True)
    called = []
    assert ai.should_interject("k", ANDY, NEW, lambda m: called.append(1) or "NO", base=tmp_path)
    assert not called


def test_conversation_not_doubled():
    text = ai._conversation(ANDY, NEW)
    assert text.count("assist with tasks") == 1


def test_addressed_by_name_is_in_exchange():
    h = _h(("user", "Nick: hi", T0),
           ("assistant", "ShinyAndreon, what's your last name?", T1),
           ("user", "ShinyAndreon: Smith", "2026-11-19T03:25:46+00:00"))
    assert ai.in_exchange(h, "Smith", now=NOW)


def test_unbacked_flags_invented_path():
    from gaia_core.cognition.second_mind import unbacked
    assert "/proc/driver/nvidia/gpu/0" in unbacked("That matches what I read from /proc/driver/nvidia/gpu/0.", "GPU: 10.8 GB")
    assert not unbacked("I read /proc/meminfo.", "source /proc/meminfo: 31 GB")


def test_open_question_by_name_survives_others_talking():
    h = _h(("assistant", "Andy, could you share the /in/ link?", "2026-11-19T03:22:55+00:00"),
           ("user", "Nick: That's what I basically said.", "2026-11-19T03:23:12+00:00"),
           ("assistant", "Nick, you're right.", "2026-11-19T03:23:21+00:00"),
           ("user", "Andy: Prince is my last name.", "2026-11-19T03:25:46+00:00"))
    assert ai.in_exchange(h, "Prince is my last name.", now=NOW)


def test_open_question_answered_already_does_not_count():
    h = _h(("assistant", "Andy, what's your last name?", "2026-11-19T03:22:55+00:00"),
           ("user", "Andy: Prince", "2026-11-19T03:23:00+00:00"),
           ("user", "Nick: told you", "2026-11-19T03:23:10+00:00"),
           ("assistant", "Nick, you're right.", "2026-11-19T03:23:21+00:00"),
           ("user", "Andy: lol", "2026-11-19T03:25:46+00:00"))
    assert not ai.in_exchange(h, "lol", now=NOW)
