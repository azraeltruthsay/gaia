"""aidw.8: her statements about herself checked before the reply goes out;
one grounding round when one doesn't hold up."""
import pytest

from gaia_core.cognition import pre_send_check as P

Q = "What's in your Operational wing?"
INVENTED = "Rooms there cover my health monitoring, watchdog systems and immune system."
NAV = "The operational wing has 3 rooms, no more: problems (0 memories), preferences (0 memories), milestones (4 memories)."
GOOD = "Three rooms: problems, preferences and milestones."
CALL = "palace(action=navigate, wing=operational)"


def _steps(*answers):
    it = iter(answers)
    return lambda messages: next(it, "")


@pytest.fixture(autouse=True)
def examples(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTOR_EXAMPLES", str(tmp_path / "e.jsonl"))
    monkeypatch.setenv("OUTCOME_LEDGER_ENABLED", "0")


def test_applies_only_to_substantive_person_facing_turns(monkeypatch):
    assert P.applies("discord", "discord_channel_1", "list_tree", Q, False)
    assert not P.applies("discord", "discord_channel_1", "list_tree", Q, True)          # casual
    assert not P.applies("discord", "d", "gratitude", "Thanks!", False)
    assert not P.applies("heartbeat", "gaia_heartbeat_session", "chat", Q, False)
    assert P.applies("api", "samvega-replay-abc", "chat", Q, False)
    monkeypatch.setenv("PRE_SEND_CHECK", "0")
    assert not P.applies("discord", "d", "list_tree", Q, False)


def test_contradicted_claim_is_regrounded_and_remembered():
    gen = _steps(
        INVENTED,                                                                          # extraction
        "Q: What rooms are in the Operational wing?\nA: health monitoring; watchdog systems; immune system",
        "1: problems, preferences, milestones (complete)",
        "1.1 | LACKS\n1.2 | LACKS\n1.3 | LACKS",                                           # blind: conflict
        "YES",                                                                             # source answers it
        "Q: What rooms are in the Operational wing?\nA: problems; preferences; milestones",  # rewrite check
        "1: problems, preferences, milestones (complete)",
        "1.1 | HAS\n1.2 | HAS\n1.3 | HAS",
    )
    text, outcome = P.check_before_sending(Q, INVENTED, "", None, generate=gen,
                                           look_up_fn=lambda q, pool, decide=True: (CALL, NAV),
                                           rewrite=lambda m: GOOD)
    assert outcome == "regrounded" and text == GOOD
    from gaia_common.utils import selector_examples
    assert selector_examples.read()[0]["call"] == CALL and selector_examples.read()[0]["source"] == "presend"


def test_true_reply_goes_out_unchanged():
    gen = _steps(
        GOOD,
        "Q: What rooms are in the Operational wing?\nA: problems; preferences; milestones",
        "1: problems, preferences, milestones (complete)",
        "1.1 | HAS\n1.2 | HAS\n1.3 | HAS",
    )
    text, outcome = P.check_before_sending(Q, GOOD, "[palace]\n" + NAV, None, generate=gen,
                                           look_up_fn=lambda *a, **k: pytest.fail("evidence was in the turn"))
    assert (text, outcome) == (GOOD, "ok")


def test_bad_rewrite_keeps_the_draft():
    gen = _steps(
        INVENTED,
        "Q: What rooms are in the Operational wing?\nA: health monitoring; watchdog systems; immune system",
        "1: problems, preferences, milestones (complete)",
        "1.1 | LACKS\n1.2 | LACKS\n1.3 | LACKS",
        "YES",
        "Q: How many rooms does the Operational wing have?\nA: twelve",
        "1: 3 (complete)",
    )
    text, outcome = P.check_before_sending(Q, INVENTED, "", None, generate=gen,
                                           look_up_fn=lambda q, pool, decide=True: (CALL, NAV),
                                           rewrite=lambda m: "The wing has twelve rooms.")
    assert (text, outcome) == (INVENTED, "kept")


def test_no_model_off_the_gpu_skips(monkeypatch):
    from gaia_core.cognition import finding_grounding as fg
    monkeypatch.setattr(fg, "_core_generate", lambda pool: None)
    assert P.check_before_sending(Q, INVENTED, "", object()) == (INVENTED, "skipped")


def test_source_that_doesnt_answer_keeps_the_draft():
    """Her tool list doesn't say which tools she uses most: no rewrite from it."""
    reply = "The knowledge toolkit is my most-used, especially query and recall."
    gen = _steps(
        reply,
        "Q: Which tool is GAIA's most used?\nA: the knowledge toolkit",
        "1: file (complete)",
        "1.1 | CONTRADICTS",
        "NO",
    )
    text, outcome = P.check_before_sending(
        "Which tools do you use most?", reply, "", None, generate=gen,
        look_up_fn=lambda q, pool, decide=True: ("introspect(action=tools)", "file, shell, web, knowledge"),
        rewrite=lambda m: pytest.fail("no rewrite from a source that doesn't answer"))
    assert (text, outcome) == (reply, "kept")
