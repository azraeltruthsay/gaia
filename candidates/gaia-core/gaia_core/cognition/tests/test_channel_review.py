"""channel_review: she looks back over a channel (Azrael, 2026-11-19)."""
from types import SimpleNamespace as R

import pytest

from gaia_core.cognition import channel_review as cr

Q = "A 6-inch inside-diameter pipe carries water at 3 ft/s. What is the flow in cfs and gpm?"


@pytest.fixture(autouse=True)
def shared(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))


def _h(now=1000.0):
    ts = "2026-11-19T23:00:00+00:00"
    return [
        {"id": "u1", "role": "user", "content": f"Azrael: {Q}", "timestamp": ts},
        {"id": "a1", "role": "assistant", "content": "The flow rate is 5.76 cfs and 261.8 gpm.", "timestamp": ts},
        {"id": "u2", "role": "user", "content": "Nick: anyone up for lunch?", "timestamp": ts},
        {"id": "u3", "role": "user", "content": "Andy: GAIA, what IBC year has Washington adopted?", "timestamp": ts},
        {"id": "a2", "role": "assistant", "content": "I'll look it up and get back to you.", "timestamp": ts},
    ]


def gen_steps(m):
    c = m[-1]["content"]
    if c.startswith("Work this problem"):
        return "D = 0.5\nQ = pi * D**2 / 4 * 3\ngpm = Q * 448.831"
    if m[0]["role"] == "system":
        return "What IBC year has Washington adopted?"
    if c.startswith("Your answer's numbers"):
        return "Q = 0.589 cfs, 264.4 gpm."
    return "I looked it up: Washington adopted the 2021 IBC (March 2024)."


PAGE = R(source="web", read=True, url="https://sbcc.wa.gov/t",
         content="March 2024 Codes in effect: 2021 IBC. Washington adopted the 2021 IBC (March 2024).")


def test_open_promise_is_kept_and_wrong_answer_corrected(monkeypatch):
    from gaia_core.cognition import promises
    import time
    h = _h()
    for m in h:
        m["timestamp"] = "2099-01-01T00:00:00+00:00"
    monkeypatch.setattr(time, "time", lambda: 4070908800.0 + 60)
    sent = []
    out = cr.review("discord_channel_1", h,
                    generate=lambda m: "THOUGHT: the flow looks off.\nRECHECK: 1",
                    same=gen_steps, other=None,
                    send=lambda s, t: sent.append(t) or True, record=lambda t: None,
                    ground=lambda q: [PAGE], now=4070908800.0 + 60)
    assert out["promise"] and "2021 IBC" in out["promise"]
    assert out["recheck"]["corrected"] and "0.589" in sent[-1] and sent[-1].startswith("Looking back")
    # each message once: a second look does nothing new
    again = cr.review("discord_channel_1", h, generate=lambda m: "RECHECK: 1", same=gen_steps, other=None,
                      send=lambda s, t: sent.append(t) or True, record=lambda t: None,
                      ground=lambda q: [PAGE], now=4070908800.0 + 60)
    assert again["promise"] is None


def test_nothing_said_when_the_check_backs_her():
    h = _h()
    h[1]["content"] = "About 0.589 cfs, or 264.4 gpm."
    out = cr.review("discord_channel_2", h[:2], generate=lambda m: "RECHECK: 1", same=gen_steps, other=None,
                    send=lambda s, t: pytest.fail("nothing to say"), record=lambda t: None, ground=lambda q: [])
    assert out["recheck"]["corrected"] is None


def test_kept_promise_is_not_open():
    h = _h()
    h.append({"id": "a3", "role": "assistant", "content": "I looked it up: 2021 IBC.",
              "timestamp": h[-1]["timestamp"]})
    assert cr.open_promises(h, now=cr._epoch(h[-1]["timestamp"]) + 10) == []


def test_already_corrected_message_is_not_rechecked():
    h = _h()[:2] + [{"id": "u9", "role": "user", "content": "Azrael: off by 10?", "timestamp": _h()[0]["timestamp"]},
                    {"id": "a9", "role": "assistant", "content": "I re-checked the calculation: 0.589 cfs.",
                     "timestamp": _h()[0]["timestamp"], "meta": {"via": "promise"}}]
    out = cr.review("discord_channel_3", h, generate=lambda m: "RECHECK: 1", same=gen_steps, other=None,
                    send=lambda s, t: pytest.fail("already corrected"), record=lambda t: None, ground=lambda q: [])
    # only the follow-up itself is a candidate now; picking 1 re-checks it, not the corrected answer
    assert out["recheck"] is None or "5.76" not in out["recheck"]["message"]


def test_her_own_corrections_are_not_rechecked():
    h = _h()[:2] + [{"id": "r1", "role": "assistant", "meta": {"unprompted": True, "via": "review"},
                     "content": "Looking back at what I said earlier (...), I was wrong. RCW 18.43.",
                     "timestamp": _h()[0]["timestamp"]}]
    latest, mine = cr.views(h)
    assert [m["id"] for i, m in mine if not cr._own_follow_up(m)] == ["a1"]


def test_question_only_since_her_previous_message():
    h = _h()
    # a2 answered Andy's IBC question, not the pipe problem before a1
    assert cr._prev_user(h, 4).startswith("GAIA, what IBC year")
    h2 = h[:2] + [{"id": "a5", "role": "assistant", "content": "Anything else?", "timestamp": h[0]["timestamp"]}]
    assert cr._prev_user(h2, 2) == ""


def test_inconclusive_check_is_tried_again():
    h = _h()[:2]
    disagree = iter(["Q = 1", "Q = 2", "Q = 3"])

    def gen_bad(m):
        return next(disagree) if m[-1]["content"].startswith("Work this problem") else "x"
    out = cr.review("discord_channel_4", h, generate=lambda m: "RECHECK: 1", same=gen_bad, other=None,
                    send=lambda s, t: pytest.fail("nothing confirmed"), record=lambda t: None, ground=lambda q: [])
    assert out["recheck"]["concluded"] is False
    sent = []
    out = cr.review("discord_channel_4", h, generate=lambda m: "RECHECK: 1", same=gen_steps, other=None,
                    send=lambda s, t: sent.append(t) or True, record=lambda t: None, ground=lambda q: [])
    assert out["recheck"]["corrected"] and "0.589" in sent[-1]


def test_calculation_answers_are_rechecked_without_her_pick():
    sent = []
    out = cr.review("discord_channel_5", _h()[:2], generate=lambda m: "RECHECK: NOTHING", same=gen_steps, other=None,
                    send=lambda s, t: sent.append(t) or True, record=lambda t: None, ground=lambda q: [])
    assert out["recheck"]["corrected"] and "0.589" in sent[-1]


def test_same_result_is_not_a_correction():
    assert cr._same_result("The hoop stress is 7,200 psi.", "= 3600 / 0.5 = 7,200 psi. Final Answer: 7,200 psi.")
    assert not cr._same_result("The dose rate at 4m would be 20 mR/hr.", "The rate is 12.5 mR/hr.")
