"""Asked to rest, she decides (rest_request.py, 2026-11-15).

Nothing acts on a phrase: a notice only leads to asking her, she must read
the message as a request, and she must agree after seeing what is running.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from gaia_core.cognition import rest_request as rr


class FakeEmbed:
    """Bag-of-words vectors: close enough to exercise nearest-example logic."""
    def __init__(self):
        self.vocab = {}

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for t in texts:
            words = [w.strip(",.?!'").lower() for w in t.split()]
            v = np.zeros(512)
            for w in words:
                v[self.vocab.setdefault(w, len(self.vocab) % 512)] += 1
            n = np.linalg.norm(v)
            rows.append(v / n if n else v)
        return np.array(rows)


def _gen(*answers):
    seq = iter(answers)
    calls = []

    def generate(messages):
        calls.append(messages)
        return next(seq)
    generate.calls = calls
    return generate


HUD = ["Gear: focusing", "Sleep cycle state: active", "Sleep task running now: none",
       "Other conversations in the last 15 minutes: none"]


def _handle(text, generate, acted, hud=HUD):
    return rr.handle(text, kind_hint_model=FakeEmbed(), history=[], who="Azrael",
                     generate=generate, hud=lambda kind: list(hud),
                     on_failure=lambda k, w: None,
                     actor=lambda kind, cb: acted.append(kind))


def test_noticing_alone_does_nothing_when_she_reads_it_as_no_request():
    acted = []
    gen = _gen('{"who_rests": "speaker", "kind": "sleep"}')
    assert _handle("go take a nap", gen, acted) is None
    assert acted == [] and len(gen.calls) == 1   # asked once, no consent step


def test_unrelated_message_never_reaches_the_model():
    acted = []
    gen = _gen()
    assert _handle("what is the capital of australia", gen, acted) is None
    assert gen.calls == []


def test_she_agrees_and_it_acts_with_her_own_reply():
    acted = []
    gen = _gen('{"who_rests": "gaia", "kind": "nap"}', json.dumps({"rest": True, "reply": "Nothing's running, so yes, a nap sounds good."}))
    out = _handle("go take a nap", gen, acted)
    assert out == {"reply": "Nothing's running, so yes, a nap sounds good.", "kind": "nap", "rest": True}
    assert acted == ["nap"]
    consent = gen.calls[1][-1]["content"]
    assert "Sleep task running now: none" in consent and "Azrael asked you to rest" in consent


def test_she_declines_and_nothing_acts():
    acted = []
    hud = HUD + ["Prime is generating for someone right now (1)."]
    gen = _gen('{"who_rests": "gaia", "kind": "sleep"}', '{"rest": false, "reply": "Prime is still generating for someone, so not yet."}')
    out = _handle("go to sleep", gen, acted, hud)
    assert out["rest"] is False and acted == []
    assert "Prime is still generating" in out["reply"]


def test_unusable_answer_means_no_action_and_an_ordinary_turn():
    acted = []
    gen = _gen('{"who_rests": "gaia", "kind": "nap"}', "sure thing!")
    assert _handle("take a nap", gen, acted) is None
    assert acted == []


def test_reply_naming_work_not_on_the_hud_is_replaced():
    acted = []
    gen = _gen('{"who_rests": "gaia", "kind": "nap"}', json.dumps(
        {"rest": True, "reply": "Let me finish the knowledge_graph refactor with Alex first, then nap."}))
    out = _handle("take a nap", gen, acted)
    assert "Alex" not in out["reply"] and "knowledge_graph" not in out["reply"]
    assert acted == ["nap"]


def test_thinking_is_stripped_before_reading_json():
    acted = []
    gen = _gen('<think>is this for me? {"kind": "none"}</think>{"who_rests": "gaia", "kind": "sleep"}',
               '{"rest": true, "reply": "Okay, going to sleep now."}')
    assert _handle("go to sleep", gen, acted)["kind"] == "sleep"


def test_hud_says_unknown_rather_than_guessing():
    lines = rr.read_hud(rr.SLEEP, gear=None, sleep_status=None, sleep_tasks=None, others=None,
                        busy={"Prime": None}, approvals=None)
    assert all("couldn't read" in line for line in lines)


def test_hud_lists_what_is_running():
    lines = rr.read_hud(rr.SLEEP, gear="awake",
                        sleep_status={"state": "active", "current_task": "docs_maintenance",
                                      "sleep_hold": {"active": True, "expires_at": "T", "reason": "penpal"},
                                      "voice_active": True, "auto_sleep_enabled": True},
                        sleep_tasks=["journal_write", "knowledge_ingestion"], others=["a Discord DM"],
                        busy={"Core": 0, "Prime": 2}, approvals=1)
    text = "\n".join(lines)
    for want in ("docs_maintenance", "penpal", "voice session", "journal_write, knowledge_ingestion",
                 "1 (a Discord DM)", "Prime is generating for someone right now (2)", "approval: 1"):
        assert want in text
    assert "Core is generating" not in text
    assert "gear 1, awake" in text


def test_other_conversations_window():
    now = datetime.now(timezone.utc)

    class S:
        def __init__(self, ts):
            self.ts = ts

        def last_message_timestamp(self):
            return self.ts
    sessions = {"me": S(now), "discord_dm_1": S(now - timedelta(minutes=3)),
                "web_x": S(now - timedelta(hours=2)), "heartbeat": S(now)}
    assert rr.other_conversations(sessions, "me", now) == ["a Discord DM"]


def test_act_parks_for_a_nap_and_reports_failure():
    posted, failed = [], []

    def post(url, timeout):
        posted.append(url)
        return {"ok": False, "detail": "matrix busy"}
    rr.act(rr.NAP, failed.append, post=post, delay=0).join(2)
    assert posted[0].endswith("/consciousness/park") and failed == ["matrix busy"]


def test_act_sleep_goes_through_force_sleep():
    posted, failed = [], []
    rr.act(rr.SLEEP, failed.append, post=lambda u, t: posted.append(u) or {"accepted": True}, delay=0).join(2)
    assert posted[0].endswith("/sleep/force") and failed == []
