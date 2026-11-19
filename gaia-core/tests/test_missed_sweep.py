"""A missed message is found once, by her pick, checked by code."""
from gaia_core.cognition import missed_sweep as ms

NOW = 1795058746.0  # 2026-11-19T03:25:46Z


def _h(*rows):
    return [{"role": r, "content": c, "timestamp": t} for r, c, t in rows]


H = _h(("assistant", "Andy, could you share the /in/ link?", "2026-11-19T03:20:00+00:00"),
       ("user", "Nick: lol", "2026-11-19T03:21:00+00:00"),
       ("user", "Andy: Try this linkedin.com/in/x", "2026-11-19T03:22:00+00:00"),
       ("user", "Azrael: she's restarting", "2026-11-19T03:22:30+00:00"))


def test_pick_and_only_once(tmp_path):
    seen = []
    gen = lambda m: seen.append(m) or "2"
    p = ms.pick_missed(H, "s", gen, now=NOW, base=tmp_path)
    assert p["speaker"] == "Andy" and p["text"] == "Try this linkedin.com/in/x"
    assert "Your last message: Andy, could you share" in seen[0][1]["content"]
    assert ms.pick_missed(H, "s", gen, now=NOW, base=tmp_path) is None   # looked at once


def test_none_and_out_of_range(tmp_path):
    assert ms.pick_missed(H, "a", lambda m: "NONE", now=NOW, base=tmp_path) is None
    assert ms.pick_missed(H, "b", lambda m: "7", now=NOW, base=tmp_path) is None
    assert ms.pick_missed(H, "c", lambda m: "", now=NOW, base=tmp_path) is None


def test_old_and_approval_messages_skipped(tmp_path):
    h = _h(("assistant", "Enter the code.", "2026-11-19T03:20:00+00:00"),
           ("user", "Azrael: 123456", "2026-11-19T03:21:00+00:00"),
           ("user", "Nick: ancient", "2026-11-19T01:00:00+00:00"))
    called = []
    assert ms.pick_missed(h, "d", lambda m: called.append(1) or "1", now=NOW, base=tmp_path) is None
    assert not called


def test_answered_messages_not_considered(tmp_path):
    h = H + _h(("assistant", "Got it.", "2026-11-19T03:23:00+00:00"))
    called = []
    assert ms.pick_missed(h, "e", lambda m: called.append(1) or "1", now=NOW, base=tmp_path) is None
    assert not called
