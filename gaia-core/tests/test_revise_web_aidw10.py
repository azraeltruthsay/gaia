"""aidw.10: a web-chat message lives only in the browser, so her history is
the record: she edits only a message her history holds, and the revision
keeps the old text from there."""
from gaia_core.cognition import revise as R


class SM:
    def __init__(self, history):
        self.h, self.saved = history, 0

    def get_history(self, sid):
        return self.h

    def _save_state(self):
        self.saved += 1


def test_web_message_in_history_is_edited(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    sm = SM([{"role": "user", "content": "rooms?"},
             {"role": "assistant", "content": "[Core]\n\nTwelve rooms, including a kennel."}])
    sent = []
    out = R.revise(sm, "web_ab12cd34", "Twelve rooms", "Three rooms.", "samvega x", major=True,
                   note="I checked my memory palace and corrected this",
                   edit=lambda sid, c, t: sent.append((sid, c, t)) or "")
    assert out.startswith("Three rooms.") and "Edited:" in out
    assert sent and sent[0][0] == "web_ab12cd34"
    assert sm.h[-1]["content"] == out and sm.h[-1]["revisions"][0]["old"].endswith("kennel.")
    assert sm.saved == 1


def test_web_message_not_in_history_is_not_sent(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    sm = SM([{"role": "assistant", "content": "Something else entirely."}])
    out = R.revise(sm, "web_ab12cd34", "Twelve rooms", "Three rooms.", "x",
                   edit=lambda *a: (_ for _ in ()).throw(AssertionError("must not send")))
    assert out is None


def test_web_event_not_written_is_not_edited(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    sm = SM([{"role": "assistant", "content": "Twelve rooms."}])
    assert R.revise(sm, "web_ab12cd34", "Twelve rooms", "Three rooms.", "x", edit=lambda *a: None) is None
    assert sm.h[-1]["content"] == "Twelve rooms."
