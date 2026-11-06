"""Shadow intent reading: rules when they fire, nearest examples otherwise (d3px)."""
import json
import time

import numpy as np

from gaia_core.cognition import intent_shadow as sh


class Embed:
    """Bag-of-words vectors: enough to test the voting, no model needed."""
    vocab = ["sword", "arthur", "haiku", "poem", "hello", "evening", "news", "today"]

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.vocab], dtype=float) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)


EXAMPLES = [{"text": "What was King Arthur's sword?", "label": "fact"},
            {"text": "Who forged Arthur's sword?", "label": "fact"},
            {"text": "Write me a haiku", "label": "create"},
            {"text": "A short poem please, a haiku", "label": "create"},
            {"text": "Good evening, hello", "label": "chat"},
            {"text": "Hello there", "label": "chat"}]


def _setup(tmp_path, monkeypatch, examples=EXAMPLES):
    p = tmp_path / "labeled.json"
    if examples is not None:
        p.write_text(json.dumps(examples))
    monkeypatch.setattr(sh, "EXAMPLES_PATH", str(p))
    monkeypatch.setattr(sh, "LOG_PATH", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setattr(sh, "_examples", None)
    monkeypatch.setattr(sh, "_matrix", None)


def test_nearest_examples_decide_when_no_rule_fires(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    got = sh.read("Where did Arthur get the sword?", [], Embed())
    assert got["rule"] is None and got["embed"] == "fact" and got["label"] == "fact"
    assert sh.read("Could you make a haiku?", [], Embed())["label"] == "create"


def test_a_rule_that_fires_wins(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    got = sh.read("Remember that the iron gate Grimhold-3 is painted green.", [], Embed())
    assert got["rule"] == "tell" and got["label"] == "tell"


def test_without_examples_nothing_is_read(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, examples=None)
    assert sh.read("Hello", [], Embed()) is None


def test_a_turn_is_logged_off_thread_and_background_turns_are_not(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    pool = type("Pool", (), {"get_embed_model": lambda self, **k: Embed()})()
    sh.observe("Good evening", [{"role": "user", "content": "hi"}], "s1", "web", pool)
    sh.observe("a seed", [], "s2", "heartbeat", pool)
    log = tmp_path / "shadow.jsonl"
    for _ in range(100):
        if log.exists():
            break
        time.sleep(0.02)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["session_id"] == "s1" and rows[0]["label"] == "chat"
    assert rows[0]["prev_user"] == "hi"
