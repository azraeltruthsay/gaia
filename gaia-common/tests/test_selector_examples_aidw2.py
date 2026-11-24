"""aidw.2: examples found by meaning (embedding similarity), not keywords."""
import numpy as np
import pytest

from gaia_common.utils import selector_examples as X


class FakeEmbed:
    """Bag-of-words vectors: enough to test ranking, thresholds and caching."""
    VOCAB = ["operational", "wing", "rooms", "poem", "rain", "adapters", "loaded", "technical", "palace"]

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.VOCAB], dtype="float32") + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTOR_EXAMPLES", str(tmp_path / "e.jsonl"))
    X._cache.update(mtime=None, rows=[], vecs=None, model=None)
    X.add("What's in your Operational wing?", "palace(action=navigate, wing=operational)", "samvega", "sv-1")
    X.add("Which adapters are loaded?", "study(action=adapter_list)", "turn")
    return tmp_path


def test_nearest_by_meaning(store):
    e = FakeEmbed()
    got = X.nearest("Azrael: which rooms are in the operational wing?", e)
    assert got and got[0]["call"].startswith("palace(") and got[0]["similarity"] >= X.MIN_SIMILARITY
    assert X.nearest("Write a poem about rain", e) == []


def test_cache_reused_until_file_changes(store):
    e = FakeEmbed()
    X.nearest("operational wing rooms", e)
    n = e.calls
    X.nearest("operational wing rooms", e)
    assert e.calls == n + 1          # only the question is encoded again
    import time
    time.sleep(0.01)
    X.add("What's in your technical wing?", "palace(action=navigate, wing=technical)", "turn")
    X.nearest("operational wing rooms", e)
    assert e.calls == n + 3          # rows re-encoded once, plus the question


def test_prompt_block_and_no_embed(store):
    assert X.nearest("anything", None) == []
    b = X.prompt_block([{"question": "What's in your Operational wing?", "call": "palace(action=navigate, wing=operational)"}])
    assert b.startswith("EARLIER QUESTIONS LIKE THIS") and '-> palace(action=navigate, wing=operational)' in b
    assert X.prompt_block([]) == ""


def test_call_label():
    assert X.call_label("palace", {"action": "navigate", "wing": "operational"}) == "palace(action=navigate, wing=operational)"
