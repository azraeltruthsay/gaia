"""d3px: a question about herself gets her founding documents, constitution first."""
import numpy as np

from gaia_core.cognition import self_grounding as S


class Embed:
    vocab = ["purpose", "consent", "time", "sleep", "mission", "rights"]

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.vocab], dtype=float) + 1e-3
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def _docs(tmp_path, monkeypatch):
    (tmp_path / "gaia_constitution.md").write_text(
        "# GAIA Constitution\n### Article XI: Purpose and Mission\nYour purpose and mission.\n"
        "### Article VII: Consent Principle\nConsent consent.\n")
    (tmp_path / "how_your_mind_works.md").write_text("# How Your Mind Works\n## Sleep\nSleep sleep time.\n")
    monkeypatch.setattr(S, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(S, "_index", None)


def test_constitution_first_when_it_speaks_to_the_question(tmp_path, monkeypatch):
    _docs(tmp_path, monkeypatch)
    got = S.passages_for("What is your purpose?", Embed())
    assert got[0]["doc"] == "gaia_constitution.md" and "Purpose" in got[0]["heading"]
    note = S.note_for("What is your purpose?", Embed())
    assert note.startswith("[From your own founding documents, written by Azrael") and "Article XI" in note


def test_an_unrelated_article_is_left_out(tmp_path, monkeypatch):
    _docs(tmp_path, monkeypatch)
    got = S.passages_for("How does your sleep work?", Embed())
    assert got and all(p["doc"] != "gaia_constitution.md" for p in got)


def test_only_when_both_minds_read_memory():
    assert S.wanted({"minds": 2, "sources": ["memory"], "act": "answer"})
    assert not S.wanted({"minds": 1, "sources": ["memory"], "act": "answer"})
    assert not S.wanted({"minds": 2, "sources": ["memory", "own"], "act": "answer"})
    assert not S.wanted({"minds": 2, "sources": ["memory"], "act": "do"})
    assert not S.wanted(None)
