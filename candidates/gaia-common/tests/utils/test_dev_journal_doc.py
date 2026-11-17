"""2026-11-17: asked about her dev journal, she reads the newest one."""
from gaia_common.utils import founding_docs as fd


def _notebook(tmp_path):
    for n in ("2026-11-08_dev_journal_delta.md", "2026-11-15_dev_journal_delta.md",
              "2026-11-16_plan.md", "context_brief.md"):
        (tmp_path / n).write_text(n)
    return tmp_path


def test_latest_dev_journal(tmp_path):
    path, title = fd.latest_dev_journal(str(_notebook(tmp_path)))
    assert path.endswith("2026-11-15_dev_journal_delta.md")
    assert "2026-11-15" in title


def test_named_document_finds_the_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "DEV_NOTEBOOK_DIR", str(_notebook(tmp_path)))
    for q in ("What do you think of your recent dev journal and your development cycles overall?",
              "can you read your development journal?",
              "GAIA, what's in the dev notes?"):
        found = fd.named_document(q)
        assert found and found[0].endswith("2026-11-15_dev_journal_delta.md"), q


def test_not_hers_or_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "DEV_NOTEBOOK_DIR", str(tmp_path / "nowhere"))
    assert fd.named_document("what do you think of your dev journal?") is None
    monkeypatch.setattr(fd, "DEV_NOTEBOOK_DIR", str(_notebook(tmp_path)))
    assert fd.named_document("how should I structure a dev journal") is None


def test_authorship():
    assert "Claude" in fd.authorship_for("/gaia/GAIA_Project/knowledge/Dev_Notebook/x.md")
    assert fd.authorship_for("/knowledge/system_reference/core_documents/gaia_constitution.md") == fd.AUTHORSHIP
