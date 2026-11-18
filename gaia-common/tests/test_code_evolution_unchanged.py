"""x9h9: an unchanged snapshot is not rewritten (it was, every 60 s)."""
from gaia_common.utils import code_evolution as ce


def test_unchanged_snapshot_keeps_its_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "list_changed_candidates", lambda root: [])
    monkeypatch.setattr(ce, "recent_git_log", lambda *a, **k: [])
    out = tmp_path / "snap.md"
    ce.generate_code_evolution_snapshot(str(tmp_path), str(out))
    first = out.read_text()
    out.write_text(first.replace("Generated:", "Generated: (kept)"))
    ce.generate_code_evolution_snapshot(str(tmp_path), str(out))
    assert "(kept)" in out.read_text()


def test_changed_snapshot_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "recent_git_log", lambda *a, **k: [])
    out = tmp_path / "snap.md"
    monkeypatch.setattr(ce, "list_changed_candidates", lambda root: [])
    ce.generate_code_evolution_snapshot(str(tmp_path), str(out))
    monkeypatch.setattr(ce, "list_changed_candidates", lambda root: [
        {"service_id": "gaia-web", "files_changed": ["a.py"]}])
    ce.generate_code_evolution_snapshot(str(tmp_path), str(out))
    assert "gaia-web" in out.read_text()
