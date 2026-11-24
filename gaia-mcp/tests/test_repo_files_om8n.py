"""She can find her own code, and can't read the repo's secrets (om8n live test, 2026-11-24)."""
from pathlib import Path

import pytest

from gaia_mcp import tools as t


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "GAIA_Project"
    (r / "contracts" / "services").mkdir(parents=True)
    (r / "contracts" / "services" / "gaia-core.yaml").write_text("api: x")
    (r / "google-cloud-sdk").mkdir()
    (r / "google-cloud-sdk" / "gaia-core.yaml").write_text("not hers")
    (r / "old.bak").mkdir()
    (r / "old.bak" / "gaia-core.yaml").write_text("backup")
    (r / ".env").write_text("GROQ_API_KEY=x")
    (r / "README.md").write_text("hi")
    monkeypatch.setattr(t, "_REPO", str(r))
    monkeypatch.setattr(t, "_LIST_ROOTS", ("/knowledge", "/sandbox", str(r)))
    return r


def test_find_searches_her_repo_by_default(repo):
    out = t._find_files_impl({"query": "gaia-core.yaml"})
    assert out["results"] == [str(repo / "contracts" / "services" / "gaia-core.yaml")]


def test_find_takes_path_as_its_root(repo):
    out = t._find_files_impl({"query": "core", "path": str(repo / "contracts")})
    assert out["root"] == str(repo / "contracts") and len(out["results"]) == 1


def test_an_empty_find_says_where_it_looked(repo):
    out = t._find_files_impl({"query": "zzz"})
    assert out["results"] == [] and str(repo) in out["note"] and "/knowledge" in out["note"]


def test_a_guessed_root_names_the_real_ones(repo):
    with pytest.raises(ValueError, match="restricted to: .*GAIA_Project"):
        t._find_files_impl({"query": "api", "path": "/home/gaia/projects/gaia-core"})


def test_tree_lists_her_repo(repo):
    tree = t._list_tree_impl({"path": str(repo)})["tree"]
    assert "contracts/" in tree and "README.md" in tree
    assert ".env" not in tree and "google-cloud-sdk" not in tree and "old.bak" not in tree


@pytest.mark.parametrize("name", [".env", ".env.discord", "server.key", "credentials.json", "groq_api_key"])
def test_secret_files_are_not_read(repo, name, monkeypatch):
    (repo / name).write_text("s3cret")
    with pytest.raises(ValueError, match="holds secrets"):
        t._read_file_impl({"path": str(repo / name)})


def test_ordinary_files_still_read():
    for name in ("tokenizer.json", "test_secrets_links.py", "CLAUDE.md", "keys.md"):
        assert not t._secret_shaped(Path("/gaia/GAIA_Project") / name)
