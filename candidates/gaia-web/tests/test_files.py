"""Tests for file browser endpoints."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gaia_web.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_roots(tmp_path):
    """Create temporary directory structure for file browser tests."""
    # Build a test file tree
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')")
    (project / "README.md").write_text("# Project")
    (project / "sub").mkdir()
    (project / "sub" / "util.py").write_text("def helper(): pass")
    (project / ".hidden").write_text("secret")

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "notes.md").write_text("# Notes\nSome notes here.")
    (knowledge / "data.csv").write_text("a,b,c\n1,2,3")

    roots = f"project:{project},knowledge:{knowledge}"
    with patch.dict(os.environ, {"FILE_ROOTS": roots}):
        yield {"project": project, "knowledge": knowledge}


class TestRoots:
    def test_list_roots(self):
        resp = client.get("/api/files/roots")
        assert resp.status_code == 200
        data = resp.json()
        names = [r["name"] for r in data]
        assert "project" in names
        assert "knowledge" in names


class TestBrowse:
    def test_browse_root(self):
        resp = client.get("/api/files/browse/project/")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["name"] for e in data["entries"]]
        assert "sub" in names  # dir listed
        assert "main.py" in names
        assert "README.md" in names

    def test_browse_subdirectory(self):
        resp = client.get("/api/files/browse/project/sub")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["name"] for e in data["entries"]]
        assert "util.py" in names

    def test_hidden_files_excluded(self):
        resp = client.get("/api/files/browse/project/")
        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()["entries"]]
        assert ".hidden" not in names

    def test_dirs_sorted_first(self):
        resp = client.get("/api/files/browse/project/")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        types = [e["type"] for e in entries]
        # All dirs should come before files
        dir_indices = [i for i, t in enumerate(types) if t == "dir"]
        file_indices = [i for i, t in enumerate(types) if t == "file"]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)

    def test_browse_unknown_root(self):
        resp = client.get("/api/files/browse/nonexistent/")
        assert resp.status_code == 404

    def test_browse_nonexistent_path(self):
        resp = client.get("/api/files/browse/project/no_such_dir")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self):
        # HTTP layer normalizes ../  so traversal results in 403 or 404 (both block access)
        resp = client.get("/api/files/browse/project/../../etc")
        assert resp.status_code in (403, 404)


class TestRead:
    def test_read_file(self):
        resp = client.get("/api/files/read/project/main.py")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "main.py"
        assert data["extension"] == ".py"
        assert "print" in data["content"]

    def test_read_markdown(self):
        resp = client.get("/api/files/read/knowledge/notes.md")
        assert resp.status_code == 200
        data = resp.json()
        assert "# Notes" in data["content"]

    def test_read_nonexistent(self):
        resp = client.get("/api/files/read/project/no_such_file.py")
        assert resp.status_code == 404

    def test_read_directory_fails(self):
        resp = client.get("/api/files/read/project/sub")
        assert resp.status_code == 400

    def test_path_traversal_blocked(self):
        resp = client.get("/api/files/read/project/../../etc/passwd")
        assert resp.status_code in (403, 404)

    def test_read_unknown_root(self):
        resp = client.get("/api/files/read/nonexistent/file.txt")
        assert resp.status_code == 404


class TestWrite:
    def test_write_to_writable_root(self):
        resp = client.put(
            "/api/files/write/project/new_file.py",
            json={"content": "# new file\nprint('created')"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        # Verify content was written
        read_resp = client.get("/api/files/read/project/new_file.py")
        assert read_resp.status_code == 200
        assert "print('created')" in read_resp.json()["content"]

    def test_write_to_readonly_root(self):
        resp = client.put(
            "/api/files/write/knowledge/test.md",
            json={"content": "# should fail"},
        )
        assert resp.status_code == 403

    def test_write_path_traversal_blocked(self):
        resp = client.put(
            "/api/files/write/project/../../etc/evil.py",
            json={"content": "malicious"},
        )
        assert resp.status_code in (403, 404)

    def test_write_creates_subdirectory(self):
        resp = client.put(
            "/api/files/write/project/newdir/deep/file.py",
            json={"content": "nested"},
        )
        assert resp.status_code == 200
        read_resp = client.get("/api/files/read/project/newdir/deep/file.py")
        assert read_resp.status_code == 200
        assert read_resp.json()["content"] == "nested"
