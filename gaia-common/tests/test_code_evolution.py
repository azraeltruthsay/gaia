"""Tests for Code Evolution — GAIA's code self-awareness utilities."""

import os
from pathlib import Path

import pytest

from gaia_common.utils.code_evolution import (
    _basenames,
    _collect_files,
    _parse_bak_timestamp,
    archive_inventory,
    diff_candidate_vs_production,
    generate_code_evolution_snapshot,
    index_bak_files,
    recent_git_log,
)


# ── Internal helpers ─────────────────────────────────────────────────────


class TestHelpers:
    def test_basenames(self):
        assert _basenames(["a/b/foo.py", "c/bar.js"]) == ["foo.py", "bar.js"]

    def test_parse_bak_timestamp_valid(self):
        result = _parse_bak_timestamp("20260118_143015")
        assert result is not None
        assert "2026-01-18" in result

    def test_parse_bak_timestamp_invalid(self):
        assert _parse_bak_timestamp("notadate") is None
        assert _parse_bak_timestamp("") is None

    def test_collect_files(self, tmp_path):
        # Create a small source tree
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")
        (tmp_path / "src" / "style.css").write_text("body {}")
        (tmp_path / "src" / "data.bin").write_bytes(b"\x00")  # not in extensions
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("# skip")

        files = _collect_files(tmp_path)
        assert "src/main.py" in files
        assert "src/style.css" in files
        assert "src/data.bin" not in files
        assert "__pycache__/cached.py" not in files


# ── Candidate diff ───────────────────────────────────────────────────────


class TestDiffCandidateVsProduction:
    def test_unknown_service(self):
        result = diff_candidate_vs_production("nonexistent", "/tmp")
        assert "error" in result

    def test_identical_dirs(self, tmp_path):
        """When candidate and production match, no changes reported."""
        # Set up matching dirs
        cand = tmp_path / "candidates" / "gaia-common" / "gaia_common" / "utils"
        prod = tmp_path / "gaia-common" / "gaia_common" / "utils"
        cand.mkdir(parents=True)
        prod.mkdir(parents=True)
        (cand / "shared.py").write_text("# same")
        (prod / "shared.py").write_text("# same")

        result = diff_candidate_vs_production("gaia-common", str(tmp_path))
        assert result.get("files_changed") == []
        assert result.get("files_added") == []
        assert result.get("files_removed") == []
        assert result.get("unchanged", 0) == 1

    def test_detects_changes(self, tmp_path):
        """When files differ, reports them."""
        cand = tmp_path / "candidates" / "gaia-common" / "gaia_common"
        prod = tmp_path / "gaia-common" / "gaia_common"
        cand.mkdir(parents=True)
        prod.mkdir(parents=True)
        (cand / "module.py").write_text("# v2 — candidate")
        (prod / "module.py").write_text("# v1 — production")
        (cand / "new_file.py").write_text("# new")

        result = diff_candidate_vs_production("gaia-common", str(tmp_path))
        assert "gaia_common/module.py" in result["files_changed"]
        assert "gaia_common/new_file.py" in result["files_added"]


# ── Git log ──────────────────────────────────────────────────────────────


class TestRecentGitLog:
    def test_returns_list(self):
        """Git log should return a list (may be empty if not in a repo)."""
        result = recent_git_log("/gaia/GAIA_Project", limit=3)
        assert isinstance(result, list)
        if result:  # we're in the actual repo
            assert "hash" in result[0]
            assert "date" in result[0]
            assert "subject" in result[0]

    def test_nonexistent_dir(self):
        result = recent_git_log("/nonexistent/path", limit=5)
        assert result == []


# ── Bak file index ───────────────────────────────────────────────────────


class TestIndexBakFiles:
    def test_finds_bak_files(self, tmp_path):
        (tmp_path / "file.bak").write_text("old")
        (tmp_path / "data.json.bak.20260118_143015").write_text("{}")

        results = index_bak_files(str(tmp_path))
        assert len(results) == 2
        # Timestamped one should parse
        timestamped = [r for r in results if r.get("timestamp")]
        assert len(timestamped) == 1
        assert "2026-01-18" in timestamped[0]["timestamp"]

    def test_empty_dir(self, tmp_path):
        assert index_bak_files(str(tmp_path)) == []


# ── Archive inventory ────────────────────────────────────────────────────


class TestArchiveInventory:
    def test_nonexistent(self, tmp_path):
        result = archive_inventory(str(tmp_path / "noarchive"))
        assert result["exists"] is False

    def test_with_subdirs(self, tmp_path):
        archive = tmp_path / "archive"
        (archive / "monolith").mkdir(parents=True)
        (archive / "monolith" / "main.py").write_text("# old")
        (archive / "monolith" / "config.py").write_text("# old")

        result = archive_inventory(str(archive))
        assert result["exists"] is True
        assert result["total_files"] == 2
        assert result["subdirs"][0]["name"] == "monolith"


# ── Snapshot generation ──────────────────────────────────────────────────


class TestGenerateSnapshot:
    def test_generates_markdown(self, tmp_path):
        output = str(tmp_path / "snapshot.md")

        # Create minimal project structure
        proj = tmp_path / "project"
        (proj / "candidates" / "gaia-common" / "gaia_common").mkdir(parents=True)
        (proj / "gaia-common" / "gaia_common").mkdir(parents=True)

        result = generate_code_evolution_snapshot(
            project_root=str(proj),
            output_path=output,
        )
        assert result == output
        content = Path(output).read_text()
        assert "# Code Evolution Snapshot" in content
        assert "Generated:" in content
