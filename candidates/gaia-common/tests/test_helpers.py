"""Tests for gaia_common utility helpers."""

import os
import pytest
from pathlib import Path

from gaia_common.utils.helpers import (
    safe_mkdir,
    get_timestamp,
    get_timestamp_for_filename,
    ensure_parent_dir,
    normalize_path,
)


class TestSafeMkdir:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "new_dir"
        safe_mkdir(str(target))
        assert target.is_dir()

    def test_idempotent(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        # Should not raise
        safe_mkdir(str(target))
        assert target.is_dir()


class TestTimestamps:
    def test_get_timestamp_returns_string(self):
        ts = get_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_get_timestamp_for_filename_no_colons(self):
        ts = get_timestamp_for_filename()
        assert ":" not in ts


class TestEnsureParentDir:
    def test_creates_parent(self, tmp_path):
        target_file = tmp_path / "sub" / "deep" / "file.txt"
        ensure_parent_dir(str(target_file))
        assert target_file.parent.is_dir()


class TestNormalizePath:
    def test_strips_trailing_slash(self):
        result = normalize_path("/some/path/")
        assert not result.endswith("/") or result == "/"

    def test_handles_empty_string(self):
        result = normalize_path("")
        assert isinstance(result, str)
