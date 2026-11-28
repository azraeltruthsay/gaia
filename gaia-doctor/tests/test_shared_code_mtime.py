"""A change to shared code a service mounts is a change to the service (2026-11-27)."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import doctor  # noqa: E402


def test_a_gaia_common_edit_counts_for_the_services_that_mount_it(tmp_path, monkeypatch):
    core, common = tmp_path / "gaia-core", tmp_path / "gaia-common"
    core.mkdir(); common.mkdir()
    (core / "a.py").write_text("x = 1\n")
    (common / "b.py").write_text("y = 1\n")
    old = time.time() - 1000
    os.utime(core / "a.py", (old, old))
    os.utime(common / "b.py", (old, old))
    monkeypatch.setitem(doctor.SERVICE_CODE_DIRS, "gaia-core", core)
    monkeypatch.setitem(doctor.SERVICE_SHARED_DIRS, "gaia-core", [common])
    before = doctor._get_service_mtime("gaia-core")
    (common / "b.py").write_text("y = 2\n")
    assert doctor._get_service_mtime("gaia-core") > before
