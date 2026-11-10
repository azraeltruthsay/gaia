"""ba6o: test runs and candidates must not write into the live service log that
self_review reads as her own turns."""

import logging
import logging.handlers

import pytest

from gaia_common.utils import logging_setup as LS


@pytest.fixture(autouse=True)
def _restore_root():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
        if h not in saved:
            h.close()
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)


def _files():
    return [h for h in logging.getLogger().handlers
            if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler, LS.LevelRingHandler))]


def test_no_log_file_under_pytest(tmp_path, monkeypatch):
    monkeypatch.delenv("GAIA_LOG_FILES_UNDER_PYTEST", raising=False)
    LS.setup_logging(log_dir=str(tmp_path), service_name="gaia-core")
    logging.getLogger("x").warning("fixture output")
    assert not _files() and not (tmp_path / "gaia-core.log").exists()


def test_log_file_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_LOG_FILES_UNDER_PYTEST", "1")
    monkeypatch.setenv("HOSTNAME", "gaia-core")
    monkeypatch.delenv("GAIA_LOG_SERVICE_NAME", raising=False)
    LS.setup_logging(log_dir=str(tmp_path), service_name="gaia-core")
    assert (tmp_path / "gaia-core.log").exists()


def test_candidate_logs_under_its_own_name(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_LOG_FILES_UNDER_PYTEST", "1")
    monkeypatch.setenv("HOSTNAME", "gaia-core-candidate")
    monkeypatch.delenv("GAIA_LOG_SERVICE_NAME", raising=False)
    LS.setup_logging(log_dir=str(tmp_path), service_name="gaia-core")
    logging.getLogger("x").warning("candidate turn")
    assert (tmp_path / "gaia-core-candidate.log").exists()
    assert not (tmp_path / "gaia-core.log").exists()
    assert "[gaia-core-candidate]" in (tmp_path / "gaia-core-candidate.log").read_text()


def test_service_name_override(monkeypatch):
    monkeypatch.setenv("GAIA_LOG_SERVICE_NAME", "custom")
    assert LS.log_service_name("gaia-core") == "custom"
    monkeypatch.delenv("GAIA_LOG_SERVICE_NAME")
    monkeypatch.setenv("HOSTNAME", "abc123")
    assert LS.log_service_name("gaia-core") == "gaia-core"
    assert LS.log_service_name(None) is None
