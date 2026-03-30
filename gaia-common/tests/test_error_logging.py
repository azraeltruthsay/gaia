"""Tests for gaia_common.utils.error_logging — structured error logging."""

import logging

from gaia_common.utils.error_logging import log_gaia_error


class TestLogGaiaError:
    """Test the log_gaia_error helper."""

    def test_logs_at_registered_level(self, caplog):
        logger = logging.getLogger("test.error_logging")
        with caplog.at_level(logging.WARNING, logger="test.error_logging"):
            log_gaia_error(logger, "GAIA-CORE-025", "test loop detected")
        assert "GAIA-CORE-025" in caplog.text
        assert "Loop detected" in caplog.text

    def test_logs_at_error_for_unknown_code(self, caplog):
        logger = logging.getLogger("test.error_logging.unknown")
        with caplog.at_level(logging.ERROR, logger="test.error_logging.unknown"):
            log_gaia_error(logger, "GAIA-FAKE-999", "unknown code")
        assert "GAIA-FAKE-999" in caplog.text
        assert "unregistered" in caplog.text

    def test_level_override(self, caplog):
        logger = logging.getLogger("test.error_logging.override")
        with caplog.at_level(logging.DEBUG, logger="test.error_logging.override"):
            log_gaia_error(logger, "GAIA-CORE-001", "overridden", level_override=logging.DEBUG)
        assert "GAIA-CORE-001" in caplog.text

    def test_extra_fields_attached(self, caplog):
        """Verify error_code, error_hint, error_category are in log record extras."""
        logger = logging.getLogger("test.error_logging.extras")
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            log_gaia_error(logger, "GAIA-CORE-001", "test extras")
            assert len(records) == 1
            rec = records[0]
            assert rec.error_code == "GAIA-CORE-001"
            assert rec.error_hint != ""
            assert rec.error_category == "loop"
        finally:
            logger.removeHandler(handler)

    def test_detail_included_in_message(self, caplog):
        logger = logging.getLogger("test.error_logging.detail")
        with caplog.at_level(logging.WARNING, logger="test.error_logging.detail"):
            log_gaia_error(logger, "GAIA-CORE-025", "specific context here")
        assert "specific context here" in caplog.text
