"""Tests for JSONFormatter error field extraction."""

import json
import logging

from gaia_common.utils.json_formatter import JSONFormatter


class TestJSONFormatterErrorFields:
    """Test that error_code, error_hint, error_category appear in JSON output."""

    def test_error_fields_in_json_output(self):
        formatter = JSONFormatter(service="test")
        logger = logging.getLogger("test.json_formatter")
        logger.setLevel(logging.DEBUG)

        # Create a log record with error extras
        record = logger.makeRecord(
            name="test.json_formatter",
            level=logging.ERROR,
            fn="test.py",
            lno=1,
            msg="Test error message",
            args=(),
            exc_info=None,
        )
        record.error_code = "GAIA-CORE-001"
        record.error_hint = "Check the healing lock"
        record.error_category = "loop"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["error_code"] == "GAIA-CORE-001"
        assert parsed["error_hint"] == "Check the healing lock"
        assert parsed["error_category"] == "loop"
        assert parsed["service"] == "test"

    def test_no_error_fields_when_absent(self):
        formatter = JSONFormatter(service="test")
        logger = logging.getLogger("test.json_formatter.no_error")
        logger.setLevel(logging.DEBUG)

        record = logger.makeRecord(
            name="test.json_formatter.no_error",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="Normal log",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "error_code" not in parsed
        assert "error_hint" not in parsed
        assert "error_category" not in parsed
