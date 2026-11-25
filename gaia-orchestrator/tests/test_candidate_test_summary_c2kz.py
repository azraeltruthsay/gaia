"""The candidate test runner reports pytest's summary, not shutdown noise.
Five CodeMind attempts were logged as 'Tests failed: --- Logging error ---'."""
from gaia_orchestrator.main import pytest_summary

NOISY = """..........                                                    [100%]
1 failed, 53 passed in 5.85s
--- Logging error ---
Traceback (most recent call last):
ValueError: I/O operation on closed file.
"""


def test_summary_is_pytests_line_not_logging_noise():
    assert pytest_summary(NOISY) == "1 failed, 53 passed in 5.85s"


def test_decorated_and_other_forms():
    assert pytest_summary("===== 3 passed, 1 skipped in 0.83s =====") == "3 passed, 1 skipped in 0.83s"
    assert pytest_summary("2 errors in 0.12s") == "2 errors in 0.12s"
    assert pytest_summary("no tests ran in 0.01s") == "no tests ran in 0.01s"
    assert pytest_summary("--- Logging error ---\nsomething else") == ""
