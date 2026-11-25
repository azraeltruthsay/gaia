"""r5z1: the regression-test prompt lists what the bug quotes and asks for
the failing case built from it word for word."""
from gaia_common.utils.codemind_engine import CodeMindEngine


def test_quoted_inputs_listed():
    bug = ('Asked "What is the SI unit of power?", my other mind answered '
           '"The SI unit of power is the watt (W)." Both said "watt".')
    p = CodeMindEngine.build_regression_test_prompt("f.py", "m", bug, "x = 1")
    block = p.split("QUOTED IN THE BUG", 1)[1].split("CURRENT CODE", 1)[0]
    assert '- "What is the SI unit of power?"' in block
    assert '- "The SI unit of power is the watt (W)."' in block
    assert '"watt"' not in block           # too short to be a case
    assert "word for word" in p and "Assert only what the bug description says" in p


def test_nothing_quoted_no_block():
    assert "QUOTED IN THE BUG" not in CodeMindEngine.build_regression_test_prompt("f.py", "m", "agree() is wrong", "x")
