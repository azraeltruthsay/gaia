"""Tests for skill_sandbox — the subprocess isolation harness that test-runs
a drafted skill before it's ever considered for promotion (9ar0)."""

from gaia_common.utils.skill_sandbox import run_sandboxed


GOOD_MODULE = '''
def execute(params: dict) -> dict:
    name = params.get("name", "world")
    return {"ok": True, "content": f"Hello, {name}!"}
'''

RAISING_MODULE = '''
def execute(params: dict) -> dict:
    raise RuntimeError("boom")
'''

HANGING_MODULE = '''
import time

def execute(params: dict) -> dict:
    time.sleep(30)
    return {"ok": True}
'''

BAD_RETURN_MODULE = '''
def execute(params: dict) -> dict:
    return "not a dict"
'''

BROKEN_SYNTAX_MODULE = "def execute(params:\n    return"


def test_clean_execution_returns_result():
    result = run_sandboxed(GOOD_MODULE, {"name": "GAIA"})
    assert result.ok
    assert result.result == {"ok": True, "content": "Hello, GAIA!"}
    assert result.exit_code == 0


def test_exception_in_execute_is_captured_not_raised():
    result = run_sandboxed(RAISING_MODULE, {})
    assert not result.ok
    assert result.exit_code != 0
    assert "boom" in result.stderr_preview or result.error


def test_timeout_is_caught():
    result = run_sandboxed(HANGING_MODULE, {}, timeout=1.0)
    assert not result.ok
    assert result.timed_out


def test_non_dict_result_flagged():
    result = run_sandboxed(BAD_RETURN_MODULE, {})
    assert not result.ok
    assert "ok" in result.error or "dict" in result.error


def test_syntax_broken_module_fails_cleanly():
    result = run_sandboxed(BROKEN_SYNTAX_MODULE, {})
    assert not result.ok
    assert not result.timed_out
    assert result.exit_code != 0


def test_skill_reported_error_is_not_conflated_with_sandbox_failure():
    """A skill legitimately answering ok=False for an underspecified
    synthetic call must count as a sandbox PASS (it ran cleanly and
    returned a well-formed dict) — not a failure. Conflating the two
    would discard working drafts over an imperfect synthetic test case."""
    module = '''
def execute(params: dict) -> dict:
    if "expr" not in params:
        return {"ok": False, "error": "missing expr"}
    return {"ok": True, "value": params["expr"]}
'''
    result = run_sandboxed(module, {})
    assert result.ok
    assert result.result == {"ok": False, "error": "missing expr"}
