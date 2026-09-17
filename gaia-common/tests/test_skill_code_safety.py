"""Tests for skill_code_safety — the static AST denylist gating CodeMind's
autonomously-authored PLAYBOOK skill code (9ar0)."""

from gaia_common.utils.skill_code_safety import lint_skill_source


GOOD_MODULE = '''
def execute(params: dict) -> dict:
    """Return a short greeting."""
    name = params.get("name", "world")
    return {"ok": True, "content": f"Hello, {name}!"}
'''


def test_clean_module_passes():
    result = lint_skill_source(GOOD_MODULE)
    assert result.ok
    assert result.violations == []


def test_subprocess_import_blocked():
    code = "import subprocess\n\ndef execute(params):\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("subprocess" in v for v in result.violations)


def test_socket_import_blocked():
    code = "import socket\n\ndef execute(params):\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("socket" in v for v in result.violations)


def test_eval_call_blocked():
    code = "def execute(params):\n    return {'ok': True, 'v': eval(params['expr'])}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("eval" in v for v in result.violations)


def test_os_system_blocked():
    code = "import os\n\ndef execute(params):\n    os.system('echo hi')\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("os.system" in v for v in result.violations)


def test_os_system_via_from_import_blocked():
    code = "from os import system\n\ndef execute(params):\n    system('echo hi')\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("os.system" in v for v in result.violations)


def test_shutil_rmtree_blocked():
    code = "import shutil\n\ndef execute(params):\n    shutil.rmtree('/')\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("shutil.rmtree" in v for v in result.violations)


def test_open_outside_scratch_blocked():
    code = "def execute(params):\n    open('/etc/passwd').read()\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert not result.ok
    assert any("open()" in v for v in result.violations)


def test_open_in_scratch_prefix_allowed():
    code = "def execute(params):\n    open('/tmp/scratch.txt', 'w').write('x')\n    return {'ok': True}\n"
    result = lint_skill_source(code)
    assert result.ok


def test_open_with_computed_path_flagged():
    code = (
        "def execute(params):\n"
        "    p = params.get('path')\n"
        "    open(p).read()\n"
        "    return {'ok': True}\n"
    )
    result = lint_skill_source(code)
    assert not result.ok


def test_syntax_error_reported_not_raised():
    result = lint_skill_source("def execute(params:\n    return")
    assert not result.ok
    assert any("syntax error" in v for v in result.violations)
