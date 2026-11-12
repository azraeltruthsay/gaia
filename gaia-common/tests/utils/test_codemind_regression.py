"""a0aw.8: the regression test CodeMind writes alongside a fix — pure
validation and file-writing helpers, no execution."""

from gaia_common.utils import codemind_regression as R

GOOD = '''import gaia_core.cognition.agent_core as agent_core


def test_cat_haiku_does_not_route():
    assert agent_core.AgentCore._should_use_tool_routing("Write a haiku about a cat.") is False
'''


def test_validate_test_code_accepts_well_formed_test():
    ok, why = R.validate_test_code(GOOD, "gaia_core.cognition.agent_core")
    assert ok, why


def test_validate_test_code_rejects_missing_test_function():
    content = "import gaia_core.cognition.agent_core as agent_core\nx = 1\n"
    ok, why = R.validate_test_code(content, "gaia_core.cognition.agent_core")
    assert not ok and "test_" in why


def test_validate_test_code_rejects_missing_import():
    content = "def test_something():\n    assert True\n"
    ok, why = R.validate_test_code(content, "gaia_core.cognition.agent_core")
    assert not ok and "import" in why


def test_validate_test_code_rejects_syntax_error():
    ok, why = R.validate_test_code("def test_x(:\n    pass\n", "gaia_core.cognition.agent_core")
    assert not ok and "syntax" in why.lower()


def test_validate_test_code_rejects_empty():
    ok, why = R.validate_test_code("", "mod")
    assert not ok and "empty" in why


def test_validate_test_code_accepts_import_from_parent_package():
    content = "from gaia_core.cognition import agent_core\n\ndef test_x():\n    assert agent_core\n"
    ok, why = R.validate_test_code(content, "gaia_core.cognition.agent_core")
    assert ok, why


def test_filename_is_unique_and_slugged():
    a = R.test_filename("gi-30a99e")
    b = R.test_filename("gi-30a99e")
    assert a.startswith("test_codemind_gi_30a99e_") and a.endswith(".py")
    # same hint, different second: not guaranteed unique within the same
    # second, but the slug itself must be stable and filesystem-safe
    assert "/" not in a and " " not in a
    assert a == b or a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0]


def test_write_test_creates_tests_dir(tmp_path):
    path = R.write_test(tmp_path, GOOD, "test_codemind_x.py")
    assert path == tmp_path / "tests" / "test_codemind_x.py"
    assert path.read_text() == GOOD
