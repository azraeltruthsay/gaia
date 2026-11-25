"""bqzt: a regression test wrapped in a markdown fence, or talked around, is
taken out of it instead of discarded as a syntax error."""
from gaia_common.utils.codemind_regression import extract_test_code, validate_test_code

CODE = "from gaia_core.cognition import second_mind as sm\n\n\ndef test_x():\n    assert sm\n"


def test_fenced_and_talked_around():
    for reply in (f"```python\n{CODE}```", f"Here is the test:\n\n```python\n{CODE}```\n\nIt fails before the fix.",
                  f"{CODE}```", f"```\n{CODE}"):
        out = extract_test_code(reply)
        assert out == CODE.strip(), reply
        assert validate_test_code(out, "gaia_core.cognition.second_mind")[0]


def test_the_block_with_the_test_is_chosen():
    reply = f"```bash\npytest tests/x.py\n```\n\n```python\n{CODE}```"
    assert extract_test_code(reply) == CODE.strip()


def test_plain_text_unchanged():
    assert extract_test_code(CODE) == CODE.strip()
