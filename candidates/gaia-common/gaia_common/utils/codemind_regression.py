"""CodeMind's regression-test step (a0aw.8): a test written alongside the fix.

Syntax/lint/diff-size checks (codemind_validator) say a patch is well-formed;
the module's pre-existing tests (codemind_tests, a0aw.2) say nothing already
broke. Neither proves the specific bug is actually gone — only a test that
fails on the untouched original and passes once the fix is applied does that.

Pure helpers only, no side effects beyond the explicit write_test() call —
same shape as codemind_validator.py (pure checks) and codemind_patch.py
(pure parse/apply). The content this module validates is never executed
here; it is only ever run later, in the candidate container, by
codemind_tests.run_tests().
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import Tuple

from gaia_common.utils.codemind_validator import validate_syntax
from gaia_common.utils.skill_creator import slugify_pattern

_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+test_\w+", re.M)


_FENCED = re.compile(r"```[\w+-]*[ \t]*\n(.*?)\n[ \t]*```", re.S)   # paired left to right
_OPEN_FENCE = re.compile(r"```[\w+-]*[ \t]*\n(.*)\Z", re.S)


def extract_test_code(text: str) -> str:
    """The test module out of a reply that wraps it in a markdown fence or
    talks around it (bqzt: Prime added a fence though asked not to, and the
    whole test was discarded as a syntax error at the ``` line). The fenced
    block holding a `def test_` (the longest, if several); an unclosed fence
    runs to the end; with no fence, the text itself."""
    text = text or ""
    if "```" not in text:
        return text.strip()
    blocks = [b.strip() for b in _FENCED.findall(text) if b.strip()]
    if not blocks:
        tail = _OPEN_FENCE.search(text)   # an opening fence never closed
        blocks = [tail.group(1).strip()] if tail and _TEST_DEF.search(tail.group(1)) else []
    tests = [b for b in blocks if _TEST_DEF.search(b)]
    if tests:
        return max(tests, key=len)
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()


def validate_test_code(content: str, module: str) -> Tuple[bool, str]:
    """Is `content` plausible as a regression test for `module`?

    Requires: clean syntax (validate_syntax), at least one `def test_...`,
    and an import that names `module` or its parent package. Never executes
    the content — that happens only later, inside the candidate container.
    """
    content = (content or "").strip()
    if not content:
        return False, "empty response"
    syntax = validate_syntax(content, file_path="<regression_test>")
    if not syntax.passed:
        return False, f"syntax: {'; '.join(syntax.errors) or 'invalid'}"
    if not _TEST_DEF.search(content):
        return False, "no def test_... found"
    parent = module.rsplit(".", 1)[0] if "." in module else module
    try:
        tree = ast.parse(content)
    except SyntaxError as e:  # pragma: no cover - validate_syntax already caught this
        return False, f"syntax: {e}"
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    if not any(n == module or n == parent or module.startswith(n + ".") for n in names):
        return False, f"does not import {module}"
    return True, "ok"


def test_filename(hint: str) -> str:
    """test_codemind_<slug-of-hint>_<unix-ts>.py — unique per write, traceable
    back to the issue/detection that produced it."""
    return f"test_codemind_{slugify_pattern(hint)}_{int(time.time())}.py"


def write_test(root: Path, content: str, filename: str) -> Path:
    """Write content to <root>/tests/<filename> (creating tests/ if needed).
    Returns the absolute path."""
    tests_dir = Path(root) / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    path = tests_dir / filename
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    return path
