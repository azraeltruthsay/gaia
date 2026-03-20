"""CodeMind Validator — pure validation functions for proposed code changes.

Wraps py_compile, ruff, ast.parse. No side effects — safe for unit testing.
"""

from __future__ import annotations

import ast
import logging
import os
import py_compile
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.CodeMind.Validator")


@dataclass
class ValidationResult:
    """Result of validating a proposed code change."""
    file_path: str
    passed: bool = True
    py_compile_ok: bool = True
    ast_parse_ok: bool = True
    ruff_ok: bool = True
    ruff_issues: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "passed": self.passed,
            "py_compile_ok": self.py_compile_ok,
            "ast_parse_ok": self.ast_parse_ok,
            "ruff_ok": self.ruff_ok,
            "ruff_issues": self.ruff_issues,
            "errors": self.errors,
        }


def validate_syntax(content: str, file_path: str = "<proposed>") -> ValidationResult:
    """Validate Python syntax using py_compile and ast.parse."""
    result = ValidationResult(file_path=file_path)

    # AST parse check
    try:
        ast.parse(content, filename=file_path)
    except SyntaxError as e:
        result.ast_parse_ok = False
        result.passed = False
        result.errors.append(f"ast.parse: {e.msg} (line {e.lineno})")

    # py_compile check (needs a temp file)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            result.py_compile_ok = False
            result.passed = False
            result.errors.append(f"py_compile: {e}")
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        result.py_compile_ok = False
        result.passed = False
        result.errors.append(f"py_compile setup error: {e}")

    return result


def validate_ruff(content: str, file_path: str = "<proposed>") -> ValidationResult:
    """Run ruff linter on proposed content. Requires ruff on PATH."""
    result = ValidationResult(file_path=file_path)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["ruff", "check", tmp_path, "--output-format=text"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                issues = [
                    line.strip()
                    for line in proc.stdout.splitlines()
                    if line.strip()
                ]
                result.ruff_issues = issues
                # Don't fail on ruff warnings — only syntax is blocking
                if any("E999" in i for i in issues):  # E999 = syntax error
                    result.ruff_ok = False
                    result.passed = False
                    result.errors.append("ruff: syntax error detected")
        finally:
            os.unlink(tmp_path)
    except FileNotFoundError:
        logger.debug("ruff not found on PATH — skipping lint check")
        # Not a failure — ruff is optional
    except subprocess.TimeoutExpired:
        result.errors.append("ruff: timed out after 30s")
    except Exception as e:
        logger.debug("ruff check failed: %s", e)

    return result


def validate_full(
    content: str,
    file_path: str = "<proposed>",
    checks: Optional[Dict[str, bool]] = None,
) -> ValidationResult:
    """Run all configured validation checks on proposed content.

    Args:
        content: The Python source code to validate.
        file_path: Display name for the file.
        checks: Dict of check_name → enabled. Defaults to all enabled.

    Returns:
        Combined ValidationResult.
    """
    if checks is None:
        checks = {"py_compile": True, "ast_parse": True, "ruff": True}

    result = ValidationResult(file_path=file_path)

    if checks.get("py_compile") or checks.get("ast_parse"):
        syntax_result = validate_syntax(content, file_path)
        if not checks.get("py_compile"):
            syntax_result.py_compile_ok = True  # skip
        if not checks.get("ast_parse"):
            syntax_result.ast_parse_ok = True  # skip

        result.py_compile_ok = syntax_result.py_compile_ok
        result.ast_parse_ok = syntax_result.ast_parse_ok
        result.errors.extend(syntax_result.errors)
        if not syntax_result.passed:
            result.passed = False

    if checks.get("ruff"):
        ruff_result = validate_ruff(content, file_path)
        result.ruff_ok = ruff_result.ruff_ok
        result.ruff_issues = ruff_result.ruff_issues
        result.errors.extend(ruff_result.errors)
        if not ruff_result.passed:
            result.passed = False

    return result


def validate_diff_safety(original: str, proposed: str, max_change_ratio: float = 0.5) -> Dict[str, Any]:
    """Sanity check that a proposed change isn't too destructive.

    Returns dict with 'safe' bool and 'reason' if unsafe.
    """
    orig_lines = original.splitlines()
    prop_lines = proposed.splitlines()

    if not orig_lines:
        return {"safe": True, "reason": "new file"}

    # Check for total content wipe
    if not proposed.strip():
        return {"safe": False, "reason": "proposed change deletes all content"}

    # Check change ratio
    orig_len = len(orig_lines)
    diff_count = abs(len(prop_lines) - orig_len)
    # Count changed lines (simple comparison)
    changed = 0
    for i in range(min(orig_len, len(prop_lines))):
        if i < len(orig_lines) and i < len(prop_lines):
            if orig_lines[i] != prop_lines[i]:
                changed += 1
    total_changes = changed + diff_count
    ratio = total_changes / max(orig_len, 1)

    if ratio > max_change_ratio:
        return {
            "safe": False,
            "reason": f"change ratio {ratio:.1%} exceeds max {max_change_ratio:.0%}",
        }

    return {"safe": True, "change_ratio": round(ratio, 3)}
