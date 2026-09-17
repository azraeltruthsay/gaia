"""Static safety linter for CodeMind-authored PLAYBOOK skill source (9ar0).

Deterministic AST-walk denylist — independent of LLM judgment, in the same
spirit as the project's existing Blast Shield (gaia-mcp/tools.py path-prefix
blocks on write_file, run_shell command blocks). The authoring prompt asks
the model not to reach for process control / raw sockets / eval, but this
module is the harness's own defense — it does not trust the prompt to be
honored.

Not a general-purpose Python sandbox analyzer: scoped specifically to the
narrow shape CodeMind asks for (a single `execute(params) -> dict` body with
no imports beyond stdlib essentials). A drafted skill that trips this lint
is rerolled once with the violations fed back, then abandoned — never
silently accepted.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional

# Modules that grant capabilities well beyond "implement one MCP-style
# execute() function": process control, raw sockets, ctypes, dynamic
# import/eval infrastructure.
DENIED_IMPORTS = frozenset({
    "subprocess", "socket", "ctypes", "pty", "multiprocessing",
})

# Dotted call targets that are dangerous regardless of which module they
# were reached through (e.g. `from os import system` still resolves to a
# bare `system(...)` call — handled separately in visit_ImportFrom).
DENIED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__",
    "os.system", "os.popen",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "shutil.rmtree",
})

# from os import <name> that are equivalent to the denied os.* calls above.
DENIED_OS_FROM_IMPORTS = frozenset({
    "system", "popen", "execv", "execve", "execvp", "execvpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
})

ALLOWED_OPEN_PREFIXES = ("/tmp/", "/shared/scratch/")


@dataclass
class LintResult:
    ok: bool
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "violations": self.violations}


def _dotted_name(node: ast.AST) -> Optional[str]:
    """Resolve a Name/Attribute chain (e.g. `os.system`) to a dotted string."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    return ".".join(reversed(parts))


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in DENIED_IMPORTS:
                self.violations.append(f"forbidden import: {alias.name} (line {node.lineno})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if root in DENIED_IMPORTS:
            self.violations.append(f"forbidden import: {module} (line {node.lineno})")
        if root == "os":
            for alias in node.names:
                if alias.name in DENIED_OS_FROM_IMPORTS:
                    self.violations.append(f"forbidden import: os.{alias.name} (line {node.lineno})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target: Optional[str]
        if isinstance(node.func, ast.Name):
            target = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target = _dotted_name(node.func)
        else:
            target = None

        if target in DENIED_CALLS:
            self.violations.append(f"forbidden call: {target}() (line {node.lineno})")

        if target == "open" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if not first.value.startswith(ALLOWED_OPEN_PREFIXES):
                    self.violations.append(
                        f"open() outside allowed scratch prefixes {ALLOWED_OPEN_PREFIXES}: "
                        f"{first.value!r} (line {node.lineno})"
                    )
            else:
                # Path computed at runtime — can't statically verify scope.
                # Flagged for human review rather than silently allowed.
                self.violations.append(
                    f"open() with non-literal path — cannot verify scope statically (line {node.lineno})"
                )

        self.generic_visit(node)


def lint_skill_source(code: str) -> LintResult:
    """Deterministic static safety check for autonomously-authored skill code.

    Returns LintResult(ok=False, violations=[...]) rather than raising —
    callers (the CodeMind drafting harness in sleep_task_scheduler.py, the
    manual promote_capability_skill.sh gate) decide what to do with a failed
    lint (reroll, abandon, refuse to promote).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return LintResult(ok=False, violations=[f"syntax error: {e.msg} (line {e.lineno})"])

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return LintResult(ok=not visitor.violations, violations=visitor.violations)
