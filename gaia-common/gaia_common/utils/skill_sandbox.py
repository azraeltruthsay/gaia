"""Subprocess sandbox for test-running a drafted PLAYBOOK skill (9ar0).

Runs a candidate skill module's `execute(params)` in a **separate OS
process**, never imported into the live gaia-core/gaia-mcp process. This is
the last line of defense: even a drafted module that slips past
skill_code_safety.lint_skill_source (a static check, necessarily
incomplete) cannot touch live state from here — a subprocess crash, hang,
or misbehavior is contained and simply reported as a failed sandbox run.

The runner script is generated fresh per call rather than a checked-in file
so the sandboxed process only ever sees module_source + params as data
(passed via a temp file and JSON on argv), never as something it could
import back into a shared namespace.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_TIMEOUT = 5.0

_RUNNER_TEMPLATE = """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("_sandboxed_skill", {module_path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

params = json.loads({params_json!r})
result = mod.execute(params)
sys.stdout.write("\\n___SANDBOX_RESULT___\\n")
sys.stdout.write(json.dumps(result, default=str))
"""


@dataclass
class SandboxResult:
    ok: bool
    timed_out: bool = False
    exit_code: Optional[int] = None
    result: Any = None
    stdout_preview: str = ""
    stderr_preview: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "timed_out": self.timed_out,
            "exit_code": self.exit_code,
            "result": self.result,
            "stdout_preview": self.stdout_preview,
            "stderr_preview": self.stderr_preview,
            "error": self.error,
        }


_RESULT_MARKER = "___SANDBOX_RESULT___"


def run_sandboxed(
    module_source: str,
    params: Dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> SandboxResult:
    """Execute `module_source`'s execute(params) in an isolated subprocess.

    Returns a SandboxResult — never raises for the sandboxed code's own
    failures (syntax errors, exceptions, timeouts); those are captured and
    reported. A raised exception here would only come from this function's
    own setup (e.g. disk full writing the temp file).
    """
    with tempfile.TemporaryDirectory(prefix="gaia_skill_sandbox_") as tmpdir:
        module_path = Path(tmpdir) / "_sandboxed_skill.py"
        try:
            module_path.write_text(module_source, encoding="utf-8")
        except OSError as e:
            return SandboxResult(ok=False, error=f"failed to stage module: {e}")

        runner_script = _RUNNER_TEMPLATE.format(
            module_path=str(module_path),
            params_json=json.dumps(params, default=str),
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={"PATH": "/usr/bin:/bin"},  # minimal — no inherited secrets/env
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                ok=False,
                timed_out=True,
                error=f"sandboxed execute() exceeded {timeout}s timeout",
                stdout_preview=(e.stdout or "")[:500] if isinstance(e.stdout, str) else "",
                stderr_preview=(e.stderr or "")[:500] if isinstance(e.stderr, str) else "",
            )
        except Exception as e:
            return SandboxResult(ok=False, error=f"sandbox launch failed: {e}")

        stdout_preview = (proc.stdout or "")[:2000]
        stderr_preview = (proc.stderr or "")[:2000]

        if proc.returncode != 0:
            return SandboxResult(
                ok=False,
                exit_code=proc.returncode,
                error="sandboxed process exited non-zero",
                stdout_preview=stdout_preview,
                stderr_preview=stderr_preview,
            )

        if _RESULT_MARKER not in (proc.stdout or ""):
            return SandboxResult(
                ok=False,
                exit_code=proc.returncode,
                error="execute() did not return a JSON result via the expected marker",
                stdout_preview=stdout_preview,
                stderr_preview=stderr_preview,
            )

        result_json = proc.stdout.split(_RESULT_MARKER, 1)[1].strip()
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError as e:
            return SandboxResult(
                ok=False,
                exit_code=proc.returncode,
                error=f"execute() result was not valid JSON: {e}",
                stdout_preview=stdout_preview,
                stderr_preview=stderr_preview,
            )

        # Sandbox "ok" means the process ran cleanly and returned a
        # well-formed dict — NOT the same as the skill's own reported
        # ok/error. A skill can legitimately answer the synthetic seed
        # payload with `{"ok": False, "error": "..."}` (e.g. the seed's
        # own hallucinated params were underspecified) without that being
        # a sandbox failure; conflating the two would discard working
        # drafts over an imperfect synthetic test case. The skill's own
        # verdict is preserved in `result` for human review.
        well_formed = isinstance(result, dict) and "ok" in result
        return SandboxResult(
            ok=well_formed,
            exit_code=proc.returncode,
            result=result,
            stdout_preview=stdout_preview,
            stderr_preview=stderr_preview,
            error="" if well_formed else "execute() result was not a dict with an 'ok' key",
        )
