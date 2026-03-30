"""
gaia-core/gaia_core/cognition/code_evaluator.py — Code Challenge Evaluator

Grades GAIA-generated code through three tiers:
  1. Syntax check (py_compile)
  2. Runtime execution in MCP sandbox
  3. Test assertion validation

Used by the code_skill_loop to evaluate GAIA's coding ability
and generate samvega artifacts from failures.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("GAIA.CodeEvaluator")

MCP_ENDPOINT = "http://gaia-mcp:8765/jsonrpc"
SANDBOX_DIR = "/sandbox"
CHALLENGE_FILE = f"{SANDBOX_DIR}/challenge.py"
RPC_TIMEOUT = 30


@dataclass
class Challenge:
    id: str
    level: int
    category: str
    prompt: str
    test_code: str
    timeout_seconds: int = 10
    tags: list = field(default_factory=list)


@dataclass
class CodeGrade:
    challenge_id: str
    passed: bool
    syntax_ok: bool
    runtime_ok: bool
    tests_passed: int
    tests_total: int
    error_output: str
    execution_time_ms: int
    generated_code: str


def load_challenges(path: str = "/knowledge/curricula/code_challenges.jsonl",
                    level: Optional[int] = None) -> list[Challenge]:
    """Load challenges from JSONL file, optionally filtered by level."""
    challenges = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if level is not None and data.get("level") != level:
                    continue
                challenges.append(Challenge(**data))
    except Exception as e:
        logger.error("Failed to load challenges from %s: %s", path, e)
    return challenges


def _mcp_call(method: str, params: dict) -> dict:
    """Make a JSON-RPC 2.0 call to gaia-mcp."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": method, "arguments": params},
        "id": 1,
    }).encode()

    req = Request(MCP_ENDPOINT, data=payload,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=RPC_TIMEOUT) as resp:
            result = json.loads(resp.read())
            if "error" in result:
                return {"ok": False, "error": result["error"]}
            return result.get("result", {})
    except (URLError, Exception) as e:
        return {"ok": False, "error": str(e)}


def _write_to_sandbox(code: str, filename: str = "challenge.py") -> bool:
    """Write code to the MCP sandbox directory."""
    result = _mcp_call("write_file", {
        "path": f"{SANDBOX_DIR}/{filename}",
        "content": code,
        "pre_approved": True,
    })
    return result.get("ok", False) or "success" in str(result).lower()


def _run_in_sandbox(command: str, timeout: int = 10) -> dict:
    """Execute a command in the MCP sandbox. Returns {stdout, stderr, exit_code}."""
    result = _mcp_call("run_shell", {
        "command": command,
        "pre_approved": True,
    })
    return result


def _count_assertions(test_code: str) -> int:
    """Count assert statements in test code."""
    return sum(1 for line in test_code.splitlines()
               if line.strip().startswith("assert "))


def evaluate_code(generated_code: str, challenge: Challenge) -> CodeGrade:
    """
    Evaluate generated code against a challenge.

    1. Syntax check via py_compile
    2. Write code + tests to sandbox
    3. Execute and check assertions
    """
    n_tests = _count_assertions(challenge.test_code)
    start = time.monotonic()

    # ── Syntax check ─────────────────────────────────────────────────────
    # Build the full file: generated code + test assertions
    full_code = f"{generated_code.strip()}\n\n# === Tests ===\n{challenge.test_code}\nprint('ALL_TESTS_PASSED')\n"

    # Write to sandbox
    if not _write_to_sandbox(full_code):
        return CodeGrade(
            challenge_id=challenge.id,
            passed=False,
            syntax_ok=False,
            runtime_ok=False,
            tests_passed=0,
            tests_total=n_tests,
            error_output="Failed to write code to sandbox",
            execution_time_ms=int((time.monotonic() - start) * 1000),
            generated_code=generated_code,
        )

    # ── Syntax check via py_compile ──────────────────────────────────────
    syntax_result = _run_in_sandbox(
        f"python3 -m py_compile {CHALLENGE_FILE}"
    )
    syntax_ok = "error" not in str(syntax_result).lower() or \
                syntax_result.get("exit_code", 1) == 0

    if not syntax_ok:
        return CodeGrade(
            challenge_id=challenge.id,
            passed=False,
            syntax_ok=False,
            runtime_ok=False,
            tests_passed=0,
            tests_total=n_tests,
            error_output=str(syntax_result.get("stderr", syntax_result)),
            execution_time_ms=int((time.monotonic() - start) * 1000),
            generated_code=generated_code,
        )

    # ── Runtime execution ────────────────────────────────────────────────
    run_result = _run_in_sandbox(
        f"timeout {challenge.timeout_seconds} python3 {CHALLENGE_FILE}"
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    stdout = str(run_result.get("stdout", run_result.get("output", "")))
    stderr = str(run_result.get("stderr", run_result.get("error", "")))
    exit_code = run_result.get("exit_code", run_result.get("returncode", -1))

    runtime_ok = exit_code == 0 or "ALL_TESTS_PASSED" in stdout
    passed = "ALL_TESTS_PASSED" in stdout

    # Count passed tests by running individually (if overall failed)
    tests_passed = n_tests if passed else _count_passed_tests(
        generated_code, challenge.test_code
    )

    error_output = ""
    if not passed:
        error_output = stderr if stderr else stdout
        # Truncate long errors
        if len(error_output) > 1000:
            error_output = error_output[:1000] + "\n... (truncated)"

    return CodeGrade(
        challenge_id=challenge.id,
        passed=passed,
        syntax_ok=True,
        runtime_ok=runtime_ok,
        tests_passed=tests_passed,
        tests_total=n_tests,
        error_output=error_output,
        execution_time_ms=elapsed_ms,
        generated_code=generated_code,
    )


def _count_passed_tests(generated_code: str, test_code: str) -> int:
    """Run each assertion individually to count how many pass."""
    passed = 0
    assertions = [line.strip() for line in test_code.splitlines()
                  if line.strip().startswith("assert ")]

    for i, assertion in enumerate(assertions):
        code = f"{generated_code.strip()}\n\ntry:\n    {assertion}\n    print('PASS_{i}')\nexcept:\n    print('FAIL_{i}')\n"
        _write_to_sandbox(code, f"test_{i}.py")
        result = _run_in_sandbox(f"timeout 5 python3 {SANDBOX_DIR}/test_{i}.py")
        output = str(result.get("stdout", result.get("output", "")))
        if f"PASS_{i}" in output:
            passed += 1

    return passed


def grade_to_dict(grade: CodeGrade) -> dict:
    """Convert CodeGrade to a serializable dict."""
    return asdict(grade)
