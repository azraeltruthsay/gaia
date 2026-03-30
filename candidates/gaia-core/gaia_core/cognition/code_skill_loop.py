"""
gaia-core/gaia_core/cognition/code_skill_loop.py — Self-Supervised Coding Skill Loop

Orchestrates the cycle: challenge → generate → execute → grade → learn.

GAIA writes code for progressively harder challenges. Failures create samvega
artifacts which accumulate into LoRA training data. The trained adapter is
loaded dynamically via gaia_cpp — never merged into base weights.

Trigger modes:
  - Manual: via MCP tool `code_skill_drill`
  - Autonomous: during SLEEP cycle
  - Scheduled: via orchestrator
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

from gaia_core.cognition.code_evaluator import (
    Challenge, CodeGrade, evaluate_code, load_challenges, grade_to_dict,
)

logger = logging.getLogger("GAIA.CodeSkillLoop")

STATE_FILE = "/knowledge/curricula/code_skill_state.json"
SAMVEGA_DIR = "/knowledge/samvega"
TRAINING_BUFFER = "/shared/observer/code_skill_buffer.jsonl"
PRIME_ENDPOINT = os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777")
STUDY_ENDPOINT = os.environ.get("STUDY_ENDPOINT", "http://gaia-study:8766")

# Thresholds
ESCALATION_PASS_RATE = 0.8   # Advance to next level when >= 80%
TRAINING_TRIGGER_COUNT = 10  # Trigger LoRA training after N failures
SELF_CORRECT_MAX_ATTEMPTS = 2


@dataclass
class DrillResult:
    level: int
    challenges_attempted: int
    challenges_passed: int
    pass_rate: float
    grades: list
    training_triggered: bool
    level_escalated: bool
    timestamp: str


def _load_state() -> dict:
    """Load the skill loop state from disk."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "current_level": 1,
            "total_attempts": 0,
            "total_passed": 0,
            "pass_rate_by_level": {},
            "active_adapter": None,
            "last_drill": None,
            "samvega_buffer_count": 0,
            "drill_history": [],
        }


def _save_state(state: dict) -> None:
    """Persist skill loop state."""
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


CODE_GEN_MAX_RETRIES = 3

def _call_prime(prompt: str, max_tokens: int = 512,
                temperature: float = 0.1) -> str:
    """Generate code via Prime's OpenAI-compatible endpoint.

    Uses low temperature (0.1) for deterministic code output.
    Retries up to CODE_GEN_MAX_RETRIES times if the output is off-topic
    (doesn't contain 'def ' or 'class ') or contains ChatML tokens.
    """
    messages = [
        {"role": "system", "content": (
            "You are a Python code generator. Output ONLY valid Python code. "
            "No explanations, no markdown fences, no comments about your approach. "
            "Just the function definition(s) requested. "
            "Use proper indentation with 4 spaces per level."
        )},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(CODE_GEN_MAX_RETRIES):
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode()

        req = Request(
            f"{PRIME_ENDPOINT}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"]
                code = _extract_code(text)

                # Validate: must look like Python code, not hallucination
                if _is_valid_code_output(code):
                    return code

                logger.warning(
                    "Generation attempt %d/%d rejected (off-topic or malformed): %.60s...",
                    attempt + 1, CODE_GEN_MAX_RETRIES, code,
                )
                # Bump temperature slightly on retry to escape bad sampling path
                temperature = min(temperature + 0.05, 0.4)

        except Exception as e:
            logger.error("Prime generation failed (attempt %d): %s", attempt + 1, e)

    return ""


def _is_valid_code_output(code: str) -> bool:
    """Check if the generated text looks like valid Python code."""
    if not code or len(code) < 10:
        return False
    # Reject ChatML token leakage
    if "<|im_start|>" in code or "<|im_end|>" in code:
        return False
    # Must contain a function or class definition
    if "def " not in code and "class " not in code:
        return False
    # Reject single-line collapse: multi-statement code on one line
    # (the model sometimes emits code with spaces instead of newlines)
    if "\n" not in code and len(code) > 80 and code.count("    ") > 2:
        return False
    return True


def _extract_code(text: str) -> str:
    """Extract Python code from LLM output, stripping markdown fences and ChatML."""
    # Strip ChatML tokens
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    text = text.replace("<|im_start|>assistant\n", "").strip()

    # Strip ```python ... ``` fences if present
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        # Skip language tag on same line
        newline = text.index("\n", start) if "\n" in text[start:] else start
        end = text.index("```", newline) if "```" in text[newline:] else len(text)
        return text[newline:end].strip()
    return text.strip()


def _self_correct(original_code: str, error_output: str,
                  challenge_prompt: str) -> str:
    """Ask Prime to fix the code given the error."""
    correction_prompt = (
        f"The following Python code has a bug:\n\n"
        f"```python\n{original_code}\n```\n\n"
        f"Error output:\n```\n{error_output}\n```\n\n"
        f"Original task: {challenge_prompt}\n\n"
        f"Fix the code. Output ONLY the corrected Python code, no explanations."
    )
    return _call_prime(correction_prompt, max_tokens=512, temperature=0.1)


def _create_code_samvega(challenge: Challenge, original_code: str,
                         grade: CodeGrade,
                         corrected_code: Optional[str] = None) -> None:
    """Create a samvega artifact for a code evaluation failure."""
    # Weight scales with challenge level
    base_weight = 0.3 + (challenge.level * 0.1)  # L1=0.4, L5=0.8
    weight = min(base_weight, 1.0)

    artifact = {
        "artifact_type": "samvega",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger": "code_evaluation_failure",
        "challenge_id": challenge.id,
        "challenge_level": challenge.level,
        "challenge_prompt": challenge.prompt,
        "original_code": original_code,
        "error_output": grade.error_output,
        "tests_passed": grade.tests_passed,
        "tests_total": grade.tests_total,
        "corrected_code": corrected_code,
        "weight": weight,
        "promoted_to_tier5": weight >= 0.7,
        "reviewed": False,
    }

    # Save samvega artifact
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"samvega_code_{challenge.id}_{ts}.json"
    filepath = Path(SAMVEGA_DIR) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(artifact, f, indent=2)

    logger.info("Samvega artifact created: %s (weight=%.2f)", filename, weight)

    # Append to training buffer if we have a correction
    if corrected_code:
        _append_training_buffer(challenge.prompt, corrected_code)


def _append_training_buffer(instruction: str, output: str) -> None:
    """Append a training pair to the code skill buffer."""
    entry = {
        "instruction": instruction,
        "output": output,
        "metadata": {
            "source": "code_skill_loop",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    Path(TRAINING_BUFFER).parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_BUFFER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _count_training_buffer() -> int:
    """Count entries in the training buffer."""
    try:
        with open(TRAINING_BUFFER) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def _trigger_training() -> bool:
    """Trigger LoRA training via gaia-study from the accumulated buffer."""
    buffer_count = _count_training_buffer()
    if buffer_count < TRAINING_TRIGGER_COUNT:
        return False

    # Determine adapter version
    state = _load_state()
    existing = state.get("active_adapter", "")
    version = 1
    if existing and existing.startswith("code_skill_v"):
        try:
            version = int(existing.split("v")[1]) + 1
        except (IndexError, ValueError):
            pass

    adapter_name = f"code_skill_v{version}"

    payload = json.dumps({
        "adapter_name": adapter_name,
        "documents": [TRAINING_BUFFER],
        "tier": 3,
        "pillar": "cognition",
        "description": f"Self-supervised code skill adapter (v{version}, {buffer_count} examples)",
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "max_steps": min(buffer_count * 3, 200),
        "target_loss": 0.05,
        "tags": ["code", "self-supervised"],
    }).encode()

    req = Request(
        f"{STUDY_ENDPOINT}/study/start",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                logger.info("Training triggered: %s (%d examples)", adapter_name, buffer_count)
                state["active_adapter"] = adapter_name
                state["samvega_buffer_count"] = 0
                _save_state(state)
                return True
            else:
                logger.warning("Training trigger failed: %s", result)
                return False
    except Exception as e:
        logger.error("Failed to trigger training: %s", e)
        return False


def run_drill(level: Optional[int] = None, count: int = 5) -> DrillResult:
    """
    Run a coding skill drill.

    1. Pick challenges at the current (or specified) level
    2. Generate code for each via Prime
    3. Execute + grade in sandbox
    4. Self-correct on failures, create samvega artifacts
    5. Trigger training if buffer is full
    6. Escalate level if pass rate is high enough
    """
    state = _load_state()

    if level is None:
        level = state.get("current_level", 1)

    challenges = load_challenges(level=level)
    if not challenges:
        logger.warning("No challenges found for level %d", level)
        return DrillResult(
            level=level, challenges_attempted=0, challenges_passed=0,
            pass_rate=0.0, grades=[], training_triggered=False,
            level_escalated=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Limit to requested count
    if len(challenges) > count:
        import random
        challenges = random.sample(challenges, count)

    grades = []
    passed_count = 0

    for challenge in challenges:
        logger.info("Challenge %s (L%d): %s", challenge.id, challenge.level,
                     challenge.prompt[:80])

        # ── Generate code via Prime ──────────────────────────────────────
        code = _call_prime(challenge.prompt)
        if not code:
            grade = CodeGrade(
                challenge_id=challenge.id, passed=False, syntax_ok=False,
                runtime_ok=False, tests_passed=0,
                tests_total=len(challenge.test_code.splitlines()),
                error_output="Prime returned empty response",
                execution_time_ms=0, generated_code="",
            )
            grades.append(grade)
            _create_code_samvega(challenge, "", grade)
            continue

        # ── Evaluate ─────────────────────────────────────────────────────
        grade = evaluate_code(code, challenge)
        logger.info("  Result: %s (%d/%d tests)",
                     "PASS" if grade.passed else "FAIL",
                     grade.tests_passed, grade.tests_total)

        if grade.passed:
            passed_count += 1
            grades.append(grade)
            continue

        # ── Self-correct on failure ──────────────────────────────────────
        corrected_code = None
        for attempt in range(SELF_CORRECT_MAX_ATTEMPTS):
            fixed = _self_correct(code, grade.error_output, challenge.prompt)
            if not fixed:
                break

            fixed_grade = evaluate_code(fixed, challenge)
            logger.info("  Correction attempt %d: %s",
                         attempt + 1,
                         "PASS" if fixed_grade.passed else "FAIL")

            if fixed_grade.passed:
                corrected_code = fixed
                break
            else:
                code = fixed
                grade = fixed_grade

        # Create samvega artifact with the failure + optional correction
        _create_code_samvega(challenge, grade.generated_code, grade,
                            corrected_code=corrected_code)
        grades.append(grade)

    # ── Update state ─────────────────────────────────────────────────────
    pass_rate = passed_count / len(challenges) if challenges else 0.0

    state["total_attempts"] = state.get("total_attempts", 0) + len(challenges)
    state["total_passed"] = state.get("total_passed", 0) + passed_count
    state["samvega_buffer_count"] = _count_training_buffer()
    state["last_drill"] = datetime.now(timezone.utc).isoformat()

    level_key = str(level)
    state.setdefault("pass_rate_by_level", {})[level_key] = pass_rate

    # Append to drill history (keep last 50)
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "attempted": len(challenges),
        "passed": passed_count,
        "pass_rate": round(pass_rate, 3),
    }
    state.setdefault("drill_history", []).append(history_entry)
    state["drill_history"] = state["drill_history"][-50:]

    # ── Trigger training if buffer is full ───────────────────────────────
    training_triggered = False
    if _count_training_buffer() >= TRAINING_TRIGGER_COUNT:
        training_triggered = _trigger_training()

    # ── Escalate level if pass rate is high enough ───────────────────────
    level_escalated = False
    if pass_rate >= ESCALATION_PASS_RATE and level == state.get("current_level", 1):
        state["current_level"] = level + 1
        level_escalated = True
        logger.info("Level escalated: %d → %d (pass rate %.1f%%)",
                     level, level + 1, pass_rate * 100)

    _save_state(state)

    return DrillResult(
        level=level,
        challenges_attempted=len(challenges),
        challenges_passed=passed_count,
        pass_rate=pass_rate,
        grades=[grade_to_dict(g) for g in grades],
        training_triggered=training_triggered,
        level_escalated=level_escalated,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def get_skill_status() -> dict:
    """Return current skill loop status for health/dashboard."""
    state = _load_state()
    state["training_buffer_count"] = _count_training_buffer()
    return state
