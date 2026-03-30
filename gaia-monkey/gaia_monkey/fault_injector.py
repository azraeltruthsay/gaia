"""Fault Injector — picks target files and injects semantic faults."""
import logging
import os
import random
import re
from pathlib import Path

log = logging.getLogger("gaia-monkey.fault")

# ---------------------------------------------------------------------------
# Difficulty levels — auto-scaled by serenity score
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = {
    1: ["comment_assignment"],                            # Easy
    2: ["remove_import", "break_return"],                 # Medium
    3: ["remove_import", "break_return", "swap_args"],    # Hard
    4: ["multi_fault"],                                   # Expert (2-3 faults)
}


def get_difficulty_for_serenity(score: float) -> int:
    """Map serenity score to difficulty level."""
    if score >= 10.0:
        return 4
    if score >= 7.0:
        return 3
    if score >= 3.0:
        return 2
    return 1

GAIA_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/gaia/GAIA_Project"))

SERVICE_CODE_DIRS = {
    "gaia-core": GAIA_PROJECT_ROOT / "gaia-core",
    "gaia-web": GAIA_PROJECT_ROOT / "gaia-web",
    "gaia-mcp": GAIA_PROJECT_ROOT / "gaia-mcp",
    "gaia-study": GAIA_PROJECT_ROOT / "gaia-study",
    "gaia-core-candidate": GAIA_PROJECT_ROOT / "candidates" / "gaia-core",
    "gaia-mcp-candidate": GAIA_PROJECT_ROOT / "candidates" / "gaia-mcp",
}

VITAL_STEMS = {"main", "__init__", "agent_core", "tools", "prompt_builder",
               "immune_system", "cognition_packet", "discord_interface"}


def pick_target_file(service_name: str) -> Path | None:
    """Pick a non-vital, non-critical candidate file for code fault injection."""
    code_dir = SERVICE_CODE_DIRS.get(service_name)
    if not code_dir or not code_dir.exists():
        return None

    candidates = []
    for p in code_dir.rglob("*.py"):
        if "__pycache__" in str(p) or ".pytest" in str(p):
            continue
        if p.stem in VITAL_STEMS:
            continue
        try:
            if p.stat().st_size > 100:
                candidates.append(p)
        except OSError:
            continue

    return random.choice(candidates) if candidates else None


def inject_semantic_fault(content: str, difficulty: int = 1) -> tuple[str, str]:
    """Inject a semantic fault that passes ast.parse but breaks runtime behavior.

    Args:
        content: Source code to inject fault into.
        difficulty: 1-4, controls which fault types are available.

    Returns (broken_content, fault_description).
    """
    difficulty = max(1, min(4, difficulty))
    available = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS[1])
    fault_type = random.choice(available)

    # Multi-fault applies 2-3 single faults sequentially
    if fault_type == "multi_fault":
        return _inject_multi_fault(content)

    # Swap args fault
    if fault_type == "swap_args":
        result = _inject_swap_args(content)
        if result:
            return result
        # Fall through to other types if no suitable call found
        fault_type = random.choice(["remove_import", "break_return", "comment_assignment"])

    lines = content.split("\n")

    if fault_type == "remove_import":
        import_lines = [(i, l) for i, l in enumerate(lines)
                        if (l.strip().startswith("import ") or l.strip().startswith("from "))
                        and "__future__" not in l and l.strip()]
        if import_lines:
            idx, line = random.choice(import_lines)
            lines[idx] = f"# CHAOS_MONKEY_REMOVED: {line}"
            return "\n".join(lines), f"removed import at line {idx + 1}: {line.strip()}"

    if fault_type == "break_return":
        return_lines = [(i, l) for i, l in enumerate(lines)
                        if "return " in l and "return None" not in l
                        and not l.strip().startswith("#")]
        if return_lines:
            idx, line = random.choice(return_lines)
            indent = len(line) - len(line.lstrip())
            lines[idx] = " " * indent + "return None  # CHAOS_MONKEY_BREAK"
            return "\n".join(lines), f"broke return at line {idx + 1}: {line.strip()} → return None"

    if fault_type == "comment_assignment":
        assign_lines = [(i, l) for i, l in enumerate(lines)
                        if "=" in l and not l.strip().startswith("#")
                        and not l.strip().startswith("def ")
                        and not l.strip().startswith("class ")
                        and not l.strip().startswith("if ")
                        and not l.strip().startswith("for ")
                        and not l.strip().startswith("while ")
                        and "==" not in l and "!=" not in l
                        and ">=" not in l and "<=" not in l
                        and l.strip()]
        if assign_lines:
            idx, line = random.choice(assign_lines)
            lines[idx] = f"# CHAOS_MONKEY_DISABLED: {line}"
            return "\n".join(lines), f"disabled assignment at line {idx + 1}: {line.strip()}"

    # Fallback
    lines.insert(0, "_chaos_undefined_var = _this_does_not_exist  # CHAOS_MONKEY_INJECT")
    return "\n".join(lines), "injected NameError via undefined variable reference"


def _inject_swap_args(content: str) -> tuple[str, str] | None:
    """Swap two arguments in a function call. Returns None if no suitable call found."""
    # Match function calls with 2+ arguments: func(a, b, ...)
    pattern = re.compile(r'(\w+)\(([^)]+,\s*[^)]+)\)')
    lines = content.split("\n")
    candidates = []
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            continue
        for m in pattern.finditer(line):
            args = [a.strip() for a in m.group(2).split(",")]
            if len(args) >= 2 and all(a and "=" not in a for a in args[:2]):
                candidates.append((i, m, args))

    if not candidates:
        return None

    idx, match, args = random.choice(candidates)
    # Swap first two args
    args[0], args[1] = args[1], args[0]
    new_call = f"{match.group(1)}({', '.join(args)})"
    original_line = lines[idx]
    lines[idx] = lines[idx][:match.start()] + new_call + lines[idx][match.end():]
    lines[idx] += "  # CHAOS_MONKEY_SWAP"
    return "\n".join(lines), f"swapped args at line {idx + 1}: {original_line.strip()}"


def _inject_multi_fault(content: str) -> tuple[str, str]:
    """Apply 2-3 single faults sequentially (expert difficulty)."""
    num_faults = random.randint(2, 3)
    descriptions = []
    result = content
    single_types = ["remove_import", "break_return", "comment_assignment", "swap_args"]

    for _ in range(num_faults):
        fault_type = random.choice(single_types)
        if fault_type == "swap_args":
            swap_result = _inject_swap_args(result)
            if swap_result:
                result, desc = swap_result
                descriptions.append(desc)
                continue
            fault_type = "comment_assignment"

        # Use difficulty=1 to get single-fault behavior (picks from comment_assignment)
        # but override with specific type
        broken, desc = inject_semantic_fault(result, difficulty=2 if fault_type != "comment_assignment" else 1)
        if broken != result:
            result = broken
            descriptions.append(desc)

    if not descriptions:
        # Fallback — at least inject one fault
        return inject_semantic_fault(content, difficulty=2)

    return result, f"multi_fault ({len(descriptions)}): " + " | ".join(descriptions)
