"""Fault Injector — picks target files and injects semantic faults."""
import logging
import os
import random
from pathlib import Path

log = logging.getLogger("gaia-monkey.fault")

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


def inject_semantic_fault(content: str) -> tuple[str, str]:
    """Inject a semantic fault that passes ast.parse but breaks runtime behavior.

    Returns (broken_content, fault_description).
    """
    lines = content.split("\n")
    fault_type = random.choice(["remove_import", "break_return", "comment_assignment"])

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
