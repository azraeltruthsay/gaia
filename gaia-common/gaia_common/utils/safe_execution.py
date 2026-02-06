from __future__ import annotations
import logging
import subprocess
from typing import Set

logger = logging.getLogger("GAIA.SafeExecution")

def run_shell_safe(command: str, safe_cmds: Set[str]) -> str:
    """Check first token against whitelist and execute."""
    parts = (command or "").strip().split()
    if not parts:
        return "❌ Shell error: Empty command."
    if parts[0] not in safe_cmds:
        return f"❌ Shell error: '{parts[0]}' not in SAFE_EXECUTE_FUNCTIONS."
    try:
        res = subprocess.run(
            command, shell=True, check=True,
            capture_output=True, text=True, timeout=10
        )
        return res.stdout.strip() or res.stderr.strip()
    except Exception as e:
        return f"❌ Shell error: {e}"
