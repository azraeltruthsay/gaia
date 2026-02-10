"""
GAIA Primitives (pillar-compliant, robust)
- Exposes: read, write, vector_query, shell (all core safe primitives)
"""

import logging
import subprocess
from pathlib import Path
# TODO: [GAIA-REFACTOR] vector_indexer.py module not yet migrated.
# from app.utils.vector_indexer import vector_query as _vector_query
import pipes
from gaia_core.utils.mcp_client import ai_execute as mcp_ai_execute

logger = logging.getLogger("GAIA.Primitives")

def read(filepath):
    try:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return f"⚠️ File not found: {filepath}"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return f"❌ Error reading file: {e}"

def write(filepath, content):
    try:
        path = Path(filepath)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "✅ Write successful."
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {e}")
        return f"❌ Error writing file: {e}"

# def vector_query(query):
#     try:
#         return _vector_query(query)
#     except Exception as e:
#         logger.error(f"Error in vector_query: {e}")
#         return f"❌ Error in vector_query: {e}"

def shell(command):
    from gaia_core.config import Config
    config = Config()
    SAFE_EXECUTE_FUNCTIONS = config.SAFE_EXECUTE_FUNCTIONS
    try:
        # Sanitize the command to prevent command injection
        command = " ".join([pipes.quote(arg) for arg in command.split()])
        cmd_name = command.split()[0] if command.strip() else ""
        if cmd_name not in SAFE_EXECUTE_FUNCTIONS:
            logger.warning(f"Rejected unsafe shell command: {command}")
            return {"stdout": "", "stderr": f"❌ Unsafe shell command: {cmd_name} is not permitted.", "returncode": -1}
        # Route execution through MCP wrapper so it can be intercepted or audited later.
        res = mcp_ai_execute(command, timeout=10, shell=True, dry_run=False)

        if not res.get("ok"):
            # Map MCP error to the expected return shape
            return {"stdout": "", "stderr": res.get("error", "MCP execution failed"), "returncode": -1}

        # MCP returns stdout/stderr/returncode when ok
        return {
            "stdout": (res.get("stdout") or "").strip(),
            "stderr": (res.get("stderr") or "").strip(),
            "returncode": int(res.get("returncode") or 0)
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Shell command timed out: {command}")
        return {"stdout": "", "stderr": "❌ Shell command timed out.", "returncode": -1}
    except Exception as e:
        logger.error(f"Shell execution error: {e}")
        return {"stdout": "", "stderr": f"❌ Shell execution error: {e}", "returncode": -1}