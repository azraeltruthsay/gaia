import os
from typing import List
import logging

logger = logging.getLogger("GAIA.FileScanner")

EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "__init__.py"}
EXCLUDED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mp4", ".exe", ".zip", ".db"}

def scan_code_directory(root: str) -> List[str]:
    """
    Recursively scan a directory and return all code file paths.

    Args:
        root (str): Root directory to scan

    Returns:
        List[str]: Relative paths to candidate code files
    """
    code_files = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in EXCLUDED_EXTENSIONS:
                continue

            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)
            code_files.append(rel_path)

    logger.info(f"🗂️ Scanned {len(code_files)} code files from {root}")
    return code_files
