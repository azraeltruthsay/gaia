"""
file_loader.py

Handles loading code files and checking for text files.
"""

import logging
import os

logger = logging.getLogger("GAIA.FileLoader")

def load_file_safely(path: str) -> str:
    """
    Safely read file contents as text, skipping binaries or unreadable files.

    Args:
        path (str): Absolute file path

    Returns:
        str: File content or empty string
    """
    try:
        if not os.path.exists(path):
            logger.warning(f"❗ File not found: {path}")
            return ""

        if os.path.isdir(path):
            logger.debug(f"🚫 Skipping directory: {path}")
            return ""

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    except UnicodeDecodeError:
        logger.warning(f"⚠️ Skipping binary or non-UTF-8 file: {path}")
        return ""

    except Exception as e:
        logger.error(f"❌ Error reading file {path}: {e}", exc_info=True)
        return ""
