"""
Generic helper utilities for GAIA services.

This module provides common utility functions used across services:
- File system helpers
- Timestamp generation
- Path utilities
"""

import os
import datetime
import logging
import json
from typing import Optional

logger = logging.getLogger("gaia_common.helpers")


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects gracefully."""
    def default(self, obj):
        # Handle common non-serializable types
        try:
            # Try standard serialization first
            return super().default(obj)
        except TypeError:
            # Return a safe string representation
            return f"<non-serializable: {type(obj).__name__}>"


def safe_mkdir(path: str) -> bool:
    """
    Create a directory if it doesn't already exist.

    Args:
        path: Directory path to create

    Returns:
        True if directory exists or was created, False on error
    """
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
        return False


def get_timestamp(compact: bool = False, utc: bool = True) -> str:
    """
    Return a timestamp string.

    Args:
        compact: If True, use YYYYMMDD_HHMMSS format. Otherwise ISO 8601.
        utc: If True, use UTC time. Otherwise use local time.

    Returns:
        Timestamp string
    """
    now = datetime.datetime.utcnow() if utc else datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S") if compact else now.isoformat()


def get_timestamp_for_filename() -> str:
    """
    Return a timestamp suitable for use in filenames.

    Returns:
        Timestamp string in YYYYMMDD_HHMMSS format
    """
    return get_timestamp(compact=True, utc=True)


def ensure_parent_dir(file_path: str) -> bool:
    """
    Ensure the parent directory of a file path exists.

    Args:
        file_path: Path to a file

    Returns:
        True if parent directory exists or was created
    """
    parent = os.path.dirname(file_path)
    if parent:
        return safe_mkdir(parent)
    return True


def normalize_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Normalize a path, optionally relative to a base directory.

    Args:
        path: Path to normalize
        base_dir: Optional base directory to resolve relative paths

    Returns:
        Normalized absolute path
    """
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(os.path.abspath(path))


__all__ = [
    "safe_mkdir",
    "get_timestamp",
    "get_timestamp_for_filename",
    "ensure_parent_dir",
    "normalize_path",
    "SafeJSONEncoder",
]
