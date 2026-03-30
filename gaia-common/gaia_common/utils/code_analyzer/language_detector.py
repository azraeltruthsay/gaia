import os

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".sh": "bash",
    ".html": "html",
    ".css": "css"
}

def detect_language(file_path: str, code: str = None) -> str:
    """
    Detect programming language based on file extension.
    Optionally validate with code content in future.

    Args:
        file_path (str): Path to file
        code (str): File contents (optional)

    Returns:
        str: Detected language
    """
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "unknown")
