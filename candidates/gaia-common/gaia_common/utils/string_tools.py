def trim_text(text: str, max_length: int) -> str:
    """Trims text to a maximum length, adding an ellipsis if truncated."""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text
