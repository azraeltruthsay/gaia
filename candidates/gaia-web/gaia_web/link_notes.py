"""What she can't see in a message, said plainly beside it.

A GIF link reaches her as its address only; Discord shows people the image.
Asked about one, she guessed its content from the words in the URL ("a cute
one about AI babies", 2026-11-19). The note lets her say what she sees.
"""
import re

GIF_NOTE = "[GIF link: only the address came through, not the image.]"
_GIF_RE = re.compile(
    r"https?://\S*(?:tenor\.com|giphy\.com|klipy\.com|gfycat\.com|/gifs?/|\.gif(?:\b|$))\S*", re.I)


def has_gif_link(text: str) -> bool:
    return bool(_GIF_RE.search(text or ""))


def with_link_notes(text: str) -> str:
    """The message with a note for each kind of link she only sees the address of."""
    if has_gif_link(text) and GIF_NOTE not in text:
        return f"{text}\n{GIF_NOTE}"
    return text
