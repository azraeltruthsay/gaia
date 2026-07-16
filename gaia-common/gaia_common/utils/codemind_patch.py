"""SEARCH/REPLACE patch format for CodeMind fixes (s4r2).

Whole-file regeneration by a small model is maximal output tokens and
every untouched line is a mangling opportunity. Patch-mode asks the
model for surgical blocks instead:

    <<<<<<< SEARCH
    exact lines from the current file
    =======
    replacement lines
    >>>>>>> REPLACE

Discipline (borrowed from the Engineer's Edit tool): the SEARCH text
must match the file EXACTLY and UNIQUELY. Ambiguity or a non-match is
an error, never a guess — the caller falls back to the legacy
whole-file prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_BLOCK_RE = re.compile(
    r"<{7} SEARCH\n(.*?)\n?={7}\n(.*?)\n?>{7} REPLACE",
    re.DOTALL,
)


@dataclass(frozen=True)
class PatchBlock:
    search: str
    replace: str


def parse_patch_blocks(text: str) -> Tuple[List[PatchBlock], Optional[str]]:
    """Extract SEARCH/REPLACE blocks from an LLM response.

    Returns (blocks, error). error is set when the response contains no
    well-formed blocks or contains marker lines outside any block
    (a sign the model mangled the grammar).
    """
    if not text or not text.strip():
        return [], "empty response"

    blocks = [PatchBlock(search=m.group(1), replace=m.group(2))
              for m in _BLOCK_RE.finditer(text)]
    if not blocks:
        return [], "no SEARCH/REPLACE blocks found in response"

    # Marker hygiene: any marker lines left over after removing the
    # well-formed blocks mean the grammar was mangled.
    stripped = _BLOCK_RE.sub("", text)
    for marker in ("<<<<<<<", "=======", ">>>>>>>"):
        if marker in stripped:
            return [], f"stray {marker!r} marker outside a well-formed block"

    for i, b in enumerate(blocks):
        if not b.search.strip():
            return [], f"block {i + 1}: empty SEARCH section"
        if b.search == b.replace:
            return [], f"block {i + 1}: SEARCH and REPLACE are identical"
    return blocks, None


def apply_patch(original: str, blocks: List[PatchBlock]) -> Tuple[Optional[str], Optional[str]]:
    """Apply blocks to original content. Returns (new_content, error).

    Each SEARCH must occur exactly once in the (progressively patched)
    content. Zero matches or more than one match aborts the whole patch.
    """
    if not blocks:
        return None, "no blocks to apply"
    content = original
    for i, b in enumerate(blocks):
        count = content.count(b.search)
        if count == 0:
            return None, f"block {i + 1}: SEARCH text not found in file"
        if count > 1:
            return None, f"block {i + 1}: SEARCH text matches {count} locations (must be unique)"
        content = content.replace(b.search, b.replace, 1)
    if content == original:
        return None, "patch produced no change"
    return content, None
