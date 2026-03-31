"""
Code Generator — targeted code generation for GAIA's planning pipeline.

Two modes:
1. New file: Generate a complete module from scratch
2. Existing file: Generate targeted patches (insertions/replacements)

The key insight: for existing files, don't ask the model to reproduce
the entire file. Instead, ask it to generate ONLY the new/changed code
with clear insertion points. This is how skilled developers work —
they read the file, identify where the change goes, and write only
the delta.

For new files, the model writes the complete module with proper imports,
class structure, and documentation — informed by the contracts of
related files so it fits the existing architecture.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("GAIA.CodeGenerator")


def generate_new_file(
    model,
    file_path: str,
    description: str,
    code_snippet: str = "",
    related_contracts: str = "",
) -> Optional[str]:
    """
    Generate a complete new file.

    Args:
        model: The LLM to use for generation
        file_path: Target path (determines language/framework)
        description: What the file should do
        code_snippet: Any starter code from the plan
        related_contracts: Contracts of related files for context
    """
    ext = Path(file_path).suffix.lower()
    lang_hint = {".py": "Python", ".js": "JavaScript", ".html": "HTML",
                 ".css": "CSS", ".yaml": "YAML"}.get(ext, "")

    prompt = f"Create a new {lang_hint} file: {file_path}\n\n"
    prompt += f"**Purpose:** {description}\n\n"

    if related_contracts:
        prompt += f"**Related files (for compatibility):**\n{related_contracts}\n\n"

    if code_snippet:
        prompt += f"**Starter code from plan:**\n```\n{code_snippet}\n```\n\n"

    prompt += (
        f"Write the complete file. Include:\n"
        f"- Proper imports\n"
        f"- Module docstring explaining the purpose\n"
        f"- Complete, working implementation\n"
        f"- Match the patterns of the related files shown above"
    )

    return _call_model(model, prompt)


def generate_patch(
    model,
    file_path: str,
    file_content: str,
    description: str,
    code_snippet: str = "",
    contract_text: str = "",
) -> Optional[List[Dict]]:
    """
    Generate targeted patches for an existing file.

    Returns a list of patches:
        [{"action": "insert_after"|"insert_before"|"replace",
          "anchor": "line or pattern to find",
          "code": "new code to insert/replace with"}]

    This approach avoids reproducing the entire file — the model only
    generates the delta, which is much more reliable.
    """
    # Show enough of the file for the model to identify insertion points
    # For large files, show the structure (imports, class/function signatures)
    if len(file_content) > 3000:
        structure = _extract_structure(file_content)
    else:
        structure = file_content

    prompt = (
        f"I need to modify an existing file. Generate TARGETED PATCHES — "
        f"do NOT reproduce the entire file.\n\n"
        f"**File:** {file_path}\n"
        f"**Change needed:** {description}\n\n"
    )

    if contract_text:
        prompt += f"**Current API:**\n{contract_text}\n\n"

    prompt += f"**File structure:**\n```\n{structure}\n```\n\n"

    if code_snippet:
        prompt += f"**Suggested code:**\n```\n{code_snippet}\n```\n\n"

    prompt += (
        "Generate patches in this exact format (one or more):\n\n"
        "PATCH 1:\n"
        "ACTION: insert_after\n"
        "ANCHOR: <exact line from the file to insert after>\n"
        "CODE:\n"
        "```\n"
        "<new code to insert>\n"
        "```\n\n"
        "Valid actions: insert_after, insert_before, replace\n"
        "For replace, ANCHOR is the line(s) to replace.\n"
        "For insert_after/insert_before, ANCHOR is the reference line.\n"
        "The ANCHOR must be an EXACT line from the file structure above."
    )

    raw = _call_model(model, prompt, max_tokens=1500)
    if not raw:
        return None

    return _parse_patches(raw)


def apply_patches(file_content: str, patches: List[Dict]) -> Tuple[str, List[str]]:
    """
    Apply patches to file content.

    Returns (modified_content, list_of_applied_descriptions).
    """
    lines = file_content.split("\n")
    applied = []

    for patch in patches:
        action = patch.get("action", "")
        anchor = patch.get("anchor", "").strip()
        code = patch.get("code", "")

        if not anchor or not code:
            continue

        # Find the anchor line
        anchor_idx = None
        for i, line in enumerate(lines):
            if anchor in line.strip():
                anchor_idx = i
                break

        if anchor_idx is None:
            logger.warning("Patch anchor not found: %s", anchor[:80])
            continue

        new_lines = code.split("\n")

        if action == "insert_after":
            lines = lines[:anchor_idx + 1] + new_lines + lines[anchor_idx + 1:]
            applied.append(f"Inserted {len(new_lines)} lines after: {anchor[:60]}")

        elif action == "insert_before":
            lines = lines[:anchor_idx] + new_lines + lines[anchor_idx:]
            applied.append(f"Inserted {len(new_lines)} lines before: {anchor[:60]}")

        elif action == "replace":
            # Replace the anchor line with the new code
            lines = lines[:anchor_idx] + new_lines + lines[anchor_idx + 1:]
            applied.append(f"Replaced line with {len(new_lines)} lines at: {anchor[:60]}")

    return "\n".join(lines), applied


# ── Internal ─────────────────────────────────────────────────────────────

def _extract_structure(content: str) -> str:
    """Extract file structure — imports, class/function signatures, key lines."""
    lines = content.split("\n")
    structure_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Always include these
        if any([
            stripped.startswith("import "),
            stripped.startswith("from "),
            stripped.startswith("class "),
            stripped.startswith("def "),
            stripped.startswith("async def "),
            stripped.startswith("@app."),
            stripped.startswith("@router."),
            stripped.startswith("# ──"),        # Section dividers
            stripped.startswith('"""') and i < 5,  # Module docstring
            stripped.startswith("TOOL_REGISTRY"),
            stripped.startswith("TOOL_METADATA"),
            not stripped,                       # Blank lines (preserve structure)
        ]):
            structure_lines.append(line)

        # Include first 20 lines always (header section)
        elif i < 20:
            structure_lines.append(line)

    # Cap at reasonable size
    result = "\n".join(structure_lines[:150])

    # Add line count note
    result += f"\n\n# ... ({len(lines)} total lines)"
    return result


def _parse_patches(raw: str) -> List[Dict]:
    """Parse the model's patch output into structured patches."""
    patches = []

    # Split on PATCH markers
    patch_sections = re.split(r'PATCH\s+\d+:', raw)

    for section in patch_sections:
        if not section.strip():
            continue

        action_match = re.search(r'ACTION:\s*(insert_after|insert_before|replace)', section, re.IGNORECASE)
        anchor_match = re.search(r'ANCHOR:\s*(.+?)(?:\n|CODE:)', section, re.DOTALL)
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', section, re.DOTALL)

        if not code_match:
            # Try without fences
            code_match = re.search(r'CODE:\s*\n((?:(?!PATCH).)+)', section, re.DOTALL)

        if action_match and anchor_match and code_match:
            code = code_match.group(1).strip()
            # Strip any remaining markdown artifacts
            code = re.sub(r'^```\w*\s*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'^\s*```\s*$', '', code, flags=re.MULTILINE)
            code = code.strip()
            patches.append({
                "action": action_match.group(1).lower().strip(),
                "anchor": anchor_match.group(1).strip(),
                "code": code,
            })

    return patches


def _call_model(model, prompt: str, max_tokens: int = 2048) -> Optional[str]:
    """Call the model with the code generation system prompt."""
    try:
        messages = [
            {"role": "system", "content": (
                "You are a precise code generator. Follow the requested format exactly. "
                "Output only what is asked — no explanations, no commentary. "
                "Match existing code patterns and style."
            )},
            {"role": "user", "content": prompt},
        ]
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        if isinstance(result, dict):
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Strip markdown fences from complete-file outputs
            content = re.sub(r'^```(?:python|javascript|html)?\n', '', content)
            content = re.sub(r'\n```\s*$', '', content)
            return content if content.strip() else None
    except Exception as e:
        logger.warning("Code generation failed: %s", e)
    return None
