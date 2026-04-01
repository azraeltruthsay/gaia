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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        "Write the complete file. Include:\n"
        "- Proper imports\n"
        "- Module docstring explaining the purpose\n"
        "- Complete, working implementation\n"
        "- Match the patterns of the related files shown above"
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
    # Show enough context for the model to generate accurate patches
    if len(file_content) > 4000:
        # Large files: show structure + relevant section
        structure = _extract_structure(file_content)
        relevant = _find_relevant_section(file_content, description, code_snippet)
        if relevant:
            structure += f"\n\n# ── Relevant section (for insertion context) ──\n{relevant}"
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
        "Output one or more changes. For each change, use this format:\n\n"
        "FIND: <an exact line from the file>\n"
        "INSERT_AFTER:\n"
        "```\n"
        "<new code>\n"
        "```\n\n"
        "The FIND line MUST be copied exactly from the file above."
    )

    messages = [
        {"role": "system", "content": "You generate targeted code changes. Use FIND/INSERT_AFTER format."},
        {"role": "user", "content": (
            "Add a health endpoint.\n\n"
            "**File:**\n```\nfrom fastapi import APIRouter\nrouter = APIRouter()\n\n"
            "@router.get('/status')\ndef status():\n    return {'ok': True}\n```\n"
        )},
        {"role": "assistant", "content": (
            "FIND: return {'ok': True}\n"
            "INSERT_AFTER:\n```\n\n\n@router.get('/health')\ndef health():\n    return {'status': 'healthy'}\n```"
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        result = model.create_chat_completion(messages=messages, max_tokens=1500, temperature=0.1)
        if isinstance(result, dict):
            raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if raw:
                patches = _parse_patches(raw)
                if patches:
                    logger.info("Patch gen: %d patches for %s", len(patches), file_path)
                    for p in patches:
                        logger.info("  %s at: %s (%d chars code)", p["action"], p["anchor"][:60], len(p["code"]))
                else:
                    logger.warning("Patch gen: no parseable patches for %s. Raw: %s", file_path, raw[:200])
                return patches
            else:
                logger.warning("Patch gen: empty response for %s", file_path)
    except Exception as e:
        logger.warning("Patch generation failed for %s: %s", file_path, e)
    return None


def apply_patches(file_content: str, patches: List[Dict], file_path: str = "") -> Tuple[str, List[str]]:
    """
    Apply patches to file content.

    Returns (modified_content, list_of_applied_descriptions).
    """
    lines = file_content.split("\n")
    applied = []
    ext = Path(file_path).suffix.lower() if file_path else ".py"

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

        # Detect indentation of the anchor line and apply to new code
        anchor_line = lines[anchor_idx]
        anchor_indent = len(anchor_line) - len(anchor_line.lstrip())
        indent_str = anchor_line[:anchor_indent]

        new_lines = code.split("\n")

        # Smart indentation: if inserting inside a class/function body, indent one level deeper
        if action in ("insert_after", "insert_before"):
            anchor_stripped = anchor_line.strip()
            needs_body_indent = anchor_stripped.endswith(":") or anchor_stripped.startswith("class ") or anchor_stripped.startswith("def ")
            if needs_body_indent:
                body_indent = indent_str + "    "
                new_lines = [body_indent + line if line.strip() else line for line in new_lines]
            elif new_lines and not new_lines[0].startswith(" ") and indent_str:
                new_lines = [indent_str + line if line.strip() else line for line in new_lines]

        # Apply the patch to a copy first, validate, then commit
        test_lines = lines.copy()
        if action == "insert_after":
            test_lines = test_lines[:anchor_idx + 1] + new_lines + test_lines[anchor_idx + 1:]
        elif action == "insert_before":
            test_lines = test_lines[:anchor_idx] + new_lines + test_lines[anchor_idx:]
        elif action == "replace":
            test_lines = test_lines[:anchor_idx] + new_lines + test_lines[anchor_idx + 1:]

        # Quick validation: does the patched file still parse?
        test_content = "\n".join(test_lines)
        if ext == ".py":
            try:
                import ast
                ast.parse(test_content)
            except SyntaxError as e:
                logger.warning("Patch failed validation (skipping): %s — %s", anchor[:50], e)
                applied.append(f"⚠️ Skipped (syntax error): {anchor[:60]}")
                continue

        # Patch validated — commit it
        lines = test_lines
        if action == "insert_after":
            applied.append(f"Inserted {len(new_lines)} lines after: {anchor[:60]}")
        elif action == "insert_before":
            applied.append(f"Inserted {len(new_lines)} lines before: {anchor[:60]}")
        elif action == "replace":
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


def _find_relevant_section(content: str, description: str, code_snippet: str = "", context_lines: int = 30) -> str:
    """
    Find the section of a file most relevant to the planned change.

    Searches for keywords from the description in the file content,
    then extracts surrounding context. This gives the model a focused
    view of WHERE in the file the change should go.
    """
    lines = content.split("\n")
    if not lines:
        return ""

    # Extract keywords from description
    desc_words = set(description.lower().split()) - {
        "a", "the", "to", "for", "and", "in", "of", "add", "new", "with",
        "function", "method", "class", "file", "update", "create",
    }

    # Score each line by keyword hits
    scored = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for w in desc_words if len(w) > 3 and w in line_lower)
        if score > 0:
            scored.append((score, i))

    if not scored:
        # Fallback: show the last function/class definition area (likely insertion point)
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith(("def ", "async def ", "class ", "@router.", "@app.")):
                start = max(0, i - 5)
                end = min(len(lines), i + context_lines)
                return "\n".join(lines[start:end])
        return ""

    # Take the highest-scoring region
    scored.sort(key=lambda x: -x[0])
    best_line = scored[0][1]
    start = max(0, best_line - 10)
    end = min(len(lines), best_line + context_lines)
    return "\n".join(lines[start:end])


def _parse_patches(raw: str) -> List[Dict]:
    """Parse the model's patch output — supports FIND/INSERT_AFTER and PATCH formats."""
    patches = []

    # Try FIND/INSERT_AFTER format first (simpler, more reliable)
    find_matches = re.finditer(
        r'FIND:\s*(.+?)\n\s*INSERT_AFTER:\s*\n```(?:\w+)?\n(.*?)```',
        raw, re.DOTALL
    )
    for match in find_matches:
        anchor = match.group(1).strip()
        code = match.group(2).strip()
        code = re.sub(r'^```\w*\s*$', '', code, flags=re.MULTILINE).strip()
        if anchor and code:
            patches.append({"action": "insert_after", "anchor": anchor, "code": code})

    if patches:
        return patches

    # Fallback: try PATCH format
    patch_sections = re.split(r'PATCH\s+\d+:', raw)
    for section in patch_sections:
        if not section.strip():
            continue

        action_match = re.search(r'ACTION:\s*(insert_after|insert_before|replace)', section, re.IGNORECASE)
        anchor_match = re.search(r'ANCHOR:\s*(.+?)(?:\n|CODE:)', section, re.DOTALL)
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', section, re.DOTALL)

        if not code_match:
            code_match = re.search(r'CODE:\s*\n((?:(?!PATCH).)+)', section, re.DOTALL)

        if action_match and anchor_match and code_match:
            code = code_match.group(1).strip()
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
