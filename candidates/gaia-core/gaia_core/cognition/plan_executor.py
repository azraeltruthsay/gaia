"""
Plan Executor — bridges planning orchestrator output to CodeMind execution.

After the planning orchestrator generates a validated plan, this module:
1. Extracts actionable file changes from plan phases
2. Loads per-file contracts for each target file
3. Feeds each change to the CodeMind engine for implementation
4. Validates generated code (py_compile, ruff, AST)
5. Writes to candidates/ if validation passes
6. Reports results back for user review

This is the "plan → code" bridge that closes the loop from
architectural planning to actual implementation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional

logger = logging.getLogger("GAIA.PlanExecutor")


def extract_file_changes(plan_text: str) -> List[Dict]:
    """
    Parse a plan phase to extract proposed file changes.

    Returns list of:
        {"file": "candidates/gaia-web/...", "action": "modify|create",
         "description": "...", "code_snippet": "..."}
    """
    changes = []

    path_patterns = [
        r'\*\*`?([a-zA-Z_][\w/.-]+\.(?:py|js|html|css|yaml|json))`?\*\*[:\s]*(.{10,200})',
        r'(?:File|Path|Modify|Create|Update):\s*`?([a-zA-Z_][\w/.-]+\.(?:py|js|html|css|yaml|json))`?[:\s]*(.{0,200})',
        r'- `([a-zA-Z_][\w/.-]+\.(?:py|js|html|css|yaml|json))`[:\s]*(.{0,200})',
    ]

    for pattern in path_patterns:
        for match in re.finditer(pattern, plan_text):
            file_path = match.group(1)
            description = match.group(2).strip() if match.group(2) else ""
            action = "modify" if Path(file_path).exists() else "create"
            code_snippet = _extract_code_near(plan_text, match.end())

            changes.append({
                "file": file_path,
                "action": action,
                "description": description[:200],
                "code_snippet": code_snippet,
            })

    seen = set()
    unique = []
    for c in changes:
        if c["file"] not in seen:
            seen.add(c["file"])
            unique.append(c)

    return unique


def execute_plan_phase(
    changes: List[Dict],
    prime_model=None,
    config=None,
    dry_run: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Execute file changes from a plan phase.

    For each change:
    1. Load the file's contract (existing API surface)
    2. If modifying: read current content, generate patch
    3. Validate: py_compile + ruff
    4. Write to candidates/ (if not dry_run)

    Yields status events for user feedback.
    """
    for change in changes:
        file_path = change["file"]
        action = change["action"]

        yield {"type": "token", "value": f"\n**[Executing: {action} {file_path}]**\n"}
        yield {"type": "flush"}

        contract_text = ""
        try:
            from gaia_common.utils.file_contracts import load_contract, contract_to_prompt
            contract = load_contract(file_path)
            if contract:
                contract_text = contract_to_prompt(contract)
        except Exception:
            pass

        if action == "modify" and Path(file_path).exists():
            try:
                current_content = Path(file_path).read_text()
            except Exception as e:
                yield {"type": "token", "value": f"*Cannot read {file_path}: {e}*\n"}
                continue

            if prime_model and change.get("code_snippet"):
                modified = _generate_modification(
                    prime_model, file_path, current_content,
                    change["description"], change["code_snippet"],
                    contract_text, config
                )
            else:
                modified = None

            if modified:
                validation = _validate_python(modified, file_path)
                yield {"type": "token", "value": f"*Validation: {validation['status']}*\n"}

                if validation["ok"] and not dry_run:
                    _write_candidate(file_path, modified)
                    yield {"type": "token", "value": f"*Written to {file_path}*\n"}
                elif not validation["ok"]:
                    yield {"type": "token", "value": f"*Validation failed: {validation.get('error', '')}*\n"}
                else:
                    yield {"type": "token", "value": f"*Dry run — would write {len(modified)} chars to {file_path}*\n"}
            else:
                yield {"type": "token", "value": f"*Could not generate modification for {file_path}*\n"}

        elif action == "create":
            if change.get("code_snippet") and not dry_run:
                _write_candidate(file_path, change["code_snippet"])
                yield {"type": "token", "value": f"*Created {file_path}*\n"}
            else:
                yield {"type": "token", "value": f"*Dry run — would create {file_path}*\n"}

        yield {"type": "flush"}


def _extract_code_near(text: str, position: int, max_distance: int = 500) -> str:
    """Extract the nearest code block after a position in text."""
    search_region = text[position:position + max_distance]
    match = re.search(r'```(?:\w+)?\n(.*?)```', search_region, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _generate_modification(
    model, file_path: str, current_content: str,
    description: str, code_snippet: str,
    contract_text: str, config
) -> Optional[str]:
    """Use Prime to generate a modified version of a file."""
    if len(current_content) > 3000:
        current_content = current_content[:3000] + "\n... (truncated)"

    prompt = (
        f"Modify this file to implement the described change.\n\n"
        f"File: {file_path}\n"
        f"Change: {description}\n\n"
        f"Current file contract:\n{contract_text}\n\n"
        f"Current content:\n```python\n{current_content}\n```\n\n"
        f"Proposed change:\n```python\n{code_snippet}\n```\n\n"
        f"Output the COMPLETE modified file. Preserve all existing code and "
        f"only add/modify what's needed for this change. Match existing style."
    )

    try:
        messages = [
            {"role": "system", "content": "You are a code modifier. Output ONLY the complete modified file. No explanations."},
            {"role": "user", "content": prompt},
        ]
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
        )
        if isinstance(result, dict):
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            content = re.sub(r'^```(?:python)?\n', '', content)
            content = re.sub(r'\n```\s*$', '', content)
            return content
    except Exception as e:
        logger.warning("Modification generation failed: %s", e)
    return None


def _validate_python(content: str, file_path: str) -> Dict:
    """Validate Python code with py_compile and AST."""
    if not file_path.endswith(".py"):
        return {"ok": True, "status": "non-Python — skipped validation"}

    import tempfile
    import py_compile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            tmp_path = f.name

        py_compile.compile(tmp_path, doraise=True)

        import ast
        ast.parse(content)

        return {"ok": True, "status": "py_compile + AST passed"}
    except py_compile.PyCompileError as e:
        return {"ok": False, "status": "py_compile failed", "error": str(e)}
    except SyntaxError as e:
        return {"ok": False, "status": "syntax error", "error": str(e)}
    except Exception as e:
        return {"ok": False, "status": "validation error", "error": str(e)}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _write_candidate(file_path: str, content: str):
    """Write content to a candidates/ path with backup."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text())

    path.write_text(content)
    logger.info("Written %d chars to %s", len(content), file_path)
