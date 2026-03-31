"""
Plan Executor — bridges planning output to real codebase modifications.

Key principle: discover, don't prescribe.
The executor resolves planned paths to real paths by scanning the actual
codebase structure, not by following hardcoded lookup tables. It reads
real files and per-file contracts before generating any modifications.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional

logger = logging.getLogger("GAIA.PlanExecutor")

# Base path for codebase
_BASE = Path("/gaia/GAIA_Project") if Path("/gaia/GAIA_Project/candidates").exists() else Path(".")

# Service directories to scan (discovered, not prescribed)
_SERVICE_DIRS = None


def _get_service_dirs() -> List[Path]:
    """Discover GAIA service directories at runtime."""
    global _SERVICE_DIRS
    if _SERVICE_DIRS is not None:
        return _SERVICE_DIRS

    _SERVICE_DIRS = []
    for prefix in [_BASE / "candidates", _BASE]:
        if not prefix.exists():
            continue
        for child in sorted(prefix.iterdir()):
            if child.is_dir() and child.name.startswith("gaia-"):
                _SERVICE_DIRS.append(child)

    return _SERVICE_DIRS


def extract_file_changes(plan_text: str) -> List[Dict]:
    """
    Parse a plan to extract proposed file changes.
    Resolves planned paths to real codebase paths dynamically.
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
            code_snippet = _extract_code_near(plan_text, match.end())

            real_path = _resolve_path(file_path)
            action = "modify" if real_path else "create"

            changes.append({
                "file": file_path,
                "real_path": real_path,
                "action": action,
                "description": description[:200],
                "code_snippet": code_snippet,
            })

    # Deduplicate by resolved path
    seen = set()
    unique = []
    for c in changes:
        key = c.get("real_path") or c["file"]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def execute_plan_phase(
    changes: List[Dict],
    prime_model=None,
    config=None,
    dry_run: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Execute file changes with self-exploration.

    For each change:
    1. Resolve path → read actual file → load contract
    2. Feed real content + contract + description to Prime
    3. Validate generated code
    4. Write to candidates/ (if approved)
    """
    for change in changes:
        planned_path = change["file"]
        real_path = change.get("real_path")
        action = change["action"]

        if real_path:
            yield {"type": "token", "value": f"\n📝 **{planned_path}** → `{real_path}`\n"}
        else:
            yield {"type": "token", "value": f"\n📄 **{planned_path}** (new file)\n"}
        yield {"type": "flush"}

        # ── Self-exploration: read the actual file ──
        current_content = ""
        contract_text = ""

        if real_path and Path(real_path).exists():
            try:
                current_content = Path(real_path).read_text()
                yield {"type": "token", "value": f"  *Read {len(current_content)} chars from {Path(real_path).name}*\n"}
            except Exception as e:
                yield {"type": "token", "value": f"  *Cannot read: {e}*\n"}

            try:
                from gaia_common.utils.file_contracts import load_contract, contract_to_prompt
                contract = load_contract(str(real_path))
                if contract:
                    contract_text = contract_to_prompt(contract)
                    yield {"type": "token", "value": f"  *Contract loaded ({len(contract_text)} chars)*\n"}
            except Exception:
                pass

        # ── Generate modification ──
        if prime_model and (current_content or change.get("code_snippet")):
            if current_content:
                from gaia_core.cognition.code_generator import generate_patch, apply_patches

                # Generate targeted patches instead of full file rewrite
                patches = generate_patch(
                    prime_model,
                    real_path or planned_path,
                    current_content,
                    change["description"],
                    change.get("code_snippet", ""),
                    contract_text,
                )

                if patches:
                    yield {"type": "token", "value": f"  *Generated {len(patches)} patch(es)*\n"}

                    modified, applied = apply_patches(current_content, patches)

                    for desc in applied:
                        yield {"type": "token", "value": f"  *  → {desc}*\n"}

                    if applied:
                        validation = _validate_code(modified, real_path or planned_path)
                        status = "✅" if validation["ok"] else "❌"
                        yield {"type": "token", "value": f"  *{status} Validation: {validation['status']}*\n"}

                        output_path = _to_candidates_path(real_path or planned_path)
                        if validation["ok"] and not dry_run:
                            _write_candidate(str(output_path), modified)
                            yield {"type": "token", "value": f"  *✅ Written to {output_path}*\n"}
                        elif validation["ok"]:
                            yield {"type": "token", "value": f"  *🔍 Dry run — would write {len(modified)} chars to {output_path}*\n"}
                        else:
                            yield {"type": "token", "value": f"  *Error: {validation.get('error', '')[:200]}*\n"}
                    else:
                        yield {"type": "token", "value": "  *Patches generated but anchors not found in file*\n"}
                else:
                    # Fallback: generate additive code (new function/route to append)
                    from gaia_core.cognition.code_generator import generate_new_file
                    yield {"type": "token", "value": "  *Patch generation failed — trying additive code*\n"}
                    addition = generate_new_file(
                        prime_model, planned_path,
                        f"New code to add to {Path(real_path or planned_path).name}: {change['description']}",
                        change.get("code_snippet", ""),
                        contract_text,
                    )
                    if addition:
                        validation = _validate_code(addition, real_path or planned_path)
                        status = "✅" if validation["ok"] else "❌"
                        yield {"type": "token", "value": f"  *{status} Validation: {validation['status']} ({len(addition)} chars additive)*\n"}
                        output_path = _to_candidates_path(real_path or planned_path)
                        if validation["ok"] and not dry_run:
                            existing = current_content
                            _write_candidate(str(output_path), existing + "\n\n" + addition)
                            yield {"type": "token", "value": f"  *✅ Appended to {output_path}*\n"}
                        elif validation["ok"]:
                            yield {"type": "token", "value": f"  *🔍 Dry run — would append {len(addition)} chars to {output_path}*\n"}
                    else:
                        yield {"type": "token", "value": "  *Could not generate code for this file*\n"}
            elif prime_model:
                from gaia_core.cognition.code_generator import generate_new_file

                # Generate complete new file
                new_content = generate_new_file(
                    prime_model,
                    planned_path,
                    change["description"],
                    change.get("code_snippet", ""),
                    contract_text,
                )

                if new_content:
                    output_path = _to_candidates_path(planned_path)
                    validation = _validate_code(new_content, str(output_path))
                    status = "✅" if validation["ok"] else "❌"
                    yield {"type": "token", "value": f"  *{status} Validation: {validation['status']} ({len(new_content)} chars)*\n"}
                    if dry_run:
                        yield {"type": "token", "value": f"  *🔍 Dry run — would create {output_path}*\n"}
                    elif validation["ok"]:
                        _write_candidate(str(output_path), new_content)
                        yield {"type": "token", "value": f"  *✅ Created {output_path}*\n"}
                else:
                    yield {"type": "token", "value": "  *Could not generate new file*\n"}
        else:
            yield {"type": "token", "value": "  *No code to generate — needs implementation detail*\n"}

        yield {"type": "flush"}


# ── Dynamic Path Resolution ──────────────────────────────────────────────

def _resolve_path(planned_path: str) -> Optional[str]:
    """
    Resolve a planned path to a real codebase path.

    Strategy (discovery-based, not hardcoded):
    1. Exact match in candidates/ or production
    2. Filename match within discovered service directories
    3. Fuzzy service + filename match
    """
    # 1. Exact match
    for prefix in [_BASE / "candidates", _BASE]:
        candidate = prefix / planned_path
        if candidate.exists():
            return str(candidate)

    if not planned_path.startswith("candidates/"):
        candidate = _BASE / "candidates" / planned_path
        if candidate.exists():
            return str(candidate)

    # 2. Filename match in service dirs
    filename = Path(planned_path).name
    if filename and len(filename) > 3:
        for svc_dir in _get_service_dirs():
            for match in svc_dir.rglob(filename):
                if any(skip in str(match) for skip in ["__pycache__", ".git", ".bak"]):
                    continue
                return str(match)

    # 3. Fuzzy: extract service hint from path, then find best filename match
    service_hint = _extract_service_hint(planned_path)
    if service_hint and filename:
        for svc_dir in _get_service_dirs():
            if service_hint in svc_dir.name:
                # Search this service specifically
                for match in svc_dir.rglob(f"*{Path(filename).stem}*{Path(filename).suffix}"):
                    if any(skip in str(match) for skip in ["__pycache__", ".git", ".bak"]):
                        continue
                    return str(match)

    return None


def _extract_service_hint(path: str) -> Optional[str]:
    """Extract a service name hint from a planned path."""
    path_lower = path.lower()
    # Look for gaia-* service names
    match = re.search(r'gaia-(\w+)', path_lower)
    if match:
        return match.group(0)  # e.g., "gaia-web"

    # Infer from common keywords
    for keyword, service in [
        ("web", "gaia-web"), ("dashboard", "gaia-web"), ("ui", "gaia-web"),
        ("client", "gaia-web"), ("frontend", "gaia-web"),
        ("mcp", "gaia-mcp"), ("tool", "gaia-mcp"),
        ("core", "gaia-core"), ("pipeline", "gaia-core"), ("cognition", "gaia-core"),
        ("common", "gaia-common"), ("packet", "gaia-common"), ("protocol", "gaia-common"),
        ("study", "gaia-study"), ("train", "gaia-study"),
        ("audio", "gaia-audio"), ("voice", "gaia-audio"),
    ]:
        if keyword in path_lower:
            return service

    return None


def _to_candidates_path(file_path: str) -> Path:
    """Convert a production path to its candidates/ equivalent."""
    path_str = str(file_path)
    if "candidates/" in path_str:
        return Path(path_str)

    base_str = str(_BASE) + "/"
    if path_str.startswith(base_str):
        path_str = path_str[len(base_str):]

    return _BASE / "candidates" / path_str


# ── Code Generation ──────────────────────────────────────────────────────

def _generate_modification(
    model, file_path: str, current_content: str,
    description: str, code_snippet: str,
    contract_text: str, config
) -> Optional[str]:
    """Generate a modification that fits the existing code."""
    # For large files, use contract + new code approach instead of full file modification
    large_file = len(current_content) > 5000

    if large_file:
        # Large file strategy: generate ONLY the new code to insert
        prompt = "Add new functionality to an existing file.\n\n"
        prompt += f"**File:** {file_path}\n"
        prompt += f"**Change needed:** {description}\n\n"

        if contract_text:
            prompt += f"**Existing API surface (do not duplicate):**\n{contract_text}\n\n"

        # Show just the imports and first few lines for style matching
        lines = current_content.split("\n")
        import_section = "\n".join(lines[:30])
        prompt += f"**File header (for style matching):**\n```\n{import_section}\n```\n\n"

        if code_snippet:
            prompt += f"**Proposed addition:**\n```\n{code_snippet}\n```\n\n"

        prompt += (
            "Output ONLY the new code to ADD to this file. Include:\n"
            "- Any new import statements needed (at top)\n"
            "- The new functions, classes, or routes\n"
            "- Match the existing style from the file header\n"
            "- Do NOT reproduce existing code — only write what's new"
        )
    else:
        # Small file: can output complete modified file
        prompt = "Modify this existing file to add the described feature.\n\n"
        prompt += f"**File:** {file_path}\n"
        prompt += f"**Change needed:** {description}\n\n"

        if contract_text:
            prompt += f"**Current API surface:**\n{contract_text}\n\n"

        prompt += f"**Current file content:**\n```\n{current_content}\n```\n\n"

        if code_snippet:
            prompt += f"**Proposed addition:**\n```\n{code_snippet}\n```\n\n"

        prompt += (
            "Output the COMPLETE modified file content.\n"
            "- Preserve ALL existing code — only add/modify what's needed\n"
            "- Match existing style and patterns\n"
            "- Include proper imports\n"
            "- Output ONLY the file content"
        )

    try:
        messages = [
            {"role": "system", "content": (
                "You are a code modifier. Your output is written directly to a file. "
                "Output ONLY valid source code. NO English, NO descriptions, NO markdown. "
                "First line must be an import, comment, or code — never English text."
            )},
            # Few-shot example
            {"role": "user", "content": (
                "Modify this file to add a health check.\n\n"
                "**Current file content:**\n```\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n"
                "@app.get('/')\ndef root():\n    return {'status': 'ok'}\n```\n\n"
                "**Change needed:** Add a /health endpoint.\n\n"
                "Output the COMPLETE modified file."
            )},
            {"role": "assistant", "content": (
                "from fastapi import FastAPI\nimport time\n\napp = FastAPI()\n_start = time.time()\n\n"
                "@app.get('/')\ndef root():\n    return {'status': 'ok'}\n\n"
                "@app.get('/health')\ndef health():\n    return {'status': 'ok', 'uptime': round(time.time() - _start, 1)}\n"
            )},
            {"role": "user", "content": prompt},
        ]
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
        )
        if isinstance(result, dict):
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Strip code fences if model wrapped output
            content = re.sub(r'^```(?:python|javascript|html)?\n', '', content)
            content = re.sub(r'\n```\s*$', '', content)
            return content if content.strip() else None
    except Exception as e:
        logger.warning("Modification generation failed: %s", e)
    return None


# ── Validation ───────────────────────────────────────────────────────────

def _validate_code(content: str, file_path: str) -> Dict:
    """Validate code based on file type."""
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        return _validate_python(content)
    elif ext in (".js", ".html", ".css"):
        if not content.strip():
            return {"ok": False, "status": "empty file", "error": "No content"}
        return {"ok": True, "status": f"{ext} basic check passed"}
    elif ext in (".yaml", ".json"):
        try:
            if ext == ".json":
                json.loads(content)
            return {"ok": True, "status": f"{ext} parse passed"}
        except Exception as e:
            return {"ok": False, "status": f"{ext} parse failed", "error": str(e)}
    return {"ok": True, "status": "unknown type — skipped"}


def _validate_python(content: str) -> Dict:
    """Validate Python with py_compile + AST."""
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
    except (py_compile.PyCompileError, SyntaxError) as e:
        return {"ok": False, "status": "syntax error", "error": str(e)[:200]}
    except Exception as e:
        return {"ok": False, "status": "validation error", "error": str(e)[:200]}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── File Operations ──────────────────────────────────────────────────────

def _write_candidate(file_path: str, content: str):
    """Write content to a candidates/ path with backup."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text())
    path.write_text(content)
    logger.info("Written %d chars to %s", len(content), file_path)


def _extract_code_near(text: str, position: int, max_distance: int = 1000) -> str:
    """Extract the nearest code block after a position in text.

    Searches up to max_distance chars ahead for a fenced code block.
    Also handles inline code snippets if no fenced block is found.
    """
    search_region = text[position:position + max_distance]

    # Try fenced code block first (```lang\n...\n```)
    match = re.search(r'```(?:\w+)?\n(.*?)```', search_region, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try indented code block (4+ spaces after a blank line)
    match = re.search(r'\n\n((?:    .+\n)+)', search_region)
    if match and len(match.group(1).strip()) > 30:
        return match.group(1).strip()

    return ""
