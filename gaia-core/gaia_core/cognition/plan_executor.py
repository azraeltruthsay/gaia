"""
Plan Executor — bridges planning output to real codebase modifications.

Key principle: self-exploration before code generation.
For each proposed file change, the executor:
1. Resolves the planned path to the real codebase path
2. Reads the actual file content
3. Loads the per-file contract (API surface)
4. Feeds real content + contract + proposed change to Prime
5. Validates the generated code (py_compile + AST)
6. Writes to candidates/ with backup (if approved)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional

logger = logging.getLogger("GAIA.PlanExecutor")

# Base path for codebase — works inside container or on host
_BASE = Path("/gaia/GAIA_Project") if Path("/gaia/GAIA_Project/candidates").exists() else Path(".")


def extract_file_changes(plan_text: str) -> List[Dict]:
    """
    Parse a plan to extract proposed file changes.
    Resolves fictional paths to real codebase paths.
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

            # Resolve to real path
            real_path = _resolve_to_real_path(file_path)
            action = "modify" if real_path else "create"

            changes.append({
                "file": file_path,
                "real_path": real_path,
                "action": action,
                "description": description[:200],
                "code_snippet": code_snippet,
            })

    # Deduplicate by real_path (or file if no real_path)
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
    2. Feed real content + contract + plan description to Prime
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

            # Load per-file contract
            try:
                from gaia_common.utils.file_contracts import load_contract, contract_to_prompt
                contract = load_contract(str(real_path))
                if contract:
                    contract_text = contract_to_prompt(contract)
                    yield {"type": "token", "value": f"  *Contract loaded ({len(contract_text)} chars)*\n"}
            except Exception:
                pass

        # ── Generate modification using Prime ──
        if prime_model and (current_content or change.get("code_snippet")):
            if current_content:
                # Modify existing file
                modified = _generate_modification(
                    prime_model,
                    real_path or planned_path,
                    current_content,
                    change["description"],
                    change.get("code_snippet", ""),
                    contract_text,
                    config,
                )
                if modified:
                    validation = _validate_code(modified, real_path or planned_path)
                    status = "✅" if validation["ok"] else "❌"
                    yield {"type": "token", "value": f"  *{status} Validation: {validation['status']}*\n"}

                    if validation["ok"]:
                        # Determine candidates/ output path
                        output_path = _to_candidates_path(real_path or planned_path)
                        if dry_run:
                            yield {"type": "token", "value": f"  *🔍 Dry run — would write {len(modified)} chars to {output_path}*\n"}
                        else:
                            _write_candidate(str(output_path), modified)
                            yield {"type": "token", "value": f"  *✅ Written to {output_path}*\n"}
                    elif not validation["ok"]:
                        yield {"type": "token", "value": f"  *Error: {validation.get('error', '')[:200]}*\n"}
                else:
                    yield {"type": "token", "value": f"  *Could not generate modification*\n"}
            elif change.get("code_snippet"):
                # New file from code snippet
                output_path = _to_candidates_path(planned_path)
                validation = _validate_code(change["code_snippet"], str(output_path))
                status = "✅" if validation["ok"] else "❌"
                yield {"type": "token", "value": f"  *{status} Validation: {validation['status']}*\n"}

                if dry_run:
                    yield {"type": "token", "value": f"  *🔍 Dry run — would create {output_path}*\n"}
                elif validation["ok"]:
                    _write_candidate(str(output_path), change["code_snippet"])
                    yield {"type": "token", "value": f"  *✅ Created {output_path}*\n"}
        else:
            yield {"type": "token", "value": f"  *No code to generate — needs implementation detail*\n"}

        yield {"type": "flush"}


# ── Path Resolution ──────────────────────────────────────────────────────

# Map of common fictional path fragments to real GAIA paths
_PATH_RESOLUTION_MAP = {
    # Web
    "web/api/chat": "gaia-web/gaia_web/routes/hooks.py",
    "web/chat": "gaia-web/gaia_web/routes/hooks.py",
    "web/router": "gaia-web/gaia_web/routes/hooks.py",
    "web/upload": "gaia-web/gaia_web/routes/files.py",
    "web/ui": "gaia-web/static/app.js",
    "web/client": "gaia-web/static/app.js",
    "web/attachment": "gaia-web/static/app.js",
    "web/config": "gaia-web/gaia_web/main.py",
    "web/main": "gaia-web/gaia_web/main.py",
    "web/src/api": "gaia-web/gaia_web/routes/hooks.py",
    "web/src/component": "gaia-web/static/app.js",
    "web/src/service": "gaia-web/gaia_web/main.py",
    # Core
    "core/pipeline": "gaia-core/gaia_core/cognition/agent_core.py",
    "core/processor": "gaia-core/gaia_core/cognition/agent_core.py",
    "core/prompt": "gaia-core/gaia_core/utils/prompt_builder.py",
    "core/api": "gaia-core/gaia_core/main.py",
    "core/main": "gaia-core/gaia_core/main.py",
    "core/src/service": "gaia-core/gaia_core/cognition/agent_core.py",
    "core/src/pipeline": "gaia-core/gaia_core/cognition/agent_core.py",
    # MCP
    "mcp/tools": "gaia-mcp/gaia_mcp/tools.py",
    "mcp/file": "gaia-mcp/gaia_mcp/tools.py",
    "mcp/runner": "gaia-mcp/gaia_mcp/tools.py",
    "mcp/src/tool": "gaia-mcp/gaia_mcp/tools.py",
    # Common
    "common/packet": "gaia-common/gaia_common/protocols/cognition_packet.py",
    "shared/packet": "gaia-common/gaia_common/protocols/cognition_packet.py",
    "shared/validator": "gaia-common/gaia_common/utils/cfr_manager.py",
}


def _resolve_to_real_path(planned_path: str) -> Optional[str]:
    """
    Resolve a fictional/approximate path from the plan to a real codebase path.

    Strategy:
    1. Check if the exact path exists (in candidates/ or production)
    2. Fuzzy match against the resolution map
    3. Search by filename in the codebase
    """
    # 1. Exact match
    for prefix in [str(_BASE / "candidates"), str(_BASE)]:
        candidate = Path(prefix) / planned_path
        if candidate.exists():
            return str(candidate)

    # Also check with candidates/ prefix added
    if not planned_path.startswith("candidates/"):
        candidate = _BASE / "candidates" / planned_path
        if candidate.exists():
            return str(candidate)

    # 2. Resolution map — match longest prefix
    planned_lower = planned_path.lower().replace("\\", "/")
    best_match = None
    best_len = 0
    for pattern, real_path in _PATH_RESOLUTION_MAP.items():
        if pattern in planned_lower and len(pattern) > best_len:
            # Check both candidates/ and production
            for prefix in ["candidates", ""]:
                full = _BASE / prefix / real_path if prefix else _BASE / real_path
                if full.exists():
                    best_match = str(full)
                    best_len = len(pattern)
                    break

    if best_match:
        return best_match

    # 3. Filename search — find the filename in GAIA service directories only
    filename = Path(planned_path).name
    if filename and len(filename) > 3:
        # Only search in actual GAIA service dirs, not google-cloud-sdk/venv/archive
        gaia_dirs = [
            _BASE / "candidates" / d for d in
            ["gaia-core", "gaia-web", "gaia-mcp", "gaia-common", "gaia-study", "gaia-audio"]
        ] + [
            _BASE / d for d in
            ["gaia-core", "gaia-web", "gaia-mcp", "gaia-common", "gaia-study", "gaia-audio"]
        ]
        for search_dir in gaia_dirs:
            if not search_dir.exists():
                continue
            for match in search_dir.rglob(filename):
                if any(skip in str(match) for skip in ["__pycache__", ".git", "test_", "tests/"]):
                    continue
                return str(match)

    return None


def _to_candidates_path(file_path: str) -> Path:
    """Convert a production path to its candidates/ equivalent."""
    path_str = str(file_path)

    # Already in candidates/
    if "candidates/" in path_str:
        return Path(path_str)

    # Strip base path prefix
    base_str = str(_BASE) + "/"
    if path_str.startswith(base_str):
        path_str = path_str[len(base_str):]

    # Map production → candidates
    for service in ["gaia-core", "gaia-web", "gaia-mcp", "gaia-common", "gaia-study", "gaia-audio"]:
        if path_str.startswith(f"{service}/"):
            return _BASE / "candidates" / path_str

    return _BASE / "candidates" / path_str


# ── Code Generation ──────────────────────────────────────────────────────

def _generate_modification(
    model, file_path: str, current_content: str,
    description: str, code_snippet: str,
    contract_text: str, config
) -> Optional[str]:
    """Use Prime to generate a modification that fits the existing code."""
    # Truncate intelligently — keep imports + class/function structure
    if len(current_content) > 4000:
        # Keep first 1500 (imports, class defs) and last 1500 (recent code)
        current_content = (
            current_content[:1500]
            + "\n\n# ... (middle truncated) ...\n\n"
            + current_content[-1500:]
        )

    prompt = (
        f"Modify this existing file to add the described feature.\n\n"
        f"**File:** {file_path}\n"
        f"**Change needed:** {description}\n\n"
    )

    if contract_text:
        prompt += f"**Current API surface (contract):**\n{contract_text}\n\n"

    prompt += (
        f"**Current file content:**\n```\n{current_content}\n```\n\n"
    )

    if code_snippet:
        prompt += f"**Proposed addition:**\n```\n{code_snippet}\n```\n\n"

    prompt += (
        "Output the COMPLETE modified file content. Rules:\n"
        "- Preserve ALL existing code — only add/modify what's needed\n"
        "- Match the existing style (indentation, naming, import patterns)\n"
        "- Add the new functionality in the appropriate location\n"
        "- Include proper imports for any new code\n"
        "- Output ONLY the file content, no explanations"
    )

    try:
        messages = [
            {"role": "system", "content": (
                "You are a code modifier. Your output is written directly to a file. "
                "Output ONLY valid source code. NO English, NO descriptions, NO markdown. "
                "First line must be an import, comment, or code — never English text."
            )},
            # Few-shot: show what correct output looks like
            {"role": "user", "content": (
                "Modify this file to add a health check.\n\n"
                "**Current file content:**\n```\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n"
                "@app.get('/')\ndef root():\n    return {'status': 'ok'}\n```\n\n"
                "**Change needed:** Add a /health endpoint that returns uptime.\n\n"
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
    """Validate code with py_compile + AST (Python) or basic checks (JS/HTML)."""
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        return _validate_python(content)
    elif ext in (".js", ".html", ".css"):
        # Basic checks for non-Python
        if not content.strip():
            return {"ok": False, "status": "empty file", "error": "No content"}
        return {"ok": True, "status": f"{ext} basic check passed"}
    elif ext in (".yaml", ".json"):
        try:
            if ext == ".json":
                json.loads(content)
            else:
                import yaml
                yaml.safe_load(content)
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
    except py_compile.PyCompileError as e:
        return {"ok": False, "status": "py_compile failed", "error": str(e)[:200]}
    except SyntaxError as e:
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


def _extract_code_near(text: str, position: int, max_distance: int = 500) -> str:
    """Extract the nearest code block after a position in text."""
    search_region = text[position:position + max_distance]
    match = re.search(r'```(?:\w+)?\n(.*?)```', search_region, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
