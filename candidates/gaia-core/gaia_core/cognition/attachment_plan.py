"""
Attachment Implementation — discover what exists, generate what's missing.

Principle: explore before generating. Before writing any code, survey
the actual codebase to find what already exists for attachment handling.
Then generate only the delta — what's missing.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Generator

logger = logging.getLogger("GAIA.AttachmentPlan")

_BASE = Path("/gaia/GAIA_Project") if Path("/gaia/GAIA_Project/candidates").exists() else Path(".")


def generate_attachment_code(
    prime_model,
    reviewer_model=None,
    dry_run: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Discover what exists, then generate what's missing for attachment support.
    """
    from gaia_common.utils.file_contracts import load_contract, contract_to_prompt

    yield {"type": "token", "value": "**[Attachment Implementation]**\n\n"}
    yield {"type": "token", "value": "**[Phase 0: Discovery — what already exists?]**\n"}
    yield {"type": "flush"}

    # ── Discovery: scan the actual codebase ──
    discoveries = _discover_existing()
    for component, status in discoveries.items():
        icon = "✅" if status["exists"] else "❌"
        yield {"type": "token", "value": f"  {icon} **{component}**: {status['summary']}\n"}
    yield {"type": "token", "value": "\n"}
    yield {"type": "flush"}

    # ── Determine what's missing ──
    changes_needed = _plan_changes(discoveries)
    if not changes_needed:
        yield {"type": "token", "value": "*Everything already exists! No changes needed.*\n"}
        yield {"type": "flush"}
        return

    yield {"type": "token", "value": f"**[{len(changes_needed)} changes needed]**\n\n"}
    yield {"type": "flush"}

    results = {"passed": 0, "failed": 0, "total": len(changes_needed)}

    # ── Generate code for each missing piece ──
    for change in changes_needed:
        file_path = str(_BASE / change["file"])
        short_path = change["file"]
        ext = Path(file_path).suffix.lower()

        yield {"type": "token", "value": f"📝 **{short_path}**: {change['summary']}\n"}
        yield {"type": "flush"}

        if not Path(file_path).exists():
            yield {"type": "token", "value": "  *File not found — skipping*\n"}
            results["failed"] += 1
            continue

        content = Path(file_path).read_text()

        # Load contract
        contract_text = ""
        try:
            contract = load_contract(file_path)
            if contract:
                contract_text = contract_to_prompt(contract)
        except Exception:
            pass

        # Find the example pattern
        example = ""
        if change.get("example_pattern"):
            example = _extract_example(content, change["example_pattern"])

        # Generate the change
        if ext == ".py":
            success = yield from _generate_python_change(
                prime_model, file_path, content, change,
                contract_text, example, dry_run
            )
        elif ext == ".js":
            success = yield from _generate_js_change(
                prime_model, content, change, example, file_path, dry_run
            )
        else:
            yield {"type": "token", "value": f"  *Unsupported file type: {ext}*\n"}
            success = False

        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1

        yield {"type": "flush"}

    # ── Summary + Approval ──
    yield {"type": "token", "value": f"\n**[Result: {results['passed']}/{results['total']} changes ready]**\n"}
    if results["passed"] > 0:
        try:
            from gaia_common.utils.approval_challenge import create_challenge, format_challenge_prompt
            challenge = create_challenge(
                action="write_attachment_code",
                context=f"{results['passed']} file(s) validated, {'dry run' if dry_run else 'written'}",
            )
            yield {"type": "token", "value": "\n" + format_challenge_prompt(challenge) + "\n"}
        except Exception:
            pass
    yield {"type": "flush"}


# ── Discovery ────────────────────────────────────────────────────────────

def _discover_existing() -> Dict[str, Dict]:
    """Scan the codebase to find what already exists for attachments."""
    results = {}

    # 1. Attachment dataclass
    pkt_path = _BASE / "candidates/gaia-common/gaia_common/protocols/cognition_packet.py"
    if pkt_path.exists():
        pkt = pkt_path.read_text()
        has_class = "class Attachment" in pkt
        fields = re.findall(r'(\w+):\s*(?:Optional\[)?(\w+)', pkt[pkt.find("class Attachment"):pkt.find("class Attachment")+300]) if has_class else []
        field_names = [f[0] for f in fields]
        results["Attachment dataclass"] = {
            "exists": has_class,
            "summary": f"Fields: {', '.join(field_names)}" if has_class else "Missing",
            "fields": field_names,
            "file": "candidates/gaia-common/gaia_common/protocols/cognition_packet.py",
        }

    # 2. Upload endpoint
    files_path = _BASE / "candidates/gaia-web/gaia_web/routes/files.py"
    if files_path.exists():
        files = files_path.read_text()
        has_upload = "/attachments/upload" in files or "/upload" in files
        existing_routes = re.findall(r'@router\.\w+\("([^"]+)"', files)
        results["Upload endpoint"] = {
            "exists": has_upload,
            "summary": "Has /attachments/upload" if has_upload else f"Missing (existing routes: {', '.join(existing_routes)})",
            "file": "candidates/gaia-web/gaia_web/routes/files.py",
        }

    # 3. read_attachment MCP tool
    tools_path = _BASE / "candidates/gaia-mcp/gaia_mcp/tools.py"
    if tools_path.exists():
        tools = tools_path.read_text()
        has_tool = "read_attachment" in tools
        results["read_attachment tool"] = {
            "exists": has_tool,
            "summary": "Registered in tool registry" if has_tool else "Missing",
            "file": "candidates/gaia-mcp/gaia_mcp/tools.py",
        }

    # 4. Frontend upload UI
    app_path = _BASE / "candidates/gaia-web/static/app.js"
    if app_path.exists():
        app = app_path.read_text()
        has_upload = "handleFileSelect" in app or "uploadAttachment" in app or "pendingAttachments" in app
        results["Frontend upload UI"] = {
            "exists": has_upload,
            "summary": "Upload handler exists" if has_upload else "Missing",
            "file": "candidates/gaia-web/static/app.js",
        }

    # 5. Prompt builder integration
    pb_path = _BASE / "candidates/gaia-core/gaia_core/utils/prompt_builder.py"
    if pb_path.exists():
        pb = pb_path.read_text()
        has_attach = "attachment" in pb.lower() and "Attached files" in pb
        results["Prompt builder integration"] = {
            "exists": has_attach,
            "summary": "Injects attachment context" if has_attach else "Missing",
            "file": "candidates/gaia-core/gaia_core/utils/prompt_builder.py",
        }

    return results


def _plan_changes(discoveries: Dict) -> List[Dict]:
    """Determine what changes are needed based on discoveries."""
    changes = []

    attach = discoveries.get("Attachment dataclass", {})
    if attach.get("exists"):
        needed_fields = {"attachment_id", "storage_path", "content_preview"}
        existing = set(attach.get("fields", []))
        missing = needed_fields - existing
        if missing:
            changes.append({
                "file": attach["file"],
                "summary": f"Add missing fields: {', '.join(missing)}",
                "description": f"Add fields to the existing Attachment dataclass: {', '.join(f'{f}: Optional[str] = None' for f in missing)}",
                "example_pattern": "class Attachment",
            })
    else:
        changes.append({
            "file": "candidates/gaia-common/gaia_common/protocols/cognition_packet.py",
            "summary": "Create Attachment dataclass",
            "description": "Add Attachment dataclass with fields: attachment_id, filename, mime_type, size_bytes, storage_path, content_preview",
            "example_pattern": "@dataclass",
        })

    upload = discoveries.get("Upload endpoint", {})
    if not upload.get("exists"):
        changes.append({
            "file": upload.get("file", "candidates/gaia-web/gaia_web/routes/files.py"),
            "summary": "Add POST /attachments/upload endpoint",
            "description": (
                "Add an async upload endpoint: @router.post('/attachments/upload'). "
                "Accept UploadFile, validate extension against ALLOWED_TYPES + image types, "
                "validate size (10MB max), store in /shared/attachments/{session_id}/, "
                "return attachment_id, filename, size."
            ),
            "example_pattern": "@router.put",
        })

    tool = discoveries.get("read_attachment tool", {})
    if not tool.get("exists"):
        changes.append({
            "file": tool.get("file", "candidates/gaia-mcp/gaia_mcp/tools.py"),
            "summary": "Add read_attachment tool",
            "description": (
                "Add 'read_attachment' to TOOL_REGISTRY that reads a file from /shared/attachments/ "
                "by attachment_id. Prevent path traversal. Return file content (text) or metadata (binary)."
            ),
            "example_pattern": "\"read_file\"",
        })

    ui = discoveries.get("Frontend upload UI", {})
    if not ui.get("exists"):
        changes.append({
            "file": ui.get("file", "candidates/gaia-web/static/app.js"),
            "summary": "Add file upload to chat panel",
            "description": (
                "Add to chatPanel(): pendingAttachments array, handleFileSelect(event), "
                "uploadAttachments() that POSTs to /api/files/attachments/upload. "
                "Modify send() to upload first and include attachment_ids."
            ),
            "example_pattern": "async send()",
        })

    pb = discoveries.get("Prompt builder integration", {})
    if not pb.get("exists"):
        changes.append({
            "file": pb.get("file", "candidates/gaia-core/gaia_core/utils/prompt_builder.py"),
            "summary": "Inject attachment context into prompts",
            "description": (
                "After the for loop that processes data_fields (the line: "
                "'for df in getattr(packet.content, 'data_fields', []) or []:'), "
                "add a new section that checks getattr(packet.content, 'attachments', []). "
                "If attachments exist, append to identity_lines a section listing each "
                "attachment's filename and size."
            ),
            "example_pattern": "for df in getattr(packet.content",
        })

    return changes


# ── Per-file code generation ─────────────────────────────────────────────

def _generate_python_change(
    model, file_path, content, change, contract_text, example, dry_run
) -> Generator[Dict[str, Any], None, bool]:
    """Generate and validate a Python code change."""
    from gaia_core.cognition.code_generator import generate_patch, apply_patches

    patches = generate_patch(
        model, file_path, content,
        change["description"], "", contract_text,
    )

    if patches:
        yield {"type": "token", "value": f"  *Generated {len(patches)} patch(es)*\n"}

        modified, applied = apply_patches(content, patches, file_path)
        for desc in applied:
            yield {"type": "token", "value": f"  *  → {desc}*\n"}

        valid_applied = [a for a in applied if "Skipped" not in a]
        if valid_applied:
            import ast
            try:
                ast.parse(modified)
                yield {"type": "token", "value": "  *✅ Validation passed*\n"}
                if not dry_run:
                    _write_file(file_path, modified)
                    yield {"type": "token", "value": "  *✅ Written*\n"}
                else:
                    yield {"type": "token", "value": f"  *🔍 Dry run — {len(modified)} chars ready*\n"}
                return True
            except SyntaxError as e:
                yield {"type": "token", "value": f"  *❌ Syntax error after patching: {e}*\n"}
                return False
        else:
            yield {"type": "token", "value": "  *⚠️ All patches skipped (bad anchors)*\n"}
            return False
    else:
        yield {"type": "token", "value": "  *Could not generate patches*\n"}
        return False


def _generate_js_change(
    model, content, change, example, file_path, dry_run
) -> Generator[Dict[str, Any], None, bool]:
    """Generate JavaScript code to add."""
    prompt = "Add this functionality to an Alpine.js application:\n\n"
    prompt += f"**Change:** {change['description']}\n\n"
    if example:
        prompt += f"**Existing pattern to match:**\n```javascript\n{example}\n```\n\n"
    prompt += "Output ONLY the new JavaScript code. Use Alpine.js patterns. No markdown fences."

    try:
        messages = [
            {"role": "system", "content": "Output only JavaScript code for Alpine.js. No explanations."},
            {"role": "user", "content": prompt},
        ]
        result = model.create_chat_completion(messages=messages, max_tokens=1024, temperature=0.1)
        if isinstance(result, dict):
            code = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            code = re.sub(r'^```(?:javascript)?\n', '', code)
            code = re.sub(r'\n```\s*$', '', code)
            if code.strip():
                yield {"type": "token", "value": f"  *Generated {len(code)} chars of JS*\n"}
                if not dry_run:
                    _write_file(file_path, content + "\n\n" + code)
                    yield {"type": "token", "value": "  *✅ Appended*\n"}
                else:
                    yield {"type": "token", "value": f"  *🔍 Dry run — {len(code)} chars ready*\n"}
                return True
    except Exception as e:
        logger.warning("JS generation failed: %s", e)
    yield {"type": "token", "value": "  *Could not generate JS*\n"}
    return False


def _extract_example(content: str, pattern: str, context_lines: int = 25) -> str:
    """Extract an example section matching the pattern."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if pattern in line:
            start = max(0, i - 2)
            end = min(len(lines), i + context_lines)
            return "\n".join(lines[start:end])
    return ""


def _write_file(file_path: str, content: str):
    """Write with backup."""
    path = Path(file_path)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text())
    path.write_text(content)
    logger.info("Written %d chars to %s", len(content), file_path)
