"""
Per-file contracts — lightweight AST-derived context anchors.

Each Python file gets a mini-contract describing its public API:
- Module purpose (from docstring)
- Classes with methods and signatures
- Functions with parameters and return types
- FastAPI endpoints
- Key imports (gaia-* packages)
- HTTP calls to other services

Contracts are auto-generated from AST summaries and stored in
/shared/contracts/files/. They serve as quick context for planning
and code generation — models can read a 20-line contract instead
of a 500-line source file to understand what a file does.

Usage:
    from gaia_common.utils.file_contracts import generate_contract, load_contract

    # Generate from source
    contract = generate_contract("/path/to/file.py")

    # Load cached contract
    contract = load_contract("candidates/gaia-web/gaia_web/routes/files.py")

    # Bulk generate for a service
    generate_service_contracts("candidates/gaia-web")
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.FileContracts")

_CONTRACTS_DIR = Path(os.environ.get("SHARED_DIR", "/shared")) / "contracts" / "files"
_CONTRACT_VERSION = "1.0"


def generate_contract(file_path: str, force: bool = False) -> Optional[Dict]:
    """
    Generate a per-file contract from AST summary.

    Args:
        file_path: Path to a Python source file
        force: Regenerate even if cached contract is fresh

    Returns:
        Contract dict or None if file can't be parsed
    """
    from gaia_common.utils.ast_summarizer import summarize_file

    path = Path(file_path)
    if not path.exists() or not path.suffix == ".py":
        return None

    # Check cache freshness
    contract_path = _contract_path_for(file_path)
    if not force and contract_path.exists():
        try:
            cached = json.loads(contract_path.read_text())
            source_mtime = path.stat().st_mtime
            if cached.get("source_mtime", 0) >= source_mtime:
                return cached
        except Exception:
            pass

    try:
        source = path.read_text()
        summary = summarize_file(source, filename=str(file_path))
    except Exception as e:
        logger.debug("Failed to parse %s: %s", file_path, e)
        return None

    # Build compact contract
    contract = {
        "version": _CONTRACT_VERSION,
        "file": str(file_path),
        "source_mtime": path.stat().st_mtime,
        "generated_at": time.time(),
        "purpose": (summary.module_docstring or "").split("\n")[0][:200],
        "classes": [
            {
                "name": c.name,
                "bases": c.bases,
                "methods": [
                    f"{'async ' if m.is_async else ''}{m.name}({', '.join(m.params[:4])})"
                    + (f" -> {m.return_type}" if m.return_type else "")
                    for m in c.methods[:15]  # Cap methods
                ],
            }
            for c in summary.classes[:10]  # Cap classes
        ],
        "functions": [
            f"{'async ' if f.is_async else ''}{f.name}({', '.join(f.params[:4])})"
            + (f" -> {f.return_type}" if f.return_type else "")
            for f in summary.functions[:20]  # Cap functions
        ],
        "endpoints": [
            f"{e.method.upper()} {e.path} → {e.function_name}"
            for e in summary.endpoints
        ],
        "imports": summary.gaia_imports[:15],
        "http_calls": [
            f"{h.call_method} {h.url_or_path} (in {h.enclosing_function})"
            for h in summary.http_calls[:10]
        ],
        "constants": [c.name for c in summary.constants[:10]],
    }

    # Persist
    _contracts_dir().mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2))
    logger.debug("Generated contract for %s (%d bytes)", file_path, contract_path.stat().st_size)

    return contract


def load_contract(file_path: str) -> Optional[Dict]:
    """Load a cached contract, or generate if missing."""
    contract_path = _contract_path_for(file_path)
    if contract_path.exists():
        try:
            cached = json.loads(contract_path.read_text())
            # Check staleness
            source_path = Path(file_path)
            if source_path.exists():
                if cached.get("source_mtime", 0) < source_path.stat().st_mtime:
                    return generate_contract(file_path, force=True)
            return cached
        except Exception:
            pass
    return generate_contract(file_path)


def contract_to_prompt(contract: Dict) -> str:
    """Render a contract as compact text for LLM prompt inclusion."""
    if not contract:
        return ""

    # Shorten full container paths to candidates/ relative form
    file_path = contract.get('file', '?')
    if 'candidates/' in file_path:
        file_path = 'candidates/' + file_path.split('candidates/', 1)[-1]
    lines = [f"**{file_path}**"]

    purpose = contract.get("purpose", "")
    if purpose:
        lines.append(f"  Purpose: {purpose}")

    for cls in contract.get("classes", []):
        bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
        lines.append(f"  class {cls['name']}{bases}:")
        for m in cls.get("methods", [])[:8]:
            lines.append(f"    {m}")

    for func in contract.get("functions", [])[:10]:
        lines.append(f"  {func}")

    for ep in contract.get("endpoints", []):
        lines.append(f"  endpoint: {ep}")

    imports = contract.get("imports", [])
    if imports:
        lines.append(f"  imports: {', '.join(imports[:8])}")

    calls = contract.get("http_calls", [])
    if calls:
        lines.append(f"  calls: {', '.join(calls[:5])}")

    return "\n".join(lines)


def generate_service_contracts(service_dir: str, force: bool = False) -> Dict:
    """Generate contracts for all Python files in a service directory."""
    results = {"generated": 0, "skipped": 0, "errors": 0}
    service_path = Path(service_dir)
    if not service_path.exists():
        return results

    for py_file in service_path.rglob("*.py"):
        # Skip tests, __pycache__, migrations
        if any(skip in str(py_file) for skip in ["__pycache__", ".pyc", "test_", "tests/"]):
            continue
        try:
            contract = generate_contract(str(py_file), force=force)
            if contract:
                results["generated"] += 1
            else:
                results["skipped"] += 1
        except Exception:
            results["errors"] += 1

    return results


def get_contracts_for_planning(file_paths: List[str]) -> str:
    """
    Load contracts for a list of files and return assembled prompt text.
    Used by the planning orchestrator for codebase context.
    """
    parts = []
    for fp in file_paths:
        contract = load_contract(fp)
        if contract:
            parts.append(contract_to_prompt(contract))

    if not parts:
        return ""

    return "## File Contracts (actual codebase API)\n\n" + "\n\n".join(parts)


def _contract_path_for(file_path: str) -> Path:
    """Map a source file path to its contract storage path."""
    # Normalize: strip leading / and replace / with __
    normalized = str(file_path).lstrip("/").replace("/", "__")
    return _contracts_dir() / f"{normalized}.json"


def _contracts_dir() -> Path:
    return _CONTRACTS_DIR
