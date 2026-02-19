"""
File browser routes for Mission Control dashboard.

Provides directory listing and file reading with path traversal
protection and configurable root directories.
"""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("GAIA.Web.Files")

router = APIRouter(prefix="/api/files", tags=["files"])

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".txt", ".rst", ".csv", ".log", ".sh", ".bash", ".zsh",
    ".env", ".dockerfile", ".xml", ".sql", ".graphql",
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
WRITABLE_ROOTS = {"project"}


def _parse_roots() -> dict[str, Path]:
    """Parse FILE_ROOTS env var into {name: Path} mapping."""
    raw = os.environ.get("FILE_ROOTS", "project:/app,knowledge:/knowledge")
    roots = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, path_str = entry.split(":", 1)
        p = Path(path_str.strip())
        if p.is_dir():
            roots[name.strip()] = p.resolve()
    return roots


def _safe_path(root: Path, subpath: str) -> Path:
    """Resolve subpath under root, raising 403 on traversal."""
    resolved = (root / subpath).resolve()
    if not str(resolved).startswith(str(root)):
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    return resolved


# ── Roots ───────────────────────────────────────────────────────────────────

@router.get("/roots")
async def list_roots():
    """Return configured file browser roots."""
    roots = _parse_roots()
    return [{"name": name, "path": str(path), "writable": name in WRITABLE_ROOTS} for name, path in sorted(roots.items())]


# ── Browse ──────────────────────────────────────────────────────────────────

@router.get("/browse/{root}/{path:path}")
async def browse(root: str, path: str = ""):
    """List directory contents under a named root."""
    roots = _parse_roots()
    if root not in roots:
        raise HTTPException(status_code=404, detail=f"Unknown root: {root}")

    target = _safe_path(roots[root], path)

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue
            stat = item.stat()
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
                "modified": stat.st_mtime,
                "extension": item.suffix.lower() if item.is_file() else None,
            })
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return {"path": path, "root": root, "entries": entries}


# ── Read ────────────────────────────────────────────────────────────────────

@router.get("/read/{root}/{path:path}")
async def read_file(root: str, path: str):
    """Read text file content from a named root."""
    roots = _parse_roots()
    if root not in roots:
        raise HTTPException(status_code=404, detail=f"Unknown root: {root}")

    target = _safe_path(roots[root], path)

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Cannot read a directory")

    ext = target.suffix.lower()
    # Allow extensionless files (Dockerfile, Makefile, etc.) and text extensions
    if ext and ext not in TEXT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    stat = target.stat()
    if stat.st_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large ({stat.st_size} bytes, max {MAX_FILE_SIZE})")

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return {
        "path": path,
        "root": root,
        "name": target.name,
        "extension": ext or None,
        "size": stat.st_size,
        "content": content,
    }


# ── Write ───────────────────────────────────────────────────────────────────

class WriteRequest(BaseModel):
    content: str


@router.put("/write/{root}/{path:path}")
async def write_file(root: str, path: str, req: WriteRequest):
    """Write text file content to a writable root."""
    roots = _parse_roots()
    if root not in roots:
        raise HTTPException(status_code=404, detail=f"Unknown root: {root}")

    if root not in WRITABLE_ROOTS:
        raise HTTPException(status_code=403, detail=f"Root '{root}' is read-only")

    target = _safe_path(roots[root], path)

    ext = target.suffix.lower()
    if ext and ext not in TEXT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.content, encoding="utf-8")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return {"ok": True, "path": path, "root": root, "size": len(req.content.encode("utf-8"))}
