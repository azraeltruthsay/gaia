"""
Discord Attachment Handler for GAIA Web Gateway

Downloads Discord file attachments, extracts text content where possible,
and returns structured metadata for the CognitionPacket pipeline.
"""

import hashlib
import logging
import os
import subprocess
import tempfile
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.Web.Attachments")

# Supported text-extractable extensions
_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".toml", ".xml", ".log", ".py", ".js", ".html"}
_PDF_EXTENSIONS = {".pdf"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

_ATTACHMENT_DIR = "/tmp/gaia_attachments"
_MAX_TEXT_BYTES = 25 * 1024 * 1024  # 25MB (Discord's limit)


def _ensure_attachment_dir():
    os.makedirs(_ATTACHMENT_DIR, exist_ok=True)


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _extract_pdf_text(path: str) -> Optional[str]:
    """Extract text from PDF using pdftotext (poppler-utils)."""
    try:
        result = subprocess.run(
            ["pdftotext", path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        logger.warning("pdftotext not available — attempting raw UTF-8 decode fallback")
    except subprocess.TimeoutExpired:
        logger.warning("pdftotext timed out for %s", path)
    except Exception as e:
        logger.warning("pdftotext failed for %s: %s", path, e)

    # Fallback: try raw UTF-8 decode (works for some text-based PDFs)
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="ignore")
        # Filter to printable content — crude but better than nothing
        lines = [ln for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("%")]
        if len(lines) > 5:
            return "\n".join(lines[:500])
    except Exception as _exc:
        logger.debug("Attachment: text extraction failed for %s: %s", path, _exc)
    return None


def _extract_text(path: str, ext: str) -> Optional[str]:
    """Extract text content based on file extension."""
    if ext in _TEXT_EXTENSIONS:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(_MAX_TEXT_BYTES)
        except Exception as e:
            logger.warning("Failed to read text file %s: %s", path, e)
            return None

    if ext in _PDF_EXTENSIONS:
        return _extract_pdf_text(path)

    if ext in _IMAGE_EXTENSIONS:
        # Future multimodal — store path only, no text extraction
        return None

    return None


async def process_attachments(message: Any) -> List[Dict]:
    """
    Download and extract content from Discord message attachments.

    Args:
        message: discord.Message object

    Returns:
        List of dicts: {filename, mime, path, text_content, size_bytes, content_hash}
    """
    if not hasattr(message, "attachments") or not message.attachments:
        return []

    _ensure_attachment_dir()
    results = []

    for attachment in message.attachments:
        filename = attachment.filename
        size_bytes = attachment.size
        ext = os.path.splitext(filename)[1].lower()

        # Generate unique local path
        local_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        local_path = os.path.join(_ATTACHMENT_DIR, local_name)

        try:
            await attachment.save(local_path)
            logger.info("Downloaded attachment: %s (%d bytes) → %s", filename, size_bytes, local_path)
        except Exception as e:
            logger.error("Failed to download attachment %s: %s", filename, e)
            continue

        # Compute content hash
        try:
            with open(local_path, "rb") as f:
                chash = _content_hash(f.read())
        except Exception:
            chash = "unknown"

        # Extract text content
        text_content = _extract_text(local_path, ext)

        # Infer MIME type
        mime_map = {
            ".pdf": "application/pdf",
            ".txt": "text/plain", ".md": "text/markdown",
            ".json": "application/json", ".csv": "text/csv",
            ".yaml": "application/yaml", ".yml": "application/yaml",
            ".xml": "application/xml", ".html": "text/html",
            ".py": "text/x-python", ".js": "text/javascript",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp",
        }
        mime = mime_map.get(ext, attachment.content_type or "application/octet-stream")

        results.append({
            "filename": filename,
            "mime": mime,
            "path": local_path,
            "text_content": text_content,
            "size_bytes": size_bytes,
            "content_hash": chash,
        })

    logger.info("Processed %d attachment(s): %d with text content",
                len(results), sum(1 for r in results if r["text_content"]))
    return results
