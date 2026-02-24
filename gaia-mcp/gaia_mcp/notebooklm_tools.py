"""
NotebookLM MCP Tools — structured access to Google NotebookLM notebooks.

Provides 8 tools:
  - notebooklm_list_notebooks   (read)
  - notebooklm_get_notebook     (read)
  - notebooklm_list_sources     (read)
  - notebooklm_list_notes       (read)
  - notebooklm_list_artifacts   (read)
  - notebooklm_chat             (read)
  - notebooklm_download_audio   (read)
  - notebooklm_create_note      (write, sensitive)

All calls go through the async NotebookLMClient (httpx-based).
Auth via Playwright storage state at NOTEBOOKLM_HOME/storage_state.json.
"""

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.NotebookLMTools")

# Lazy imports — notebooklm-py may not be installed in all environments
_notebooklm_available = True
try:
    from notebooklm import NotebookLMClient
    from notebooklm.exceptions import NotebookLMError
except ImportError:
    _notebooklm_available = False
    NotebookLMClient = None  # type: ignore
    NotebookLMError = Exception  # type: ignore

# Optional: httpx for audio transcription relay to gaia-audio
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

_GAIA_AUDIO_URL = os.getenv("GAIA_AUDIO_URL", "http://gaia-audio:8080")


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[Any] = None
_client_entered = False


async def _get_client() -> Any:
    """Lazily initialise and return the NotebookLMClient singleton."""
    global _client, _client_entered

    if not _notebooklm_available:
        raise RuntimeError(
            "notebooklm-py is not installed. Add it to requirements.txt and rebuild."
        )

    if _client is not None and _client_entered:
        if _client.is_connected:
            return _client
        # Connection dropped — re-enter
        try:
            await _client.__aenter__()
            return _client
        except Exception:
            _client = None
            _client_entered = False

    storage_path = os.getenv("NOTEBOOKLM_STORAGE_STATE")
    _client = await NotebookLMClient.from_storage(
        path=storage_path, timeout=45
    )
    await _client.__aenter__()
    _client_entered = True
    logger.info("NotebookLM client initialised (storage=%s)", storage_path or "default")
    return _client


async def _close_client():
    """Shut down the client (call on app shutdown)."""
    global _client, _client_entered
    if _client is not None and _client_entered:
        try:
            await _client.__aexit__(None, None, None)
        except Exception:
            pass
    _client = None
    _client_entered = False


def _err(msg: str) -> dict:
    return {"ok": False, "error": msg}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def notebooklm_list_notebooks(params: dict) -> dict:
    """List all notebooks accessible to the authenticated user."""
    try:
        client = await _get_client()
        notebooks = await client.notebooks.list()
        return {
            "ok": True,
            "notebooks": [
                {
                    "id": nb.id,
                    "title": nb.title,
                    "sources_count": getattr(nb, "sources_count", 0),
                    "is_owner": getattr(nb, "is_owner", True),
                    "created_at": str(nb.created_at) if getattr(nb, "created_at", None) else None,
                }
                for nb in notebooks
            ],
            "count": len(notebooks),
        }
    except NotebookLMError as e:
        logger.error("notebooklm_list_notebooks failed: %s", e)
        return _err(str(e))
    except Exception as e:
        logger.error("notebooklm_list_notebooks unexpected error: %s", e)
        return _err(f"Unexpected error: {e}")


async def notebooklm_get_notebook(params: dict) -> dict:
    """Get notebook details including AI summary and suggested topics."""
    notebook_id = (params.get("notebook_id") or "").strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")

    try:
        client = await _get_client()
        notebook = await client.notebooks.get(notebook_id)

        # Fetch description (includes summary + suggested topics)
        description = None
        try:
            desc = await client.notebooks.get_description(notebook_id)
            description = {
                "summary": getattr(desc, "summary", None),
                "suggested_topics": [
                    {"question": t.question, "prompt": t.prompt}
                    for t in getattr(desc, "suggested_topics", [])
                ],
            }
        except Exception:
            # Description may not be available for all notebooks
            pass

        return {
            "ok": True,
            "notebook": {
                "id": notebook.id,
                "title": notebook.title,
                "sources_count": getattr(notebook, "sources_count", 0),
                "is_owner": getattr(notebook, "is_owner", True),
                "created_at": str(notebook.created_at) if getattr(notebook, "created_at", None) else None,
            },
            "description": description,
        }
    except NotebookLMError as e:
        logger.error("notebooklm_get_notebook failed: %s", e)
        return _err(str(e))


async def notebooklm_list_sources(params: dict) -> dict:
    """List sources in a notebook."""
    notebook_id = (params.get("notebook_id") or "").strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")

    try:
        client = await _get_client()
        sources = await client.sources.list(notebook_id)
        return {
            "ok": True,
            "notebook_id": notebook_id,
            "sources": [
                {
                    "id": s.id,
                    "title": s.title,
                    "url": getattr(s, "url", None),
                    "kind": str(s.kind) if hasattr(s, "kind") else None,
                    "status": s.status.name if hasattr(s.status, "name") else str(s.status),
                    "is_ready": s.is_ready,
                }
                for s in sources
            ],
            "count": len(sources),
        }
    except NotebookLMError as e:
        logger.error("notebooklm_list_sources failed: %s", e)
        return _err(str(e))


async def notebooklm_list_notes(params: dict) -> dict:
    """List user notes in a notebook."""
    notebook_id = (params.get("notebook_id") or "").strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")

    try:
        client = await _get_client()
        notes = await client.notes.list(notebook_id)
        return {
            "ok": True,
            "notebook_id": notebook_id,
            "notes": [
                {
                    "id": n.id,
                    "title": n.title,
                    "content": (n.content or "")[:500],  # Truncate for listing
                }
                for n in notes
            ],
            "count": len(notes),
        }
    except NotebookLMError as e:
        logger.error("notebooklm_list_notes failed: %s", e)
        return _err(str(e))


async def notebooklm_list_artifacts(params: dict) -> dict:
    """List artifacts (audio overviews, reports, quizzes, etc.) in a notebook."""
    notebook_id = (params.get("notebook_id") or "").strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")

    artifact_type = (params.get("artifact_type") or "").strip().lower() or None

    try:
        client = await _get_client()
        artifacts = await client.artifacts.list(notebook_id)

        # Optional type filter
        if artifact_type:
            artifacts = [
                a for a in artifacts
                if str(getattr(a, "kind", "")).lower() == artifact_type
            ]

        return {
            "ok": True,
            "notebook_id": notebook_id,
            "artifacts": [
                {
                    "id": a.id,
                    "title": a.title,
                    "kind": str(a.kind) if hasattr(a, "kind") else None,
                    "status": a.status.name if hasattr(a.status, "name") else str(a.status),
                    "is_completed": a.is_completed,
                    "created_at": str(a.created_at) if a.created_at else None,
                }
                for a in artifacts
            ],
            "count": len(artifacts),
        }
    except NotebookLMError as e:
        logger.error("notebooklm_list_artifacts failed: %s", e)
        return _err(str(e))


async def notebooklm_chat(params: dict) -> dict:
    """Ask a question to a notebook. Supports follow-up via conversation_id."""
    notebook_id = (params.get("notebook_id") or "").strip()
    question = (params.get("question") or "").strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")
    if not question:
        raise ValueError("question is required")

    source_ids = params.get("source_ids")
    conversation_id = params.get("conversation_id")

    try:
        client = await _get_client()
        result = await client.chat.ask(
            notebook_id=notebook_id,
            question=question,
            source_ids=source_ids if isinstance(source_ids, list) else None,
            conversation_id=conversation_id,
        )
        return {
            "ok": True,
            "notebook_id": notebook_id,
            "answer": result.answer,
            "conversation_id": result.conversation_id,
            "turn_number": result.turn_number,
            "is_follow_up": result.is_follow_up,
            "references": [
                {
                    "source_id": getattr(ref, "source_id", None),
                    "citation_number": getattr(ref, "citation_number", None),
                    "cited_text": getattr(ref, "cited_text", None),
                }
                for ref in (result.references or [])
            ],
        }
    except NotebookLMError as e:
        logger.error("notebooklm_chat failed: %s", e)
        return _err(str(e))


async def notebooklm_download_audio(params: dict) -> dict:
    """Download an audio overview and optionally transcribe it via gaia-audio."""
    notebook_id = (params.get("notebook_id") or "").strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")

    artifact_id = (params.get("artifact_id") or "").strip() or None

    try:
        client = await _get_client()

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        download_path = await client.artifacts.download_audio(
            notebook_id=notebook_id,
            output_path=tmp_path,
            artifact_id=artifact_id,
        )

        result: Dict[str, Any] = {
            "ok": True,
            "notebook_id": notebook_id,
            "download_path": str(download_path),
        }

        # Try to transcribe via gaia-audio
        if httpx is not None:
            try:
                audio_bytes = Path(download_path).read_bytes()
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

                async with httpx.AsyncClient(timeout=120) as http:
                    resp = await http.post(
                        f"{_GAIA_AUDIO_URL}/transcribe",
                        json={"audio": audio_b64, "format": "mp4"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        result["transcription"] = data.get("text", data.get("transcription", ""))
                        result["transcribed"] = True
                    else:
                        result["transcribed"] = False
                        result["transcribe_error"] = f"gaia-audio returned {resp.status_code}"
            except Exception as e:
                result["transcribed"] = False
                result["transcribe_error"] = f"Transcription relay failed: {e}"
        else:
            result["transcribed"] = False
            result["transcribe_error"] = "httpx not available for transcription relay"

        # Clean up temp file
        try:
            Path(download_path).unlink(missing_ok=True)
        except Exception:
            pass

        return result

    except NotebookLMError as e:
        logger.error("notebooklm_download_audio failed: %s", e)
        return _err(str(e))


async def notebooklm_create_note(params: dict) -> dict:
    """Create a note in a notebook. Requires approval."""
    notebook_id = (params.get("notebook_id") or "").strip()
    title = (params.get("title") or "").strip()
    content = params.get("content", "")

    if not notebook_id:
        raise ValueError("notebook_id is required")
    if not title:
        raise ValueError("title is required")

    try:
        client = await _get_client()
        note = await client.notes.create(
            notebook_id=notebook_id,
            title=title,
            content=content,
        )
        return {
            "ok": True,
            "notebook_id": notebook_id,
            "created": {
                "id": note.id,
                "title": note.title,
                "content": (note.content or "")[:500],
            },
        }
    except NotebookLMError as e:
        logger.error("notebooklm_create_note failed: %s", e)
        return _err(str(e))
