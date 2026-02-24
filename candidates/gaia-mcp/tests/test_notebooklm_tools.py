"""Tests for GAIA MCP NotebookLM tools."""

import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import gaia_mcp.notebooklm_tools as nbt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level client singleton between tests."""
    nbt._client = None
    nbt._client_entered = False
    yield
    nbt._client = None
    nbt._client_entered = False


def _mock_notebook(id="nb-1", title="Test Notebook", sources_count=5):
    nb = MagicMock()
    nb.id = id
    nb.title = title
    nb.sources_count = sources_count
    nb.is_owner = True
    nb.created_at = None
    return nb


def _mock_source(id="src-1", title="Source 1", url="https://example.com"):
    s = MagicMock()
    s.id = id
    s.title = title
    s.url = url
    s.kind = "web_page"
    s.status = MagicMock()
    s.status.name = "READY"
    s.is_ready = True
    return s


def _mock_note(id="note-1", title="Note 1", content="Some content"):
    n = MagicMock()
    n.id = id
    n.title = title
    n.content = content
    return n


def _mock_artifact(id="art-1", title="Audio Overview", kind="audio"):
    a = MagicMock()
    a.id = id
    a.title = title
    a.kind = kind
    a.status = MagicMock()
    a.status.name = "COMPLETED"
    a.is_completed = True
    a.created_at = None
    return a


def _mock_ask_result():
    ref = MagicMock()
    ref.source_id = "src-1"
    ref.citation_number = 1
    ref.cited_text = "relevant quote"

    result = MagicMock()
    result.answer = "The answer is 42."
    result.conversation_id = "conv-abc"
    result.turn_number = 1
    result.is_follow_up = False
    result.references = [ref]
    return result


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestNotebookLMValidation:

    @pytest.mark.asyncio
    async def test_get_notebook_missing_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_get_notebook({})

    @pytest.mark.asyncio
    async def test_list_sources_missing_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_list_sources({})

    @pytest.mark.asyncio
    async def test_list_notes_missing_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_list_notes({})

    @pytest.mark.asyncio
    async def test_list_artifacts_missing_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_list_artifacts({})

    @pytest.mark.asyncio
    async def test_chat_missing_notebook_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_chat({"question": "test?"})

    @pytest.mark.asyncio
    async def test_chat_missing_question(self):
        with pytest.raises(ValueError, match="question is required"):
            await nbt.notebooklm_chat({"notebook_id": "nb-1"})

    @pytest.mark.asyncio
    async def test_download_audio_missing_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_download_audio({})

    @pytest.mark.asyncio
    async def test_create_note_missing_notebook_id(self):
        with pytest.raises(ValueError, match="notebook_id is required"):
            await nbt.notebooklm_create_note({"title": "Test"})

    @pytest.mark.asyncio
    async def test_create_note_missing_title(self):
        with pytest.raises(ValueError, match="title is required"):
            await nbt.notebooklm_create_note({"notebook_id": "nb-1"})


# ---------------------------------------------------------------------------
# Success path tests
# ---------------------------------------------------------------------------

class TestNotebookLMListNotebooks:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_returns_notebooks(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.notebooks.list = AsyncMock(return_value=[
            _mock_notebook("nb-1", "Notebook A"),
            _mock_notebook("nb-2", "Notebook B"),
        ])
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_list_notebooks({})
        assert result["ok"] is True
        assert result["count"] == 2
        assert result["notebooks"][0]["title"] == "Notebook A"


class TestNotebookLMGetNotebook:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_returns_notebook_with_description(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.notebooks.get = AsyncMock(return_value=_mock_notebook())

        desc = MagicMock()
        desc.summary = "This is a codebase notebook."
        topic = MagicMock()
        topic.question = "How does GAIA work?"
        topic.prompt = "Explain the architecture"
        desc.suggested_topics = [topic]
        mock_client.notebooks.get_description = AsyncMock(return_value=desc)
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_get_notebook({"notebook_id": "nb-1"})
        assert result["ok"] is True
        assert result["notebook"]["title"] == "Test Notebook"
        assert result["description"]["summary"] == "This is a codebase notebook."
        assert len(result["description"]["suggested_topics"]) == 1


class TestNotebookLMListSources:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_returns_sources(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.sources.list = AsyncMock(return_value=[
            _mock_source("src-1", "Doc A"),
            _mock_source("src-2", "Doc B"),
        ])
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_list_sources({"notebook_id": "nb-1"})
        assert result["ok"] is True
        assert result["count"] == 2
        assert result["sources"][0]["title"] == "Doc A"


class TestNotebookLMListNotes:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_returns_notes(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.notes.list = AsyncMock(return_value=[_mock_note()])
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_list_notes({"notebook_id": "nb-1"})
        assert result["ok"] is True
        assert result["count"] == 1
        assert result["notes"][0]["title"] == "Note 1"


class TestNotebookLMListArtifacts:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_returns_artifacts(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.artifacts.list = AsyncMock(return_value=[
            _mock_artifact("a1", "Audio 1", "audio"),
            _mock_artifact("a2", "Report 1", "report"),
        ])
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_list_artifacts({"notebook_id": "nb-1"})
        assert result["ok"] is True
        assert result["count"] == 2

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_filters_by_type(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.artifacts.list = AsyncMock(return_value=[
            _mock_artifact("a1", "Audio 1", "audio"),
            _mock_artifact("a2", "Report 1", "report"),
        ])
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_list_artifacts({"notebook_id": "nb-1", "artifact_type": "audio"})
        assert result["ok"] is True
        assert result["count"] == 1
        assert result["artifacts"][0]["title"] == "Audio 1"


class TestNotebookLMChat:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_returns_answer(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.chat.ask = AsyncMock(return_value=_mock_ask_result())
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_chat({"notebook_id": "nb-1", "question": "What is GAIA?"})
        assert result["ok"] is True
        assert result["answer"] == "The answer is 42."
        assert result["conversation_id"] == "conv-abc"
        assert len(result["references"]) == 1

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_passes_conversation_id(self, mock_gc):
        mock_client = AsyncMock()
        ask_result = _mock_ask_result()
        ask_result.is_follow_up = True
        ask_result.turn_number = 2
        mock_client.chat.ask = AsyncMock(return_value=ask_result)
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_chat({
            "notebook_id": "nb-1",
            "question": "Tell me more about that.",
            "conversation_id": "conv-abc",
        })
        assert result["ok"] is True
        assert result["is_follow_up"] is True
        assert result["turn_number"] == 2
        mock_client.chat.ask.assert_called_once_with(
            notebook_id="nb-1",
            question="Tell me more about that.",
            source_ids=None,
            conversation_id="conv-abc",
        )


class TestNotebookLMCreateNote:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_creates_note(self, mock_gc):
        mock_client = AsyncMock()
        mock_client.notes.create = AsyncMock(return_value=_mock_note("new-1", "My Note", "Content here"))
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_create_note({
            "notebook_id": "nb-1",
            "title": "My Note",
            "content": "Content here",
        })
        assert result["ok"] is True
        assert result["created"]["title"] == "My Note"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestNotebookLMErrors:

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_client_error_returns_ok_false(self, mock_gc):
        mock_gc.side_effect = RuntimeError("notebooklm-py is not installed")

        result = await nbt.notebooklm_list_notebooks({})
        assert result["ok"] is False
        assert "not installed" in result["error"]

    @pytest.mark.asyncio
    @patch.object(nbt, "_get_client")
    async def test_api_error_returns_ok_false(self, mock_gc):
        mock_client = AsyncMock()
        # Simulate a NotebookLMError (using Exception since we can't import it in test env)
        mock_client.notebooks.list = AsyncMock(side_effect=Exception("Auth expired"))
        mock_gc.return_value = mock_client

        result = await nbt.notebooklm_list_notebooks({})
        assert result["ok"] is False
        assert "Auth expired" in result["error"]
