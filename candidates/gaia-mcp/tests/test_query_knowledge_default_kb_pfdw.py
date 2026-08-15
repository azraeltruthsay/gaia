"""Regression test for GAIA_Project-pfdw — query_knowledge (and the other
VectorIndexer-backed tool_map lambdas) crashed with an unhandled 500 when
the caller omitted knowledge_base_name.

VectorIndexer.instance(cls, knowledge_base_name: str = "system") only
applies its "system" default when the argument is OMITTED. The lambdas
called VectorIndexer.instance(p.get("knowledge_base_name")) — p.get()
returns None for a missing key, and an explicit None overrides the
default, so VectorIndexer.__init__ raised
ValueError("Knowledge base 'None' not found in configuration.") every
single time a caller (in production: Prime's native <tool_call>, which
never includes knowledge_base_name) omitted the param. Confirmed live via
a real gaia-mcp 500 on 2026-08-15 (session discord_channel_1538193317781573).
"""
from unittest.mock import patch, MagicMock

import pytest

from gaia_mcp.tools import execute_limb
from gaia_mcp.approval import ApprovalStore


@pytest.fixture
def approval_store():
    return ApprovalStore(ttl_seconds=10)


@pytest.mark.asyncio
async def test_query_knowledge_defaults_to_system_kb_when_omitted(approval_store):
    with patch("gaia_mcp.tools.VectorIndexer") as mock_vi:
        mock_instance = MagicMock()
        mock_instance.query.return_value = []
        mock_vi.instance.return_value = mock_instance

        result = await execute_limb(
            "query_knowledge", {"query": "King Arthur discussion"}, approval_store
        )

        mock_vi.instance.assert_called_once_with("system")
        assert result == []


@pytest.mark.asyncio
async def test_query_knowledge_respects_explicit_kb(approval_store):
    with patch("gaia_mcp.tools.VectorIndexer") as mock_vi:
        mock_instance = MagicMock()
        mock_instance.query.return_value = []
        mock_vi.instance.return_value = mock_instance

        await execute_limb(
            "query_knowledge",
            {"query": "campaign lore", "knowledge_base_name": "dnd_campaign"},
            approval_store,
        )

        mock_vi.instance.assert_called_once_with("dnd_campaign")


@pytest.mark.asyncio
async def test_add_document_defaults_to_system_kb_when_omitted(approval_store):
    with patch("gaia_mcp.tools.VectorIndexer") as mock_vi:
        mock_instance = MagicMock()
        mock_vi.instance.return_value = mock_instance

        await execute_limb(
            "add_document", {"file_path": "/tmp/doc.md"}, approval_store
        )

        mock_vi.instance.assert_called_once_with("system")
