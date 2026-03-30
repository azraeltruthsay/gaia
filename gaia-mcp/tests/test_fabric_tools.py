"""Tests for Fabric pattern loader and tool registration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _create_test_patterns(tmpdir: Path):
    """Create a test patterns directory with sample patterns."""
    patterns_dir = tmpdir / "fabric_patterns"
    patterns_dir.mkdir(exist_ok=True)

    # Pattern 1: extract_test
    (patterns_dir / "extract_test").mkdir()
    (patterns_dir / "extract_test" / "system.md").write_text(
        "# IDENTITY and PURPOSE\n\n"
        "You extract key insights from text.\n\n"
        "# STEPS\n\n- Extract ideas\n- Summarize\n\n"
        "# OUTPUT INSTRUCTIONS\n\n- Only output Markdown.\n"
    )

    # Pattern 2: summarize_test
    (patterns_dir / "summarize_test").mkdir()
    (patterns_dir / "summarize_test" / "system.md").write_text(
        "# IDENTITY and PURPOSE\n\n"
        "You are an expert summarizer.\n\n"
        "# STEPS\n\n- Create summary\n"
    )

    # Pattern 3: disabled_test (won't be in allowlist)
    (patterns_dir / "disabled_test").mkdir()
    (patterns_dir / "disabled_test" / "system.md").write_text(
        "# IDENTITY\n\nDisabled pattern.\n"
    )

    # Config: allowlist mode
    (patterns_dir / "_config.json").write_text(json.dumps({
        "mode": "allowlist",
        "default_target": "core",
        "default_max_tokens": 2048,
        "default_no_think": True,
        "patterns": {
            "extract_test": {"enabled": True, "target": "prime"},
            "summarize_test": {"enabled": True},
        }
    }))

    return patterns_dir


class TestPatternLoading:
    """Test pattern discovery and schema generation."""

    def _load_with_tmpdir(self):
        """Load patterns from a temp directory by patching the module constant."""
        import gaia_mcp.fabric_tools as ft
        tmpdir = Path(tempfile.mkdtemp())
        patterns_dir = _create_test_patterns(tmpdir)
        with patch.object(ft, "FABRIC_PATTERNS_DIR", patterns_dir), \
             patch.object(ft, "_CONFIG_FILE", patterns_dir / "_config.json"):
            return ft.load_fabric_patterns()

    def test_load_patterns_finds_enabled(self):
        schemas, prompts = self._load_with_tmpdir()
        assert "fabric_extract_test" in schemas
        assert "fabric_summarize_test" in schemas
        assert "fabric_extract_test" in prompts
        assert "fabric_summarize_test" in prompts

    def test_load_patterns_skips_disabled(self):
        schemas, prompts = self._load_with_tmpdir()
        assert "fabric_disabled_test" not in schemas
        assert "fabric_disabled_test" not in prompts

    def test_schema_structure(self):
        schemas, _ = self._load_with_tmpdir()
        schema = schemas["fabric_extract_test"]
        assert "description" in schema
        assert schema["description"].startswith("[Fabric]")
        assert "params" in schema
        assert "input" in schema["params"]["properties"]
        assert "input" in schema["params"]["required"]

    def test_description_extracted_from_purpose(self):
        schemas, _ = self._load_with_tmpdir()
        desc = schemas["fabric_extract_test"]["description"]
        assert "extract" in desc.lower() or "insights" in desc.lower()

    def test_system_prompt_loaded(self):
        _, prompts = self._load_with_tmpdir()
        prompt = prompts["fabric_extract_test"]
        assert "IDENTITY" in prompt
        assert "STEPS" in prompt
        assert "OUTPUT INSTRUCTIONS" in prompt

    def test_skips_underscore_dirs(self):
        schemas, _ = self._load_with_tmpdir()
        for name in schemas:
            assert not name.startswith("fabric__")


class TestAllMode:
    """Test 'all' mode loading (no allowlist filtering)."""

    def test_all_mode_loads_everything(self):
        import gaia_mcp.fabric_tools as ft
        tmpdir = Path(tempfile.mkdtemp())
        patterns_dir = _create_test_patterns(tmpdir)
        # Override config to all mode
        (patterns_dir / "_config.json").write_text(json.dumps({
            "mode": "all",
            "default_target": "core",
            "default_max_tokens": 2048,
            "default_no_think": True,
            "patterns": {}
        }))
        with patch.object(ft, "FABRIC_PATTERNS_DIR", patterns_dir), \
             patch.object(ft, "_CONFIG_FILE", patterns_dir / "_config.json"):
            schemas, _ = ft.load_fabric_patterns()
        assert "fabric_disabled_test" in schemas
        assert "fabric_extract_test" in schemas
        assert "fabric_summarize_test" in schemas


class TestPurposeExtraction:
    """Test _extract_purpose helper."""

    def test_extracts_from_standard_format(self):
        from gaia_mcp.fabric_tools import _extract_purpose
        result = _extract_purpose(
            "# IDENTITY and PURPOSE\n\n"
            "You are an expert analyst.\n"
            "You specialize in threat reports.\n\n"
            "# STEPS\n"
        )
        assert "expert analyst" in result
        assert "threat reports" in result

    def test_handles_missing_section(self):
        from gaia_mcp.fabric_tools import _extract_purpose
        result = _extract_purpose("# STEPS\n- Do stuff\n")
        assert result == "Execute Fabric pattern."

    def test_truncates_long_descriptions(self):
        from gaia_mcp.fabric_tools import _extract_purpose
        long_text = "You do things. " * 50
        result = _extract_purpose(f"# IDENTITY and PURPOSE\n\n{long_text}\n# STEPS\n")
        assert len(result) <= 200


class TestExecuteFabricTool:
    """Test the async execution path."""

    @pytest.mark.asyncio
    async def test_missing_input_returns_error(self):
        from gaia_mcp.fabric_tools import execute_fabric_tool, fabric_system_prompts
        fabric_system_prompts["fabric_extract_test"] = "test prompt"
        result = await execute_fabric_tool("fabric_extract_test", {"input": ""})
        assert result["ok"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_unknown_pattern_returns_error(self):
        from gaia_mcp.fabric_tools import execute_fabric_tool
        result = await execute_fabric_tool("fabric_nonexistent", {"input": "test"})
        assert result["ok"] is False
        assert "not loaded" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        from gaia_mcp import fabric_tools

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value={
            "content": "## SUMMARY\n\nTest output.",
            "target": "core",
        })

        fabric_tools.fabric_system_prompts["fabric_summarize_test"] = "You summarize."
        fabric_tools.fabric_config = {
            "default_target": "core",
            "default_max_tokens": 2048,
            "default_no_think": True,
            "patterns": {"summarize_test": {}},
        }

        with patch.object(fabric_tools, "_core_client", mock_client):
            result = await fabric_tools.execute_fabric_tool(
                "fabric_summarize_test", {"input": "Some article text."}
            )

        assert result["ok"] is True
        assert "Test output" in result["content"]
        assert result["pattern"] == "summarize_test"

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/api/cognitive/query"
        payload = call_args[1]["data"]
        assert payload["prompt"] == "Some article text."
        assert payload["system"] == "You summarize."
