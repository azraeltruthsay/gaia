import logging
import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from gaia_core.memory.semantic_codex import SemanticCodex, CodexEntry
from gaia_core.config import Config

# Fixture for a temporary directory to simulate knowledge/
@pytest.fixture
def temp_knowledge_dir(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "self_generated_docs").mkdir() # Ensure self_generated_docs exists
    return knowledge_dir

# Fixture for a mock Config object
@pytest.fixture
def mock_config(temp_knowledge_dir):
    config = MagicMock(spec=Config)
    config.KNOWLEDGE_CODEX_DIR = str(temp_knowledge_dir)
    config.CODEX_FILE_EXTS = (".md", ".yaml", ".yml", ".json")
    config.CODEX_ALLOW_HOT_RELOAD = True
    return config

# Fixture for a SemanticCodex instance
@pytest.fixture
def semantic_codex(mock_config):
    return SemanticCodex(mock_config)

def test_write_entry_creates_markdown_file(semantic_codex, temp_knowledge_dir):
    entry = CodexEntry(
        symbol="TEST_CONCEPT",
        title="Test Concept Title",
        body="This is the body of the test concept.",
        tags=("test", "concept"),
        version="v1.0",
        scope="project"
    )
    
    file_path = semantic_codex.write_entry(entry)
    
    expected_path = temp_knowledge_dir / "self_generated_docs" / "TEST_CONCEPT.md"
    assert file_path == expected_path
    assert file_path.exists()
    
    content = file_path.read_text()
    assert "---" in content
    assert "symbol: TEST_CONCEPT" in content
    assert "title: Test Concept Title" in content
    assert "- test" in content
    assert "- concept" in content
    assert "version: v1.0" in content
    assert "scope: project" in content
    assert "This is the body of the test concept." in content

    # Verify that the entry is loaded into the index after hot_reload
    loaded_entry = semantic_codex.get("TEST_CONCEPT")
    assert loaded_entry == entry

def test_load_one_markdown_with_front_matter(semantic_codex, temp_knowledge_dir):
    # Create a dummy Markdown file directly
    md_content = """---
symbol: ANOTHER_CONCEPT
title: Another Test Concept
tags:
  - second
  - concept
version: v2.0
scope: global
---

This is the body for another concept. It has some text.
"""
    file_path = temp_knowledge_dir / "self_generated_docs" / "ANOTHER_CONCEPT.md"
    file_path.write_text(md_content)

    # Manually trigger load_all to pick up the new file
    semantic_codex._load_all()

    loaded_entry = semantic_codex.get("ANOTHER_CONCEPT")
    assert loaded_entry is not None
    assert loaded_entry.symbol == "ANOTHER_CONCEPT"
    assert loaded_entry.title == "Another Test Concept"
    assert loaded_entry.body == "This is the body for another concept. It has some text."
    assert loaded_entry.tags == ("second", "concept")
    assert loaded_entry.version == "v2.0"
    assert loaded_entry.scope == "global"

def test_load_one_markdown_missing_symbol(semantic_codex, temp_knowledge_dir, caplog):
    md_content = """---
title: Missing Symbol
tags: []
---
Body content.
"""
    file_path = temp_knowledge_dir / "self_generated_docs" / "MISSING_SYMBOL.md"
    file_path.write_text(md_content)
    
    with caplog.at_level(logging.WARNING):
        semantic_codex._load_one(file_path)
        assert "missing 'symbol' or 'body' in front matter" in caplog.text
    assert semantic_codex.get("MISSING_SYMBOL") is None

def test_load_one_markdown_invalid_yaml(semantic_codex, temp_knowledge_dir, caplog):
    md_content = """---
symbol: INVALID_YAML
title: Invalid YAML Test
tags: [tag1, tag2
version: v1.0
---
Body content.
"""
    file_path = temp_knowledge_dir / "self_generated_docs" / "INVALID_YAML.md"
    file_path.write_text(md_content)
    
    with caplog.at_level(logging.DEBUG):
        semantic_codex._load_one(file_path)
        assert "non-codex front matter" in caplog.text
    assert semantic_codex.get("INVALID_YAML") is None

def test_load_one_json_still_works(semantic_codex, temp_knowledge_dir):
    json_content = """{
    "symbol": "JSON_CONCEPT",
    "title": "JSON Test Concept",
    "body": "This is a JSON body.",
    "tags": ["json", "test"],
    "version": "v1",
    "scope": "global"
}"""
    file_path = temp_knowledge_dir / "JSON_CONCEPT.json"
    file_path.write_text(json_content)
    
    semantic_codex._load_all() # Reload to pick up JSON
    loaded_entry = semantic_codex.get("JSON_CONCEPT")
    assert loaded_entry is not None
    assert loaded_entry.symbol == "JSON_CONCEPT"
    assert loaded_entry.body == "This is a JSON body."

def test_iter_files_includes_self_generated_docs(semantic_codex, temp_knowledge_dir):
    # Create a file in the root knowledge dir
    (temp_knowledge_dir / "root_doc.json").write_text("{}")
    
    # Write an entry via write_entry to self_generated_docs
    entry = CodexEntry(
        symbol="GEN_DOC", title="Generated Document", body="Content", tags=(), version="v1", scope="global"
    )
    semantic_codex.write_entry(entry)

    # Ensure _iter_files finds both
    found_files = [p.name for p in semantic_codex._iter_files()]
    assert "root_doc.json" in found_files
    assert "GEN_DOC.md" in found_files

def test_hot_reload_updates_markdown_entry(semantic_codex, temp_knowledge_dir):
    entry = CodexEntry(
        symbol="UPDATABLE_CONCEPT",
        title="Initial Title",
        body="Initial Body.",
        tags=("initial",),
        version="v1",
        scope="global"
    )
    semantic_codex.write_entry(entry) # Writes and hot_reloads

    # Verify initial load
    loaded_entry = semantic_codex.get("UPDATABLE_CONCEPT")
    assert loaded_entry.title == "Initial Title"

    # Modify the file directly (simulating external change)
    file_path = temp_knowledge_dir / "self_generated_docs" / "UPDATABLE_CONCEPT.md"
    updated_content = """---
symbol: UPDATABLE_CONCEPT
title: Updated Title
tags:
  - updated
version: v1
scope: global
---

Updated Body Content.
"""
    file_path.write_text(updated_content)

    # Trigger hot reload
    changed = semantic_codex.hot_reload()
    assert changed is True

    # Verify the entry is updated in the index
    updated_loaded_entry = semantic_codex.get("UPDATABLE_CONCEPT")
    assert updated_loaded_entry.title == "Updated Title"
    assert updated_loaded_entry.body == "Updated Body Content."
    assert updated_loaded_entry.tags == ("updated",)
