# Dev Journal Entry: 2026-02-10 - Knowledge Ingestion Pipeline for D&D Campaign Content

**Date:** 2026-02-10
**Author:** Claude Code (Opus 4.6) via Happy

## Context

When D&D information is shared with GAIA via Discord, she responds conversationally but doesn't retain the information long-term. The existing infrastructure had all the primitives — persona detection, RAG retrieval, CodexWriter, vector embedding — but no pipeline to detect incoming knowledge, check for duplicates, format a document, write it, and embed it for future retrieval. This implementation connects those pieces.

Plan file: `/home/azrael/.claude/plans/prancy-gathering-popcorn.md`

## Changes Implemented

### 1. New Module: `knowledge_ingestion.py`

**File:** `gaia-core/gaia_core/cognition/knowledge_ingestion.py` (~270 lines)

Single module with all ingestion logic:

| Function | Purpose |
|----------|---------|
| `detect_save_command(user_input)` | Regex match for explicit save requests ("save this about X", "remember this", legacy DOCUMENT format). Returns `{subject, raw_content}` or `None` |
| `detect_knowledge_dump(user_input, kb_name)` | Heuristic: fires when `kb_name == "dnd_campaign"`, message > 300 chars, contains D&D entity names (Braeneage, BlueShot, etc.) or structural signals (bullets, headers, stat blocks). No LLM call. Returns `bool` |
| `classify_content(text)` | Keyword-based categorization → `{category, tags, suggested_title, suggested_symbol}`. Categories: lore, character, rules, session_recap |
| `check_dedup(content, kb_name)` | Semantic similarity via `mcp_client.embedding_query(content[:500], top_k=1)`. If top hit similarity >= 0.85, returns match info. Otherwise `None` |
| `format_document(content, classification, subject)` | Generates markdown with YAML front matter (symbol, title, category, tags, source, created, version, scope). Filename: `{category}_{sanitized_subject}_{YYYYMMDD}.md` |
| `write_and_embed(filename, doc_content, kb_name)` | Two MCP calls: (1) `write_file` to persist, (2) `embed_documents` to index for RAG |
| `run_explicit_save(user_input, kb_name)` | Full pipeline orchestrator for explicit saves |
| `run_auto_detect(user_input, kb_name)` | Heuristic detection orchestrator for auto-detect path |

### 2. Wired into `agent_core.py` — Location A (after RAG block)

**File:** `gaia-core/gaia_core/cognition/agent_core.py` (~line 852)

After the RAG query and knowledge acquisition workflow, before intent detection:
- When `knowledge_base_name` is set, check `run_explicit_save()` first (explicit > auto)
- Explicit saves: classify → dedup → write_and_embed → yield confirmation → return
- Dedup blocked: yield message about existing document → return
- Auto-detect: `run_auto_detect()` → tag packet with `knowledge_ingestion_offer` DataField → continue normal flow

### 3. Replaced DOCUMENT Handler — Location B

**File:** `gaia-core/gaia_core/cognition/agent_core.py` (~line 1897)

Replaced the rigid regex `GAIA, DOCUMENT "..." AS "..." ABOUT "..."` with a call to `detect_save_command()` from the new module. This consolidates both old-format and new natural-language save commands into one code path. Falls back to CodexWriter for non-knowledge-base contexts.

### 4. Ingestion Offer Hint — `knowledge_enhancer.py`

**File:** `gaia-core/gaia_core/cognition/knowledge_enhancer.py` (~line 70)

When `knowledge_ingestion_offer` is in the packet's data_fields, appends a `system_hint` DataField instructing GAIA to proactively offer to save the information. The prompt builder already injects DataFields into context, so this "just works."

## Data Flow

### Explicit Save Path
```
User: "save this info about the Tower Faction: [content]"
  → run_explicit_save() → detect_save_command() ✓
  → classify_content() → {category: "lore", ...}
  → check_dedup() → None (no duplicate)
  → format_document() → (filename, markdown_with_yaml_frontmatter)
  → write_and_embed() → write_file (MCP) + embed_documents (MCP)
  → yield confirmation token → return
```

### Auto-Detect Path
```
User: [500+ char D&D lore dump with entity names]
  → run_auto_detect() → detect_knowledge_dump() ✓ (entity_hits >= 2)
  → classify_content() → {category: "lore", ...}
  → check_dedup() → None
  → tag packet with knowledge_ingestion_offer DataField
  → knowledge_enhancer injects system_hint
  → GAIA responds conversationally + offers to save
  → User: "yes, save it"
  → Next turn: run_explicit_save() triggers → write + embed
```

## Approval Flow

- **Explicit save**: User intent is clear → `write_file` goes through MCP approval layer
- **Auto-detect**: GAIA only *offers* to save. Actual write fires only when user confirms. Two safety layers: user confirmation + MCP approval.

## Files Modified

| File | Change |
|------|--------|
| `gaia-core/gaia_core/cognition/knowledge_ingestion.py` | **NEW** — all ingestion logic |
| `gaia-core/gaia_core/cognition/agent_core.py` | Wire detection after RAG block + replace DOCUMENT handler |
| `gaia-core/gaia_core/cognition/knowledge_enhancer.py` | Add ingestion offer hint injection |

All mirrored to `candidates/` counterparts.

## Verification Steps

1. **Syntax check**: All three files pass `ast.parse()` — no syntax errors
2. **Unit test detection**: Send D&D messages through `detect_knowledge_dump()` and `detect_save_command()` — verify triggers
3. **Dedup test**: Send content similar to existing `braeneage_general_info.md` — verify dedup catches it
4. **E2E via Discord**: Send a D&D info-dump → GAIA offers to save → confirm → verify doc appears in `knowledge/projects/dnd-campaign/core-documentation/` with YAML front matter
5. **Explicit save test**: "save this info about the Tower Faction: [info]" → verify immediate save
6. **Regression**: Normal questions (no save intent) should work unchanged
