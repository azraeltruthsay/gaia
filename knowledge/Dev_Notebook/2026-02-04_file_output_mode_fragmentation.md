# GAIA Development Journal

## Date: 2026-02-04

### Subject: File Output Mode for Fragmented Generation

**Summary:**

Added a file output mode to the fragmentation/reassembly system. When GAIA performs long-form content generation (like reciting "The Raven"), the assembled content is now written to a file in `/sandbox/` rather than being returned directly as a chat response.

**Problem:**

The existing fragmentation system worked as follows:
1. Generate content in fragments, storing each to the sketchpad
2. Run an assembly turn where the model reads and assembles its own fragments
3. Return the assembled content as a chat response string

This worked but had a UX issue: long-form content (poems, documents, etc.) would flood the chat with thousands of characters. The user wanted the content delivered as a **file artifact** that could be viewed separately.

**Solution:**

Added `output_as_file` mode to the fragmentation system:

1. **New parameters** on `_run_with_fragmentation`:
   - `output_as_file: bool = False` - When True, writes to file instead of returning as response
   - `output_filename: Optional[str] = None` - Optional explicit filename (auto-generates if not provided)

2. **New method `_write_assembled_to_file`**:
   - Extracts a meaningful filename from the request (looks for quoted titles, "The X" patterns)
   - Sanitizes filename: lowercase, underscores, no special chars
   - Writes to `/sandbox/{filename}.txt` using `mcp_client.ai_write()`
   - Returns a formatted message with file path and size
   - Falls back gracefully if file write fails

3. **Updated callers** - Both recitation paths now use file output:
   - `_run_with_document_recitation` - for known documents loaded from knowledge base
   - `_run_with_fragmentation` - for confidence-based recitation from model memory

**Files Changed:**

- `gaia-core/gaia_core/cognition/agent_core.py`:
  - Modified `_run_with_fragmentation` signature (lines ~1944-1975)
  - Added `_write_assembled_to_file` method (lines ~2277-2338)
  - Modified `_run_with_document_recitation` signature (lines ~1849-1872)
  - Updated both callers to pass `output_as_file=True` (lines ~1732, ~1778)

**New Flow:**

```
User: "Recite The Raven by Edgar Allan Poe"
  │
  ▼
[Intent Detection] → recitation intent detected
  │
  ▼
[Find Recitable Document] → found in knowledge base OR assess confidence
  │
  ▼
[Fragmented Generation]
  ├── Fragment 1 → sketchpad_write("recitation_fragment_abc123_0", content)
  ├── Fragment 2 → sketchpad_write("recitation_fragment_abc123_1", content)
  └── Fragment N → ...
  │
  ▼
[Assembly Turn]
  └── Model reads all fragments, removes overlaps, assembles clean output
  │
  ▼
[File Output Mode] ← NEW
  ├── Generate filename: "the_raven.txt"
  ├── Write to: /sandbox/the_raven.txt
  └── Return: file header + full content
  │
  ▼
User sees: Brief file notification header + the full assembled content
```

**Example Output:**

```
*[Saved to `/sandbox/the_raven.txt` (6,789 bytes)]*

Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
...
[full poem content follows]
```

**Design Decisions:**

1. **Dual output mode**: Returns BOTH the actual content (for Discord/chat display) AND saves to file (for persistence/download). The file path is shown as a subtle header.

2. **Filename extraction**: Uses regex to find quoted strings or "The X" patterns in the request. Falls back to `assembled_content_{request_id}.txt` if no title found.

3. **Sandbox location**: All files go to `/sandbox/` which is the designated safe write area for GAIA.

4. **Fallback behavior**: If `ai_write` fails, the method returns the content directly with an error note rather than failing silently.

5. **Always-on for recitation**: Both recitation paths now default to file output. This could be made configurable per-request if needed.

**Testing:**

To test inside the container:
```bash
docker exec -it gaia-assistant python3 gaia_rescue.py
# Then try: "Recite The Raven by Edgar Allan Poe"
# Should see file notification instead of raw poem
# Then: cat /sandbox/the_raven.txt
```

**Future Considerations:**

- Could add a user preference to toggle file vs inline output
- Could support different output formats (markdown, HTML, PDF)
- Could add sketchpad cleanup after successful file write
- Could notify via Discord/web with file attachment

---
