# Improving the Read-Reflect-Respond Loop — 2026-02-28

## Context
During an interaction on Discord, the user requested GAIA to recite "The Raven." While GAIA successfully fetched the text via `web_fetch`, the resulting behavior exhibited "Lazy Echoing"—dumping the raw source text into the chat with minimal synthesis. Additionally, the high volume of text in the tool result triggered a JSON parsing "Extra data" error in the `ToolSelector`, causing a fallback to a "Slim Prompt" which bypassed GAIA's high-rigor reflection stage.

## The Problem
1. **Lazy Synthesis**: The model defaults to pattern-matching the tool output (recitation) rather than fulfilling its role as an analytical assistant.
2. **Parsing Fragility**: The `ToolSelector` regex for JSON extraction was too greedy, causing it to fail when tool results contained complex characters or appended model "thoughts."
3. **Context Pressure**: Large web results consume significant tokens, potentially crowding out the "Reflection" space or hitting output limits.

## Actions Taken (Immediate Fixes)
- **Robust JSON Extraction**: Modified `gaia_core/cognition/tool_selector.py` to use non-greedy regex (`r'\{[\s\S]*?\}'`). This prevents "Extra data" errors when the model appends text outside the JSON block.
- **Increased Token Limits**: 
    - `max_tokens`: Increased from 4096 to **8192** (to allow for full-text reviews).
    - `reflection_max_tokens`: Increased from 2048 to **4096** (to give the "Reflector" more room to analyze large inputs).
- **Candidate Sync**: Mirrored these changes to the `candidates/` directory to ensure development parity.

## Proposed Plan: The "Deep Read" Enhancement

### Phase 1: Structured Tool Output (gaia-mcp)
Enhance the `web_fetch` tool result to include a "Metadata Header." By providing the model with `word_count`, `domain_trust`, and `title` as distinct fields, we help it "orient" before it dives into the raw content.

### Phase 2: Synthesis Scaffolding (prompt_builder.py)
Update the `PromptBuilder` to inject stronger "Anti-Recitation" constraints when tool results are present:
- **Mandatory Synthesis**: Add an instruction: *"When presenting tool results, do not dump raw content. Provide a synthesis that highlights relevance to the user's goal, citing specific sections only as needed for validation."*
- **Assistant Prefill Steering**: Refine the `"Based on the results,"` prefill to specifically steer toward analysis: *"Based on my review of the source from [domain]..."*

### Phase 3: The "Validator" persona check
Update the `Reflector` persona instructions in `gaia_constants.json` to explicitly check for "Recitation vs. Synthesis." If the Reflector detects a raw data dump, it should trigger a "Refinement" turn to condense the output.

### Phase 4: Summarization Fallback
If `web_fetch` results exceed a certain token threshold (e.g., > 4000 tokens), implement an automatic "Summary-First" pass where a Lite model generates a TL;DR that is prepended to the full text, allowing the Prime model to use the summary for its primary reasoning.

## Success Criteria
- GAIA can review a 50KB research paper and provide a 4-paragraph synthesis without cutting off.
- GAIA responds to "Recite [X]" with a structured presentation (e.g., "Here is the text you requested, noted for its [thematic element]...") rather than a raw copy-paste.
- Zero "Extra data" parsing errors in `ToolSelector` logs during large-scale data retrieval.
