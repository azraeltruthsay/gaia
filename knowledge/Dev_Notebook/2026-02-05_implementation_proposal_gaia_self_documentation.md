## Implementation Proposal: GAIA's Self-Generated Long-Term Formal Documentation

### Objective

To enhance GAIA's learning capabilities by enabling her to autonomously create, persistently store, and effectively retrieve her own formal documentation, thereby extending her knowledge base beyond session-specific memory and pre-defined datasets.

### Core Components to Modify/Create

The implementation will focus on extending existing components and introducing new modules within the `gaia-core` service.

1.  **Extend `SemanticCodex` (`gaia-core/gaia_core/memory/semantic_codex.py`)**
    *   **`write_entry(codex_entry: CodexEntry) -> Path` Method:**
        *   **Purpose:** This new public method will be responsible for serializing a `CodexEntry` object into a Markdown file with YAML front matter and saving it to a designated persistent storage location.
        *   **Mechanism:** It will construct the Markdown file content, including a YAML front matter block (containing `symbol`, `title`, `tags`, `version`, `scope`) and the `body` as Markdown. The filename will be derived from the `symbol` for uniqueness (e.g., `knowledge/self_generated_docs/GAIA_concept_example.md`).
        *   **Return Value:** Returns the `Path` to the newly created document.
    *   **Enhanced Loading (`_load_one` method modification):**
        *   **Purpose:** To enable `SemanticCodex` to correctly parse and load the self-generated Markdown files, which will feature YAML front matter.
        *   **Mechanism:** The existing `_load_one` method will be updated to first check for YAML front matter in `.md` files. If present, it will extract the metadata for the `CodexEntry` fields (symbol, title, tags, etc.) and treat the remaining content as the `body`. If no YAML front matter, it will continue with existing JSON/YAML parsing.
    *   **Configuration (`Config` update):**
        *   **Purpose:** To inform `SemanticCodex` about the new directory for self-generated documents.
        *   **Mechanism:** The `Config` object will be updated to include `knowledge/self_generated_docs/` as part of the directories scanned by `SemanticCodex`, or a dedicated configuration variable will point to this new location.
    *   **Dependency:** Ensure `PyYAML` is a declared dependency for `gaia-core` to handle YAML front matter.

2.  **New `CodexWriter` Module (`gaia-core/gaia_core/memory/codex_writer.py`)**
    *   **Purpose:** This new module will encapsulate the high-level logic for initiating and managing the documentation process. It will act as an interface between `AgentCore` and `SemanticCodex`.
    *   **Key Functions/Classes:**
        *   `CodexWriter.document_information(packet: CognitionPacket, info_to_document: str, symbol: str, title: str, tags: List[str]) -> CodexEntry`
        *   This function will orchestrate:
            *   Taking relevant information (e.g., from `packet.content.original_prompt`, `packet.response.candidate`, or specific `info_to_document`).
            *   Possibly engaging an LLM to refine or summarize `info_to_document` into a concise `body` suitable for a `CodexEntry`.
            *   Creating a `CodexEntry` object.
            *   Calling `SemanticCodex.write_entry` to persist the document.
            *   Handling potential errors or conflicts.

3.  **Integration with `AgentCore` (`gaia-core/gaia_core/cognition/agent_core.py`)**
    *   **User-Initiated Documentation:**
        *   **Mechanism:** Introduce a new explicit user command (e.g., a special directive or tool call) that, when detected, triggers `CodexWriter.document_information`. This could be parsed within `_run_slim_prompt` or as part of the intent detection.
    *   **Autonomous Documentation Trigger (Future/Stretch Goal):**
        *   **Mechanism:** Implement heuristics or a dedicated LLM prompt within `AgentCore.run_turn` to identify "document-worthy" information from the current turn's `CognitionPacket` (e.g., after a successful tool execution, a particularly insightful reflection, or the resolution of a complex query). If triggered, `CodexWriter.document_information` would be called.

4.  **Knowledge Retrieval Integration (Updates to existing mechanisms)**
    *   **Purpose:** Ensure that self-generated documents are discoverable by GAIA's existing knowledge retrieval systems.
    *   **Mechanism:** Since `SemanticCodex` will load these files, `semantic_codex.get()` and `semantic_codex.search()` will automatically include them. Further, if a vector store is used for RAG, a process to automatically index these new Markdown files into the vector store will be required.

### Technical Details & Considerations

*   **File Naming Convention:** A consistent naming convention for self-generated Markdown files (e.g., `GAIA_concept_SYMBOL.md` or `GAIA_solution_TIMESTAMP.md`) to avoid collisions and aid organization.
*   **Symbol Uniqueness:** Mechanisms to ensure the `symbol` for each `CodexEntry` is unique or handles conflicts gracefully.
*   **LLM Prompting for Documentation:** Designing effective prompts for GAIA's LLM to generate high-quality, concise, and structured documentation content. This will be critical for the "formal" aspect.
*   **Error Handling:** Robust error handling for file I/O, parsing, and LLM generation.
*   **Metadata Validation:** Ensure generated metadata (title, tags) adheres to expected formats.
*   **Scalability:** Consider the impact of a rapidly growing number of Markdown files on `SemanticCodex` loading times and overall search performance. Pre-indexing or lazy loading strategies might be necessary in the long term.

### Phased Approach

The implementation will proceed incrementally:

1.  **Phase 1: Core `SemanticCodex` Functionality:**
    *   Add `PyYAML` dependency.
    *   Implement `SemanticCodex.write_entry` method.
    *   Update `SemanticCodex._load_one` to parse Markdown with YAML front matter.
    *   Write unit tests for these new functionalities.
2.  **Phase 2: User-Initiated Documentation:**
    *   Create the `CodexWriter` module with its core `document_information` method.
    *   Integrate a user command/tool call in `AgentCore` to trigger `CodexWriter`.
    *   Develop an initial LLM prompt for documentation generation.
3.  **Phase 3: Autonomous Triggering & Refinement:**
    *   Implement heuristics or a reflective mechanism in `AgentCore` to autonomously identify documentation opportunities.
    *   Refine LLM prompts for higher quality documentation.
    *   Integrate with vector stores if applicable.
