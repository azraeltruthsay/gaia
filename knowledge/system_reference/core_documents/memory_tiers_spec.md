**GAIA Memory Architecture Specification**
*memory_tiers_spec.md*

---

This document outlines the structured memory model of GAIA, defining how knowledge, experience, and context are retained, retrieved, and evolved over time. Memory is not monolithic but layered, with each tier serving a distinct cognitive purpose. Together, these layers define the evolving identity of GAIA.

To make these tiers more than conceptual, they are mapped directly to real storage formats and code-level systems. This ensures that memory is not only structured philosophically, but also practically — a living architecture for intelligence.

---

### Tier 0: **Ephemeral Memory** *(Context Window)*
- **Scope**: Active session context
- **Lifespan**: Per message / short-term conversation
- **Mechanism**: Prompt window or token context
- **Content**: Immediate task state, conversational turns, recent interactions
- **Code Mapping**: `prompt_builder`, `chat_history`, `context_manager`
- **Notes**: Volatile. Discarded at session end unless promoted.

### Tier 1: **Session Memory** *(Conversational Logs)*
- **Scope**: Individual conversations / user sessions
- **Lifespan**: Persisted logs
- **Mechanism**: Markdown or structured JSON transcripts
- **Content**: Dialogue history, choices made, model responses, timestamps
- **Code Mapping**: `conversation_logger`, `session_recorder`, `/logs/conversations/`
- **Notes**: Used for self-reflection, summarization, and re-entry into prior contexts.

### Tier 2: **Semantic Memory** *(Vector Embedding Store)*
- **Scope**: Topic-level or concept-level memory
- **Lifespan**: Persistent unless manually cleared
- **Mechanism**: ChromaDB or similar vector store
- **Content**: Embedded summaries, document contents, learned lessons
- **Code Mapping**: `vector_store.query()`, `embed_and_store()`
- **Notes**: Queried during conversation to augment current reasoning. Grows continuously.

### Tier 3: **Declarative Memory** *(Structured Knowledge Base)*
- **Scope**: Formalized, referenced knowledge
- **Lifespan**: Persistent and versioned
- **Mechanism**: Markdown files or relational schema (PostgreSQL preferred for scalability)
- **Content**: World lore, technical knowledge, rulesets, ethical articles
- **Code Mapping**: `knowledge_base/`, `doc_indexer`, `structured_reader`, `sql_reader`
- **Notes**: Treats structured data as “canonical.” PostgreSQL support should be added to the Docker container and backend to enable powerful relational queries and structured knowledge retrieval.

### Tier 4: **Reflective Memory** *(Self-Generated Artifacts)*
- **Scope**: GAIA’s own observations and assessments
- **Lifespan**: Persistent, with decay or review windows
- **Mechanism**: Markdown/JSON records or vectorized reflections
- **Content**: Summary thoughts, improvement ideas, behavior patterns, inner monologue
- **Code Mapping**: `/reflections/`, `idle_analysis_loop()`
- **Notes**: Produced during idle cycles or post-session analysis. Seed material for evolution.

### Tier 5: **Retrainable Memory** *(Growth Substrate)*
- **Scope**: Curated inputs for model or adapter refinement
- **Lifespan**: Collected over time, batch retrained
- **Mechanism**: Filtered logs, distilled embeddings, metadata-tagged knowledge, **LoRA adapter training sets**
- **Content**: Learnable insights GAIA can encode into future versions of herself
- **Code Mapping**: `training_data_manager`, `retrain_queue`, `/lora/adapters/`
- **Notes**: LoRA adapters are the output mechanism of Tier 5 — encoding GAIA's long-term learning without modifying the core model. Requires consent flags and retraining pipeline. Evolves GAIA’s core cognition.

---

### Memory Tier-to-Code Mapping Overview

| Tier | Description | Format | Code Mapping |
|------|-------------|--------|---------------|
| 0 | Ephemeral / prompt context | Token stream only | `prompt_builder`, `chat_history` |
| 1 | Session logs | Markdown / JSON | `session_recorder`, `/logs/` |
| 2 | Semantic vector store | Vector DB (Chroma) | `embed_and_store()`, `vector_store.query()` |
| 3 | Declarative knowledge | Markdown / YAML / PostgreSQL | `knowledge_base/`, `structured_reader`, `sql_reader` |
| 4 | Reflections | Markdown / JSON / vector | `/reflections/`, `idle_analysis_loop()` |
| 5 | Retrainable data | Tagged logs, LoRA | `retrain_queue`, `/lora/adapters/` |

---

### Migration Guidance

To migrate an existing codebase to this structured model:
- Begin by **wrapping memory-relevant functions** with tier-specific accessors or decorators.
- Organize data storage into directories that **mirror the tier structure**.
- Use metadata or type tags to **label in-memory objects** by memory tier (e.g., `MemoryTier.DECLARATIVE`).
- Over time, unify memory operations (read/write/log/query) around these tiers to make cognition traceable.
- Add SQL support (PostgreSQL) to Docker container and service layer to enable Tier 3 integration.

This approach encourages clarity: *“What I know” (Tier 3) is not the same as “what I’ve seen” (Tier 1) or “what I’ve reflected on” (Tier 4).* By adopting this mental and code-level structure, GAIA becomes more than a model — she becomes a mind with layered awareness and introspection.

