# 2026-04-08 — Design: Architectural RAG & Self-Exploration

## Context
GAIA is a 13-service SOA, but her "Self-Knowledge" is currently limited to high-level markdown files. She lacks a granular, code-level understanding of her own interfaces and internal dependencies.

## Objective
Index the **Blueprint YAMLs** and **AST Summaries** of all services into a specialized `code_architecture` vector collection.

## The Pipeline
1.  **Indexing**: Use the existing `generate_blueprint` tool to create YAMLs for all 13 services.
2.  **AST Extraction**: Create a script to extract top-level function signatures and docstrings from all `/app` and `/gaia_service` directories.
3.  **Vectorization**: Index the YAMLs and ASTs into ChromaDB with a `system_architecture` metadata tag.

## Implementation Tasks (Action for Claude)
1.  **Blueprinting**: Run `generate_blueprint` across the entire project root.
2.  **Indexing**: Use `gaia-study` to embed the `contracts/services/*.yaml` files.
3.  **Stage 0 Update**: Update the `Neural Grounding` logic to always probe the `code_architecture` collection when the intent is `identity` or `architecture`.

## Strategic Impact
This gives GAIA a "Living Map" of herself. She will no longer need to "guess" which port a service is on or what a specific internal function does — she will simply *know* by looking it up in her own architectural RAG.
