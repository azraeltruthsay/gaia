# GAIA Error Codes Reference (`gaia_errors.md`)

This document maps all assigned error codes to their descriptions and suggested debugging steps.

---

## Core AI Manager (ERR-GAIA-1xxx)

| Error Code | Description | Suggested Action |
|:-----------|:------------|:-----------------|
| ERR-GAIA-1001 | Failed to load core instructions or default personality file | Check file path and permissions. Ensure default_personality.json exists. |
| ERR-GAIA-1002 | Failed to initialize the language model (LLM) | Confirm model path, model format (gguf), and hardware support. |
| ERR-GAIA-1003 | Failed to load or create vector store | Inspect ChromaDB config and document availability. |
| ERR-GAIA-1004 | AI Manager startup sequence crashed | Check detailed logs for root exception. |

---

## Web Interface and API (ERR-GAIA-2xxx)

| Error Code | Description | Suggested Action |
|:-----------|:------------|:-----------------|
| ERR-GAIA-2001 | /api/status route failure | Check if AI_MANAGER is initialized before accessing status. |
| ERR-GAIA-2002 | /api/query route failure | Verify AI_MANAGER and behavior manager are operational. |
| ERR-GAIA-2003 | Missing or invalid request body in API call | Inspect API client and ensure correct POST body. |

---

## Background Processor (ERR-GAIA-3xxx)

| Error Code | Description | Suggested Action |
|:-----------|:------------|:-----------------|
| ERR-GAIA-3001 | Failed to process conversation summary in background | Review conversation archive system and permissions. |
| ERR-GAIA-3002 | Failed to embed new documents in background | Check document processor and vector store manager. |

---

## Vector Store and Embeddings (ERR-GAIA-4xxx)

| Error Code | Description | Suggested Action |
|:-----------|:------------|:-----------------|
| ERR-GAIA-4001 | Vector similarity search failure | Ensure vector database is populated and online. |
| ERR-GAIA-4002 | Embedding model failed to encode document | Validate Huggingface model and retry embeddings. |

---

## Document Processing (ERR-GAIA-5xxx)

| Error Code | Description | Suggested Action |
|:-----------|:------------|:-----------------|
| ERR-GAIA-5001 | Raw file parsing failed | Validate file type, encoding, and content structure. |
| ERR-GAIA-5002 | Markdown generation failure | Inspect document processing pipelines. |

---

## General Internal System (ERR-GAIA-9xxx)

| Error Code | Description | Suggested Action |
|:-----------|:------------|:-----------------|
| ERR-GAIA-9001 | Unhandled exception in unknown module | Check logs for full traceback, improve exception handling. |
| ERR-GAIA-9002 | Critical service missing during boot | Verify docker-compose volumes, models, and environment variables. |

---

# Notes:
- New modules must reserve an error code block.
- Every critical exception should log an associated error code.
- Maintain this document as GAIA evolves!

