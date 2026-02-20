# gaia-study — The Subconscious

Background processing service. SOLE WRITER to the vector store and LoRA adapters. Runs on GPU when gaia-prime is asleep.

## Responsibilities

- Index knowledge base documents into the vector store
- Generate and update sentence embeddings
- Run LoRA fine-tuning on conversation data
- Execute scheduled sleep tasks (codebase analysis, knowledge summarization)
- Manage the evolving conversation summaries

## SOLE WRITER Principle

gaia-study is the **only service** with write access to:

- `vector_store/` — sentence embeddings and FAISS indices
- `gaia-models/` — LoRA adapter checkpoints

All other services mount these as read-only. This eliminates write contention and ensures index consistency.

## GPU Time-Sharing

gaia-study shares the GPU with gaia-prime through the orchestrator's handoff protocol:

1. gaia-core's sleep cycle triggers → orchestrator puts gaia-prime to sleep
2. Orchestrator hands GPU lease to gaia-study
3. gaia-study runs training/embedding tasks
4. On wake signal → orchestrator reclaims GPU for gaia-prime

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `VECTOR_STORE_PATH` | `/vector_store` | FAISS index location |
| `MODELS_DIR` | `/models` | LoRA adapter output |
| `SHARED_DIR` | `/shared` | Inter-service state |
