# GAIA Service Blueprint: `gaia-study` (The Subconscious)

## Role and Overview

`gaia-study` is the GPU-enabled background processing service for knowledge management and model fine-tuning. It is the **sole writer** to the vector store, ensuring data consistency. It generates embeddings, builds search indices, and manages QLoRA adapter training. Other services (primarily `gaia-core`) have read-only access to the vector store.

## Container Configuration

**Base Image**: `nvidia/cuda:12.4.0-devel-ubuntu22.04` (GPU-enabled with CUDA toolkit)

**Port**: 8766 (live), 8768 (candidate)

**Health Check**: `curl -f http://localhost:8766/health` (30s interval, 60s start_period)

**Startup**: `uvicorn gaia_study.main:app --host 0.0.0.0 --port 8766`

### GPU Allocation

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Note**: On single-GPU systems, this creates contention with `gaia-prime` (which claims 1 GPU). May need CPU-only embedding mode or careful scheduling via `gaia-orchestrator`.

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `KNOWLEDGE_DIR` | `/knowledge` | Knowledge base directory |
| `VECTOR_STORE_PATH` | `/vector_store` | Vector index storage |
| `MODELS_DIR` | `/models` | Model files directory |

### Volume Mounts

- `./gaia-study:/app:rw` — Source code
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:rw` — Full knowledge base (for indexing)
- `gaia-vector-store:/vector_store:rw` — Vector index (**sole writer**)
- `./gaia-models:/models:rw` — Model files + LoRA adapters
- `gaia-shared:/shared:rw` — Shared state

## Internal Architecture

### Key Components

1. **`main.py`** — FastAPI entry point, initializes VectorIndexer

2. **`indexer.py`** (VectorIndexer) — Core vector store manager
   - Singleton per knowledge base
   - **Sole writer** pattern for vector store integrity
   - Uses `sentence-transformers` for embedding generation (default: `all-MiniLM-L6-v2`)
   - Index format: JSON with documents, embeddings, metadata
   - Methods:
     - `build_index_from_docs(directory)` — Initial full index build
     - `add_document(file_path)` — Incremental document addition
     - `query(query_text, top_k=5)` — Similarity search
     - `save_index()` — Persist to JSON

3. **`qlora_trainer.py`** (QLoRA Trainer) — Memory-efficient fine-tuning
   - Targets consumer GPUs (RTX 5080 16GB)
   - Lazy imports for heavy libraries (transformers, PEFT, bitsandbytes)
   - **QLoRAConfig**:
     - Load in 4-bit quantization (bitsandbytes)
     - LoRA rank=8, alpha=16, dropout=0.05
     - Batch size=1, gradient accumulation=4
     - Learning rate=2e-4, max_steps=100
     - Max sequence length=512

4. **`study_mode_manager.py`** — Controls study/training modes, coordinates with orchestrator

5. **`training_utils.py`** — Helper functions for model training

## Data Flow

1. **Ingestion**: Receives documents from knowledge directory or API
2. **Preprocessing**: Clean, structure, tokenize
3. **Embedding**: Generate vectors via sentence-transformers
4. **Indexing**: Store in vector database (JSON-backed)
5. **Fine-tuning** (conditional): QLoRA adapter training when triggered
6. **Availability**: Updated index available to gaia-core for RAG queries

## Interaction with Other Services

- **`gaia-core`** (reader): Performs read-only queries against the vector store for RAG
- **`gaia-orchestrator`** (coordinator): Coordinates GPU handoffs between study and prime
- **`gaia-common`** (library): Shared data structures and utilities

## Sole Writer Pattern

`gaia-study` has exclusive write access to `/vector_store`. This prevents concurrent write corruption:
- `gaia-study`: RW access to vector store
- `gaia-core`: RO access via `VectorClient` (from `gaia-common`)
- All other services: No direct vector store access
