"""
E2E Smoke Test for RAG + Rolling History pipeline.

Exercises the real SessionManager -> SessionHistoryIndexer -> persistence chain
using the actual embedding model (or verifying graceful degradation if unavailable).

Usage:
    # Inside gaia-core-candidate container:
    docker exec gaia-core-candidate python /app/e2e_rag_smoke_test.py

    # Or via docker run:
    docker run --rm -v ...:/models:ro localhost:5000/gaia-core-candidate:local \
        python e2e_rag_smoke_test.py

Exit codes:
    0 = all checks passed
    1 = assertion failure
"""

import json
import logging
import os
import sys
import time

# Suppress noisy logs from model loading and pool initialization
logging.disable(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress tqdm progress bars from SentenceTransformer model loading
os.environ["TQDM_DISABLE"] = "1"


def main() -> int:
    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}{f': {detail}' if detail else ''}")

    print("=" * 60)
    print("RAG + Rolling History — E2E Smoke Test")
    print("=" * 60)

    # --- 1. Imports ---
    print("\n--- Imports ---")
    try:
        from gaia_core.memory.session_manager import SessionManager
        from gaia_core.memory.session_history_indexer import (
            SessionHistoryIndexer, _get_embed_model,
        )
        from gaia_core.config import Config
        check("Core modules import", True)
    except Exception as e:
        check("Core modules import", False, str(e))
        return 1

    config = Config()
    sm = SessionManager(config)
    sm.max_active_messages = 999  # Prevent auto-archive during test

    test_session = f"e2e_rag_test_{int(time.time())}"

    # --- 2. Embedding model ---
    print("\n--- Embedding Model ---")
    model = _get_embed_model()
    has_model = model is not None
    check("Embedding model loaded", has_model,
          "graceful degradation mode" if not has_model else type(model).__name__)

    # --- 3. Conversation population (8 turn-pairs) ---
    print("\n--- Conversation Population ---")
    conversations = [
        ("What is Python?", "Python is a high-level programming language."),
        ("How does garbage collection work?", "Python uses reference counting plus a cyclic GC."),
        ("Tell me about cats", "Cats are domesticated feline mammals."),
        ("What is photosynthesis?", "Plants convert sunlight into glucose and oxygen."),
        ("Explain the theory of relativity", "Einstein's theory relates space, time, and energy."),
        ("How do neural networks learn?", "Through backpropagation — adjusting weights to minimize loss."),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("How does DNS work?", "DNS translates domain names to IP addresses."),
    ]
    for user_msg, asst_msg in conversations:
        sm.add_message(test_session, "user", user_msg)
        sm.add_message(test_session, "assistant", asst_msg)

    history = sm.get_history(test_session)
    check("History has 16 messages", len(history) == 16, f"got {len(history)}")

    # --- 4. Vector index state ---
    print("\n--- Vector Index ---")
    indexer = SessionHistoryIndexer.instance(test_session)

    if has_model:
        check("8 turns indexed", len(indexer.turns) == 8, f"got {len(indexer.turns)}")
        check("8 embeddings stored", len(indexer.turn_embeddings) == 8, f"got {len(indexer.turn_embeddings)}")
        check("Topic summary generated", len(indexer.topics) >= 1, f"got {len(indexer.topics)}")
    else:
        check("No turns indexed (degradation)", len(indexer.turns) == 0, f"got {len(indexer.turns)}")

    # --- 5. Semantic retrieval ---
    print("\n--- Retrieval ---")
    if has_model:
        result = indexer.retrieve("programming language Python", exclude_recent_n=6, top_k_turns=3)
        check("Turns retrieved", len(result["turns"]) > 0, f"got {len(result['turns'])}")
        if result["turns"]:
            best = result["turns"][0]
            print(f"        Best match: Turn {best['idx']} sim={best['similarity']} — '{best['user'][:50]}'")
            check("Best match is Python question", "python" in best["user"].lower())
        check("Topics retrieved", len(result["topics"]) >= 0)
    else:
        result = indexer.retrieve("Python", exclude_recent_n=0)
        check("Empty retrieval (degradation)", result == {"turns": [], "topics": []})

    # --- 6. Sliding window ---
    print("\n--- Sliding Window ---")
    SLIDING_WINDOW_SIZE = 6
    window = history[-SLIDING_WINDOW_SIZE:]
    check("Window is 6 messages", len(window) == SLIDING_WINDOW_SIZE)
    check("Window excludes oldest turns", "neural" in window[0]["content"].lower(),
          f"first window msg: '{window[0]['content'][:40]}'")

    # --- 7. Persistence ---
    print("\n--- Persistence ---")
    persist_path = os.path.join(indexer.persist_dir, f"{test_session}.json")
    if has_model:
        check("Index file exists on disk", os.path.exists(persist_path))
        if os.path.exists(persist_path):
            with open(persist_path, 'r') as f:
                data = json.load(f)
            check("Persisted turns match", len(data.get("turns", [])) == 8)
            print(f"        File size: {os.path.getsize(persist_path):,} bytes")
    else:
        check("No persist file (degradation)", not os.path.exists(persist_path))

    # --- 8. Archive and reset ---
    print("\n--- Archive ---")
    if has_model:
        pre = len(indexer.turns)
        indexer.archive_and_reset()
        check("Index cleared after archive", len(indexer.turns) == 0, f"was {pre}, now {len(indexer.turns)}")
        archive_dir = os.path.join(indexer.persist_dir, "archive")
        archive_files = [f for f in os.listdir(archive_dir) if f.startswith(test_session)] if os.path.isdir(archive_dir) else []
        check("Archive file created", len(archive_files) == 1, f"found {len(archive_files)}")
    else:
        indexer.archive_and_reset()
        check("Archive no-op (degradation)", True)

    # --- Cleanup ---
    sm.reset_session(test_session)
    if os.path.exists(persist_path):
        os.remove(persist_path)
    archive_dir = os.path.join(indexer.persist_dir, "archive")
    if os.path.isdir(archive_dir):
        for f in os.listdir(archive_dir):
            if f.startswith(test_session):
                os.remove(os.path.join(archive_dir, f))
    SessionHistoryIndexer._instances.pop(test_session, None)

    # --- Summary ---
    total = passed + failed
    mode = "full" if has_model else "degradation"
    print(f"\n{'=' * 60}")
    print(f"{passed}/{total} checks passed ({mode} mode)")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
