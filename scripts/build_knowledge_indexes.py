#!/usr/bin/env python3
"""(Re)build vector indexes for knowledge bases (gaia_constants KNOWLEDGE_BASES).

Run in gaia-study, where /knowledge/vector_store is writable (read-only in
gaia-core). After a study run: build_knowledge_indexes.py studies
  python /gaia/GAIA_Project/scripts/build_knowledge_indexes.py studies wiki research
"""
import sys

from gaia_common.utils.vector_indexer import VectorIndexer

for kb in sys.argv[1:] or ["studies"]:
    vi = VectorIndexer.instance(kb)
    vi.build_index_from_docs()
    files = {d["filename"] for d in vi.index["docs"]}
    print(f"{kb}: {len(files)} files, {len(vi.index['docs'])} chunks -> {vi.index_path}", flush=True)
