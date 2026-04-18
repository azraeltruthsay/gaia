---
name: palace-recall
description: Search long-term memories in the MemPalace
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: palace_recall
sensitive: false
domain: palace
parameters:
  - name: query
    type: string
    required: true
    description: What to search for in memory
  - name: top_k
    type: integer
    required: false
    description: Maximum number of results (default 3)
---
Search the MemPalace for stored memories. Returns matches ranked by relevance
with cross-references to the Knowledge Graph.
