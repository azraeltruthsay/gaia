---
name: palace-store
description: Store a memory in the MemPalace long-term knowledge system
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: palace_store
sensitive: false
domain: palace
parameters:
  - name: text
    type: string
    required: true
    description: The memory content to store
  - name: source
    type: string
    required: false
    description: Source attribution (default "model")
---
Store structured knowledge in the MemPalace. Content is classified, compressed
via AAAK dialect, and entity triples are extracted for the Knowledge Graph.
