---
name: file-write
description: Write content to a file on the filesystem
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: write_file
sensitive: true
domain: file
parameters:
  - name: path
    type: string
    required: true
    description: Absolute path to write to
  - name: content
    type: string
    required: true
    description: Content to write
---
Write content to a file. Requires approval. Paths restricted to safe directories.
