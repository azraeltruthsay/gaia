---
name: file-read
description: Read the contents of a file from the filesystem
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: read_file
sensitive: false
domain: file
parameters:
  - name: path
    type: string
    required: true
    description: Absolute path to the file to read
---
Read a file and return its contents. Supports text files, JSON, YAML, and markdown.
Paths must be within allowed directories (/knowledge, /sandbox, /models).
