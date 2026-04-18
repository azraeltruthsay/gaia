---
name: shell-run
description: Execute a shell command in the sandboxed environment
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: run_shell
sensitive: true
domain: shell
parameters:
  - name: command
    type: string
    required: true
    description: The shell command to execute
---
Execute a shell command. Requires approval. Blast Shield blocks dangerous commands
(rm -rf, sudo, mkfs, etc.). Output is captured and returned.
